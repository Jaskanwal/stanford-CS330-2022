from cProfile import label
from multiprocessing.sharedctypes import Value
from typing import Dict, List, Optional, Tuple
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import transformers
import numpy as np
import random

import argparse
from collections import defaultdict
import json
import os
from rouge_score import rouge_scorer
import tqdm

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--task')
parser.add_argument('--model')
parser.add_argument('--dataset')
parser.add_argument('--k')
parser.add_argument('--prompt', default='qa')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--repeats', default=1, type=int)
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()


DEVICE = torch.device(args.device)


def get_icl_prompts(
    support_inputs: List[str],
    support_labels: List[str],
    test_input: str,
    prompt_mode: str = 'qa') -> str:
    """
    Take a list of contexts and combine them into k-shot prompts.

    **Note**: Be sure to shuffle the support examples and labels 
      *together* (i.e. so the pairings of support input/label is preserved)
      before constructing the prompt. np.random.permutation may be helpful.

    Args:
      support_inputs: The k inputs used for in-context learning (k may be zero!)
      support_labels: The k labels used for in-context learning (k may be zero!)
      test_input: The input we are evaluating on
      prompt_mode: The task description mode we're using; 'none' means we're only using
        k-shot examples, 'tl;dr' means we're using the tl;dr prompt from the GPT-2 paper,
        etc.

    Returns:
      A string containing the complete input to the model.
    """
    # YOUR CODE HERE
    prompt = ''
    k = len(support_inputs)
    if k > 0:
        indices = list(range(k))
        np.random.shuffle(indices) # Note 2 - shuffle the order of inputs and targets
        for idx in indices:
            if prompt_mode == "qa":
                prompt += support_inputs[idx] + " In the " + support_labels[idx] + "."
            elif prompt_mode == "none":
                prompt += support_inputs[idx] + " " + support_labels[idx]
            elif prompt_mode == "tldr":
                prompt += support_inputs[idx] + " TL;DR: " + support_labels[idx]
            elif prompt_mode == "custom":
                prompt += "passage: " + support_inputs[idx] + " summary: " + support_labels[idx]

    if prompt_mode == "qa":
        prompt += " " + test_input + " In the" # Refer to Note 1 in assignment - no space at the end of prompt
    elif prompt_mode == "none":
        prompt += " " + test_input
    elif prompt_mode == "tldr":
        prompt += " " + test_input + " TL;DR:"
    elif prompt_mode == "custom":
        prompt += "passage: " + test_input + " summary:"

    return prompt


def get_performance_metric(predictions: List[str], targets: List[str], metric: str) -> float:
    if metric == 'rouge':
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        scores = []
        for p, t in zip(predictions, targets):
            score = scorer.score(p, t)['rouge1'].fmeasure
            scores.append(score)
        return sum(scores) / len(scores)
    elif metric == 'exact match':
        if isinstance(targets[0], str):
            return sum([p.strip() == t.strip() for p, t in zip(predictions, targets)]) / len(predictions)
        else:
            def _normalize(prediction):
                if prediction.endswith('Q'):
                    prediction = prediction[:-1]
                elif 'Q:' in prediction:
                    prediction = prediction[:prediction.index('Q:')]
                return prediction.strip('. ').lower()

            normalized = [_normalize(p) for p in predictions]
            def contains(key, candidates):
                for c in candidates:
                    if key in c:
                        return True
                return False

            return sum([contains(n, t) for n, t in zip(normalized, targets)]) / len(normalized)
    else:
        raise NotImplementedError()


def do_sample(model, input_ids, stop_tokens, max_tokens):
    """
    Sample from the model using the given input_ids as a prefix until we either
    hit the stop token or we have sampled max_tokens tokens.

    (Don't use model.generate; implement this yourself in a loop)

    Note: when calling the model here, be sure to wrap the call with
      torch.inferece_mode() to save memory!

    Args:
        model: A transformers.PreTrainedModel that we will sample from.
        input_ids: An integer tensor of shape [1, prefix_len]
        stop_tokens: A list of token ids that indicates that we should stop sampling (e.g., a period)
        max_tokens: Stop sampling if we've sampled this many tokens
    
    Returns:
        The sampled tokens (a python list of ints/zero-dim tensors), not including the input_ids prefix
          OR the stop token (if we hit the stop token before max_tokens)
    """
    # YOUR CODE HERE
    sampled_tokens = []

    for _ in range(max_tokens):
        with torch.inference_mode():
            output = model(input_ids)
        logits = output["logits"]
        greedy_token = logits[0, -1, :].argmax(dim=-1)

        if greedy_token.item() in stop_tokens:
            break
        else:
            sampled_tokens.append(greedy_token.item())

        input_ids = torch.cat([input_ids, greedy_token.view(1, -1)], dim=-1)
    return sampled_tokens


def run_icl(models: List[str], datasets_: List[str], ks: List[int], prompt_modes: List[str], n_val: int = 125):
    results = {}
    for model_name in models:
        print(f'Loading model {model_name}...')
        model, tokenizer = utils.get_model_and_tokenizer(model_name, transformers.AutoModelForCausalLM)
        stop_tokens = utils.stop_tokens(tokenizer)
        model.to(DEVICE)

        for dataset in datasets_:
            print(f'Loading dataset {dataset}...')
            if args.debug:
                n_val = 1
            max_tokens = utils.max_sampled_tokens_for_dataset(dataset)
            train, val = utils.get_dataset(dataset, n_train=max(ks), n_val=n_val)
            for prompt_mode in prompt_modes:
                for k in ks:
                    print(f'Running in-context learning with {model_name} on {dataset} with k={k} and prompt_mode={prompt_mode}')
                    for repeat in range(args.repeats):
                        if repeat > 0:
                            print(f'Beginning repeat #{repeat}')
                        support_idxs = random.choices(range(len(train['x'])), k=k)
                        support_x = [train['x'][idx].replace('\n', ' ') for idx in support_idxs]
                        support_y = [train['simple_y'][idx].replace('\n', ' ') for idx in support_idxs]
                        targets = []
                        predictions = []
                        pbar = tqdm.tqdm(list(range(min(n_val, len(val['x'])))))
                        for row in pbar:
                            test_input = val['x'][row]
                            targets.append(val['y'][row])

                            # Ingredients you'll need:
                            #   get_icl_prompts() [which you implemented]
                            #   do_sample() [which you implemented]
                            #   tokenizer() (for encoding text into tokens) and tokenizer.decode() (for decoding tokens back into text)
                            #   See the documentation for the tokenizer encoder function here:
                            #   https://huggingface.co/docs/transformers/v4.23.1/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__
                            # Note that the tokenizer by default will give you results on the CPU, so you will need to move them to the
                            # proper device.
                            # YOUR CODE HERE

                            prompt = get_icl_prompts(
                                support_inputs=support_x,
                                support_labels=support_y,
                                test_input=test_input,
                                prompt_mode=prompt_mode)
                            tokens = tokenizer.encode(prompt, return_tensors='pt')
                            tokens = tokens.to(DEVICE)
                            sampled_tokens = do_sample(model, tokens, stop_tokens, max_tokens)
                            decoded_prediction = tokenizer.decode(sampled_tokens, skip_special_tokens=True)

                            # END YOUR CODE

                            predictions.append(decoded_prediction)
                            metric = get_performance_metric(predictions, targets, utils.metric_for_dataset(dataset))
                            pbar.set_description(f'Eval: {metric:.04f}')
                        results['_'.join([model_name, dataset, str(k), prompt_mode])] = metric

                        print('Evaluation results:', results)
                        if not os.path.exists('results/icl'):
                            os.makedirs('results/icl')

                        for k_, v in results.items():
                            with open(f'results/icl/{k_}.json', 'w') as f:
                                json.dump({'metric': v}, f)
                        results = {}


def plot(models, dataset, ks, prompt_modes):
    data = defaultdict(lambda: defaultdict(list))
    symbols = ['solid', 'dashed', 'dotted', 'dashdot']

    x_vals = set()
    for model in models:
        symbol = symbols.pop(0)
        for prompt_mode in prompt_modes:
            for k in ks:
                fn = '_'.join([model, dataset, str(k), prompt_mode])
                id_ = '_'.join([model, dataset, prompt_mode])
                with open(f'results/icl/{fn}.json', 'r') as f:
                    score = json.load(f)['metric']
                    data[id_]['x'].append(k)
                    x_vals.add(k)
                    data[id_]['y'].append(score)
                    data[id_]['linestyle'] = symbol

    for k, v in data.items():
        plt.plot(v['x'], v['y'], label=k, linestyle=v['linestyle'])

    if max(x_vals) > 4:
        plt.xscale('symlog')
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_ticks(v['x'])
    plt.legend()
    plt.title(dataset)
    plt.ylabel(utils.metric_for_dataset(dataset))
    plt.xlabel('Number of support examples')
    plt.show()


def run():
    ks = [int(k) for k in args.k.split(',')]
    if args.task == 'icl':
        run_icl(args.model.split(','), args.dataset.split(','), ks, args.prompt.split(','))
    elif args.task == 'plot':
        assert ',' not in args.dataset, "Only one dataset at a time for plotting"
        plot(args.model.split(','), args.dataset, ks, args.prompt.split(','))


if __name__ == '__main__':
    run()