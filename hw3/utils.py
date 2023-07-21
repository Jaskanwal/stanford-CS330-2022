from collections import defaultdict
import datasets
import transformers
import logging
import numpy as np

logging.basicConfig()
LOG = logging.getLogger(__name__)

datasets.logging.set_verbosity_error()


def model2hfname(model: str) -> str:
    return {
        'bert-tiny': 'prajjwal1/bert-tiny',
        'bert-med': 'prajjwal1/bert-medium',
        'small': 'gpt2',
        'med': 'gpt2-medium',
        'large': 'gpt2-large',
        'full': 'gpt2-xl',
        'gpt2-sm': 'gpt2',
        'gpt2-med': 'gpt2-medium',
        'gpt2-lg': 'gpt2-large',
        'gpt2': 'gpt2-xl',
        'neo': 'EleutherAI/gpt-neo-2.7B',
    }[model]


def dataset2hfname(dataset: str) -> str:
    return {
        'mnli': ('multi_nli',),
        'amazon': ('amazon_us_reviews', 'Video_v1_00'),
        'cnn': ('cnn_dailymail', '3.0.0'),
        'math': ('math_qa',),
        'tos': ('ought/raft', 'terms_of_service'),
        'xsum': ('xsum',),
        'babi': ('babi_qa', 'en-valid-10k-qa1')
    }[dataset]


def get_dataset(dataset: str, n_train: int, n_val: int = 100):
    if dataset == 'cnn':
        n_train = 64
        d = datasets.load_dataset('cnn_dailymail', '3.0.0', split='train')
        filter_fn = lambda rows: ['VIDEO' not in a and len(a.split(' ')) < 110 and len(a.split(' ')) > 35 and len(s.split(' ')) < 25 for a, s in zip(rows['article'], rows['highlights'])]
        d = d.filter(filter_fn, batched=True, batch_size=None)
        d = d.rename_columns({'article': 'x', 'highlights': 'y'})
        def strip_target(row):
            y = row['y']
            y = y.replace(' .', '.')
            if '. ' in y:
                y = y[:y.index('. ')]
            if '\n' in y:
                y = y[:y.index('\n')]
            row['y'] = y
            return row
        d = d.map(strip_target)
        d = d.add_column('simple_y', d['y'])
        return d[:n_train], d[n_train:n_train + n_val]
    elif dataset == 'trivia':
        n_train = 256
        d = datasets.load_dataset('trivia_qa', 'rc.nocontext', split='train[:1%]')
        targets = [[a['normalized_value']] + a['normalized_aliases'] for a in d['answer']]
        d = d.add_column('simple_y', [t[0] for t in targets])
        d = d.add_column('y', targets)
        d = d.rename_column('question', 'x')
        offset = 0
        return d[offset:offset+n_train], d[offset+n_train:offset+n_train + n_val]
    elif dataset == 'babi':
        n_train = 256
        d = datasets.load_dataset('babi_qa', 'en-valid-10k-qa1', split='train')
        answer_idxs = []
        for story in d['story']:
            for idx, answer in enumerate(story['answer']):
                if answer:
                    answer_idxs.append(idx)
                    break
        
        perm = np.random.permutation(len(d['story']))
        answers = [story['answer'][idx] for idx, story in zip(answer_idxs, d['story'])]
        stories = [' '.join(story['text'][:idx + 1]) for idx, story in zip(answer_idxs, d['story'])]

        answers = [answers[idx] for idx in perm]
        stories = [stories[idx] for idx in perm]
        data = {'x': stories, 'y': answers, 'simple_y': answers}
        d = datasets.Dataset.from_dict(data)
        return d[:n_train], d[n_train:n_train + n_val]
    elif dataset == 'amazon':
        d = datasets.load_dataset('amazon_us_reviews', 'Video_v1_00')['train']
        filter_fn = lambda rows: ['sex' not in r.lower() for r in rows['review_body']]
        d = d.filter(filter_fn, batched=True, batch_size=None)
        x = d['review_body']
        y = [s - 1 for s in d['star_rating']]
        train = defaultdict(lambda: [None] * 5 * n_train)
        val = defaultdict(lambda: [None] * 5 * n_val)
        counts = defaultdict(int)
        for idx in range(len(y)):
            c = counts[y[idx]]
            if c < n_train:
                train['x'][c * 5 + y[idx]] = x[idx]
                train['y'][c * 5 + y[idx]] = y[idx]
                counts[y[idx]] += 1
            elif c < n_train + n_val:
                val['x'][(c - n_train) * 5 + y[idx]] = x[idx]
                val['y'][(c - n_train) * 5 + y[idx]] = y[idx]
                counts[y[idx]] += 1
        return train, val
    elif dataset == 'xsum':
        n_train = 256
        d = datasets.load_dataset('xsum', split='train')
        filter_fn = lambda rows: [len(a.split(' ')) + len(s.split(' ')) < 100 for a, s in zip(rows['document'], rows['summary'])]
        d = d.filter(filter_fn, batched=True, batch_size=None)
        d = d.rename_columns({'document': 'x', 'summary': 'y'})
        d = d.add_column('simple_y', d['y'])
        return d[:n_train], d[n_train:n_train + n_val]
    else:
        raise NotImplementedError(f'{dataset}')


def is_qa_dataset(dataset: str) -> bool:
    return dataset in ['trivia', 'babi']


def stop_tokens(tokenizer, stop_string: str = '.') -> int:
    tokens = []
    for idx in range(len(tokenizer)):
        if tokenizer.decode(idx) == stop_string:
            tokens.append(idx)
    return tokens

def max_sampled_tokens_for_dataset(dataset: str) -> int:
    return {
        'cnn': 30,
        'trivia': 12,
        'babi': 6,
        'xsum': 30,
    }[dataset]


def get_model_and_tokenizer(model: str, Cls, **model_kwargs):
    hf_model_name = model2hfname(model)

    m = Cls.from_pretrained(hf_model_name, **model_kwargs)
    if isinstance(m, transformers.GPT2LMHeadModel):
        m.transformer.gradient_checkpointing_enable()

    tok = transformers.AutoTokenizer.from_pretrained(hf_model_name)

    if tok.pad_token_id is None:
        if Cls == transformers.AutoModelForCausalLM:
            tok.pad_token = tok.eos_token
        else:
            print("Adding pad token to tokenizer")
            tok.add_special_tokens({'pad_token': '[PAD]'})
            tok.pad_token = '[PAD]'
    return m, tok


def metric_for_dataset(dataset: str):
    return {
        'cnn': 'rouge',
        'xsum': 'rouge',
        'trivia': 'exact match',
        'babi': 'exact match',
        'amazon': 'classification accuracy',
    }[dataset]


def early_stop_thresold(dataset: str):
    return {
        'cnn': 0.8,
        'trivia': 0.7,
        'babi': 0.9,
        'amazon': 0.75,
        'xsum': 0.55,
    }[dataset]