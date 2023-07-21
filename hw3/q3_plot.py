import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json
from collections import defaultdict
import utils


def plot():
    dataset = 'xsum'
    data = defaultdict(lambda: defaultdict(list))
    model = 'med'
    mode = 'lora16'
    x_vals = set()
    for k in [0,1,8,64]:
        fn = '_'.join([model, dataset, str(k), mode])
        id_ = '_'.join([model, dataset, mode])
        with open(f'results/ft/{fn}.json', 'r') as f:
            score = json.load(f)['metric']
            data[id_]['x'].append(k)
            x_vals.add(k)
            data[id_]['y'].append(score)
    
    prompt_mode = 'tldr'
    for k in [0,1,4]:
        fn = '_'.join([model, dataset, str(k), prompt_mode])
        id_ = '_'.join([model, dataset, prompt_mode])
        with open(f'results/icl/{fn}.json', 'r') as f:
            score = json.load(f)['metric']
            data[id_]['x'].append(k)
            x_vals.add(k)
            data[id_]['y'].append(score)

    for k, v in data.items():
        plt.plot(v['x'], v['y'], label=k)
    if max(x_vals) > 4:
        plt.xscale('symlog')
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_ticks(sorted(x_vals))
    plt.legend()
    plt.title(dataset)
    plt.ylabel(utils.metric_for_dataset(dataset))
    plt.xlabel('Number of support examples')
    plt.show()


if __name__ == '__main__':
    plot()