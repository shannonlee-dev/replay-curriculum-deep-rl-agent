import argparse
import csv
from pathlib import Path


def plot_history(history, output=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history = Path(history)
    with history.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    for key in rows[0]:
        if not (key.startswith('test_R') or key == 'natural_full'):
            continue
        points = [(int(r['global_steps']), float(r[key])) for r in rows if r.get(key)]
        if points:
            x, y = zip(*points)
            ax.plot(x, y, marker='.', label=key.removeprefix('test_'))
    ax.set(xlabel='Global training steps', ylabel='Win rate', ylim=(0, 1))
    ax.grid(alpha=.25)
    ax.legend(loc='center left', bbox_to_anchor=(1, .5), fontsize=8)
    fig.tight_layout()
    fig.savefig(output or history.with_name('win_rate_by_target.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('history', nargs='?', default='models/history.csv')
    p.add_argument('--output')
    args = p.parse_args()
    plot_history(args.history, args.output)
