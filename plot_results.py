"""
Plot the learning curves saved by the version scripts (--save_history).

    python plot_results.py            # reads runs/*.json, writes static/

See run_all.sh for the exact runs behind the figures in the README.
"""

import glob
import json
import os

import numpy as np

from utils import plot_style, savefig


def load(pattern):
    runs = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def curve(run, key='test_acc'):
    h = run['history']
    return np.array([r['step'] for r in h]), np.array([r[key] for r in h])


def learning_curves():
    """Versions 2-5 on copying and selective copying."""
    plt = plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    models = [('v2', 'v2: S4D (LTI, convolution)'),
              ('v3', 'v3: selective SSM (S6, scan)'),
              ('v4', 'v4: Mamba (naive init)'),
              ('v5', 'v5: Mamba (S4D-Real + log-uniform Δ)')]
    for ax, task in zip(axes, ['copy', 'selective']):
        for tag, label in models:
            pat = (f'runs/v2_{task}_random.json' if tag == 'v2' else
                   f'runs/v5_{task}_real_loguniform_s0.json' if tag == 'v5'
                   else f'runs/{tag}_{task}.json')
            runs = load(pat)
            if runs:
                s, a = curve(runs[0])
                ax.plot(s, a, label=label)
        ax.set_xlim(0, 2000)
        ax.set_title({'copy': 'Copying', 'selective': 'Selective copying'}[task])
        ax.set_xlabel('training step')
        ax.axhline(1 / 8, color='k', ls=':', lw=1)
    axes[0].set_ylabel('test accuracy (marker positions)')
    axes[1].legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    savefig(fig, 'learning_curves.png')
    plt.close(fig)


def init_comparison():
    """Version 5: initialisations of Mamba on selective copying (seeds)."""
    configs = [('random', 'default', 'naive: -A ~ U(0.1,1), Δ ≈ 0.69 (v4)'),
               ('random', 'loguniform', 'naive A + log-uniform Δ'),
               ('real', 'default', 'S4D-Real A + naive Δ'),
               ('real', 'loguniform', 'S4D-Real + log-uniform Δ (Mamba)'),
               ('lin', 'loguniform', 'S4D-Lin (complex) + log-uniform Δ')]
    plt = plot_style()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    print('Version 5, selective copying (test accuracy, mean over seeds):')
    for i, (A, dt, label) in enumerate(configs):
        runs = load(f'runs/v5_selective_{A}_{dt}_s*.json')
        if not runs:
            continue
        accs = np.stack([curve(r)[1] for r in runs])
        steps = curve(runs[0])[0]
        m = accs.mean(0)
        ax.plot(steps, m, color=f'C{i}', label=f'{label} ({len(runs)} seeds)')
        if len(runs) > 1:
            ax.fill_between(steps, accs.min(0), accs.max(0), color=f'C{i}',
                            alpha=0.15)
        hit = [steps[np.argmax(a >= 0.9)] if (a >= 0.9).any() else None
               for a in accs]
        print(f'  {label:42s}: final {m[-1]:.3f} | steps to 90%: {hit}')
    ax.set_xlabel('training step')
    ax.set_ylabel('test accuracy')
    ax.set_title('Mamba on selective copying: effect of the initialisation')
    ax.legend(fontsize=7.5, loc='lower right')
    savefig(fig, 'v5_init_comparison.png')
    plt.close(fig)


def long_memory():
    """Selective copying with longer memory:
    n_mem 128 (L = 136, default model) and n_mem 256 (L = 264, one layer,
    d_model 32, batch 32)."""
    plt = plot_style()
    settings = [(tag, title) for tag, title in
                [('mid', 'L = 136 (2 layers, d_model 64)'),
                 ('long', 'L = 264 (1 layer, d_model 32)')]
                if glob.glob(f'runs/{tag}_*.json')]
    fig, axes = plt.subplots(1, len(settings), figsize=(5.5 * len(settings), 3.8),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (tag, title) in zip(axes, settings):
        print(f'Selective copying, {title}:')
        for name, label in [
                ('v2', 'v2: S4D (LTI)'),
                ('v5_random_default', 'v4: Mamba, naive init'),
                ('v5_real_loguniform', 'v5: Mamba, S4D-Real + log-uniform Δ')]:
            runs = load(f'runs/{tag}_{name}.json')
            if not runs:
                continue
            s, a = curve(runs[0])
            ax.plot(s, a, label=label)
            hit = s[np.argmax(a >= 0.9)] if (a >= 0.9).any() else None
            print(f'  {label:40s}: final {a[-1]:.3f}, best {a.max():.3f}, '
                  f'steps to 90%: {hit}, '
                  f'{runs[0]["history"][-1]["time"]:.0f}s')
        ax.axhline(1 / 8, color='k', ls=':', lw=1)
        ax.set_xlabel('training step')
        ax.set_title(f'Selective copying, {title}')
    axes[0].set_ylabel('test accuracy')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, 'long_selective_copying.png')
    plt.close(fig)


def summary():
    print('Final test accuracy of every saved run:')
    for path in sorted(glob.glob('runs/*.json')):
        r = load(path)[0]
        print(f'  {os.path.basename(path):40s} {r["history"][-1]["test_acc"]:.3f}'
              f' ({r["history"][-1]["step"]} steps, '
              f'{r["history"][-1]["time"]:.0f}s)')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    learning_curves()
    init_comparison()
    long_memory()
    summary()
