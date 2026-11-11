#!/usr/bin/env python3
"""Figures from data/results.json, plus the re-centering animation.

    python3 scripts/figures.py
"""
import json, os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA
from pyriemann.tangentspace import TangentSpace

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mibci.data import load_session
from mibci.align import covariances, recenter, reference

plt.style.use(os.path.join(ROOT, 'figures', 'neuro.mplstyle'))
FIG = os.path.join(ROOT, 'figures')
INK, MUTED, GREY, BLUE, ORANGE = '#0b0b0b', '#52514e', '#8a8985', '#2a78d6', '#eb6834'
MAIN = 'Tangent space + logistic regression'


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.png')); fig.savefig(os.path.join(FIG, name + '.svg')); plt.close(fig)


def paired(ax, cols, labels, chance95):
    V = 100 * np.array(cols)                                    # conditions x subjects
    for s in range(V.shape[1]):
        ax.plot(range(len(V)), V[:, s], color=GREY, linewidth=0.9, alpha=0.6, marker='o', markersize=3.5)
    ax.plot(range(len(V)), V.mean(1), color=BLUE, linewidth=2.2, marker='o', markersize=7, markeredgecolor='#fcfcfb')
    for i, m in enumerate(V.mean(1)):
        ax.text(i, m + 3.5, f'{m:.0f}%', ha='center', fontsize=9.5, color=INK)
    ax.axhline(100 * chance95, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.set_xticks(range(len(V)), labels); ax.set_xlim(-0.4, len(V) - 0.6); ax.set_ylim(15, 100)
    ax.grid(axis='x', visible=False)


def fig_transfer(r, chance95):
    m = r['models'][MAIN]
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.3), gridspec_kw=dict(width_ratios=[3, 2], wspace=0.22))
    paired(ax[0], [m['within'], m['session_raw'], m['session_aligned']],
           ['Same day\n(leave one run out)', 'Next day', 'Next day,\nre-centred'], chance95)
    ax[0].set_ylabel('Accuracy, four classes (%)'); ax[0].set_title('Same person, new day')
    ax[0].text(2.55, 100 * chance95 + 1.5, 'chance, 95% bound', color=MUTED, fontsize=8.5, ha='right')
    paired(ax[1], [m['subject_raw'], m['subject_aligned']], ['New person', 'New person,\nre-centred'], chance95)
    ax[1].set_title('Trained on eight other people')
    save(fig, 'transfer')


def fig_calibration(r, chance95):
    counts = [int(k) for k in r['calibration']]
    x = np.arange(len(counts))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, key, title in zip(axes, ('calibration', 'calibration_new_person'),
                              ('Same person, new day', 'New person')):
        V = 100 * np.array([r[key][str(n)] for n in counts])
        for s in range(V.shape[1]):
            ax.plot(x, V[:, s], color=GREY, linewidth=0.9, alpha=0.6)
        ax.plot(x, V.mean(1), color=BLUE, linewidth=2.2, marker='o', markersize=7, markeredgecolor='#fcfcfb')
        for i, m in enumerate(V.mean(1)):
            ax.text(i, m + 3, f'{m:.0f}', ha='center', fontsize=9, color=INK)
        ax.axhline(100 * chance95, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.set_xticks(x, ['none'] + [str(n) for n in counts[1:]]); ax.set_title(title)
        ax.set_xlabel('Unlabelled trials used to re-centre'); ax.set_ylim(15, 100); ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('Accuracy on the second half of the session (%)')
    save(fig, 'calibration')


def gif_recentering(a_id, b_id):
    """Two people's trials in tangent space, before and after each is re-centred."""
    CA = covariances(load_session(a_id, 'T')[0]); CB = covariances(load_session(b_id, 'T')[0])
    ts = TangentSpace(metric='riemann'); ts.reference_ = reference(np.concatenate([CA, CB]))
    raw = ts.transform(np.concatenate([CA, CB]))
    ts2 = TangentSpace(metric='riemann'); ts2.reference_ = np.eye(CA.shape[1])
    ali = ts2.transform(np.concatenate([recenter(CA), recenter(CB)]))
    pca = PCA(2).fit(np.concatenate([raw, ali]))
    a, b = pca.transform(raw), pca.transform(ali)
    who = np.r_[np.zeros(len(CA)), np.ones(len(CB))].astype(int)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    sc = ax.scatter(a[:, 0], a[:, 1], s=14, c=np.array([BLUE, ORANGE])[who], alpha=0.75, linewidth=0)
    lim = np.abs(np.concatenate([a, b])).max() * 1.08
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.text(0.03, 0.97, f'subject {a_id}', color=BLUE, transform=ax.transAxes, va='top', fontweight='bold')
    ax.text(0.03, 0.91, f'subject {b_id}', color=ORANGE, transform=ax.transAxes, va='top', fontweight='bold')
    title = ax.set_title('')
    steps = np.r_[np.zeros(10), np.linspace(0, 1, 30), np.ones(14)]
    ease = steps * steps * (3 - 2 * steps)

    def draw(k):
        sc.set_offsets((1 - ease[k]) * a + ease[k] * b)
        title.set_text('One point per trial, ' + ('as recorded' if ease[k] < 0.5 else 'each person re-centred'))

    out = os.path.join(FIG, 'recentering.gif')
    FuncAnimation(fig, draw, frames=len(steps)).save(out, writer=PillowWriter(fps=12), dpi=90)
    plt.close(fig)
    print(out, f'{os.path.getsize(out) / 1e6:.1f} MB')


def main():
    r = json.load(open(os.path.join(ROOT, 'data', 'results.json')))
    fig_transfer(r, r['chance_95']); fig_calibration(r, r['chance_95'])
    m = r['models'][MAIN]
    gain = np.array(m['subject_aligned']) - np.array(m['subject_raw'])
    order = np.argsort(gain)                                     # the two people nearest the median gain
    gif_recentering(r['subjects'][int(order[len(order) // 2])], r['subjects'][int(order[len(order) // 2 - 1])])

if __name__ == '__main__':
    main()
