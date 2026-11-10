"""Nacrtaj sažetak CPU mjerenja. Potreban je Matplotlib."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from plot_styles import COLORS, PARALLEL_BACKENDS, curve_style

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('summary', type=Path)
args = parser.parse_args()
with args.summary.open(encoding='utf-8') as stream:
    rows = list(csv.DictReader(stream))

theta_values = sorted({r.get('theta', '0.5') for r in rows if r['backend'].startswith('bh_')}, key=float)
variants = [(backend, theta) for backend in COLORS
            for theta in (theta_values if backend.startswith('bh_') else [''])]
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True, layout='constrained')
legend = {}
for ax, (distribution, title) in zip(axes, [('uniform', 'Ravnomjerna'), ('clustered', 'Klasterisana')]):
    for backend, theta in variants:
        candidates = [r for r in rows if r['distribution'] == distribution and r['backend'] == backend
                      and (not theta or r.get('theta', '0.5') == theta)]
        if not candidates:
            continue
        threads = max(int(r.get('threads', 1)) for r in candidates)
        series = sorted((r for r in candidates if int(r.get('threads', 1)) == threads),
                        key=lambda r: int(r['n']))
        style = curve_style(backend)
        if theta and len(theta_values) > 1:
            index = theta_values.index(theta)
            style['linestyle'] = ['-', ':', '-.'][index % 3]
            if backend in PARALLEL_BACKENDS:
                style['linestyle'] = (0, (3 + 2 * index, 2))
        n = [int(r['n']) for r in series]
        label = backend + (f' θ={float(theta):g}' if theta else '')
        if backend in PARALLEL_BACKENDS:
            label += f' ({threads} niti)'
        line, = ax.plot(n, [1000 * float(r['median_seconds']) for r in series], **style)
        legend.setdefault(label, line)
        ax.fill_between(n, [1000 * float(r['min_seconds']) for r in series],
                        [1000 * float(r['max_seconds']) for r in series], color=style['color'], alpha=.12)
    ax.set(xlabel='Broj tijela N', title=title, xscale='log', yscale='log')
    ax.grid(alpha=.24)
    ax.set_xticks(sorted({int(r['n']) for r in rows}))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()
axes[0].set_ylabel('Vrijeme računanja (ms)')
fig.legend(legend.values(), legend.keys(), loc='outside lower center', ncol=4, fontsize=8)
subtitle = 'Sjenčenje: min–maks · obje ose logaritamske'
if any(r['backend'] in PARALLEL_BACKENDS for r in rows):
    subtitle += ' · isprekidano: OpenMP'
fig.suptitle(f'N-body CPU · medijana vremena jednog računa\n{subtitle}', fontsize=12)
fig.savefig(args.summary.with_name('cpu_times.png'), dpi=180)
fig.savefig(args.summary.with_name('cpu_times.svg'))
