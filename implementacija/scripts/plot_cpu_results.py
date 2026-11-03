"""Nacrtaj sažetak CPU mjerenja; potrebna je biblioteka Matplotlib."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('summary', type=Path)
args = parser.parse_args()
with args.summary.open(encoding='utf-8') as stream:
    rows = list(csv.DictReader(stream))
# Jedna kriva predstavlja backend i, za BH, konkretnu theta vrijednost.
# Boja razlikuje backend, a stil linije theta, bez mijesanja razlicitih gresaka.
series_keys = sorted({(r['backend'], r.get('theta', '0.5') if r['backend'].startswith('bh_') else '') for r in rows})
backend_names = sorted({backend for backend, _ in series_keys})
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
theta_values = sorted({theta for _, theta in series_keys if theta}, key=float)
styles = ['-', '--', ':', '-.']
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6 + 0.2 * ((len(series_keys) + 2) // 3)), layout='constrained')
# Serijske verzije imaju jednu nit. Za paralelne krive prikazujemo najveći
# izmjereni broj niti; potpuni rezultati ostaju u summary.csv datoteci.
for ax, distribution in zip(axes, ['uniform', 'clustered']):
    for backend, theta in series_keys:
        available = [r for r in rows if r['distribution'] == distribution and r['backend'] == backend
                     and (not theta or r.get('theta', '0.5') == theta)]
        if not available:
            continue
        threads = max(int(r['threads']) for r in available)
        series = sorted((r for r in available if int(r['threads']) == threads), key=lambda r: int(r['n']))
        ax.plot([int(r['n']) for r in series], [1000 * float(r['median_seconds']) for r in series],
                marker='o', color=colors[backend_names.index(backend) % len(colors)],
                linestyle=styles[theta_values.index(theta) % len(styles)] if theta else '-',
                label=f'{backend}' + (f' theta={float(theta):g}' if theta else '')
                + f' ({threads} {"thread" if threads == 1 else "threads"})')
    ax.set(xlabel='Number of bodies N', ylabel='Solver time including tree build (ms)',
           title=distribution.capitalize(), xscale='log', yscale='log')
    ax.grid(alpha=.25)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside lower center', ncol=3, fontsize=8)
fig.suptitle('N-body CPU: median of measured repetitions')
fig.savefig(args.summary.with_name('cpu_times.png'), dpi=180)
fig.savefig(args.summary.with_name('cpu_times.svg'))
