"""Plot a run_cpu_benchmarks.py summary; requires matplotlib."""
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
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
for ax, distribution in zip(axes, ['uniform', 'clustered']):
    for backend in ['sequential', 'blocked', 'simd', 'openmp', 'openmp_simd']:
        available = [r for r in rows if r['distribution'] == distribution and r['backend'] == backend]
        threads = max(int(r['threads']) for r in available)
        series = sorted((r for r in available if int(r['threads']) == threads), key=lambda r: int(r['n']))
        ax.plot([int(r['n']) for r in series], [1000 * float(r['median_seconds']) for r in series],
                marker='o', label=f'{backend} ({threads} threads)')
    ax.set(xlabel='Number of bodies N', ylabel='Force evaluation time (ms)',
           title=distribution.capitalize(), xscale='log', yscale='log')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
fig.suptitle('Direct N-body CPU: median of measured repetitions')
fig.savefig(args.summary.with_name('cpu_times.png'), dpi=180)
fig.savefig(args.summary.with_name('cpu_times.svg'))
