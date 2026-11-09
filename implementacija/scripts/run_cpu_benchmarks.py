"""Pokreni CPU mjerenja i izračunaj statistiku bez dodatnih Python biblioteka."""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('results/local'))
    parser.add_argument('--sizes', type=int, nargs='+', default=[256, 1024, 2048])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--algorithm', choices=['direct', 'barnes_hut', 'all'], default='direct')
    parser.add_argument('--theta', type=float, nargs='+', default=[0.5])
    parser.add_argument('--leaf-capacity', type=int, default=8)
    parser.add_argument('--max-depth', type=int, default=32)
    parser.add_argument('--max-relative-error', type=float, default=0.05)
    args = parser.parse_args()
    if any(n <= 0 for n in args.sizes) or args.repeats <= 0 or args.steps <= 0:
        parser.error('Veličine, ponavljanja i koraci moraju biti pozitivni.')
    if len(set(args.sizes)) != len(args.sizes):
        parser.error('Veličine ne treba ponavljati u istom eksperimentu.')
    if any(not 0 <= theta <= 1 for theta in args.theta) or len(set(args.theta)) != len(args.theta):
        parser.error('Theta vrijednosti moraju biti razlicite i u rasponu 0..1.')
    if args.leaf_capacity <= 0 or not 1 <= args.max_depth <= 64 or not 0 < args.max_relative_error < float('inf'):
        parser.error('Neispravna velicina lista, dubina ili tolerancija.')
    args.out.mkdir(parents=True, exist_ok=True)
    exe = args.exe.resolve()
    observations = []
    commands = []
    # Svi algoritmi dobijaju iste velicine, seed iz programa i rasporede tijela.
    # Theta serija mijenja samo BH; Direct se ne mjeri ponovo za svaku theta.
    algorithms = ['direct', 'barnes_hut'] if args.algorithm == 'all' else [args.algorithm]
    for algorithm in algorithms:
        for theta in args.theta if algorithm == 'barnes_hut' else [0.5]:
            for n in args.sizes:
                for distribution in ['uniform', 'clustered']:
                    command = [str(exe), '--n', str(n),
                               '--distribution', distribution, '--repeats', str(args.repeats),
                               '--steps', str(args.steps), '--backend', 'all',
                               '--algorithm', algorithm, '--theta', str(theta),
                               '--leaf-capacity', str(args.leaf_capacity), '--max-depth', str(args.max_depth),
                               '--max-relative-error', str(args.max_relative_error)]
                    commands.append(command)
                    print(f'Running {algorithm} theta={theta} N={n} {distribution}', flush=True)
                    completed = subprocess.run(command, check=True, text=True, encoding='utf-8', capture_output=True)
                    observations.extend(csv.DictReader(completed.stdout.splitlines()))
    with (args.out / 'raw.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(observations[0]))
        writer.writeheader()
        writer.writerows(observations)
    # Svaki red ostaje u raw.csv; za isti ulaz, verziju i broj niti
    # računamo medijanu i raspon, bez uklanjanja sporijih uzoraka.
    groups = {}
    for row in observations:
        key = (row['algorithm'], row['theta'], row['n'], row['distribution'], row['backend'])
        groups.setdefault(key, []).append(row)
    medians = {key: statistics.median(float(r['seconds_per_step']) for r in rows)
               for key, rows in groups.items()}
    summary = []
    for key, rows in groups.items():
        algorithm, theta, n, distribution, backend = key
        timings = [float(r['seconds_per_step']) for r in rows]
        median = medians[key]
        serial = 'bh_sequential' if algorithm == 'barnes_hut' else 'sequential'
        sequential = medians[(algorithm, theta, n, distribution, serial)]
        direct = medians.get(('direct', '0.5', n, distribution, 'sequential'))
        summary.append(dict(algorithm=algorithm, theta=theta, n=n, distribution=distribution, backend=backend,
                            repeats=len(rows), median_seconds=median,
                            min_seconds=min(timings), max_seconds=max(timings),
                            speedup_vs_sequential=sequential / median,
                            speedup_vs_direct=direct / median if direct is not None else '',
                            median_tree_seconds=statistics.median(float(r['tree_seconds_per_step']) for r in rows),
                            median_force_seconds=statistics.median(float(r['force_seconds_per_step']) for r in rows),
                            leaf_capacity=rows[0]['leaf_capacity'], max_depth=rows[0]['max_depth'],
                            max_relative_l2=max(float(r['relative_l2']) for r in rows)))
    with (args.out / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    metadata = dict(platform=platform.platform(), processor=platform.processor(),
                    processor_identifier=os.environ.get('PROCESSOR_IDENTIFIER'),
                    logical_cpus=os.cpu_count(), executable=str(exe), commands=commands,
                    note='Ukupno BH vrijeme ukljucuje gradnju stabla. Theta i greska su u CSV-u.')
    (args.out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Saved {len(observations)} observations and {len(summary)} summaries to {args.out}')


if __name__ == '__main__':
    main()
