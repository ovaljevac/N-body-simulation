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
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--steps', type=int, default=3)
    args = parser.parse_args()
    if any(n <= 0 for n in args.sizes) or any(p <= 0 for p in args.threads) or args.repeats <= 0 or args.steps <= 0:
        parser.error('Veličine, niti, ponavljanja i koraci moraju biti pozitivni.')
    if len(set(args.threads)) != len(args.threads) or len(set(args.sizes)) != len(args.sizes):
        parser.error('Veličine i broj niti ne treba ponavljati u istom eksperimentu.')
    args.out.mkdir(parents=True, exist_ok=True)
    exe = args.exe.resolve()
    observations = []
    commands = []
    for n in args.sizes:
        for distribution in ['uniform', 'clustered']:
            for threads in args.threads:
                # Serijske verzije mjere se jednom za svaki ulaz. Za svaki
                # naredni broj niti ponavljaju se samo OpenMP verzije.
                modes = ['all'] if threads == args.threads[0] else ['openmp', 'openmp_simd']
                for backend in modes:
                    command = [str(exe), '--n', str(n), '--threads', str(threads),
                               '--distribution', distribution, '--repeats', str(args.repeats),
                               '--steps', str(args.steps), '--backend', backend]
                    commands.append(command)
                    completed = subprocess.run(command, check=True, text=True, encoding='utf-8', capture_output=True)
                    rows = list(csv.DictReader(completed.stdout.splitlines()))
                    observations.extend(rows)
                    print(f'N={n} {distribution} P={threads} {backend}', flush=True)
    with (args.out / 'raw.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(observations[0]))
        writer.writeheader()
        writer.writerows(observations)
    # Svaki red ostaje u raw.csv; za isti ulaz, verziju i broj niti
    # računamo medijanu i raspon, bez uklanjanja sporijih uzoraka.
    groups = {}
    for row in observations:
        key = (row['n'], row['distribution'], row['backend'], row['threads'])
        groups.setdefault(key, []).append(row)
    medians = {key: statistics.median(float(r['seconds_per_step']) for r in rows)
               for key, rows in groups.items()}
    summary = []
    for key, rows in groups.items():
        n, distribution, backend, threads = key
        timings = [float(r['seconds_per_step']) for r in rows]
        median = medians[key]
        sequential = medians[(n, distribution, 'sequential', '1')]
        serial_backend = {'openmp': 'sequential', 'openmp_simd': 'simd'}.get(backend, backend)
        matching_serial = medians[(n, distribution, serial_backend, '1')]
        speedup = matching_serial / median
        summary.append(dict(n=n, distribution=distribution, backend=backend, threads=threads,
                            repeats=len(rows), median_seconds=median,
                            min_seconds=min(timings), max_seconds=max(timings),
                            speedup_vs_sequential=sequential / median,
                            speedup_vs_matching_serial=speedup,
                            cpu_efficiency=speedup / int(threads) if backend.startswith('openmp') else '',
                            max_relative_l2=max(float(r['relative_l2']) for r in rows)))
    with (args.out / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    metadata = dict(platform=platform.platform(), processor=platform.processor(),
                    processor_identifier=os.environ.get('PROCESSOR_IDENTIFIER'),
                    logical_cpus=os.cpu_count(), executable=str(exe), commands=commands,
                    note='Direct CPU; OpenMP se poredi sa sequential, a OpenMP+SIMD sa simd. Preciznost i build su u CSV-u.')
    (args.out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Saved {len(observations)} observations and {len(summary)} summaries to {args.out}')


if __name__ == '__main__':
    main()
