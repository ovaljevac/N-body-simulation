"""Dodatni grafikoni i izvještaj iz stvarno izmjerenih CSV rezultata."""
import argparse
import csv
import html
import json
import math
import statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('root', type=Path, help='Folder sa scaling/ i theta/ rezultatima')
a = p.parse_args()
supported = {'sequential', 'blocked', 'simd', 'bh_sequential', 'bh_optimized'}
root = a.root.resolve()
def read(folder, name='summary.csv'):
    with (root / folder / name).open(encoding='utf-8') as f:
        return [row for row in csv.DictReader(f) if row['backend'] in supported]
rows, theta = read('scaling'), read('theta')
raw = read('scaling', 'raw.csv')
steps = int(raw[0]['steps'])
for folder, summaries in [('scaling', rows), ('theta', theta)]:
    observations = read(folder, 'raw.csv')
    groups = {}
    def group_key(r):
        return tuple(r[k] for k in ['algorithm', 'theta', 'n', 'distribution', 'backend'])
    for r in observations:
        if r['mode'] != 'force' or r['precision'] != 'float' or int(r['steps']) != steps:
            raise ValueError('Ovaj izvještaj opisuje FP32 force seriju sa istim brojem evaluacija.')
        groups.setdefault(group_key(r), []).append(r)
    if len(groups) != len(summaries):
        raise ValueError('Broj grupa u raw.csv i summary.csv se razlikuje.')
    for r in summaries:
        samples = groups[group_key(r)]
        values = [float(s['seconds_per_step']) for s in samples]
        if len(samples) != 7 or len({s['repeat'] for s in samples}) != 7:
            raise ValueError('Nedostaje sedam različitih ponavljanja.')
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError('Neispravna vremena mjerenja.')
        if not math.isclose(statistics.median(values), float(r['median_seconds']), rel_tol=1e-12):
            raise ValueError('Medijana u sažetku ne odgovara sirovim mjerenjima.')
        if any(float(s['relative_l2']) > float(s['max_relative_error']) for s in samples):
            raise ValueError('Numerička greška prelazi zadani prag.')
print(f'CSV provjera prošla: {len(rows)} scaling grupa i {len(theta)} theta grupa, po 7 ponavljanja.')
meta = json.loads((root / 'scaling/metadata.json').read_text(encoding='utf-8'))
cpu = meta.get('processor_identifier') or meta.get('processor')
try:
    import winreg
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
        cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
except (ImportError, OSError):
    pass
nmax = max(int(r['n']) for r in rows)
distributions = [('uniform', 'Ravnomjerna raspodjela'), ('clustered', 'Klasterisana raspodjela')]
def save(fig, name):
    fig.savefig(root / (name + '.png'), dpi=180)
    fig.savefig(root / (name + '.svg'))
    plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout='constrained')
for ax, (dist, title) in zip(axes, distributions):
    for backend in ['bh_sequential', 'bh_optimized']:
        candidates = [r for r in theta if r['distribution'] == dist and r['backend'] == backend]
        series = sorted(candidates, key=lambda r: float(r['theta']))
        ax.plot([100 * float(r['max_relative_l2']) for r in series], [1000 * float(r['median_seconds']) for r in series], 'o-', label=backend)
        for r in series:
            rightmost = float(r['theta']) == max(float(s['theta']) for s in series)
            ax.annotate(f"θ={float(r['theta']):g}", (100 * float(r['max_relative_l2']), 1000 * float(r['median_seconds'])), xytext=(-6 if rightmost else 5, 5), ha='right' if rightmost else 'left', textcoords='offset points', fontsize=8)
    ax.set(title=title, xlabel='Relativna L2 greška (%)', ylabel='Medijana vremena (ms)', xscale='log', yscale='log')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
fig.suptitle(f'Barnes–Hut: vrijeme i tačnost, N = {theta[0]["n"]}\nGreška prema direktnoj FP64 referenci; vrijeme uključuje gradnju stabla', fontsize=11)
save(fig, 'theta_tradeoff')

lines = ['# Izvještaj o CPU benchmarku N-body simulacije', '',
         '## Metodologija', '',
         f'- Sistem: {meta["platform"]}; procesor: {cpu}; dostupno logičkih procesora: {meta["logical_cpus"]}.',
         f'- Build: {raw[0]["build_type"]}; preciznost: {raw[0]["precision"]}; kompajler: {raw[0]["compiler"]}; native: {raw[0]["native"]}.',
         '- Režim force: položaji su zamrznuti; mjeri se računanje ubrzanja, bez GUI-ja i integracije.',
         '- N = ' + ', '.join(str(n) for n in sorted({int(r['n']) for r in rows})) + '. Sve izvedbe rade na jednoj niti.',
         f'- Sedam mjerenih ponavljanja, {steps} evaluacija po ponavljanju i jedno zagrijavanje. Prikazana je medijana vremena jedne evaluacije; sjenčenje označava minimum–maksimum.',
         '- Seed 42, G=1, epsilon=0.01; Barnes–Hut: θ=0.5, kapacitet lista 8, maksimalna dubina 32. Isti ulazi za sve izvedbe.',
         '- U vrijeme ulazi javni poziv solvera sa validacijom; Barnes–Hut uključuje sekvencijalnu gradnju i alokacije stabla. FP64 referenca i provjera greške računaju se izvan tajmera.',
         '- Skalarnom i blokovskom modulu je isključena automatska vektorizacija radi kontrolisanog poređenja sa eksplicitnim SIMD modulom.',
         '- Release build i sva tri postojeća CTest testa su prošli prije mjerenja. Svaka benchmark konfiguracija provjerava rezultat prema FP64 referenci.', '',
         '## Vrijeme u zavisnosti od broja tijela', '',
         '![Vrijeme računanja](scaling/cpu_times.png)', '',
         'Obje ose su logaritamske. Tačke su mjerenja, a linije ih samo povezuju. Prav izgled krive ne označava linearnu složenost; T=c·N² je prava na log–log grafiku. Dvije raspodjele su odvojene jer utiču na strukturu Barnes–Hut stabla.', '',
         f'### Rezultati za N={nmax}', '',
         '| Raspodjela | Izvedba | Vrijeme (ms) | Min–max (ms) | Ubrzanje prema direct sequential | L2 greška (%) |',
         '|---|---|---:|---:|---:|---:|']
findings = []
for dist, _ in distributions:
    selected = [r for r in rows if r['distribution'] == dist and int(r['n']) == nmax]
    for r in sorted(selected, key=lambda r: float(r['median_seconds'])):
        lines.append(f'| {dist} | {r["backend"]} | {1000*float(r["median_seconds"]):.3f} | {1000*float(r["min_seconds"]):.3f}–{1000*float(r["max_seconds"]):.3f} | {float(r["speedup_vs_direct"]):.2f}× | {100*float(r["max_relative_l2"]):.5f} |')
    winner = min(selected, key=lambda r: float(r['median_seconds']))
    findings.extend(['', f'Za {dist}, najbrža izmjerena izvedba na N={nmax} je **{winner["backend"]}**, sa {1000*float(winner["median_seconds"]):.3f} ms po evaluaciji.', ''])
    top = sorted(selected, key=lambda r: float(r['median_seconds']))[:2]
    gap = 100 * (float(top[1]['median_seconds']) / float(top[0]['median_seconds']) - 1)
    overlap = max(float(r['min_seconds']) for r in top) <= min(float(r['max_seconds']) for r in top)
    findings.extend([f'Druga po medijani je {top[1]["backend"]}; njeno vrijeme je veće za {gap:.1f}%. ' +
                     ('Rasponi ponavljanja se preklapaju, pa ovo treba opisati kao bliske rezultate, bez tvrdnje o pouzdanoj prednosti.' if overlap else 'Rasponi ovih sedam ponavljanja se ne preklapaju; zaključak i dalje važi samo za ovaj eksperiment.'), ''])
    for backend in ['sequential', 'simd', 'bh_optimized']:
        s = sorted([r for r in rows if r['distribution'] == dist and r['backend'] == backend], key=lambda r: int(r['n']))
        ratio = float(s[-1]['median_seconds']) / float(s[0]['median_seconds'])
        slope = math.log(ratio) / math.log(int(s[-1]['n']) / int(s[0]['n']))
        findings.append(f'- {backend}: između N={s[0]["n"]} i N={s[-1]["n"]}, vrijeme raste {ratio:.2f}×; efektivni eksponent rasta je {slope:.2f}. To je opis ovog intervala, a ne dokaz asimptotske složenosti.')
lines.extend(findings)
lines.extend(['', '## Barnes–Hut: brzina i greška', '', '![Vrijeme i tačnost](theta_tradeoff.png)', '',
              'Dodatno su izmjereni θ=0.2, 0.5 i 0.8 na N=4096. Veće θ dozvoljava više grupnih aproksimacija. Vrijeme i stvarna greška moraju se prikazati zajedno; niža cijena računanja ne znači jednaku tačnost kao Direct.', '',
              '| Raspodjela | θ | Optimizovani Barnes–Hut (ms) | L2 greška (%) |', '|---|---:|---:|---:|'])
for r in theta:
    if r['backend'] == 'bh_optimized':
        lines.append(f'| {r["distribution"]} | {float(r["theta"]):g} | {1000*float(r["median_seconds"]):.3f} | {100*float(r["max_relative_l2"]):.5f} |')
lines.extend(['', '## Trošak Barnes–Hut stabla', '',
              f'Za N={nmax}, bh_optimized i θ=0.5:', '',
              '| Raspodjela | Gradnja (ms) | Obilazak (ms) | Ukupno (ms) | Gradnja / ukupno |',
              '|---|---:|---:|---:|---:|'])
for r in rows:
    if r['backend'] == 'bh_optimized' and int(r['n']) == nmax:
        total, tree = float(r['median_seconds']), float(r['median_tree_seconds'])
        lines.append(f'| {r["distribution"]} | {1000*tree:.3f} | {1000*float(r["median_force_seconds"]):.3f} | {1000*total:.3f} | {100*tree/total:.1f}% |')
lines.extend(['', 'Faze i ukupno vrijeme imaju zasebno izračunate medijane. Njihov zbir se zato ne mora tačno poklopiti; ukupni poziv dodatno uključuje validaciju i druge poslove. Gradnja se u ovom projektu ne paralelizuje.', ''])
spreads = [(float(r['max_seconds']) / float(r['min_seconds']), r) for r in rows]
spread, worst = max(spreads, key=lambda pair: pair[0])
lines.extend(['', '## Ograničenja i tumačenje', '',
              f'- Najveći odnos maksimuma i minimuma ponavljanja je {spread:.2f}× ({worst["backend"]}, N={worst["n"]}, {worst["distribution"]}). Zato se prikazuju medijana i raspon, bez uklanjanja sporijih uzoraka.',
              '- Ovo su lokalna FP32 mjerenja na jednom računaru, jednom seedu i dvije sintetičke raspodjele. Ne predstavljaju univerzalno rangiranje algoritama niti FP64 eksperiment.',
              '- Benchmarki su pokretani redom. Drugi procesi i frekvencija CPU-a nisu kontrolisani.',
              f'- Najkraći cijeli mjereni prolaz od {steps} evaluacija traje {1000*min(float(r["total_seconds"]) for r in raw):.3f} ms. Najkraće slučajeve treba tumačiti uz njihov raspon i ponoviti ako su male razlike presudne.',
              '- Iz ovog raspona N ne treba predviđati tačnu tačku u kojoj će Barnes–Hut postati brži od najboljeg direktnog kernela. Ako presijecanje nije izmjereno, ono se ne tvrdi.',
              '- Relativna greška opisuje ubrzanja na početnom ulazu. Ovaj force benchmark ne provjerava dugoročno očuvanje energije niti tačnost putanja simulacije.', '',
              '## Objašnjenje za odbranu', '',
              '„Poredio sam direktni algoritam i Barnes–Hut na istim ulazima. Direktnom algoritmu optimizacije mijenjaju brzinu izvršavanja, ali zadržavaju kvadratni broj interakcija. Barnes–Hut smanjuje posao aproksimacijom udaljenih grupa, uz trošak stabla i izmjerenu grešku. Vrijeme prikazujem kao medijanu sedam ponavljanja. Ubrzanje prikazujem prema osnovnoj Direct verziji, a zaključke ograničavam na izmjerene veličine i ovaj računar.”', '',
              '## Datoteke i ponovljivost', '',
              '- scaling/raw.csv i theta/raw.csv: sva pojedinačna mjerenja.',
              '- scaling/summary.csv i theta/summary.csv: medijane, rasponi, greške i ubrzanja.',
              '- scaling/metadata.json i theta/metadata.json: tačne komande, sistem i izvršni fajl.',
              '- PNG grafikoni za prezentaciju; SVG verzije za uvećavanje bez gubitka kvaliteta.', ''])
(root / 'izvjestaj.md').write_text('\n'.join(lines), encoding='utf-8')
# Samostalan HTML za otvaranje i štampanje, sa istim tekstom i ugrađenim slikama.
import base64
parts = []
table = False
for line in lines:
    if line.startswith('|'):
        if set(line.replace('|', '').replace('-', '').replace(':', '').strip()) == set():
            continue
        if not table:
            parts.append('<table>')
            table = True
        parts.append('<tr>' + ''.join('<td>' + html.escape(cell.strip()) + '</td>' for cell in line.strip('|').split('|')) + '</tr>')
        continue
    if table:
        parts.append('</table>')
        table = False
    if line.startswith('!['):
        path = line.split('](', 1)[1][:-1]
        data = base64.b64encode((root / path).read_bytes()).decode()
        parts.append(f'<img src="data:image/png;base64,{data}">')
    elif line.startswith('#'):
        level = len(line) - len(line.lstrip('#'))
        parts.append(f'<h{level}>{html.escape(line.lstrip("# "))}</h{level}>')
    elif line:
        parts.append('<p>' + html.escape(line).replace('**', '') + '</p>')
(root / 'izvjestaj.html').write_text('<!doctype html><html lang="bs"><meta charset="utf-8"><title>N-body benchmark</title><style>body{font:16px/1.55 Segoe UI,Arial,sans-serif;max-width:1100px;margin:40px auto;color:#172033;padding:0 24px}h1,h2,h3{line-height:1.2}h2{margin-top:40px}img{width:100%}table{border-collapse:collapse;width:100%;font-size:13px}td{border:1px solid #ccd3df;padding:7px}tr:first-child{background:#edf2fa;font-weight:600}@media print{body{margin:0;font-size:11px}h2{break-after:avoid}img,table{break-inside:avoid}}</style>' + '\n'.join(parts) + '</html>', encoding='utf-8')
print(root / 'izvjestaj.html')
