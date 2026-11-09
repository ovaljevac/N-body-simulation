# N-body CPU implementacija: Direct i Barnes-Hut

Direct računa gravitaciono djelovanje svakog tijela na svako drugo, a Barnes-Hut udaljene grupe zamjenjuje njihovom masom i centrom mase. Implementacija poredi CPU optimizacije uz isti fizički model i provjeru greške aproksimacije.

Kod, CMake konfiguracija, testovi, benchmark i skripte objedinjeni su u ovom folderu. Sve relativne putanje i komande ispod polaze iz foldera `implementacija/`. Literatura je u zasebnom repozitorijskom folderu `docs/`. Generisani rezultati, grafikoni i lokalni metapodaci ostaju u ignorisanom folderu `results/`.

## Šta je završeno

| Varijanta | Šta se mijenja | Broj niti |
| --- | --- | --- |
| `sequential` | Direktna i-j petlja, SoA podaci, skalarni akumulatori | 1 |
| `blocked` | Blokovi ciljnih i izvornih tijela radi ponovne upotrebe podataka | 1 |
| `simd` | Unutrašnje petlje sa `omp simd` redukcijama | 1 |
| `openmp` | Iste petlje kao `sequential`, uz OpenMP raspodjelu ciljnih tijela | Zadani P |
| `openmp_simd` | Iste SIMD petlje uz OpenMP raspodjelu ciljnih tijela | Zadani P |

Ovih pet Direct varijanti podržavaju `float` i `double`, isti 3D model i iste ulaze. Njihova složenost ostaje O(N²), uz O(N) memoriju.

| Barnes-Hut varijanta | Obilazak i redoslijed ciljeva | Niti |
| --- | --- | --- |
| `bh_sequential` | Rekurzivni obilazak djece, ciljevi izvornim redoslijedom | 1 |
| `bh_optimized` | Iterativni obilazak bez steka, ciljevi prostornim redoslijedom | 1 |
| `bh_openmp` | Isti iterativni obilazak, ciljevi podijeljeni OpenMP nitima | Zadani P |

Barnes-Hut koristi FP32/FP64 i isti integrator. Zaseban modul `vizualizacija/` daje interaktivni 2D prikaz na Windowsu. FMM i GPU nisu implementirani.

`src/direct.cpp` sadrži tri direktno čitljive funkcije: `sequential`, `blocked` i `openmp`. `src/simd.cpp` sadrži `simd` i `openmp_simd`. Slične petlje su namjerno napisane zasebno da se razlike mogu pokazati i objasniti.

`src/nbody.h` sadrži strukture i deklaracije, a `src/nbody.cpp` inicijalizaciju, provjeru ulaza, izbor funkcije, referentni proračun i integrator. `main.cpp` vodi benchmark, `tests.cpp` provjerava numeriku, a tri skripte služe buildu i obradi rezultata. Nema template kernela, praznih omotača ni unaprijed dodanih foldera za buduće algoritme.

`src/barnes_hut.cpp` sadrži gradnju octree stabla i tri izvedbe obilaska. Interni čvorovi i funkcije ostaju u tom jednom izvornom fajlu. Svi testovi su objedinjeni u `tests.cpp`, bez posebnog Barnes-Hut testnog programa.

`vizualizacija/main.cpp` sadrži samo prozor, kontrole i crtanje. Njegov `CMakeLists.txt` pravi zasebni `nbody_viewer.exe`, povezan sa postojećim `nbody_core`. Benchmark i testovi ne zavise od prozora.

## Fizički model i podaci

Koristi se Newtonova gravitacija sa Plummerovim omekšavanjem:

```text
dx = x[j] - x[i]; analogno dy i dz
inverse_distance = 1 / sqrt(dx² + dy² + dz² + epsilon²)
a[i] += G * mass[j] * inverse_distance³ * (dx, dy, dz), j != i
```

Podrazumijevano su `G=1`, `epsilon=0.01`, `dt=0.001`. Potencijal za provjeru energije je `-G*m[i]*m[j]/sqrt(r²+epsilon²)`, uz brojanje svakog para jednom. Kinetička energija je `sum(m*v²/2)`.

`Bodies` koristi sedam odvojenih nizova: x/y/z, vx/vy/vz i mass. `Acceleration` sadrži tri izlazna niza. Mase moraju biti pozitivne, G nenegativan, epsilon pozitivan i njegov kvadrat predstavljiv u izabranoj preciznosti. Provjeravaju se dužine nizova i konačnost ulaza. Parametre treba birati tako da međurezultati ne prekorače numerički opseg.

Početni uniformni raspored je u kocki [-1,1]³. Klasterisani raspored su dvije manje kocke sa centrima na x=-0.65 i x=+0.65, poluveličine 0.15. Ukupna početna masa približno je 1; brzine su početno nula. To su kontrolisani testni skupovi za performanse, ne model galaksije u ravnoteži. Generator koristi mt19937_64 i eksplicitno pretvaranje bitova u koordinatu; isti seed i konfiguracija daju isti ulaz. FP64 provjera FP32 testa koristi baš zaokružene FP32 koordinate i mase, pretvorene u double.

## Kako rade optimizacije

Sekvencijalna funkcija prolazi kroz sva izvorna tijela za svaki cilj i preskače vlastito djelovanje. Odvojena funkcija `reference_forces()` računa isti izraz u FP64 iz stvarnih ulaznih koordinata, masa i parametara. Provjera reference je izvan mjerenog vremena.

Blokovski kernel obrađuje ciljni blok, a zatim izvorne blokove. Unutar izvornog bloka više ciljeva koristi iste izvorne podatke. Podrazumijevane veličine su 32 ciljna i 256 izvornih tijela; mijenjaju se parametrima `--target-block` i `--source-block`. Krajnji nepotpuni blok se obrađuje pomoću stvarnog kraja niza. Blokiranje nije garancija ubrzanja: Direct SoA već ima jednostavan pristup memoriji, a dodatne petlje mogu poskupjeti male slučajeve. Vrijednosti blokova nisu predstavljene kao univerzalno optimalne.

SIMD verzija koristi `omp simd reduction(+:sx,sy,sz)`. Da izbjegne grananje za vlastito djelovanje, izvorni interval dijeli na dio prije i poslije ciljnog indeksa. To omogućava stvarno vektorizovanje računanja udaljenosti i sila. Redoslijed sabiranja se razlikuje od sekvencijalnog i poređenje zato koristi tolerancije.

OpenMP raspoređuje ciljna tijela sa `schedule(static)`. Svaka nit upisuje samo ubrzanje svog tijela; nema atomskih operacija niti paralelnog mijenjanja položaja. Svi izvori tokom računanja sila ostaju nepromijenjeni. `target_block` i `source_block` koriste se samo u funkciji `blocked`; ostale verzije ne zavise od veličine bloka.

Za kontrolisan SIMD eksperiment automatska vektorizacija u skalarnom i blokovskom prevodilačkom modulu eksplicitno je isključena. SIMD modul se prevodi zasebno. Koriste se Release optimizacije i `-fno-math-errno`, bez globalnog `-ffast-math`. `NBODY_NATIVE=ON` dodaje `-march=native`; takav binarni program može zahtijevati instrukcije trenutnog procesora. Standardni build bez te opcije je prenosiviji.

U lokalnom GCC build-u SIMD potvrđuju i izvještaj kompajlera i instrukcije `vsqrtps`, `vsqrtpd`, `vdivps`, `vdivpd` u objektnoj datoteci. Sama direktiva u kodu nije uzeta kao dokaz vektorizacije.

## Barnes-Hut: stablo i tri CPU izvedbe

Gradnja počinje kockom koja obuhvata sva tijela. Svaki unutrašnji čvor dijeli se na najviše osam oktanata. Čvor čuva ukupnu masu, centar mase i raspon tijela koja mu pripadaju. Podjela prestaje na `leaf_capacity` tijela, na `max_depth` ili kada su sve pozicije u čvoru jednake. Višestruki list tada obrađuje svoja tijela direktno. Podrazumijevano su kapacitet lista 8 i maksimalna dubina 32; dozvoljene dubine su 1..64, a korijen ima dubinu 0.

Za udaljeni unutrašnji čvor koristi se kriterij `sirina / udaljenost_do_centra_mase < theta`. U kodu je poređenje kvadrirano da ne zahtijeva dodatni korijen. Parametar `theta` je u rasponu 0..1, podrazumijevano 0.5. Manja vrijednost obično znači više rada i manju grešku, ali greška ne mora biti strogo monotona na svakom skupu. `theta=0` otvara sve čvorove i daje Direct proračun do razlike u redoslijedu sabiranja. Listovi se uvijek računaju direktno.

Čvor koji sadrži ciljno tijelo nikad se ne prihvata kao jedna masa. Pripadnost se provjerava preko raspona indeksa u prostornom poretku, a ne preko poređenja koordinata na granicama. U listu se vlastiti indeks preskače. I udaljene mase i pojedinačna tijela koriste isti Plummerov epsilon kao Direct. Mase i centri mase sabiraju se u `double`, a u čvorovima čuvaju kao `Real`; doprinosi ubrzanja računaju se u izabranoj FP32/FP64 preciznosti.

Sve tri verzije grade isto stablo u jednom nizu. Osnovna verzija rekurzivno prati djecu. Optimizovana koristi preorder raspored: poslije otvaranja ide na sljedeći čvor, a poslije prihvatanja ili lista koristi indeks `next`, iza cijelog podstabla. Ne koristi rekurziju niti eksplicitni stek. Ciljeve dodatno obrađuje prostornim redoslijedom radi ponovne upotrebe čvorova u kešu, ali rezultat vraća na izvorni indeks tijela. Optimizacija ne mijenja theta, centar mase ni posjećene izvore; nije obećanje ubrzanja na svakoj veličini.

`bh_openmp` raspoređuje te iste ciljeve sa `schedule(static)`. Niti samo čitaju završeno stablo i upisuju različite elemente ubrzanja. Gradnja je sekvencijalna u sve tri verzije, uključena u ukupno vrijeme, i ponavlja se pri svakom novom računanju sila. Ovo nije paralelna gradnja stabla. Očekivano ponašanje na uobičajenim rasporedima je približno O(N log N), ali theta=0 i degenerisani rasporedi mogu vratiti kvadratni posao.

Primjeri iz foldera `implementacija/`:

```powershell
# Tri BH verzije na istom ulazu i pri istoj aproksimaciji:
./build/nbody_bench.exe --algorithm barnes_hut --n 4096 --theta 0.5 --threads 4
# Samo jedna verzija; theta=0 isključuje grupisanu aproksimaciju:
./build/nbody_bench.exe --backend bh_optimized --n 257 --theta 0
# Simulacija kretanja i ponovno građenje stabla:
./build/nbody_bench.exe --backend bh_openmp --mode simulate --n 257 --steps 20
# Više theta vrijednosti i broj niti; rezultati ostaju lokalno:
python scripts/run_cpu_benchmarks.py --exe build/nbody_bench.exe --algorithm barnes_hut --theta 0.2 0.5 0.8 --sizes 257 1024 4096 --threads 1 2 4 --max-relative-error 0.12 --out results/barnes_hut
python scripts/plot_cpu_results.py results/barnes_hut/summary.csv
```

Bez opcije `--algorithm` ostaje stari Direct izbor. `--algorithm all` uključuje oba algoritma. Eksplicitni `--backend bh_...` bira traženu BH verziju bez dodatne opcije algorithm.

CSV dodaje algorithm, theta, kapacitet lista, maksimalnu dubinu, toleranciju, broj čvorova, dostignutu dubinu te odvojena vremena gradnje i obilaska po koraku. Oslobađanje stabla i vanjska validacija nisu dio te dvije faze, ali jesu dio ukupnog vremena. Broj čvorova i dubina odnose se na posljednju evaluaciju. Direct u BH faznim kolonama ima nule. Poređenje sa Direct referencom je izvan tajmera i ostaje O(N²), pa veliki ulazi imaju trošak provjere i kada je sam Barnes-Hut brži.

Približni BH prolaz odbija početni rezultat ako relativna L2 greška pređe `--max-relative-error` (podrazumijevano 0.05). To je izmjereni prag za ulaz, ne matematička garancija za svaki theta i svako tijelo. Za theta=0 i za Direct ostaju stroži pragovi 1e-4 u FP32 i 1e-11 u FP64.

## Leapfrog integracija

Prije prvog koraka izračunati početna ubrzanja. `step()` zatim radi:

1. Pola promjene brzine pomoću postojećeg ubrzanja.
2. Pomjeranje svih položaja za puni dt.
3. Izračun svih novih ubrzanja iz istog novog stanja položaja.
4. Preostalih pola promjene brzine.

Ubrzanja moraju odgovarati trenutnim položajima prije poziva koraka. Solver ne računa i ne pomjera pojedinačno tijelo unutar iste force petlje, što bi miješalo stara i nova stanja. Za veće vremenske korake ili kolaps klastera treba analizirati grešku energije i smanjiti dt; softening sam ne garantuje stabilnu integraciju.

## Build i pokretanje

Potreban je C++20 GCC ili Clang, CMake i OpenMP runtime. Windows je provjeren sa prenosivim GCC/OpenMP alatom. MSVC podrška nije deklarisana jer klasični `/openmp` nije dovoljan za ovaj `omp simd` eksperiment. Linux komande su u glavnom README-u; Linux/Clang nije lokalno testiran.

Iz foldera `implementacija/` na Windowsu:

```powershell
./scripts/build_windows.ps1 -Native
./build/nbody_bench.exe --n 4096 --threads 4 --backend all --output results/local/raw.csv
./scripts/build_windows.ps1 -Native -DoublePrecision
./build-double/nbody_bench.exe --n 128 --mode simulate --steps 100 --output results/local/simulation.csv
```

Preciznost se bira pri build-u: `Real` je podrazumijevano `float`, a opcija `NBODY_DOUBLE=ON` bira `double`. Tako nema template-a kroz algoritme ni odabira preciznosti tokom izvršavanja. Stara CLI opcija `--precision` je uklonjena. Za FP64 na Linuxu koristiti `cmake -S . -B build-double -DCMAKE_BUILD_TYPE=Release -DNBODY_DOUBLE=ON`, zatim build i CTest nad tim folderom.

Windows skripta koristi alate sa PATH-a ili prenosive alate iz radnog prostora, ako su prisutni. Na drugom računaru može se zadati `-ToolsBin C:/putanja/w64devkit/bin`, uz CMake na PATH-u. Alati nisu dio izvornog repozitorija i skripta ništa ne preuzima niti sistemski instalira.

## Interaktivni 2D prikaz

Na Windowsu build skripta pravi i zaseban `build/nbody_viewer.exe`. Prije ponovnog kompajliranja zatvoriti otvoreni prozor simulatora. Pokretanje iz foldera `implementacija/`:

```powershell
./scripts/build_windows.ps1 -Native
./build/nbody_viewer.exe
```

Prozor omogućava izbor broja tijela (1..10000), seeda, početnog rasporeda, svih osam CPU izvedbi, broja niti i BH theta parametra. Broj niti je aktivan samo za paralelne izvedbe, a theta samo za Barnes-Hut. Dugme Novi bira nasumičan seed; isti seed i postavke uvijek vraćaju isti početni raspored.

Pri otvaranju prozor očita broj fizičkih jezgri preko Windows API-ja i broj logičkih procesora dostupnih OpenMP-u. Unos niti ograničen je na taj dostupni broj. Status prikazuje veličinu stvarno pokrenutog OpenMP tima, provjerenu pri resetu, odnosno jednu nit za sekvencijalne izvedbe. Niti se ne vežu za konkretne jezgre; njihov raspored određuje operativni sistem. Ovo nije mjera procenta opterećenja CPU-a. Ako očitavanje fizičkih jezgri ne uspije, prikazuje se znak `?`.

Pokreni nastavlja pauzirano stanje. Ako su postavke promijenjene, prvo priprema novo početno stanje. Pauziraj zaustavlja korake, Resetuj vraća početno stanje sa trenutnim postavkama, a Uklopi prikaz centrira kameru. Zum točkom miša mijenja samo pogled. Postavke se tokom rada zaključavaju; za promjenu prvo pauzirati.

Početni položaji dolaze iz istog generatora kao u benchmarku, uz z=0 i početne brzine nula. Ovo je ravanska Newtonova simulacija: isti 3D gravitacioni zakon djeluje u ravni XY, bez uvođenja logaritamskog 2D potencijala. Prozor poziva postojeće `accelerations()` i `step()`; ne kopira algoritme. Četiri fiksna koraka dt=0.002 izvode se pri osvježavanju. Sporiji računar usporava prikaz umjesto da povećava dt. Za jedno tijelo, uz nultu početnu brzinu i bez drugih tijela, položaj ostaje isti.

Prikaz koristi ugrađeni Windows GDI, bez dodatnih grafičkih biblioteka i bez Python zavisnosti. Sve se nalazi u modulu `vizualizacija/`. CMake opcija `-DNBODY_VIEWER=OFF` isključuje taj modul i ostavlja numerički dio, testove i benchmark. FP64 build zasebno pravi `build-double/nbody_viewer.exe`.

Vrijeme u prozoru označava posljednji prosjek računanja koraka, bez crtanja. To je informativna vrijednost, ne zamjena za benchmark sa zagrijavanjem, ponavljanjima i statistikom. Početna stanja u prozoru su ravanska, dok postojeća benchmark skripta zadržava 3D ulaze. Postojećih sedam testnih scenarija ostaje uključeno: izgled animacije ne dokazuje numeričku tačnost.

## Šta benchmark mjeri

- `--mode force` ponavlja računanje ubrzanja na zamrznutom ulazu. `--steps` je broj force evaluacija u jednom ponavljanju; CSV `seconds_per_step` je vrijeme jedne evaluacije.
- `--mode simulate` mjeri zadani broj leapfrog koraka. Početna force evaluacija i priprema izlaznih nizova nisu dio vremena koraka; redovne force evaluacije, integracija i validacija ulaza jesu. Sve naredne Barnes-Hut gradnje, sa svojim alokacijama, uključene su u vrijeme u oba režima.
- Podrazumijevano je jedno cijelo zagrijavanje i pet mjerenih ponavljanja. Svako ponavljanje počinje iz istog početnog stanja.
- Prije mjerenja provjerava se sila prema FP64 sekvencijalnoj referenci. CSV `relative_l2` i `max_absolute` odnose se na početni zamrznuti ulaz, ne na konačno stanje različitih putanja.
- Provjera energije, impulsa i checksum računaju se izvan tajmera. U force režimu energijski drift je nula jer nema integracije. Za nultu početnu energiju normalizacija koristi donju granicu 1e-30.
- U vrijeme svakog javnog poziva solvera uključena je O(N) validacija ulaza. Ne tvrdi se da su vremena isključivo unutrašnji kernel.
- CSV čuva stvarni broj niti, traženi broj, seed, parametre modela, blokove, preciznost, ponavljanje, verziju kompajlera, build tip i native opciju.

Kompletna mala CPU serija i obrada:

```powershell
python scripts/run_cpu_benchmarks.py --exe build/nbody_bench.exe --out results/local --sizes 256 1024 4096 --threads 1 2 4 --repeats 5 --steps 3
python scripts/plot_cpu_results.py results/local/summary.csv
```

Prva skripta koristi samo Python standardnu biblioteku i čuva `raw.csv`, `summary.csv` i `metadata.json` sa tačnim komandama i podacima o sistemu. Druga zahtijeva Matplotlib i pravi PNG/SVG grafikon. Namjerno se ne pokreće N=100.000 i stotinu koraka kao podrazumijevani test.

### Proširena mjerenja i izvještaj za odbranu

Sljedeće komande pokretati redom, bez istovremenog izvršavanja drugih benchmarka:

```powershell
python scripts/run_cpu_benchmarks.py --exe build/nbody_bench.exe --algorithm all --sizes 512 1024 2048 4096 8192 16384 --threads 1 2 4 --repeats 7 --steps 10 --out results/odbrana_2026-11-08/scaling
python scripts/run_cpu_benchmarks.py --exe build/nbody_bench.exe --algorithm barnes_hut --sizes 4096 --theta 0.2 0.5 0.8 --threads 1 4 --repeats 7 --steps 10 --max-relative-error 0.12 --out results/odbrana_2026-11-08/theta
python scripts/plot_cpu_results.py results/odbrana_2026-11-08/scaling/summary.csv
python scripts/defence_report.py results/odbrana_2026-11-08
```

Graf vremena koristi šest stvarno izmjerenih veličina N i sjenčenje minimum–maksimum; obje ose su logaritamske. `defence_report.py` dodatno pravi grafik skaliranja nitima na najvećem N, odnos vremena i greške za theta seriju te `izvjestaj.md` i samostalan `izvjestaj.html` sa ugrađenim slikama. Izvještaj opisuje ovu FP32 seriju sa navedenim parametrima; za drugačiji eksperiment treba prilagoditi i tekst metodologije. Rezultati ostaju u lokalnom ignorisanom folderu `results/`.

`speedup_vs_sequential` poredi sa osnovnom verzijom istog algoritma (`sequential` ili `bh_sequential`). `speedup_vs_matching_serial` za OpenMP koristi `sequential`, za OpenMP+SIMD `simd`, a za BH OpenMP `bh_optimized` pri istom theta. CPU efikasnost je taj drugi odnos podijeljen stvarnim brojem niti. BH ukupno ubrzanje uključuje i sekvencijalnu gradnju. Ako je zadan `--algorithm all`, `speedup_vs_direct` dodatno poredi sa Direct sekvencijalnom verzijom; BH pri tome ima izmjerenu aproksimacijsku grešku. Stari lokalni rezultati ostaju odvojeno.

## Provjera ispravnosti

`tests.cpp` sadrži sedam scenarija: pet zajedničkih i dva za Barnes-Hut. Isti program pokreće se u FP32 i FP64 buildu.

| Scenario | Šta se provjerava |
| --- | --- |
| Prazan sistem i jedno tijelo | Prazni izlazi i nulto vlastito djelovanje, u svim verzijama. |
| Dva tijela sa poznatim rezultatom | Smjer i iznos ubrzanja za mase 2 i 3; BH list kapaciteta 1 i theta=1 provjeravaju otvaranje roditelja koji sadrži cilj. |
| Jednake pozicije | Za 17 tijela na istom položaju sva ubrzanja moraju ostati nula, bez dijeljenja nulom ili beskonačne podjele stabla. |
| Referenca i paralelizacija | N=33,257, seed=42, oba rasporeda, blokovi 7/19; rezultat prema Direct FP64 referenci, a paralelne verzije dodatno identično na 1 i 4 niti. |
| Kratka simulacija | Deset koraka dt=0.001 za 33 tijela; poređenje sve tri koordinate i brzine sa Direct sekvencijalnom putanjom. |
| BH aproksimacija | N=257, theta=0.5, oba rasporeda; relativna L2 greška ispod 0.01 na ovim skupovima i identični rezultati tri BH izvedbe. |
| BH maksimalna dubina | Jedno udaljeno i 32 bliska tijela, list kapaciteta 1, dubina 4; podjela staje na granici i nijedno tijelo nije izgubljeno. |

Zajedničko poređenje i simulacija koriste theta=0, sa tolerancijom 2e-5 u FP32 i 2e-12 u FP64. Poseban BH scenario provjerava nenultu theta. Testovi koriste eksplicitne provjere aktivne i u Release buildu. CTest pokreće ovaj program i još dva kratka testa pokretanja: mjerenje sila i simulaciju sa `--algorithm all`. Dugi kombinatorni testovi i pojedinačni CLI testovi nisu dio ovog skupa; provjere ulaza u programu ostaju uključene.

Performanse se porede odvojenom zajedničkom skriptom `run_cpu_benchmarks.py`, nad istim ulazima i brojem ponavljanja. Sedam scenarija provjerava ispravnost, a nije sedam različitih benchmarka. Kratki testovi nisu garancija dugoročne stabilnosti; jednostrani Barnes-Hut približni proračun ne nameće tačno očuvanje impulsa.

## Izvori i vlastiti doprinos

Kod je implementiran za ovaj projekat. Fizički model i all-pairs princip odgovaraju pripremljenoj literaturi, posebno Nyland/Harris/Prins, *Fast N-Body Simulation with CUDA*, i Belleman i saradnici, *High performance direct gravitational N-body simulations ... II*. Njihovi GPU rezultati nisu predstavljeni kao naši. GPU tiling iz literature još nije implementiran; sadašnje blokiranje odnosi se na CPU keš i SoA podatke.

Barnes-Hut koristi octree, masu/centar mase i kriterij otvaranja iz pripremljene literature (Barnesove *N-Body Methods* bilješke i Burtscher/Pingali, *An Efficient CUDA Implementation of the Tree-Based Barnes Hut n-Body Algorithm*). Ovdje su implementirane CPU verzije, bez preuzimanja njihovih GPU rezultata.
