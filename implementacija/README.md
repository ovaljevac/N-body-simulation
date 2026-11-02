# Direct N-body prvi CPU dio

Direktni algoritam računa gravitaciono djelovanje svakog tijela na svako drugo tijelo. Implementacija je namijenjena poređenju CPU optimizacija, uz jasno vidljive petlje i isti fizički model.

Kod, CMake konfiguracija, testovi, benchmark i skripte objedinjeni su u ovom folderu. Sve relativne putanje i komande ispod polaze iz foldera `implementacija/`. Literatura je u zasebnom repozitorijskom folderu `docs/`. Generisani rezultati, grafikoni i lokalni metapodaci ostaju u ignorisanom folderu `results/`.

## Šta je završeno

| Varijanta | Šta se mijenja | Broj niti |
| --- | --- | --- |
| `sequential` | Direktna i-j petlja, SoA podaci, skalarni akumulatori | 1 |
| `blocked` | Blokovi ciljnih i izvornih tijela radi ponovne upotrebe podataka | 1 |
| `simd` | Unutrašnje petlje sa `omp simd` redukcijama | 1 |
| `openmp` | Iste petlje kao `sequential`, uz OpenMP raspodjelu ciljnih tijela | Zadani P |
| `openmp_simd` | Iste SIMD petlje uz OpenMP raspodjelu ciljnih tijela | Zadani P |

Sve varijante podržavaju `float` i `double`, isti 3D model i iste ulaze. Računska složenost svih pet ostaje O(N²), uz O(N) memoriju. Ovo je implementacija Direct algoritma. Barnes–Hut, FMM, GPU i prikaz nisu implementirani u ovoj fazi.

`src/direct.cpp` sadrži tri direktno čitljive funkcije: `sequential`, `blocked` i `openmp`. `src/simd.cpp` sadrži `simd` i `openmp_simd`. Slične petlje su namjerno napisane zasebno da se razlike mogu pokazati i objasniti.

`src/nbody.h` sadrži strukture i deklaracije, a `src/nbody.cpp` inicijalizaciju, provjeru ulaza, izbor funkcije, referentni proračun i integrator. `main.cpp` vodi benchmark, `tests.cpp` provjerava numeriku, a tri skripte služe buildu i obradi rezultata. Nema template kernela, praznih omotača ni unaprijed dodanih foldera za buduće algoritme.

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

## Šta benchmark mjeri

- `--mode force` ponavlja računanje ubrzanja na zamrznutom ulazu. `--steps` je broj force evaluacija u jednom ponavljanju; CSV `seconds_per_step` je vrijeme jedne evaluacije.
- `--mode simulate` mjeri zadani broj leapfrog koraka. Početna force evaluacija i alokacije nisu dio vremena koraka; redovne force evaluacije, integracija i validacija ulaza jesu.
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

`speedup_vs_sequential` pokazuje ukupno ubrzanje prema skalarnoj osnovi. `speedup_vs_matching_serial` za OpenMP poredi sa `sequential`, a za OpenMP+SIMD sa `simd`, jer su to sada odgovarajuće serijske petlje. CPU efikasnost računa se iz tog drugog odnosa podijeljenog stvarnim brojem niti. Stare lokalne rezultate iz prethodne strukture treba zadržati odvojeno; njihove OpenMP verzije koristile su blokovske petlje.

## Provjera ispravnosti

CTest pokreće numeričke provjere, kratko mjerenje sila, kratku simulaciju i provjere neispravnih CLI parametara. Isti testovi se pokreću u FP32 i FP64 build-u. Numeričke provjere uključuju:

- prazan i jednočlani sistem, vlastito djelovanje i jednaku poziciju tijela uz softening;
- analitičko ubrzanje para sa različitim masama i odgovarajuću promjenu impulsa;
- FP32/FP64, uniformne/klasterisane ulaze i veličine koje nisu djeljive blokovima;
- sve backend-e na 1, 2 i 4 niti, poređenje sa FP64 referencom i nepromijenjene ulazne podatke;
- translacionu i permutacionu invarijantnost i skaliranje mase;
- slobodno kretanje, dosljednost integracije između backend-a i poboljšanje orbitne greške smanjenjem dt;
- neispravne mase, SoA dužine, NaN podatke, parametre blokova i nepoznat backend.

Testovi koriste eksplicitne provjere koje ostaju uključene u Release build-u. Broj izvršenih provjera ispisuje `nbody_tests`. Testovi nisu zamjena za profilisanje velikih skupova, analizu dugih putanja ili konačnu evaluaciju za predmet.

## Izvori i vlastiti doprinos

Kod je implementiran za ovaj projekat. Fizički model i all-pairs princip odgovaraju pripremljenoj literaturi, posebno Nyland/Harris/Prins, *Fast N-Body Simulation with CUDA*, i Belleman i saradnici, *High performance direct gravitational N-body simulations ... II*. Njihovi GPU rezultati nisu predstavljeni kao naši. GPU tiling iz literature još nije implementiran; sadašnje blokiranje odnosi se na CPU keš i SoA podatke.
