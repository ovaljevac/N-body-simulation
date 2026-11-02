# Paralelna N-body simulacija gravitacionog djelovanja

Projekat iz predmeta **Paralelni računarski sistemi**: simulacija gravitacionog djelovanja više tijela pomoću algoritama **Direct N-body**, **Barnes–Hut** i **Fast Multipole Method (FMM)**. Za svaki algoritam planirane su sekvencijalna i optimizovana CPU verzija, višejezgrena CPU verzija pomoću **OpenMP-a** te GPU verzija pomoću **CUDA-e**, na grafičkoj kartici **NVIDIA GeForce RTX 3050**.

Cilj je odvojeno ispitati **ubrzanje paralelizacijom istog algoritma** i **razlike između algoritama za isti fizički problem**. Uz vrijeme izvršavanja analiziraju se skalabilnost, troškovi pojedinih faza i tačnost aproksimacija.

Implementiran je **prvi CPU dio za Direct N-body**: sekvencijalna osnova, blokovska verzija, SIMD, OpenMP i OpenMP + SIMD, uz FP32/FP64, leapfrog integraciju, numeričke testove i CSV benchmark. Sav kod, build konfiguracija, testovi, skripte i upute nalaze se u folderu [implementacija/](implementacija/). Rezultati mjerenja generišu se lokalno i nisu dio repozitorija. Barnes–Hut, FMM, CUDA i vizualizacija u nastavku dokumenta i dalje predstavljaju plan rada.

## Dokumentacija

Folder [docs/](docs/) sadrži PDF literaturu. Detaljan popis, izvori i preporučeni redoslijed čitanja nalaze se u [docs/README.md](docs/README.md).

- **[docs/algoritmi/](docs/algoritmi/)**: direktno računanje, hijerarhijske metode i poređenje algoritama.
  - [Fast N-Body Simulation with CUDA](docs/algoritmi/01_Direct_NBody_CUDA_GPU_Gems_3.pdf), Nyland, Harris i Prins: gravitacioni model, direktni algoritam, GPU blokiranje i dijeljena memorija.
  - [N-Body Methods](docs/algoritmi/02_Barnes_NBody_Methods_Lecture_Notes.pdf), Joshua E. Barnes: teorijska osnova N-body i hijerarhijskih metoda.
  - [An Efficient CUDA Implementation of the Tree-Based Barnes Hut n-Body Algorithm](docs/algoritmi/03_Barnes_Hut_CUDA_Burtscher_Pingali_2011.pdf), Burtscher i Pingali: organizacija stabla i GPU paralelizacija.
  - *12 Steps to a Fast Multipole Method on GPUs*, Rio Yokota: [predavanje 1](docs/algoritmi/04_FMM_Yokota_Lecture_1.pdf), [predavanje 2](docs/algoritmi/05_FMM_Yokota_Lecture_2.pdf) i [predavanje 3](docs/algoritmi/06_FMM_Yokota_Lecture_3.pdf).
  - [A Tuned and Scalable Fast Multipole Method as a Preeminent Algorithm for Exascale Systems](docs/algoritmi/07_FMM_Tuning_OpenMP_SIMD_Yokota_Barba.pdf), Yokota i Barba: OpenMP, SIMD i skalabilnost FMM-a.
  - [A Practical Comparison of N-Body Algorithms](docs/algoritmi/08_Practical_Comparison_NBody_Algorithms.pdf), Blelloch i Narlikar: poređenje algoritama pri sličnom nivou greške.
- **[docs/paralelizacija/](docs/paralelizacija/)**: [OpenMP primjeri](docs/paralelizacija/01_OpenMP_Examples_5.2.2.pdf) i [CUDA Programming Guide](docs/paralelizacija/02_CUDA_Programming_Guide.pdf). OpenCL dokumenti ostaju dodatna literatura; planirani GPU backend ovog projekta je CUDA.

Literatura opisuje različite modele i implementacije. Preuzete formule i tehnike prilagođavat će se zajedničkom gravitacionom modelu projekta.

## Algoritmi i obim implementacije

| Algoritam | Princip | Očekivana složenost računanja sila |
| --- | --- | --- |
| **Direct N-body** | Za svako tijelo direktno sabira djelovanja svih ostalih tijela. | O(N²) |
| **Barnes–Hut** | Prostor dijeli stablom; dovoljno udaljene grupe zamjenjuje aproksimacijom preko ukupne mase i centra mase. | Prosječno O(N log N), zavisno od rasporeda i kriterija otvaranja |
| **FMM** | Koristi multipolne i lokalne razvoje za prenos djelovanja između grupa tijela. Bliska djelovanja računa direktno. | Može postići O(N) uz fiksan red razvoja i odgovarajuće uslove |

Za sva tri algoritma planirane su sljedeće varijante:

| Algoritam | Osnovni sekvencijalni CPU | Optimizovani sekvencijalni CPU | Paralelni CPU | GPU |
| --- | --- | --- | --- | --- |
| Direct N-body | C++ | Blokiranje i vektorizacija | OpenMP, zatim OpenMP + SIMD | CUDA, osnovni i tiled kernel |
| Barnes–Hut | C++ i osnovni obilazak stabla | Ravno stablo, iterativni obilazak i bolja lokalnost | OpenMP računanje sila | Hibridni CPU/CUDA pristup |
| FMM | C++ i sve osnovne faze FMM-a | Ponovna upotreba operatora i grupisanje računanja | OpenMP po ćelijama i nivoima stabla | Hibridni CPU/CUDA pristup |

Time se dobija **12 osnovnih varijanti**, uz dodatno poređenje osnovnog i optimizovanog CUDA kernela za Direct N-body. Optimizacije ne mijenjaju fizički model ni parametre tačnosti algoritma.

## Početni plan implementacije

### 1. Zajednički simulator i podaci

Planirani alati su **C++20**, **CMake**, **OpenMP** i **CUDA Toolkit** kompatibilan s RTX 3050 i izabranim kompajlerom. Python i Matplotlib koriste se za analizu CSV rezultata i izradu grafikona.

Simulacija je planirana u **3D prostoru**, s octree stablom za hijerarhijske algoritme. Jednostavan 2D prikaz projekcije tijela služi za demonstraciju. Svako tijelo ima masu, položaj, brzinu i ubrzanje. Podaci se čuvaju u **SoA** rasporedu: odvojeni nizovi za komponente položaja, brzine i ubrzanja te za mase.

Koristit će se zajednički Newtonov gravitacioni model s Plummerovim omekšavanjem, u bezdimenzionalnim jedinicama s `G = 1`:

```text
a_i = G * sum(j != i) [m_j * (r_j - r_i) / (|r_j - r_i|² + epsilon²)^(3/2)]
```

Parametar `epsilon` ograničava sile pri vrlo malim udaljenostima. Isti `epsilon`, vremenski korak `dt` i početno stanje koriste se u svim verzijama. **Multipolni razvoj FMM-a mora odgovarati ovom omekšanom jezgru**; klasične formule za neomekšani potencijal `1/r` zahtijevaju prilagodbu. Potencijal `log(r)` iz nekih 2D FMM primjera predstavlja drugi fizički model.

Položaji i brzine ažuriraju se **Leapfrog kick–drift–kick** integracijom. Svi izračuni sile u jednoj fazi koriste isto stanje položaja. Računanje sila završava prije ažuriranja podataka koje će čitati druga tijela ili niti.

Ulazni skupovi generišu se s fiksnim seed-om: ravnomjerno raspoređena tijela i tijela grupisana u više prostornih klastera. Rezultati mjerenja izvoze se u CSV. Vizualizacija se izvršava odvojeno od mjerenja računskog dijela.

### 2. Direct N-body

Prvo se implementira sekvencijalno sabiranje svih djelovanja. Vanjska petlja obrađuje ciljna tijela, a unutrašnja sva izvorna tijela. Ubrzanje svakog ciljnog tijela ima zaseban akumulator. Ova implementacija služi kao referenca za provjeru ostalih verzija.

Zatim se uvode blokiranje podataka i SIMD, pa OpenMP raspodjela ciljnih tijela između niti. CUDA verzija prvo koristi jednostavan kernel s jednom niti po ciljnom tijelu. Optimizovani kernel učitava blokove izvornih tijela u **shared memory**, tako da podatke ponovo koristi više niti. Svi članovi bloka moraju pravilno učestvovati u sinhronizaciji, uključujući zadnji nepotpuni blok.

### 3. Barnes–Hut

Implementiraju se određivanje granica prostora, izgradnja octree stabla, računanje ukupnih masa i centara masa te obilazak za svako tijelo. Udaljena ćelija prihvata se prema kriteriju `s / d < theta`, uz otvaranje ćelija koje sadrže ciljno tijelo i izostavljanje vlastitog djelovanja.

Osnovna sekvencijalna verzija zatim se optimizuje predstavljanjem čvorova u nizovima, indeksima umjesto pokazivača, iterativnim obilaskom i prostornim sortiranjem tijela. List može sadržavati više tijela; maksimalna dubina stabla sprječava beskonačnu podjelu pri jednakim ili veoma bliskim položajima.

OpenMP verzija paralelizuje računanje sila nad izgrađenim stablom, uz poređenje `static`, `dynamic` i `guided` raspoređivanja. Početna CUDA verzija koristi stablo pripremljeno na CPU-u i obilazak na GPU-u. Posebno se mjere izgradnja stabla i prijenos podataka.

### 4. Fast Multipole Method

FMM koristi prostorno stablo i koeficijente razvoja za odabrano gravitaciono jezgro. Početna implementacija obrađuje sljedeće faze:

1. **P2M:** formiranje multipolnog razvoja iz tijela u listu.
2. **M2M:** prenos i sabiranje multipolnih razvoja od djece prema roditeljima.
3. **M2L:** pretvaranje djelovanja dovoljno udaljene izvorne ćelije u lokalni razvoj ciljne ćelije.
4. **L2L:** prenos lokalnih razvoja prema djeci.
5. **L2P:** računanje udaljenog djelovanja na tijela iz lokalnog razvoja.
6. **P2P:** direktno računanje bliskih djelovanja.

Prvo se provjeravaju pojedinačne faze, zatim kompletno računanje sila. Eksperimenti mijenjaju red razvoja `p` i kapacitet lista, uz praćenje greške i vremena.

Optimizovana CPU verzija ponovo koristi operatore koji zavise samo od relativnog rasporeda ćelija i zajedničkih parametara. OpenMP obrađuje nezavisne ciljne ćelije, uz sinhronizaciju između zavisnih nivoa i faza. Više niti ne smije nekontrolisano ažurirati iste koeficijente.

Za početnu hibridnu CUDA verziju CPU priprema stablo, liste interakcija i operatore, a GPU izvršava računski zahtjevne **P2P i M2L** faze. Ostale faze prenose se na GPU nakon provjere ispravnosti i profilisanja. U rezultatima se navodi tačna podjela rada između CPU-a i GPU-a.

## Tok rada i provjera ispravnosti

1. **Zajednički model i Direct referenca.** Provjeriti pojedinačno tijelo, par tijela i mali ručno provjerljiv skup. Vlastito djelovanje mora biti nula, a rezultati konačni i bez `NaN` vrijednosti.
2. **Integracija.** Provjeriti kretanje bez sile i jednostavnu orbitu dva tijela. Pratiti energiju, ukupni impuls i uticaj smanjenja `dt`; numeričko očuvanje ocjenjuje se tolerancijom i promjenom kroz vrijeme.
3. **Sekvencijalni Barnes–Hut.** Provjeriti masu i centar mase stabla te ubrzanja u odnosu na Direct referencu. Smanjivanje `theta` treba u pravilu smanjivati aproksimacijsku grešku.
4. **Sekvencijalni FMM.** Provjeriti prenos razvoja i liste interakcija: svako djelovanje mora biti obuhvaćeno jednom kroz blisku ili udaljenu obradu. Ispitati uticaj povećanja `p` na grešku i vrijeme.
5. **CPU optimizacije i OpenMP.** Svaka verzija poredi se sa sekvencijalnom verzijom istog algoritma, s istim parametrima i preciznošću. Greške zaokruživanja zbog drugačijeg redoslijeda sabiranja ocjenjuju se dokumentovanim tolerancijama.
6. **CUDA.** Provjeriti GPU rezultate i sinhronizaciju, zatim odvojeno mjeriti kernele, prijenose i cijeli korak. Testirati i broj tijela koji nije djeljiv veličinom bloka.
7. **Konačni eksperimenti.** Tek nakon provjera porediti performanse i pripremiti tabele, grafikone i izvještaj.

Tačnost sila provjerava se na **istom zamrznutom stanju tijela**, prije integracije. Glavna metrika je relativna L2 greška ubrzanja prema Direct referenci, uz pregled najvećih apsolutnih grešaka. Poređenje dugih putanja je dodatna provjera jer se mala početna odstupanja mogu povećavati tokom simulacije.

## Plan mjerenja

| Parametar | Početni skup vrijednosti |
| --- | --- |
| Broj tijela `N` | 1.000, 5.000, 10.000, 50.000 i 100.000; veći skupovi prema raspoloživom vremenu i memoriji |
| Broj CPU niti | 1, 2, 4, 8, do raspoloživih hardverskih niti |
| Raspored tijela | Ravnomjeran i klasterisan |
| Barnes–Hut `theta` | 0,3; 0,5; 0,7 |
| FMM red razvoja `p` | Početno 2, 4, 6; izbor se potvrđuje prema konkretnoj formulaciji i grešci |
| Koraci simulacije | Početno 100; isti broj za poređene verzije |
| Ponavljanja | Zagrijavanje i najmanje 5 mjerenih ponavljanja; medijan i rasipanje vremena |
| Numerička preciznost | FP32 za glavna poređenja; FP64 Direct referenca na manjim skupovima |

Skuplji Direct eksperimenti koriste manji podskup vrijednosti `N`, uz jasno naznačene dostupne rezultate. Ubrzanje se računa samo za zajedničke konfiguracije koje su stvarno izmjerene.

Mjeri se ukupno vrijeme koraka, a za hijerarhijske algoritme posebno izgradnja stabla, priprema interakcija, računanje sila i integracija. GPU vremena kernela mjere se CUDA događajima, a ukupno vrijeme zidnim satom uz sinhronizaciju. Jednokratna inicijalizacija uređaja prikazuje se odvojeno. Redovni prijenosi podataka ulaze u ukupno vrijeme hibridne verzije.

- **Paralelno ubrzanje:** `S = T_seq / T_parallel`, za isti algoritam, parametre i preciznost.
- **CPU efikasnost:** `E = S / P`, gdje je `P` broj niti. Ova formula se ne primjenjuje koristeći broj CUDA jezgri kao `P`.
- **Poređenje algoritama:** vremena Direct, Barnes–Hut i FMM verzija pri istom načinu izvršavanja; aproksimativne metode dodatno se porede pri sličnom nivou greške.
- **Skalabilnost:** vrijeme u zavisnosti od `N` i strong scaling pri fiksnom `N` i promjeni broja CPU niti.
- **Uticaj optimizacija:** osnovna naspram optimizovane verzije, uz isti matematički izračun i kriterije tačnosti.

Za svako mjerenje bilježe se CPU model, RAM, tačna varijanta RTX 3050 i količina VRAM-a, operativni sistem, kompajler i njegove opcije, verzije CUDA-e i drajvera, broj niti, seed i svi simulacijski parametri. Sekvencijalna CPU osnova je optimizovan Release build s jednom niti; postavke automatske vektorizacije evidentiraju se za svaku varijantu.

## Pokretanje implementiranog CPU dijela

Potrebni su CMake 3.20+, C++20 GCC/Clang i OpenMP. Na Windowsu je provjeren GCC iz prenosivog w64devkit-a; klasični MSVC `/openmp` nije podržan za ovaj SIMD eksperiment.

```powershell
# Windows: koristi GCC/CMake sa PATH-a ili lokalne prenosive alate ako postoje.
cd implementacija
./scripts/build_windows.ps1 -Native
./build/nbody_bench.exe --n 1024 --threads 4 --backend all --output results/local/raw.csv
# Isti kod u FP64 preciznosti, bez template-a u implementaciji:
./scripts/build_windows.ps1 -Native -DoublePrecision
./build-double/nbody_bench.exe --n 128 --mode simulate --steps 100
```

```bash
# Linux / GCC, ili Clang uz dostupan libomp
cd implementacija
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/nbody_bench --n 1024 --threads 4 --backend all
```

Python skripte za ponavljanja, statistiku i grafikone opisane su u [uputama implementacije](implementacija/README.md). CSV datoteke, grafikoni i podaci o lokalnom hardveru spremaju se u ignorisani folder `implementacija/results/`.

## Pojmovi i skraćenice

| Pojam | Značenje |
| --- | --- |
| **N-body** | Simulacija sistema više tijela koja međusobno djeluju. |
| **Direct / All-pairs** | Direktno računanje svih parova djelovanja. |
| **Octree** | Stablo koje 3D prostor dijeli na osam podprostora. |
| **Centar mase** | Prosjek položaja ponderisan masama tijela. |
| **`theta`** | Parametar Barnes–Hut kriterija prihvatanja udaljene grupe. |
| **FMM / `p`** | Fast Multipole Method; `p` određuje red aproksimacijskog razvoja. |
| **P2M, M2M, M2L, L2L, L2P, P2P** | Faze formiranja, prenosa i evaluacije FMM razvoja te direktne obrade bliskih tijela. |
| **Softening / `epsilon`** | Omekšavanje gravitacionog jezgra radi ograničavanja sila na malim udaljenostima. |
| **Leapfrog / `dt`** | Metoda vremenske integracije; `dt` je vremenski korak. |
| **SoA** | Structure of Arrays: odvojeni nizovi po fizičkoj veličini. |
| **OpenMP** | CPU paralelizacija pomoću više niti i programskih direktiva. |
| **SIMD** | Jedna procesorska instrukcija obrađuje više vrijednosti. |
| **CUDA kernel** | Funkcija koja se paralelno izvršava na NVIDIA GPU-u. |
| **Tiling / shared memory** | Obrada u blokovima uz ponovno korištenje podataka iz memorije dijeljene unutar GPU bloka. |
| **Hibridna implementacija** | Različite faze istog algoritma izvršavaju se na CPU-u i GPU-u. |
| **FP32 / FP64** | Računanje u jednostrukoj odnosno dvostrukoj preciznosti. |
| **Speedup / strong scaling** | Odnos vremena referentne i paralelne verzije; promjena vremena pri fiksnom problemu i promjeni broja radnika. |
