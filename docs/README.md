# Literatura za N-body projekat

Preuzeto 1. novembra 2026. Ukupno 13 PDF dokumenata. Originalni dokumenti su sačuvani bez izmjena.

## Algoritmi

| PDF | Izvor i namjena | Stranica |
| --- | --- | ---: |
| [01 Direct N-body](algoritmi/01_Direct_NBody_CUDA_GPU_Gems_3.pdf) | Lars Nyland, Mark Harris i Jan Prins: *Fast N-Body Simulation with CUDA*, GPU Gems 3, 2007. Gravitacioni model, softening, direktno računanje i GPU blokiranje s dijeljenom memorijom. [Izvor PDF-a](https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/nbody/doc/nbody_gems3_ch31.pdf) | 20 |
| [02 Barnesove bilješke o N-body metodama](algoritmi/02_Barnes_NBody_Methods_Lecture_Notes.pdf) | Joshua E. Barnes: *N-Body Methods*, Astronomy 626, University of Hawaiʻi, materijali za 2005. Teorijski uvod u numeričke N-body metode, direktno računanje i hijerarhijske metode. [Izvor PDF-a](https://home.ifa.hawaii.edu/users/barnes/ast626_05/nbm.pdf) | 8 |
| [03 Barnes–Hut na GPU-u](algoritmi/03_Barnes_Hut_CUDA_Burtscher_Pingali_2011.pdf) | Martin Burtscher i Keshav Pingali: *An Efficient CUDA Implementation of the Tree-Based Barnes Hut n-Body Algorithm*, GPU Computing Gems Emerald Edition, 2011. Stablo, računanje masa, obilazak i GPU optimizacije. [Izvor PDF-a](https://userweb.cs.txstate.edu/~burtscher/papers/gcg11.pdf) | 18 |
| [04 FMM predavanje 1](algoritmi/04_FMM_Yokota_Lecture_1.pdf) | Rio Yokota: *12 Steps to a Fast Multipole Method on GPUs*, PASI 2011, prvo predavanje. [Izvor PDF-a](https://www.bu.edu/pasi/files/2011/01/RioYokota1-06-0900.pdf) | 17 |
| [05 FMM predavanje 2](algoritmi/05_FMM_Yokota_Lecture_2.pdf) | Isti kurs, drugo predavanje. [Izvor PDF-a](https://www.bu.edu/pasi/files/2011/01/RioYokota2-07-0900.pdf) | 13 |
| [06 FMM predavanje 3](algoritmi/06_FMM_Yokota_Lecture_3.pdf) | Isti kurs, treće predavanje. [Izvor PDF-a](https://www.bu.edu/pasi/files/2011/01/RioYokota3-08-0900.pdf) | 14 |
| [07 FMM optimizacija i skalabilnost](algoritmi/07_FMM_Tuning_OpenMP_SIMD_Yokota_Barba.pdf) | Rio Yokota i Lorena A. Barba: *A Tuned and Scalable Fast Multipole Method as a Preeminent Algorithm for Exascale Systems*. Preprint rada objavljenog 2012; OpenMP, SIMD i analiza skalabilnosti. [Izvor PDF-a](https://arxiv.org/pdf/1106.2176) | 12 |
| [08 Poređenje N-body algoritama](algoritmi/08_Practical_Comparison_NBody_Algorithms.pdf) | Guy Blelloch i Girija Narlikar: *A Practical Comparison of N-Body Algorithms*, 1997. Barnes–Hut, FMM i PMTA, uz poređenje pri sličnim nivoima greške. [Izvor PDF-a](https://www.cs.cmu.edu/~scandal/papers/dimacs-nbody.pdf) | 16 |

Tri Yokotina predavanja čine jednu cjelinu. [Stranica kursa](https://www.bu.edu/pasi/courses/12-steps-to-having-a-fast-multipole-method-on-gpus/) i [popis materijala](https://www.bu.edu/pasi/materials/) sadrže i dodatne materijale i link na izvorni kod.

## Paralelizacija

| PDF | Izvor i namjena | Stranica |
| --- | --- | ---: |
| [01 OpenMP primjeri](paralelizacija/01_OpenMP_Examples_5.2.2.pdf) | OpenMP ARB: *OpenMP Application Programming Interface Examples*, 5.2.2, april 2024. Paralelne petlje, raspoređivanje, sinhronizacija i SIMD. [Izvor PDF-a](https://www.openmp.org/wp-content/uploads/openmp-examples-5.2.2-final.pdf) | 578 |
| [02 CUDA priručnik](paralelizacija/02_CUDA_Programming_Guide.pdf) | NVIDIA: *CUDA Programming Guide*, izdanje 13.4.2, 10. septembar 2026. GPU model, niti, blokovi, memorija i sinhronizacija. [Izvor PDF-a](https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf) | 734 |
| [03 OpenCL API specifikacija](paralelizacija/03_OpenCL_API_Specification.pdf) | Khronos: *The OpenCL Specification*, v3.1.2, 17. septembar 2026. Host API, uređaji, konteksti, redovi naredbi, baferi i profilisanje. [Izvor PDF-a](https://registry.khronos.org/OpenCL/specs/unified/pdf/OpenCL_API.pdf) | 672 |
| [04 OpenCL kratki priručnik](paralelizacija/04_OpenCL_Reference_Guide_3.0.pdf) | Khronos: *OpenCL 3.0 Reference Guide*, revizija 09/2022. Kratka referenca API-ja i jezika. [Izvor PDF-a](https://www.khronos.org/files/opencl30-reference-guide.pdf) | 12 |
| [05 OpenCL C specifikacija](paralelizacija/05_OpenCL_C_Specification.pdf) | Khronos: *The OpenCL C Specification*, v3.1.2, 17. septembar 2026. Jezik za GPU kernele, memorijski prostori i ugrađene funkcije. [Izvor PDF-a](https://registry.khronos.org/OpenCL/specs/unified/pdf/OpenCL_C.pdf) | 330 |

## Redoslijed čitanja

1. Direct N-body poglavlje: fizički model i osnovna GPU optimizacija.
2. Barnesove bilješke, pa Barnes–Hut GPU rad.
3. Tri Yokotina FMM predavanja, redom.
4. Rad o FMM optimizaciji i rad o poređenju algoritama.
5. OpenMP i izabrani GPU priručnik konsultovati tokom implementacije.

## Zamjene i napomene

- Originalni rad Barnesa i Huta iz 1986, *A hierarchical O(N log N) force-calculation algorithm*, nije preuzet kao PDF: izdavač je vratio HTML stranicu, a pokušaj javne arhive HTTP 403. Njegova [izdavačka stranica](https://www.nature.com/articles/324446a0) ostaje bibliografski izvor. Za dostupni teorijski PDF dodate su bilješke samog Barnesa, jasno imenovane kao zaseban dokument.
- Prethodno navedena ExaFMM stranica je web dokumentacija. PDF 07 je povezani rad autora o optimizaciji i skalabilnosti, a ne PDF kopija cijele dokumentacije. [ExaFMM dokumentacija](https://www.bu.edu/exafmm/documentation/overview/).
- Khronos OpenCL Guide je web vodič. Umjesto tvrdnje da postoji identičan PDF, dodati su službeni API i OpenCL C dokumenti te kratki priručnik. [OpenCL Guide](https://github.com/KhronosGroup/OpenCL-Guide).
- Stariji radovi koriste tadašnji hardver i alate. Njihove izmjerene brzine nisu očekivani rezultati našeg projekta; algoritamske tehnike treba provjeriti vlastitim mjerenjima.
- Verzija dokumentacije ne potvrđuje podršku konkretne kartice ili kompajlera. Implementacija će koristiti mogućnosti dostupne na izabranom hardveru.
- PDF 08 koristi starije kodiranje fontova: automatsko izdvajanje teksta može biti nepouzdano.

Izvorni linkovi i broj stranica svakog PDF-a navedeni su u tabelama iznad.
