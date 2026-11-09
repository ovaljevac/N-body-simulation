#include "src/nbody.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace nbody;
const double tolerance = sizeof(Real) == 4 ? 2e-5 : 2e-12;

// Eksplicitna provjera ostaje aktivna i u Release buildu. Testovi provjeravaju
// rezultat; vrijeme izvrsavanja mjeri odvojeni zajednicki benchmark.
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
bool same(const Acceleration& a, const Acceleration& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

// Prazan ulaz mora dati prazne izlaze, a jedno tijelo nulto ubrzanje.
// Isti scenario prolazi kroz sve verzije i otkriva vlastito djelovanje.
void empty_and_single() {
    for (auto backend : backends) {
        Acceleration a;
        accelerations(Bodies(0), a, Config{}, backend);
        require(a.x.empty() && a.y.empty() && a.z.empty(), "Prazan sistem");
        accelerations(Bodies(1), a, Config{}, backend);
        require(a.x[0] == 0 && a.y[0] == 0 && a.z[0] == 0, "Tijelo ne djeluje na sebe");
    }
}

// Dva tijela na razmaku 1 imaju rezultat koji racunamo direktno iz formule.
// Razlicite mase provjeravaju i smjer i zavisnost ubrzanja od mase drugog tijela.
void known_pair() {
    Bodies bodies(2);
    bodies.x = {Real(-0.5), Real(0.5)};
    bodies.mass = {2, 3};
    Config config;
    config.leaf_capacity = 1;
    config.theta = 1; // BH mora otvoriti roditelja koji sadrzi cilj i pri sirokom kriteriju.
    const double factor = 1 / std::pow(1 + double(config.epsilon) * config.epsilon, 1.5);
    for (auto backend : backends) {
        Acceleration a;
        accelerations(bodies, a, config, backend);
        require(std::abs(a.x[0] - 3 * factor) < tolerance && std::abs(a.x[1] + 2 * factor) < tolerance
                && a.y[0] == 0 && a.y[1] == 0 && a.z[0] == 0 && a.z[1] == 0, "Analiticki par");
    }
}

// Poklopljene pozicije provjeravaju softening i zavrsavanje BH podjele.
// Kapacitet lista 1 namjerno trazi podjelu, ali duplikati moraju ostati u listu.
void coincident_positions() {
    Config config;
    config.leaf_capacity = 1;
    for (auto backend : backends) {
        Acceleration a;
        accelerations(Bodies(17), a, config, backend);
        for (const auto* values : {&a.x, &a.y, &a.z})
            require(std::all_of(values->begin(), values->end(), [](Real v) { return v == 0; }), "Jednake pozicije");
    }
}

// Neparne velicine i blokovi 7/19 provjeravaju posljednji nepotpuni blok.
// Na istim ulazima provjeravamo rezultate prema FP64 referenci.
void reference_comparison() {
    Config config;
    config.theta = 0; // Svi algoritmi tada racunaju bez grupne aproksimacije.
    config.target_block = 7;
    config.source_block = 19;
    for (auto distribution : {"uniform", "clustered"}) {
        for (std::size_t n : {33, 257}) {
            const auto bodies = initial_state(n, 42, distribution);
            const auto reference = reference_forces(bodies, config);
            for (auto backend : backends) {
                Acceleration a;
                accelerations(bodies, a, config, backend);
                require(compare(a, reference).relative_l2 < tolerance, "Direct referenca: " + std::string(name(backend)));
            }
        }
    }
}

// Svaka verzija krece iz istog stanja i prolazi deset koraka kao Direct osnova.
// theta=0 iskljucuje aproksimaciju da razlike putanje ne sakriju gresku implementacije.
void short_simulation() {
    Config config;
    config.theta = 0;
    const auto initial = initial_state(33, 42, "uniform");
    auto expected = initial;
    Acceleration reference;
    accelerations(expected, reference, config, Backend::sequential);
    for (int k = 0; k < 10; ++k) step(expected, reference, config, Backend::sequential, 0.001);
    for (auto backend : backends) {
        auto bodies = initial;
        Acceleration a;
        accelerations(bodies, a, config, backend);
        for (int k = 0; k < 10; ++k) step(bodies, a, config, backend, 0.001);
        for (std::size_t i = 0; i < bodies.size(); ++i) {
            const double position_error = std::abs(bodies.x[i] - expected.x[i]) + std::abs(bodies.y[i] - expected.y[i]) + std::abs(bodies.z[i] - expected.z[i]);
            const double velocity_error = std::abs(bodies.vx[i] - expected.vx[i]) + std::abs(bodies.vy[i] - expected.vy[i]) + std::abs(bodies.vz[i] - expected.vz[i]);
            require(position_error + velocity_error < tolerance, "Kratka simulacija: " + std::string(name(backend)));
        }
    }
}

// Pri nenultom theta dozvoljavamo malu gresku prema Direct, ovdje do 1% L2.
// Tri BH izvedbe moraju dati isti rezultat; prag vazi za ove izabrane ulaze.
void bh_approximation() {
    Config config; // theta=0.5: sve tri BH verzije imaju istu aproksimaciju.
    for (auto distribution : {"uniform", "clustered"}) {
        const auto bodies = initial_state(257, 42, distribution);
        const auto reference = reference_forces(bodies, config);
        Acceleration baseline;
        accelerations(bodies, baseline, config, Backend::bh_sequential);
        for (auto backend : {Backend::bh_sequential, Backend::bh_optimized}) {
            Acceleration a;
            accelerations(bodies, a, config, backend);
            require(compare(a, reference).relative_l2 < 0.01, "BH greska mora biti ispod 1% na ovom skupu");
            require(same(a, baseline), "BH optimizacije ne smiju promijeniti rezultat");
        }
    }
}

// Udaljeno tijelo siri korijen, a bliska grupa trazi mnogo narednih podjela.
// Na dubini 4 preostala tijela moraju ostati u listu i sva uci u proracun.
void bh_depth_limit() {
    Config config;
    config.theta = 0;
    config.leaf_capacity = 1;
    config.max_depth = 4;
    Bodies bodies(33);
    bodies.x[0] = 1;
    for (std::size_t i = 1; i < bodies.size(); ++i) bodies.x[i] = Real(1e-8 * double(i));
    const auto reference = reference_forces(bodies, config);
    for (auto backend : {Backend::bh_sequential, Backend::bh_optimized}) {
        Acceleration a;
        SolverTiming timing;
        accelerations(bodies, a, config, backend, &timing);
        require(timing.tree_depth == config.max_depth, "Podjela mora stati na maksimalnoj dubini");
        require(compare(a, reference).relative_l2 < tolerance, "Tijela u zadnjem listu ne smiju biti izgubljena");
    }
}

int main() {
    try {
        empty_and_single();
        known_pair();
        coincident_positions();
        reference_comparison();
        short_simulation();
        bh_approximation();
        bh_depth_limit();
        std::cout << "PASS: 7 test scenarios (" << (sizeof(Real) == 4 ? "FP32" : "FP64") << ")\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
