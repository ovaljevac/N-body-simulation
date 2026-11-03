#include "nbody.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace nbody {

Bodies::Bodies(std::size_t count)
    : x(count), y(count), z(count), vx(count), vy(count), vz(count), mass(count, 1) {}

std::string_view name(Backend backend) {
    switch (backend) {
    case Backend::sequential: return "sequential";
    case Backend::blocked: return "blocked";
    case Backend::simd: return "simd";
    case Backend::openmp: return "openmp";
    case Backend::openmp_simd: return "openmp_simd";
    case Backend::bh_sequential: return "bh_sequential";
    case Backend::bh_optimized: return "bh_optimized";
    case Backend::bh_openmp: return "bh_openmp";
    }
    throw std::invalid_argument("Nepoznata verzija algoritma");
}

Backend parse_backend(std::string_view value) {
    for (Backend backend : backends) {
        if (name(backend) == value) return backend;
    }
    throw std::invalid_argument("Nepoznata verzija algoritma");
}

bool parallel(Backend backend) {
    return backend == Backend::openmp || backend == Backend::openmp_simd || backend == Backend::bh_openmp;
}

bool barnes_hut_backend(Backend backend) {
    return backend == Backend::bh_sequential || backend == Backend::bh_optimized || backend == Backend::bh_openmp;
}

void validate(const Bodies& bodies, const Config& config) {
    const Real epsilon2 = config.epsilon * config.epsilon;
    if (!std::isfinite(config.gravity) || config.gravity < 0 ||
        !std::isfinite(config.epsilon) || config.epsilon <= 0 ||
        !std::isfinite(epsilon2) || epsilon2 == 0 ||
        config.target_block == 0 || config.source_block == 0 ||
        !std::isfinite(config.theta) || config.theta < 0 || config.theta > 1 ||
        config.leaf_capacity == 0 || config.max_depth == 0 || config.max_depth > 64) {
        throw std::invalid_argument("Neispravan G, epsilon, blokovi ili Barnes-Hut parametri (theta 0..1, dubina 1..64)");
    }

    // Sve veličine moraju opisivati isti broj tijela. Ova provjera sprečava
    // izlazak iz nizova prije ulaska u paralelne petlje, gdje je greške teže pratiti.
    for (const auto* values : {&bodies.x, &bodies.y, &bodies.z, &bodies.vx,
                              &bodies.vy, &bodies.vz, &bodies.mass}) {
        if (values->size() != bodies.size()) {
            throw std::invalid_argument("Nizovi tijela nemaju jednaku dužinu");
        }
        for (Real value : *values) {
            if (!std::isfinite(value)) throw std::invalid_argument("Podaci nisu konačni brojevi");
        }
    }
    for (Real mass : bodies.mass) {
        if (mass <= 0) throw std::invalid_argument("Mase moraju biti pozitivne");
    }
}

Bodies initial_state(std::size_t count, std::uint64_t seed, std::string_view distribution) {
    if (distribution != "uniform" && distribution != "clustered") {
        throw std::invalid_argument("Raspored mora biti uniform ili clustered");
    }
    std::mt19937_64 generator(seed);
    Bodies bodies(count);

    // Isti seed daje isti niz koordinata. Eksplicitno pretvaranje bitova
    // izbjegava razlike između implementacija standardne uniformne distribucije.
    for (std::size_t i = 0; i < count; ++i) {
        const double center = distribution == "clustered" ? (i % 2 ? 0.65 : -0.65) : 0;
        const double scale = distribution == "clustered" ? 0.15 : 1;
        const double rx = double(generator() >> 11) * 0x1.0p-53 * 2 - 1;
        const double ry = double(generator() >> 11) * 0x1.0p-53 * 2 - 1;
        const double rz = double(generator() >> 11) * 0x1.0p-53 * 2 - 1;
        bodies.x[i] = Real(center + scale * rx);
        bodies.y[i] = Real(scale * ry);
        bodies.z[i] = Real(scale * rz);
        bodies.mass[i] = Real(1.0 / double(count));
    }
    return bodies;
}

// Zajednicki ulaz provjerava podatke i priprema izlaz prije izbora algoritma.
// Ostatak simulatora zato jednako poziva Direct i Barnes-Hut izvedbe.
void accelerations(const Bodies& bodies, Acceleration& acceleration,
                   const Config& config, Backend backend, SolverTiming* timing) {
    if (timing) *timing = {};
    validate(bodies, config);
    acceleration.x.resize(bodies.size());
    acceleration.y.resize(bodies.size());
    acceleration.z.resize(bodies.size());

    switch (backend) {
    case Backend::sequential: sequential(bodies, acceleration, config); break;
    case Backend::blocked: blocked(bodies, acceleration, config); break;
    case Backend::simd: simd(bodies, acceleration, config); break;
    case Backend::openmp: openmp(bodies, acceleration, config); break;
    case Backend::openmp_simd: openmp_simd(bodies, acceleration, config); break;
    case Backend::bh_sequential:
    case Backend::bh_optimized:
    case Backend::bh_openmp: barnes_hut(bodies, acceleration, config, backend, timing); break;
    default: throw std::invalid_argument("Nepoznata verzija algoritma");
    }
}

Reference reference_forces(const Bodies& bodies, const Config& config) {
    validate(bodies, config);
    Reference reference;
    reference.x.resize(bodies.size());
    reference.y.resize(bodies.size());
    reference.z.resize(bodies.size());
    const double epsilon2 = double(config.epsilon) * config.epsilon;

    // Referenca čita baš iste, već zaokružene ulazne podatke kao ostale verzije.
    // Sam proračun i sabiranje su FP64, a ova provjera nije dio mjerenog vremena.
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        double ax = 0, ay = 0, az = 0;
        for (std::size_t j = 0; j < bodies.size(); ++j) {
            if (i == j) continue;
            const double dx = double(bodies.x[j]) - bodies.x[i];
            const double dy = double(bodies.y[j]) - bodies.y[i];
            const double dz = double(bodies.z[j]) - bodies.z[i];
            const double inverse = 1 / std::sqrt(dx * dx + dy * dy + dz * dz + epsilon2);
            const double factor = double(config.gravity) * bodies.mass[j] * inverse * inverse * inverse;
            ax += dx * factor;
            ay += dy * factor;
            az += dz * factor;
        }
        reference.x[i] = ax;
        reference.y[i] = ay;
        reference.z[i] = az;
    }
    return reference;
}

// Relativna L2 greska poredi velicinu ukupnog odstupanja sa referencom.
// Dodatno cuvamo najvece apsolutno odstupanje ubrzanja jednog tijela.
Error compare(const Acceleration& acceleration, const Reference& reference) {
    const auto count = reference.x.size();
    if (reference.y.size() != count || reference.z.size() != count ||
        acceleration.x.size() != count || acceleration.y.size() != count || acceleration.z.size() != count) {
        throw std::invalid_argument("Ubrzanja nemaju jednaku dužinu");
    }
    double difference = 0, reference_norm = 0, maximum = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const double dx = double(acceleration.x[i]) - reference.x[i];
        const double dy = double(acceleration.y[i]) - reference.y[i];
        const double dz = double(acceleration.z[i]) - reference.z[i];
        if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dz)) {
            throw std::runtime_error("Proračun je proizveo nekonačno ubrzanje");
        }
        const double squared_error = dx * dx + dy * dy + dz * dz;
        difference += squared_error;
        maximum = std::max(maximum, std::sqrt(squared_error));
        reference_norm += reference.x[i] * reference.x[i] + reference.y[i] * reference.y[i]
                        + reference.z[i] * reference.z[i];
    }
    return {reference_norm > 0 ? std::sqrt(difference / reference_norm) : std::sqrt(difference), maximum};
}

void step(Bodies& bodies, Acceleration& acceleration, const Config& config,
          Backend backend, double dt, SolverTiming* timing) {
    if (!std::isfinite(dt) || dt <= 0 || !std::isfinite(Real(dt)) || Real(dt) == 0) {
        throw std::invalid_argument("Vremenski korak mora biti pozitivan i predstavljiv");
    }
    validate(bodies, config);
    if (acceleration.x.size() != bodies.size() || acceleration.y.size() != bodies.size() ||
        acceleration.z.size() != bodies.size()) {
        throw std::invalid_argument("Prije integracije treba izračunati početna ubrzanja");
    }
    const Real interval = Real(dt);
    const Real half = interval / 2;

    // Leapfrog prvo mijenja brzine za pola koraka i pomjera sva tijela.
    // Tek nakon toga se sile ponovo računaju iz zajedničkog novog stanja.
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        bodies.vx[i] += half * acceleration.x[i];
        bodies.vy[i] += half * acceleration.y[i];
        bodies.vz[i] += half * acceleration.z[i];
        bodies.x[i] += interval * bodies.vx[i];
        bodies.y[i] += interval * bodies.vy[i];
        bodies.z[i] += interval * bodies.vz[i];
    }
    accelerations(bodies, acceleration, config, backend, timing);
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        bodies.vx[i] += half * acceleration.x[i];
        bodies.vy[i] += half * acceleration.y[i];
        bodies.vz[i] += half * acceleration.z[i];
    }
}

// Dijagnostika sabira kineticku i omeksanu potencijalnu energiju.
// Direktno obilazi parove i zato se racuna izvan mjerenog vremena solvera.
double energy(const Bodies& bodies, const Config& config) {
    validate(bodies, config);
    double total = 0;
    const double epsilon2 = double(config.epsilon) * config.epsilon;
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        total += 0.5 * bodies.mass[i] * (double(bodies.vx[i]) * bodies.vx[i]
               + double(bodies.vy[i]) * bodies.vy[i] + double(bodies.vz[i]) * bodies.vz[i]);
        // Svaki par ulazi jednom u potencijalnu energiju, uz isto omekšavanje kao u sili.
        for (std::size_t j = i + 1; j < bodies.size(); ++j) {
            const double dx = double(bodies.x[j]) - bodies.x[i];
            const double dy = double(bodies.y[j]) - bodies.y[i];
            const double dz = double(bodies.z[j]) - bodies.z[i];
            total -= double(config.gravity) * bodies.mass[i] * bodies.mass[j]
                   / std::sqrt(dx * dx + dy * dy + dz * dz + epsilon2);
        }
    }
    return total;
}

// Ukupna kolicina kretanja je suma mase puta brzina po sve tri ose.
// Benchmark prati njenu promjenu da bi se vidjelo odstupanje tokom simulacije.
std::array<double, 3> momentum(const Bodies& bodies) {
    std::array<double, 3> result{};
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        result[0] += double(bodies.mass[i]) * bodies.vx[i];
        result[1] += double(bodies.mass[i]) * bodies.vy[i];
        result[2] += double(bodies.mass[i]) * bodies.vz[i];
    }
    return result;
}

} // namespace nbody
