#include "nbody.h"

#include <algorithm>
#include <cmath>

namespace nbody {

// Osnovna verzija: za jedno ciljno tijelo sabiraju se djelovanja svih drugih.
// Položaji se tokom ove petlje ne mijenjaju, a vlastito djelovanje se preskače.
void sequential(const Bodies& bodies, Acceleration& acceleration, const Config& config) {
    const Real epsilon2 = config.epsilon * config.epsilon;
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        Real ax = 0, ay = 0, az = 0;
        for (std::size_t j = 0; j < bodies.size(); ++j) {
            if (i == j) continue;
            const Real dx = bodies.x[j] - bodies.x[i];
            const Real dy = bodies.y[j] - bodies.y[i];
            const Real dz = bodies.z[j] - bodies.z[i];
            const Real inverse = Real(1) / std::sqrt(dx * dx + dy * dy + dz * dz + epsilon2);
            const Real factor = config.gravity * bodies.mass[j] * inverse * inverse * inverse;
            ax += dx * factor;
            ay += dy * factor;
            az += dz * factor;
        }
        acceleration.x[i] = ax;
        acceleration.y[i] = ay;
        acceleration.z[i] = az;
    }
}

// Blokiranje zadržava isti fizički proračun, ali više ciljeva koristi isti
// izvorni blok dok je njegov sadržaj još u kešu. Posljednji blok može biti kraći.
void blocked(const Bodies& bodies, Acceleration& acceleration, const Config& config) {
    const auto count = bodies.size();
    const Real epsilon2 = config.epsilon * config.epsilon;
    std::fill(acceleration.x.begin(), acceleration.x.end(), Real(0));
    std::fill(acceleration.y.begin(), acceleration.y.end(), Real(0));
    std::fill(acceleration.z.begin(), acceleration.z.end(), Real(0));

    for (std::size_t target = 0; target < count;) {
        const auto target_end = target + std::min(config.target_block, count - target);
        for (std::size_t source = 0; source < count;) {
            const auto source_end = source + std::min(config.source_block, count - source);
            for (std::size_t i = target; i < target_end; ++i) {
                Real ax = 0, ay = 0, az = 0;
                for (std::size_t j = source; j < source_end; ++j) {
                    if (i == j) continue;
                    const Real dx = bodies.x[j] - bodies.x[i];
                    const Real dy = bodies.y[j] - bodies.y[i];
                    const Real dz = bodies.z[j] - bodies.z[i];
                    const Real inverse = Real(1) / std::sqrt(dx * dx + dy * dy + dz * dz + epsilon2);
                    const Real factor = config.gravity * bodies.mass[j] * inverse * inverse * inverse;
                    ax += dx * factor;
                    ay += dy * factor;
                    az += dz * factor;
                }
                acceleration.x[i] += ax;
                acceleration.y[i] += ay;
                acceleration.z[i] += az;
            }
            source = source_end;
        }
        target = target_end;
    }
}

// OpenMP verzija se od osnovne razlikuje raspodjelom ciljnih tijela između niti.
// Svaka nit upisuje drugi izlazni element, pa nisu potrebni atomici ni zaključavanje.
void openmp(const Bodies& bodies, Acceleration& acceleration, const Config& config) {
    const Real epsilon2 = config.epsilon * config.epsilon;
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        Real ax = 0, ay = 0, az = 0;
        for (std::size_t j = 0; j < bodies.size(); ++j) {
            if (i == j) continue;
            const Real dx = bodies.x[j] - bodies.x[i];
            const Real dy = bodies.y[j] - bodies.y[i];
            const Real dz = bodies.z[j] - bodies.z[i];
            const Real inverse = Real(1) / std::sqrt(dx * dx + dy * dy + dz * dz + epsilon2);
            const Real factor = config.gravity * bodies.mass[j] * inverse * inverse * inverse;
            ax += dx * factor;
            ay += dy * factor;
            az += dz * factor;
        }
        acceleration.x[i] = ax;
        acceleration.y[i] = ay;
        acceleration.z[i] = az;
    }
}

} // namespace nbody
