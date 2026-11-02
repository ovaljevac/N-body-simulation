#include "nbody.h"

#include <cmath>

namespace nbody {

// SIMD obrađuje više izvornih tijela jednom vektorskom instrukcijom. Petlja
// je podijeljena prije i poslije i, tako da nema grananja za vlastito djelovanje.
void simd(const Bodies& bodies, Acceleration& acceleration, const Config& config) {
    const auto count = bodies.size();
    const Real epsilon2 = config.epsilon * config.epsilon;
    const Real* x = bodies.x.data();
    const Real* y = bodies.y.data();
    const Real* z = bodies.z.data();
    const Real* mass = bodies.mass.data();

    for (std::size_t i = 0; i < count; ++i) {
        Real ax = 0, ay = 0, az = 0;
        const Real xi = x[i], yi = y[i], zi = z[i];
        #pragma omp simd reduction(+:ax,ay,az)
        for (std::size_t j = 0; j < i; ++j) {
            const Real dx = x[j] - xi, dy = y[j] - yi, dz = z[j] - zi;
            const Real inverse = Real(1) / std::sqrt(dx * dx + dy * dy + dz * dz + epsilon2);
            const Real factor = config.gravity * mass[j] * inverse * inverse * inverse;
            ax += dx * factor;
            ay += dy * factor;
            az += dz * factor;
        }
        #pragma omp simd reduction(+:ax,ay,az)
        for (std::size_t j = i + 1; j < count; ++j) {
            const Real dx = x[j] - xi, dy = y[j] - yi, dz = z[j] - zi;
            const Real inverse = Real(1) / std::sqrt(dx * dx + dy * dy + dz * dz + epsilon2);
            const Real factor = config.gravity * mass[j] * inverse * inverse * inverse;
            ax += dx * factor;
            ay += dy * factor;
            az += dz * factor;
        }
        acceleration.x[i] = ax;
        acceleration.y[i] = ay;
        acceleration.z[i] = az;
    }
}

// Kombinovana verzija koristi iste SIMD petlje kao iznad. Dodana je samo
// raspodjela ciljeva pomoću OpenMP-a, pa se doprinos dvije optimizacije lako poredi.
void openmp_simd(const Bodies& bodies, Acceleration& acceleration, const Config& config) {
    const auto count = bodies.size();
    const Real epsilon2 = config.epsilon * config.epsilon;
    const Real* x = bodies.x.data();
    const Real* y = bodies.y.data();
    const Real* z = bodies.z.data();
    const Real* mass = bodies.mass.data();

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < count; ++i) {
        Real ax = 0, ay = 0, az = 0;
        const Real xi = x[i], yi = y[i], zi = z[i];
        #pragma omp simd reduction(+:ax,ay,az)
        for (std::size_t j = 0; j < i; ++j) {
            const Real dx = x[j] - xi, dy = y[j] - yi, dz = z[j] - zi;
            const Real inverse = Real(1) / std::sqrt(dx * dx + dy * dy + dz * dz + epsilon2);
            const Real factor = config.gravity * mass[j] * inverse * inverse * inverse;
            ax += dx * factor;
            ay += dy * factor;
            az += dz * factor;
        }
        #pragma omp simd reduction(+:ax,ay,az)
        for (std::size_t j = i + 1; j < count; ++j) {
            const Real dx = x[j] - xi, dy = y[j] - yi, dz = z[j] - zi;
            const Real inverse = Real(1) / std::sqrt(dx * dx + dy * dy + dz * dz + epsilon2);
            const Real factor = config.gravity * mass[j] * inverse * inverse * inverse;
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
