#include "src/nbody.h"

#include <omp.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace nbody;
int checks = 0;
const double tolerance = sizeof(Real) == 4 ? 1e-5 : 1e-12;

// Provjere ostaju uključene u Release build-u. Poruka označava fizikalni
// ili paralelni uslov koji je prekršen, umjesto da samo prekine program.
void require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

void analytic_tests() {
    Config config;
    for (auto backend : backends) {
        Acceleration acceleration;
        Bodies bodies;
        accelerations(bodies, acceleration, config, backend);
        require(acceleration.x.empty(), "Prazan sistem");
        bodies = Bodies(1);
        accelerations(bodies, acceleration, config, backend);
        require(acceleration.x[0] == 0 && acceleration.y[0] == 0 && acceleration.z[0] == 0, "Vlastita sila");

        // Analitički par sa različitim masama provjerava smjer sile i to da
        // ubrzanje tijela zavisi od mase drugog, a ne vlastitog tijela.
        bodies = Bodies(2);
        bodies.x = {Real(-0.5), Real(0.5)};
        bodies.mass = {2, 3};
        accelerations(bodies, acceleration, config, backend);
        const double factor = 1 / std::pow(1 + double(config.epsilon) * config.epsilon, 1.5);
        require(std::abs(acceleration.x[0] - 3 * factor) < tolerance &&
                std::abs(acceleration.x[1] + 2 * factor) < tolerance, "Sila između dva tijela");
        require(std::abs(2 * acceleration.x[0] + 3 * acceleration.x[1]) < tolerance, "Promjena ukupnog impulsa");

        bodies.x = {0, 0};
        accelerations(bodies, acceleration, config, backend);
        require(acceleration.x[0] == 0 && acceleration.x[1] == 0, "Jednake pozicije uz softening");

        bodies = Bodies(3);
        bodies.x[1] = 1;
        bodies.y[2] = 1;
        bodies.mass = {1, 2, 3};
        accelerations(bodies, acceleration, config, backend);
        require(std::abs(acceleration.x[0] - 2 * factor) < tolerance &&
                std::abs(acceleration.y[0] - 3 * factor) < tolerance, "Sabiranje sila tri tijela");
    }
}

void equivalence_tests() {
    Config config;
    config.target_block = 7;
    config.source_block = 19;

    // Neparne veličine i mali blokovi otkrivaju preskakanje posljednjeg
    // bloka. Isti skup se provjerava na više niti i u obje distribucije.
    for (auto distribution : {"uniform", "clustered"}) {
        for (std::size_t count : {2, 7, 8, 19, 31, 32, 33, 127, 129, 257}) {
            const auto bodies = initial_state(count, 42, distribution);
            const auto original = bodies;
            const auto reference = reference_forces(bodies, config);
            for (auto backend : backends) {
                for (int threads : {1, 2, 4}) {
                    omp_set_num_threads(threads);
                    Acceleration acceleration;
                    accelerations(bodies, acceleration, config, backend);
                    require(compare(acceleration, reference).relative_l2 < tolerance,
                            "Poređenje " + std::string(name(backend)) + " sa referencom");
                }
            }
            require(bodies.x == original.x && bodies.y == original.y && bodies.z == original.z &&
                    bodies.mass == original.mass && bodies.vx == original.vx &&
                    bodies.vy == original.vy && bodies.vz == original.vz, "Solver je promijenio ulaz");
        }
    }
}

void invariance_tests() {
    Config config;
    auto bodies = initial_state(33, 12, "uniform");
    const auto reference = reference_forces(bodies, config);
    Acceleration acceleration;
    for (auto& value : bodies.x) value += 2;
    for (auto& value : bodies.y) value -= 1;
    accelerations(bodies, acceleration, config, Backend::openmp_simd);
    require(compare(acceleration, reference).relative_l2 < tolerance, "Translacija sistema");

    bodies = initial_state(33, 12, "uniform");
    for (auto& mass : bodies.mass) mass *= 2;
    auto doubled_reference = reference;
    for (auto* values : {&doubled_reference.x, &doubled_reference.y, &doubled_reference.z}) {
        for (auto& value : *values) value *= 2;
    }
    accelerations(bodies, acceleration, config, Backend::openmp_simd);
    require(compare(acceleration, doubled_reference).relative_l2 < tolerance, "Skaliranje masa");

    for (auto backend : backends) {
        bodies = initial_state(33, 12, "uniform");
        for (auto* values : {&bodies.x, &bodies.y, &bodies.z, &bodies.mass}) {
            std::reverse(values->begin(), values->end());
        }
        accelerations(bodies, acceleration, config, backend);
        for (auto* values : {&acceleration.x, &acceleration.y, &acceleration.z}) {
            std::reverse(values->begin(), values->end());
        }
        require(compare(acceleration, reference).relative_l2 < tolerance, "Promjena redoslijeda tijela");
    }
}

double orbit_test(double dt) {
    Config config;
    Bodies bodies(2);
    bodies.x = {Real(-0.5), Real(0.5)};
    const double speed = std::sqrt(0.5 / std::pow(1 + double(config.epsilon) * config.epsilon, 1.5));
    bodies.vy = {Real(-speed), Real(speed)};
    Acceleration acceleration;
    accelerations(bodies, acceleration, config, Backend::openmp_simd);
    const double initial_energy = energy(bodies, config);
    double energy_drift = 0;
    const int steps = int(std::lround(2 / dt));
    for (int i = 0; i < steps; ++i) {
        step(bodies, acceleration, config, Backend::openmp_simd, dt);
        energy_drift = std::max(energy_drift, std::abs(energy(bodies, config) - initial_energy) / std::abs(initial_energy));
    }
    const auto impulse = momentum(bodies);
    require(std::abs(impulse[0]) + std::abs(impulse[1]) + std::abs(impulse[2]) < tolerance, "Impuls orbite");
    require(energy_drift < 1e-4, "Energija orbite");
    const double angle = 2 * speed * steps * dt;
    return std::hypot(double(bodies.x[0]) + 0.5 * std::cos(angle),
                      double(bodies.y[0]) + 0.5 * std::sin(angle));
}

void integration_tests() {
    Config config;
    for (auto backend : backends) {
        Bodies bodies(1);
        bodies.vx[0] = 2;
        bodies.vy[0] = -3;
        Acceleration acceleration;
        accelerations(bodies, acceleration, config, backend);
        for (int i = 0; i < 10; ++i) step(bodies, acceleration, config, backend, 0.1);
        require(std::abs(bodies.x[0] - 2) < tolerance && std::abs(bodies.y[0] + 3) < tolerance, "Slobodno kretanje");

        auto state = initial_state(17, 2, "uniform");
        auto expected = state;
        Acceleration reference;
        accelerations(state, acceleration, config, backend);
        accelerations(expected, reference, config, Backend::sequential);
        for (int i = 0; i < 10; ++i) {
            step(state, acceleration, config, backend, 0.001);
            step(expected, reference, config, Backend::sequential, 0.001);
        }
        for (std::size_t i = 0; i < state.size(); ++i) {
            require(std::abs(state.x[i] - expected.x[i]) + std::abs(state.vx[i] - expected.vx[i]) < tolerance,
                    "Dosljednost integracije između verzija");
        }
    }
    // Greška putanje mora opadati kad se korak prepolovi. Ovaj uslov
    // provjerava integrator i kada FP32 zaokruživanje prikrije vrlo mali drift energije.
    const double coarse = orbit_test(0.04);
    const double fine = orbit_test(0.02);
    require(fine < coarse / 3, "Konvergencija integratora");
}

void invalid_input_tests() {
    for (int problem = 0; problem < 8; ++problem) {
        auto bodies = initial_state(3, 42, "uniform");
        Config config;
        Acceleration acceleration;
        switch (problem) {
        case 0: config.epsilon = 0; break;
        case 1: config.source_block = 0; break;
        case 2: config.target_block = 0; break;
        case 3: bodies.mass[0] = -1; break;
        case 4: bodies.x.pop_back(); break;
        case 5: bodies.x[0] = std::numeric_limits<Real>::quiet_NaN(); break;
        case 6: config.gravity = -1; break;
        case 7: bodies.vx[0] = std::numeric_limits<Real>::infinity(); break;
        }
        bool rejected = false;
        try { accelerations(bodies, acceleration, config, Backend::openmp); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "Odbijanje neispravnog ulaza");
    }
    auto bodies = initial_state(3, 42, "uniform");
    Config config;
    Acceleration acceleration;
    accelerations(bodies, acceleration, config, Backend::sequential);
    for (double dt : {0.0, -0.1, std::numeric_limits<double>::infinity()}) {
        bool rejected = false;
        try { step(bodies, acceleration, config, Backend::sequential, dt); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "Odbijanje neispravnog dt");
    }
}

int main() {
    try {
        omp_set_dynamic(0);
        omp_set_num_threads(4);
        analytic_tests();
        equivalence_tests();
        invariance_tests();
        integration_tests();
        invalid_input_tests();
        std::cout << "PASS: " << checks << " numerical checks (" << (sizeof(Real) == 4 ? "FP32" : "FP64") << ")\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
