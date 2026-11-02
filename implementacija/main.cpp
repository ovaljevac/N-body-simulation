#include "src/nbody.h"

#include <omp.h>
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

struct Options {
    std::size_t count = 1000, steps = 5, repeats = 5, warmup = 1;
    std::uint64_t seed = 42;
    int threads = 4;
    double dt = 0.001;
    std::string backend = "all", distribution = "uniform", mode = "force", output;
    nbody::Config config;
};

// Parametri se provjeravaju prije računanja. Negativan broj tijela ili niti
// ne smije neprimjetno postati ogroman unsigned broj i pokrenuti pogrešan test.
std::uint64_t read_integer(std::string_view text) {
    std::uint64_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw std::invalid_argument("Očekivan je nenegativan cijeli broj");
    }
    return value;
}

double read_number(const std::string& text) {
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(value)) {
        throw std::invalid_argument("Očekivan je konačan broj");
    }
    return value;
}

Options read_options(int argc, char** argv) {
    Options options;
    options.threads = std::min(4, omp_get_max_threads());
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") {
            std::cout << "nbody_bench --n N --backend all|sequential|blocked|simd|openmp|openmp_simd\n"
                      << " --threads P --mode force|simulate --steps K --repeats R --warmup W\n"
                      << " --distribution uniform|clustered --seed S --epsilon E --gravity G --dt H\n"
                      << " --target-block B --source-block B --output file.csv\n"
                      << "FP64 build: cmake -S . -B build-double -DNBODY_DOUBLE=ON -DCMAKE_BUILD_TYPE=Release\n";
            std::exit(0);
        }
        if (++i == argc) throw std::invalid_argument("Nedostaje vrijednost opcije " + key);
        const std::string value = argv[i];
        if (key == "--n") options.count = read_integer(value);
        else if (key == "--steps") options.steps = read_integer(value);
        else if (key == "--repeats") options.repeats = read_integer(value);
        else if (key == "--warmup") options.warmup = read_integer(value);
        else if (key == "--seed") options.seed = read_integer(value);
        else if (key == "--threads") {
            const auto threads = read_integer(value);
            if (threads == 0 || threads > std::uint64_t(std::numeric_limits<int>::max())) {
                throw std::invalid_argument("Broj niti mora biti pozitivan int");
            }
            options.threads = int(threads);
        }
        else if (key == "--dt") options.dt = read_number(value);
        else if (key == "--epsilon") options.config.epsilon = nbody::Real(read_number(value));
        else if (key == "--gravity") options.config.gravity = nbody::Real(read_number(value));
        else if (key == "--target-block") options.config.target_block = read_integer(value);
        else if (key == "--source-block") options.config.source_block = read_integer(value);
        else if (key == "--backend") options.backend = value;
        else if (key == "--distribution") options.distribution = value;
        else if (key == "--mode") options.mode = value;
        else if (key == "--output") options.output = value;
        else throw std::invalid_argument("Nepoznata opcija " + key);
    }
    if (options.steps == 0 || options.repeats == 0 || options.dt <= 0 ||
        !std::isfinite(nbody::Real(options.dt)) || nbody::Real(options.dt) == 0 ||
        options.warmup > std::numeric_limits<std::size_t>::max() - options.repeats) {
        throw std::invalid_argument("Neispravan broj koraka, ponavljanja ili dt");
    }
    if (options.mode != "force" && options.mode != "simulate") {
        throw std::invalid_argument("Režim mora biti force ili simulate");
    }
    if (options.backend != "all") nbody::parse_backend(options.backend);
    return options;
}

void run_benchmark(const Options& options, std::ostream& output) {
    const auto initial = nbody::initial_state(options.count, options.seed, options.distribution);
    const auto reference = nbody::reference_forces(initial, options.config);
    const double initial_energy = options.mode == "simulate" ? nbody::energy(initial, options.config) : 0;
    const auto initial_momentum = nbody::momentum(initial);
    const char* precision = sizeof(nbody::Real) == 4 ? "float" : "double";

    omp_set_dynamic(0);
    omp_set_num_threads(options.threads);
    int actual_threads = 1;
    #pragma omp parallel reduction(max:actual_threads)
    actual_threads = omp_get_num_threads();

    output << "backend,precision,mode,distribution,n,threads,requested_threads,seed,gravity,epsilon,dt,steps,repeat,"
           << "total_seconds,seconds_per_step,relative_l2,max_absolute,energy_relative_drift,momentum_absolute_drift,"
           << "checksum,target_block,source_block,compiler,build_type,native\n";
    output << std::setprecision(17);

    // Svaka verzija prvo računa sile na istom početnom stanju. Tek nakon
    // provjere prema FP64 referenci počinju zagrijavanje i mjereni prolazi.
    for (auto backend : nbody::backends) {
        if (options.backend != "all" && nbody::name(backend) != options.backend) continue;
        nbody::Acceleration checked;
        nbody::accelerations(initial, checked, options.config, backend);
        const auto error = nbody::compare(checked, reference);
        if (error.relative_l2 > (sizeof(nbody::Real) == 4 ? 1e-4 : 1e-11)) {
            throw std::runtime_error("Sile se ne slažu sa referencom");
        }

        for (std::size_t repeat = 0; repeat < options.warmup + options.repeats; ++repeat) {
            auto bodies = initial;
            nbody::Acceleration acceleration;
            nbody::accelerations(bodies, acceleration, options.config, backend);

            // Alokacije i početna sila za leapfrog su izvan tajmera. Force
            // režim ponavlja isti snimak, a simulate režim pomjera tijela.
            const auto start = std::chrono::steady_clock::now();
            for (std::size_t k = 0; k < options.steps; ++k) {
                if (options.mode == "force") {
                    nbody::accelerations(bodies, acceleration, options.config, backend);
                } else {
                    nbody::step(bodies, acceleration, options.config, backend, options.dt);
                }
            }
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            if (repeat < options.warmup) continue;

            // Dijagnostika se računa nakon zaustavljanja sata. Checksum
            // koristi izlaz solvera, a energija i impuls prate simulaciju.
            double checksum = 0;
            for (std::size_t i = 0; i < bodies.size(); ++i) {
                if (!std::isfinite(acceleration.x[i]) || !std::isfinite(acceleration.y[i]) ||
                    !std::isfinite(acceleration.z[i])) {
                    throw std::runtime_error("Nekonačna sila nakon mjerenja");
                }
                checksum += double(i + 1) * (double(acceleration.x[i]) + acceleration.y[i] + acceleration.z[i]);
            }
            const double energy_drift = options.mode == "simulate"
                ? std::abs(nbody::energy(bodies, options.config) - initial_energy) / std::max(std::abs(initial_energy), 1e-30) : 0;
            const auto momentum = nbody::momentum(bodies);
            const double px = momentum[0] - initial_momentum[0];
            const double py = momentum[1] - initial_momentum[1];
            const double pz = momentum[2] - initial_momentum[2];
            const double momentum_drift = std::sqrt(px * px + py * py + pz * pz);

            output << nbody::name(backend) << ',' << precision << ',' << options.mode << ',' << options.distribution << ','
                   << options.count << ',' << (nbody::parallel(backend) ? actual_threads : 1) << ',' << options.threads << ','
                   << options.seed << ',' << options.config.gravity << ',' << options.config.epsilon << ',' << options.dt << ','
                   << options.steps << ',' << repeat - options.warmup << ',' << seconds << ',' << seconds / options.steps << ','
                   << error.relative_l2 << ',' << error.max_absolute << ',' << energy_drift << ',' << momentum_drift << ','
                   << checksum << ',' << options.config.target_block << ',' << options.config.source_block << ','
                   << __VERSION__ << ',' << NBODY_BUILD_TYPE << ',' << NBODY_NATIVE_BUILD << '\n';
        }
    }
}

int main(int argc, char** argv) {
    try {
        const auto options = read_options(argc, argv);
        std::ofstream file;
        if (!options.output.empty()) {
            const std::filesystem::path path(options.output);
            if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
            file.open(path);
            if (!file) throw std::runtime_error("Nije moguće otvoriti izlaznu CSV datoteku");
        }
        run_benchmark(options, options.output.empty() ? std::cout : file);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Greška: " << error.what() << '\n';
        return 1;
    }
}
