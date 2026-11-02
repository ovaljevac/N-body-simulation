#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace nbody {

// Jedan tip broja koristi se u cijeloj simulaciji. CMake opcija NBODY_DOUBLE
// bira FP64, a podrazumijevani build koristi FP32 kao buduća GPU verzija.
#ifdef NBODY_DOUBLE
using Real = double;
#else
using Real = float;
#endif

enum class Backend { sequential, blocked, simd, openmp, openmp_simd };
constexpr std::array backends = {Backend::sequential, Backend::blocked,
    Backend::simd, Backend::openmp, Backend::openmp_simd};

struct Config {
    Real gravity = 1;
    Real epsilon = Real(0.01);
    std::size_t target_block = 32;
    std::size_t source_block = 256;
};

// Odvojeni nizovi olakšavaju uzastopno čitanje položaja i SIMD obradu.
// Brzine se mijenjaju u integratoru, a solver samo čita položaje i mase.
struct Bodies {
    std::vector<Real> x, y, z;
    std::vector<Real> vx, vy, vz;
    std::vector<Real> mass;

    explicit Bodies(std::size_t count = 0);
    std::size_t size() const { return mass.size(); }
};

struct Acceleration {
    std::vector<Real> x, y, z;
};

// Referenca uvijek sabira u double preciznosti, čak i u FP32 programu.
// Tako se mjeri greška sile bez uvođenja generičkih struktura podataka.
struct Reference {
    std::vector<double> x, y, z;
};

struct Error {
    double relative_l2 = 0;
    double max_absolute = 0;
};

std::string_view name(Backend backend);
Backend parse_backend(std::string_view value);
bool parallel(Backend backend);
void validate(const Bodies& bodies, const Config& config);
Bodies initial_state(std::size_t count, std::uint64_t seed, std::string_view distribution);

// Ove funkcije imaju iste ulaze i izlaze. Poziv accelerations() prvo
// provjerava ulaz i priprema izlazne nizove, pa bira traženu verziju.
void sequential(const Bodies&, Acceleration&, const Config&);
void blocked(const Bodies&, Acceleration&, const Config&);
void simd(const Bodies&, Acceleration&, const Config&);
void openmp(const Bodies&, Acceleration&, const Config&);
void openmp_simd(const Bodies&, Acceleration&, const Config&);
void accelerations(const Bodies&, Acceleration&, const Config&, Backend);

Reference reference_forces(const Bodies&, const Config&);
Error compare(const Acceleration&, const Reference&);
void step(Bodies&, Acceleration&, const Config&, Backend, double dt);
double energy(const Bodies&, const Config&);
std::array<double, 3> momentum(const Bodies&);

} // namespace nbody
