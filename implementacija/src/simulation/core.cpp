#include "nbody/core.hpp"
namespace nbody {
std::string_view name(Backend b) {
    switch (b) {
    case Backend::sequential:return "sequential";
    case Backend::blocked:return "blocked";
    case Backend::simd:return "simd";
    case Backend::openmp:return "openmp";
    case Backend::openmp_simd:return "openmp_simd";
    }
    throw std::invalid_argument("Unknown backend");
}
Backend parse_backend(std::string_view value) {
    for (auto b:backends) if (name(b)==value) return b;
    throw std::invalid_argument("Unknown backend");
}
}
