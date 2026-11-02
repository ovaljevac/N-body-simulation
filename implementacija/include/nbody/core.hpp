#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace nbody {
enum class Backend { sequential, blocked, simd, openmp, openmp_simd };
inline constexpr std::array backends{Backend::sequential, Backend::blocked, Backend::simd,
                                     Backend::openmp, Backend::openmp_simd};
std::string_view name(Backend backend);
Backend parse_backend(std::string_view name);
inline bool parallel(Backend b) { return b == Backend::openmp || b == Backend::openmp_simd; }

struct Config {
    double gravity = 1.0;
    double epsilon = 0.01;
    std::size_t target_block = 32;
    std::size_t source_block = 256;
};
template<class T> struct Bodies {
    std::vector<T> x, y, z, vx, vy, vz, mass;
    explicit Bodies(std::size_t n = 0): x(n), y(n), z(n), vx(n), vy(n), vz(n), mass(n, T(1)) {}
    std::size_t size() const { return mass.size(); }
};
template<class T> struct Acceleration {
    std::vector<T> x, y, z;
    void resize(std::size_t n) { x.resize(n); y.resize(n); z.resize(n); }
};
template<class T> void validate(const Bodies<T>& b, const Config& c) {
    const auto n = b.size();
    if (!std::isfinite(c.gravity) || c.gravity < 0 || !std::isfinite(c.epsilon) || c.epsilon <= 0 ||
        !std::isfinite(T(c.epsilon)*T(c.epsilon)) || T(c.epsilon)*T(c.epsilon) == 0 ||
        !std::isfinite(T(c.gravity)) || !c.target_block || !c.source_block)
        throw std::invalid_argument("G must be finite/nonnegative, epsilon positive/representable, blocks nonzero");
    for (const auto* v : {&b.x, &b.y, &b.z, &b.vx, &b.vy, &b.vz, &b.mass}) {
        if (v->size() != n) throw std::invalid_argument("SoA lengths differ");
        for (const auto value : *v)
            if (!std::isfinite(value)) throw std::invalid_argument("Body data must be finite");
    }
    for (const auto m : b.mass) if (m <= 0) throw std::invalid_argument("Mass must be positive");
}

// Kernels read a frozen position snapshot and write disjoint acceleration entries.
template<class T> void scalar_kernel(const Bodies<T>&, Acceleration<T>&, const Config&);
template<class T> void blocked_kernel(const Bodies<T>&, Acceleration<T>&, const Config&, bool);
template<class T> void simd_kernel(const Bodies<T>&, Acceleration<T>&, const Config&, bool);
template<class T> void accelerations(const Bodies<T>& b, Acceleration<T>& a, const Config& c, Backend mode) {
    validate(b, c);
    a.resize(b.size());
    switch (mode) {
    case Backend::sequential: scalar_kernel(b,a,c); break;
    case Backend::blocked: blocked_kernel(b,a,c,false); break;
    case Backend::simd: simd_kernel(b,a,c,false); break;
    case Backend::openmp: blocked_kernel(b,a,c,true); break;
    case Backend::openmp_simd: simd_kernel(b,a,c,true); break;
    default: throw std::invalid_argument("Unknown backend");
    }
}
template<class To, class From> Bodies<To> cast(const Bodies<From>& b) {
    Bodies<To> result(b.size());
    result.x.assign(b.x.begin(),b.x.end()); result.y.assign(b.y.begin(),b.y.end());
    result.z.assign(b.z.begin(),b.z.end()); result.mass.assign(b.mass.begin(),b.mass.end());
    result.vx.assign(b.vx.begin(),b.vx.end()); result.vy.assign(b.vy.begin(),b.vy.end());
    result.vz.assign(b.vz.begin(),b.vz.end()); return result;
}
template<class T> Bodies<T> initial_state(std::size_t n, std::uint64_t seed, std::string_view distribution) {
    if (distribution != "uniform" && distribution != "clustered")
        throw std::invalid_argument("Distribution must be uniform or clustered");
    std::mt19937_64 rng(seed);
    // Explicit mapping makes this generator reproducible across standard libraries.
    auto uniform = [&]() { return double(rng() >> 11) * 0x1.0p-53 * 2.0 - 1.0; };
    Bodies<T> b(n);
    for (std::size_t i=0;i<n;++i) {
        double center = distribution == "clustered" ? (i%2 ? 0.65 : -0.65) : 0;
        double scale = distribution == "clustered" ? 0.15 : 1.0;
        b.x[i]=T(center+scale*uniform()); b.y[i]=T(scale*uniform()); b.z[i]=T(scale*uniform());
        b.mass[i]=T(1.0/double(n));
    }
    return b;
}
template<class T> void step(Bodies<T>& b, Acceleration<T>& a, const Config& c, Backend backend, double dt) {
    if (!std::isfinite(dt) || dt <= 0 || !std::isfinite(T(dt)) || T(dt)==0)
        throw std::invalid_argument("dt must be positive, finite and representable");
    validate(b,c);
    if (a.x.size()!=b.size() || a.y.size()!=b.size() || a.z.size()!=b.size())
        throw std::invalid_argument("Initialize accelerations before stepping");
    const T h=T(dt), half=T(0.5)*h;
    for (std::size_t i=0;i<b.size();++i) {
        b.vx[i]+=half*a.x[i]; b.vy[i]+=half*a.y[i]; b.vz[i]+=half*a.z[i];
        b.x[i]+=h*b.vx[i]; b.y[i]+=h*b.vy[i]; b.z[i]+=h*b.vz[i];
    }
    accelerations(b,a,c,backend);
    for (std::size_t i=0;i<b.size();++i) {
        b.vx[i]+=half*a.x[i]; b.vy[i]+=half*a.y[i]; b.vz[i]+=half*a.z[i];
    }
}
template<class T> double energy(const Bodies<T>& b, const Config& c) {
    validate(b,c);
    double e=0;
    for (std::size_t i=0;i<b.size();++i) {
        e+=0.5*double(b.mass[i])*(double(b.vx[i])*b.vx[i]+double(b.vy[i])*b.vy[i]+double(b.vz[i])*b.vz[i]);
        for (std::size_t j=i+1;j<b.size();++j) {
            double dx=double(b.x[j])-b.x[i],dy=double(b.y[j])-b.y[i],dz=double(b.z[j])-b.z[i];
            e-=c.gravity*double(b.mass[i])*b.mass[j]/std::sqrt(dx*dx+dy*dy+dz*dz+c.epsilon*c.epsilon);
        }
    }
    return e;
}
template<class T> std::array<double,3> momentum(const Bodies<T>& b) {
    std::array<double,3> p{};
    for (std::size_t i=0;i<b.size();++i) {
        p[0]+=double(b.mass[i])*b.vx[i]; p[1]+=double(b.mass[i])*b.vy[i]; p[2]+=double(b.mass[i])*b.vz[i];
    }
    return p;
}
struct Error { double relative_l2=0, max_absolute=0; };
template<class T> Error compare(const Acceleration<T>& a, const Acceleration<double>& ref) {
    if (a.x.size()!=ref.x.size() || a.y.size()!=ref.y.size() || a.z.size()!=ref.z.size())
        throw std::invalid_argument("Acceleration lengths differ");
    double numerator=0,denominator=0,largest=0;
    for (std::size_t i=0;i<a.x.size();++i) {
        const double dx=double(a.x[i])-ref.x[i],dy=double(a.y[i])-ref.y[i],dz=double(a.z[i])-ref.z[i];
        if (!std::isfinite(dx)||!std::isfinite(dy)||!std::isfinite(dz))
            throw std::runtime_error("Nonfinite acceleration");
        const double error2=dx*dx+dy*dy+dz*dz;
        numerator+=error2; largest=std::max(largest,std::sqrt(error2));
        denominator+=ref.x[i]*ref.x[i]+ref.y[i]*ref.y[i]+ref.z[i]*ref.z[i];
    }
    return {denominator>0?std::sqrt(numerator/denominator):std::sqrt(numerator),largest};
}
} // namespace nbody
