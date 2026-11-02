#include "nbody/core.hpp"
#include <omp.h>
#include <iostream>
#include <limits>
#include <string>

namespace {
int checks=0;
void require(bool ok,const std::string& message) {
    ++checks; if(!ok) throw std::runtime_error(message);
}
template<class F> void rejects(F action,const std::string& message) {
    bool rejected=false;
    try { action(); } catch(const std::invalid_argument&) { rejected=true; }
    require(rejected,message);
}
void analytic() {
    nbody::Config c;
    for(auto backend:nbody::backends) {
        nbody::Bodies<double> empty;
        nbody::Acceleration<double> a;
        nbody::accelerations(empty,a,c,backend);
        require(a.x.empty(),"empty system");
        nbody::Bodies<double> b(1);
        nbody::accelerations(b,a,c,backend);
        require(a.x[0]==0 && a.y[0]==0 && a.z[0]==0,"self force");
        b=nbody::Bodies<double>(2); b.x={-0.5,0.5}; b.mass={2,3};
        nbody::accelerations(b,a,c,backend);
        const double inverse=1/std::pow(1+c.epsilon*c.epsilon,1.5);
        require(std::abs(a.x[0]-3*inverse)<1e-12 && std::abs(a.x[1]+2*inverse)<1e-12,"analytic unequal mass pair");
        require(std::abs(2*a.x[0]+3*a.x[1])<1e-12,"pair momentum derivative");
        b.x={0,0}; nbody::accelerations(b,a,c,backend);
        require(a.x[0]==0 && a.x[1]==0,"coincident softened particles");
    }
}
template<class T> void cross_backend() {
    nbody::Config c;
    c.target_block=7; c.source_block=19;
    for(auto distribution:{"uniform","clustered"}) for(std::size_t n:{2,7,8,19,31,32,33,127,129,257}) {
        const auto b=nbody::initial_state<T>(n,42,distribution);
        const auto copy=b;
        nbody::Acceleration<double> reference;
        nbody::accelerations(nbody::cast<double>(b),reference,c,nbody::Backend::sequential);
        for(auto mode:nbody::backends) for(int threads:{1,2,4}) {
            omp_set_num_threads(threads);
            nbody::Acceleration<T> a;
            nbody::accelerations(b,a,c,mode);
            require(nbody::compare(a,reference).relative_l2<(sizeof(T)==4?5e-6:5e-13),"backend/tail/precision/thread equivalence");
        }
        require(b.x==copy.x && b.y==copy.y && b.z==copy.z && b.mass==copy.mass,"force calculation mutated inputs");
    }
}
void invariants() {
    nbody::Config c;
    auto b=nbody::initial_state<double>(33,12,"uniform");
    nbody::Acceleration<double> a,shifted;
    nbody::accelerations(b,a,c,nbody::Backend::sequential);
    for(auto& x:b.x) x+=2;
    for(auto& y:b.y) y-=1;
    nbody::accelerations(b,shifted,c,nbody::Backend::openmp_simd);
    require(nbody::compare(shifted,a).relative_l2<1e-13,"translation invariance");
    for(auto& mass:b.mass) mass*=2;
    nbody::accelerations(b,shifted,c,nbody::Backend::openmp_simd);
    for(std::size_t i=0;i<a.x.size();++i) { a.x[i]*=2; a.y[i]*=2; a.z[i]*=2; }
    require(nbody::compare(shifted,a).relative_l2<1e-13,"mass scaling");
    for(auto mode:nbody::backends) {
        auto reversed=nbody::initial_state<double>(33,12,"uniform");
        for(auto* v:{&reversed.x,&reversed.y,&reversed.z,&reversed.mass}) std::reverse(v->begin(),v->end());
        nbody::accelerations(reversed,shifted,c,mode);
        auto original=nbody::initial_state<double>(33,12,"uniform");
        nbody::accelerations(original,a,c,nbody::Backend::sequential);
        for(auto* v:{&shifted.x,&shifted.y,&shifted.z}) std::reverse(v->begin(),v->end());
        require(nbody::compare(shifted,a).relative_l2<1e-13,"permutation invariance");
    }
}
double orbit_error(double dt,nbody::Backend mode) {
    nbody::Config c;
    nbody::Bodies<double> b(2); b.x={-0.5,0.5};
    const double speed=std::sqrt(0.5/std::pow(1+c.epsilon*c.epsilon,1.5));
    b.vy={-speed,speed};
    nbody::Acceleration<double> a;
    nbody::accelerations(b,a,c,mode);
    const double initial=nbody::energy(b,c);
    double largest=0;
    for(int i=0;i<int(2/dt);++i) {
        nbody::step(b,a,c,mode,dt);
        largest=std::max(largest,std::abs(nbody::energy(b,c)-initial)/std::abs(initial));
    }
    auto p=nbody::momentum(b);
    require(std::abs(p[0])+std::abs(p[1])+std::abs(p[2])<1e-12,"orbit total momentum");
    return largest;
}
void integration() {
    nbody::Config c;
    for(auto backend:nbody::backends) {
        nbody::Bodies<double> b(1); b.vx[0]=2; b.vy[0]=-3;
        nbody::Acceleration<double> a;
        nbody::accelerations(b,a,c,backend);
        for(int k=0;k<10;++k) nbody::step(b,a,c,backend,0.1);
        require(std::abs(b.x[0]-2)<1e-14 && std::abs(b.y[0]+3)<1e-14,"free motion");
        auto state=nbody::initial_state<double>(17,2,"uniform");
        auto expected=state;
        nbody::Acceleration<double> reference;
        nbody::accelerations(state,a,c,backend);
        nbody::accelerations(expected,reference,c,nbody::Backend::sequential);
        for(int k=0;k<10;++k) {
            nbody::step(state,a,c,backend,0.001);
            nbody::step(expected,reference,c,nbody::Backend::sequential,0.001);
        }
        for(std::size_t i=0;i<state.size();++i)
            require(std::abs(state.x[i]-expected.x[i])+std::abs(state.vx[i]-expected.vx[i])<1e-12,"integration backend consistency");
    }
    const double coarse=orbit_error(0.04,nbody::Backend::sequential);
    const double fine=orbit_error(0.02,nbody::Backend::openmp_simd);
    require(fine<coarse/3 && fine<1e-5,"orbit improves with smaller timestep");
}
void invalid() {
    auto b=nbody::initial_state<double>(3,42,"uniform");
    nbody::Config c; nbody::Acceleration<double> a;
    c.epsilon=0;
    rejects([&]{nbody::accelerations(b,a,c,nbody::Backend::simd);},"reject zero epsilon");
    c.epsilon=0.01; c.source_block=0;
    rejects([&]{nbody::accelerations(b,a,c,nbody::Backend::blocked);},"reject zero tile");
    c.source_block=256; b.mass[0]=-1;
    rejects([&]{nbody::accelerations(b,a,c,nbody::Backend::sequential);},"reject negative mass");
    b.mass[0]=1; b.x.pop_back();
    rejects([&]{nbody::accelerations(b,a,c,nbody::Backend::openmp);},"reject invalid SoA");
    b.x.push_back(std::numeric_limits<double>::quiet_NaN());
    rejects([&]{nbody::accelerations(b,a,c,nbody::Backend::sequential);},"reject NaN");
    rejects([&]{nbody::parse_backend("invalid");},"reject unknown backend");
}
}
int main() {
    try {
        omp_set_dynamic(0); omp_set_num_threads(4);
        analytic(); cross_backend<float>(); cross_backend<double>(); invariants(); integration(); invalid();
        std::cout<<"PASS: "<<checks<<" numerical checks\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
}
