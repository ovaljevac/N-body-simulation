#include "tiled.hpp"
namespace nbody {
template<class T> void simd_kernel(const Bodies<T>& b, Acceleration<T>& a,const Config& c,bool parallel) {
    detail::tiled<T,true>(b,a,c,parallel);
}
template void simd_kernel<float>(const Bodies<float>&,Acceleration<float>&,const Config&,bool);
template void simd_kernel<double>(const Bodies<double>&,Acceleration<double>&,const Config&,bool);
}
