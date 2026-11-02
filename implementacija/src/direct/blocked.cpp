#include "tiled.hpp"
namespace nbody {
template<class T> void blocked_kernel(const Bodies<T>& b, Acceleration<T>& a,const Config& c,bool parallel) {
    detail::tiled<T,false>(b,a,c,parallel);
}
template void blocked_kernel<float>(const Bodies<float>&,Acceleration<float>&,const Config&,bool);
template void blocked_kernel<double>(const Bodies<double>&,Acceleration<double>&,const Config&,bool);
}
