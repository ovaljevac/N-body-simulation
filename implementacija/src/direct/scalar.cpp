#include "nbody/core.hpp"
namespace nbody {
template<class T> void scalar_kernel(const Bodies<T>& b, Acceleration<T>& a, const Config& c) {
    const T eps2=T(c.epsilon)*T(c.epsilon), G=T(c.gravity);
    for (std::size_t i=0;i<b.size();++i) {
        T ax=0,ay=0,az=0;
        for (std::size_t j=0;j<b.size();++j) {
            if (i==j) continue;
            const T dx=b.x[j]-b.x[i],dy=b.y[j]-b.y[i],dz=b.z[j]-b.z[i];
            const T inv=T(1)/std::sqrt(dx*dx+dy*dy+dz*dz+eps2);
            const T s=G*b.mass[j]*inv*inv*inv;
            ax+=dx*s; ay+=dy*s; az+=dz*s;
        }
        a.x[i]=ax; a.y[i]=ay; a.z[i]=az;
    }
}
template void scalar_kernel<float>(const Bodies<float>&,Acceleration<float>&,const Config&);
template void scalar_kernel<double>(const Bodies<double>&,Acceleration<double>&,const Config&);
}
