#pragma once
#include "nbody/core.hpp"
namespace nbody::detail {
template<class T,bool SIMD> void tile(const Bodies<T>& b, Acceleration<T>& a, const Config& c,
                                     std::size_t begin, std::size_t end) {
    const T* x=b.x.data(); const T* y=b.y.data(); const T* z=b.z.data(); const T* m=b.mass.data();
    const T eps2=T(c.epsilon)*T(c.epsilon),G=T(c.gravity);
    for (auto i=begin;i<end;++i) a.x[i]=a.y[i]=a.z[i]=0;
    // Source tiles are reused by a tile of target bodies before advancing.
    for (std::size_t jb=0;jb<b.size();) {
        const auto je=jb+std::min(c.source_block,b.size()-jb);
        for (auto i=begin;i<end;++i) {
            const T xi=x[i],yi=y[i],zi=z[i];
            T ax=0,ay=0,az=0;
            if constexpr (SIMD) {
                // Exclude self by splitting ranges, keeping the SIMD loop branch-free.
                for (int side=0;side<2;++side) {
                    const auto first=side==0?jb:std::max(jb,i+1);
                    const auto last=side==0?std::min(je,i):je;
                    T sx=0,sy=0,sz=0;
                    #pragma omp simd reduction(+:sx,sy,sz)
                    for (std::size_t j=first;j<last;++j) {
                        const T dx=x[j]-xi,dy=y[j]-yi,dz=z[j]-zi;
                        const T inv=T(1)/std::sqrt(dx*dx+dy*dy+dz*dz+eps2);
                        const T s=G*m[j]*inv*inv*inv;
                        sx+=dx*s; sy+=dy*s; sz+=dz*s;
                    }
                    ax+=sx; ay+=sy; az+=sz;
                }
            } else {
                for (std::size_t j=jb;j<je;++j) {
                    if (j==i) continue;
                    const T dx=x[j]-xi,dy=y[j]-yi,dz=z[j]-zi;
                    const T inv=T(1)/std::sqrt(dx*dx+dy*dy+dz*dz+eps2);
                    const T s=G*m[j]*inv*inv*inv;
                    ax+=dx*s; ay+=dy*s; az+=dz*s;
                }
            }
            a.x[i]+=ax; a.y[i]+=ay; a.z[i]+=az;
        }
        jb=je;
    }
}
template<class T,bool SIMD> void tiled(const Bodies<T>& b, Acceleration<T>& a,const Config& c,bool parallel) {
    const auto count=b.size()/c.target_block+(b.size()%c.target_block!=0);
    const auto run=[&](std::size_t block) {
        const auto begin=block*c.target_block;
        tile<T,SIMD>(b,a,c,begin,begin+std::min(c.target_block,b.size()-begin));
    };
    if (parallel) {
        // Each worker owns complete targets; all source arrays are read-only.
        #pragma omp parallel for schedule(static)
        for (std::size_t block=0;block<count;++block) run(block);
    } else {
        for (std::size_t block=0;block<count;++block) run(block);
    }
}
}
