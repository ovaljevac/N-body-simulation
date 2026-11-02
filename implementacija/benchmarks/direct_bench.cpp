#include "nbody/core.hpp"
#include <omp.h>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <thread>

namespace {
struct Options {
    std::size_t n=1000,steps=5,repeats=5,warmup=1;
    std::uint64_t seed=42;
    int threads=4;
    double dt=0.001;
    std::string backend="all",precision="float",distribution="uniform",mode="force",output;
    nbody::Config config;
};
std::uint64_t integer(std::string_view text) {
    std::uint64_t result=0;
    auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),result);
    if (error!=std::errc{} || end!=text.data()+text.size()) throw std::invalid_argument("Invalid unsigned integer");
    return result;
}
double real(const std::string& text) {
    std::size_t used=0; const double result=std::stod(text,&used);
    if (used!=text.size() || !std::isfinite(result)) throw std::invalid_argument("Invalid finite number");
    return result;
}
Options parse(int argc,char** argv) {
    Options o;
    o.threads=std::min(4,omp_get_max_threads());
    for (int i=1;i<argc;++i) {
        std::string key=argv[i];
        if (key=="--help") {
            std::cout<<"nbody_bench --n N --backend all|sequential|blocked|simd|openmp|openmp_simd\n"
                     <<" --precision float|double --threads P --mode force|simulate --steps K\n"
                     <<" --repeats R --warmup W --distribution uniform|clustered --seed S\n"
                     <<" --epsilon E --gravity G --dt H --target-block B --source-block B --output file.csv\n";
            std::exit(0);
        }
        if (++i>=argc) throw std::invalid_argument("Missing option value");
        const std::string value=argv[i];
        if(key=="--n") o.n=integer(value);
        else if(key=="--steps") o.steps=integer(value);
        else if(key=="--repeats") o.repeats=integer(value);
        else if(key=="--warmup") o.warmup=integer(value);
        else if(key=="--seed") o.seed=integer(value);
        else if(key=="--threads") {
            const auto count=integer(value);
            if (!count || count>std::uint64_t(std::numeric_limits<int>::max())) throw std::invalid_argument("Invalid thread count");
            o.threads=int(count);
        }
        else if(key=="--dt") o.dt=real(value);
        else if(key=="--epsilon") o.config.epsilon=real(value);
        else if(key=="--gravity") o.config.gravity=real(value);
        else if(key=="--target-block") o.config.target_block=integer(value);
        else if(key=="--source-block") o.config.source_block=integer(value);
        else if(key=="--backend") o.backend=value;
        else if(key=="--precision") o.precision=value;
        else if(key=="--distribution") o.distribution=value;
        else if(key=="--mode") o.mode=value;
        else if(key=="--output") o.output=value;
        else throw std::invalid_argument("Unknown option: "+key);
    }
    if(!o.steps || !o.repeats || o.dt<=0) throw std::invalid_argument("steps/repeats/dt must be positive");
    if(o.precision!="float" && o.precision!="double") throw std::invalid_argument("precision must be float or double");
    if(o.mode!="force" && o.mode!="simulate") throw std::invalid_argument("mode must be force or simulate");
    if(o.backend!="all") (void)nbody::parse_backend(o.backend);
    return o;
}
template<class T> void run(const Options& o,std::ostream& output) {
    const auto initial=nbody::initial_state<T>(o.n,o.seed,o.distribution);
    nbody::validate(initial,o.config);
    const auto reference_state=nbody::cast<double>(initial);
    nbody::Acceleration<double> reference;
    nbody::accelerations(reference_state,reference,o.config,nbody::Backend::sequential);
    omp_set_dynamic(0);
    omp_set_num_threads(o.threads);
    int actual_threads=1;
    #pragma omp parallel reduction(max:actual_threads)
    actual_threads=omp_get_num_threads();
    output<<"backend,precision,mode,distribution,n,threads,requested_threads,seed,gravity,epsilon,dt,steps,repeat,total_seconds,seconds_per_step,relative_l2,max_absolute,energy_relative_drift,momentum_absolute_drift,checksum,target_block,source_block,compiler,build_type,native\n";
    output<<std::setprecision(17);
    const double initial_energy=o.mode=="simulate"?nbody::energy(initial,o.config):0;
    const auto initial_momentum=nbody::momentum(initial);
    for (const auto backend:nbody::backends) {
        if(o.backend!="all" && nbody::name(backend)!=o.backend) continue;
        nbody::Acceleration<T> checked;
        nbody::accelerations(initial,checked,o.config,backend);
        const auto error=nbody::compare(checked,reference);
        if(error.relative_l2>(sizeof(T)==4?1e-4:1e-11)) throw std::runtime_error("Force validation failed");
        for(std::size_t run=0;run<o.warmup+o.repeats;++run) {
            auto state=initial;
            nbody::Acceleration<T> acceleration;
            // Allocation and initial force for leapfrog are outside the step timer.
            nbody::accelerations(state,acceleration,o.config,backend);
            const auto start=std::chrono::steady_clock::now();
            for(std::size_t k=0;k<o.steps;++k) {
                if(o.mode=="force") nbody::accelerations(state,acceleration,o.config,backend);
                else nbody::step(state,acceleration,o.config,backend,o.dt);
            }
            const auto stop=std::chrono::steady_clock::now();
            if(run<o.warmup) continue;
            const double seconds=std::chrono::duration<double>(stop-start).count();
            double checksum=0;
            for(std::size_t i=0;i<state.size();++i) {
                if (!std::isfinite(acceleration.x[i]) || !std::isfinite(acceleration.y[i]) || !std::isfinite(acceleration.z[i]))
                    throw std::runtime_error("Nonfinite force after benchmark");
                checksum+=double(i+1)*(double(acceleration.x[i])+acceleration.y[i]+acceleration.z[i]);
            }
            const double drift=o.mode=="simulate"?std::abs(nbody::energy(state,o.config)-initial_energy)/std::max(std::abs(initial_energy),1e-30):0;
            const auto p=nbody::momentum(state);
            const double momentum_drift=std::sqrt(std::pow(p[0]-initial_momentum[0],2)+std::pow(p[1]-initial_momentum[1],2)+std::pow(p[2]-initial_momentum[2],2));
            output<<nbody::name(backend)<<','<<o.precision<<','<<o.mode<<','<<o.distribution<<','<<o.n<<','
                  <<(nbody::parallel(backend)?actual_threads:1)<<','<<o.threads<<','<<o.seed<<','<<o.config.gravity<<','
                  <<o.config.epsilon<<','<<o.dt<<','<<o.steps<<','<<(run-o.warmup)<<','<<seconds<<','<<seconds/o.steps<<','
                  <<error.relative_l2<<','<<error.max_absolute<<','<<drift<<','<<momentum_drift<<','<<checksum<<','
                  <<o.config.target_block<<','<<o.config.source_block<<','<<__VERSION__<<','<<NBODY_BUILD_TYPE<<','<<NBODY_NATIVE_BUILD<<'\n';
        }
    }
}
}
int main(int argc,char** argv) {
    try {
        const auto options=parse(argc,argv);
        std::ofstream file;
        if(!options.output.empty()) {
            const std::filesystem::path path(options.output);
            if(path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
            file.open(path);
            if(!file) throw std::runtime_error("Cannot open output CSV");
        }
        auto& stream=options.output.empty()?std::cout:file;
        if(options.precision=="float") run<float>(options,stream); else run<double>(options,stream);
        return 0;
    } catch(const std::exception& e) { std::cerr<<"Error: "<<e.what()<<'\n'; return 1; }
}
