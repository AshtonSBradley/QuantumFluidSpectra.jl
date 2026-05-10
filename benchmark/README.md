# GPU/CPU Spectra Benchmark

This benchmark compares the existing CPU `density_spectrum` + `kinetic_density`
path with the CUDA spectral-analysis path:

```sh
julia --project=benchmark -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=benchmark benchmark/gpu_cpu_spectra.jl --dim=3 --n=32 --nradial=64 --seconds=10
```

The benchmark environment keeps CUDA.jl below the CUDA 12-runtime line so it can
run on Thunderbird nodes with CUDA 11.x drivers.
