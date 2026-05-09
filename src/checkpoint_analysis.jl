struct CUDADevice end

const _CUDA_LOAD_ERROR = "CUDA checkpoint analysis requires CUDA.jl to be loaded in the active environment"

"""
    gpu(psi::Psi)

Move a `Psi` field and its coordinate vectors to CUDA device arrays.
Requires CUDA.jl to be loaded and functional in the active environment.
"""
gpu(args...; kwargs...) = error(_CUDA_LOAD_ERROR)

"""
    cpu(psi::Psi)

Move a `Psi` field and its coordinate vectors back to host `Array`s.
Requires CUDA.jl to be loaded in the active environment.
"""
cpu(args...; kwargs...) = error(_CUDA_LOAD_ERROR)

"""
    checkpoint_analysis_cache(psi; backend=CUDADevice(), k=nothing, nradial=length(psi.K[1]))

Construct reusable storage for CUDA checkpoint spectrum analysis. The CUDA extension
currently supports 2D and 3D device-resident `Psi` fields and errors if CUDA is not
available.
"""
function checkpoint_analysis_cache(args...; backend = CUDADevice(), kwargs...)
    backend isa CUDADevice ||
        error("Unsupported checkpoint analysis backend: $(typeof(backend))")
    return error(_CUDA_LOAD_ERROR)
end

"""
    analyze_checkpoint!(cache, psi; spectra=(:density, :kinetic))

Update checkpoint spectra in `cache` for a device-resident `Psi`.
"""
analyze_checkpoint!(args...; kwargs...) = error(_CUDA_LOAD_ERROR)

"""
    checkpoint_results(cache; host=false)

Return checkpoint spectra. With `host=true`, copy the result arrays back to the CPU.
"""
checkpoint_results(args...; kwargs...) = error(_CUDA_LOAD_ERROR)
