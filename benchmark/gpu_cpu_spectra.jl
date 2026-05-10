using BenchmarkTools
using CUDA
using QuantumFluidSpectra

function _arg(name, default)
    prefix = "--$(name)="
    for arg in ARGS
        startswith(arg, prefix) && return split(arg, "=", limit = 2)[2]
    end
    return default
end

function _field_2d(X)
    x, y = X
    yr = reshape(y, 1, :)
    amp = @. 1 + 0.12 * cos(2π * x) + 0.08 * sin(4π * yr)
    phase = @. 2π * x + 4π * yr + 0.2 * sin(2π * x) * cos(2π * yr)
    return complex.(amp .* exp.(im .* phase))
end

function _field_3d(X)
    x, y, z = X
    yr = reshape(y, 1, :, 1)
    zr = reshape(z, 1, 1, :)
    amp = @. 1 + 0.1 * cos(2π * x) + 0.08 * sin(2π * yr) + 0.06 * cos(2π * zr)
    phase = @. 2π * x + 2π * yr + 2π * zr + 0.15 * sin(2π * x) * cos(2π * yr)
    return complex.(amp .* exp.(im .* phase))
end

function _problem(dim, n, nradial)
    L = ntuple(_ -> 1.0, dim)
    N = ntuple(_ -> n, dim)
    X, K, _, _ = xk_arrays(L, N)
    ψ = dim == 2 ? _field_2d(X) : _field_3d(X)
    psi = Psi(ψ, X, K)
    kmax = maximum(abs, K[1])
    k = collect(LinRange(0.0, kmax, nradial))
    return psi, k
end

function _summary(label, trial)
    println(label)
    display(trial)
    println()
end

function main()
    dim = parse(Int, _arg("dim", "3"))
    n = parse(Int, _arg("n", dim == 2 ? "128" : "32"))
    nradial = parse(Int, _arg("nradial", "64"))
    seconds = parse(Float64, _arg("seconds", "10"))

    dim in (2, 3) || error("--dim must be 2 or 3")
    CUDA.functional() || error("CUDA.functional() is false on this machine")

    println("QuantumFluidSpectra GPU/CPU spectra benchmark")
    println("device: ", CUDA.name(CUDA.device()))
    println("dim: ", dim, ", n: ", n, ", nradial: ", nradial)

    psi_cpu, k_cpu = _problem(dim, n, nradial)
    psi_gpu = gpu(psi_cpu)
    cache = spectrum_cache(psi_gpu; k = k_cpu)

    density_cpu = density_spectrum(k_cpu, psi_cpu)
    kinetic_cpu = kinetic_density(k_cpu, psi_cpu)

    analyze_spectra!(cache, psi_gpu)
    CUDA.synchronize()
    result_gpu = spectrum_results(cache; host = true)

    density_err =
        maximum(abs.(result_gpu.density .- density_cpu)) /
        max(maximum(abs, density_cpu), eps(real(eltype(density_cpu))))
    kinetic_err =
        maximum(abs.(result_gpu.kinetic .- kinetic_cpu)) /
        max(maximum(abs, kinetic_cpu), eps(real(eltype(kinetic_cpu))))
    println("density relative max error: ", density_err)
    println("kinetic relative max error: ", kinetic_err)

    cpu_trial = @benchmark begin
        density_spectrum($k_cpu, $psi_cpu)
        kinetic_density($k_cpu, $psi_cpu)
    end seconds = seconds

    gpu_trial = @benchmark begin
        analyze_spectra!($cache, $psi_gpu)
        CUDA.synchronize()
    end seconds = seconds

    _summary("CPU density + kinetic", cpu_trial)
    _summary("GPU density + kinetic", gpu_trial)
end

main()
