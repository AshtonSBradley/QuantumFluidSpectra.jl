module QuantumFluidSpectraCUDAExt

using CUDA
using FFTW
using QuantumFluidSpectra
using SpecialFunctions

import QuantumFluidSpectra:
    Psi, CUDADevice, gpu, cpu, spectrum_cache, analyze_spectra!, spectrum_results

const CuVecTuple{D,T} = NTuple{D,CUDA.CuVector{T}}

mutable struct CUDASpectrumCache{D,T,RT,KT,XT,R}
    X::XT
    K::KT
    k::CUDA.CuVector{RT}
    DX::NTuple{D,RT}
    DK::NTuple{D,RT}
    corr::CUDA.CuArray{T,D}
    work::CUDA.CuArray{T,D}
    fftwork::CUDA.CuArray{T,D}
    partials::CUDA.CuMatrix{RT}
    radial::R
    density::CUDA.CuVector{RT}
    kinetic::CUDA.CuVector{RT}
    incompressible::CUDA.CuVector{RT}
    compressible::CUDA.CuVector{RT}
    totals::CUDA.CuVector{RT}
    groups::Int
    chunk::Int
end

const TOTAL_KINETIC = 1
const TOTAL_INCOMPRESSIBLE = 2
const TOTAL_COMPRESSIBLE = 3

struct RadialWeightCache{I,R,J}
    rid::I
    radii::R
    weights::J
end

function _require_cuda()
    CUDA.functional() ||
        error("CUDA spectral analysis requested, but CUDA.functional() is false")
    return nothing
end

gpu(psi::Psi) = begin
    _require_cuda()
    return Psi(CUDA.CuArray(psi.ψ), map(CUDA.CuArray, psi.X), map(CUDA.CuArray, psi.K))
end

cpu(psi::Psi) = Psi(Array(psi.ψ), map(Array, psi.X), map(Array, psi.K))

_require_device_array(A) =
    A isa CUDA.CuArray || error("CUDA spectral analysis requires device-resident arrays")

function _require_device_psi(psi::Psi)
    _require_cuda()
    _require_device_array(psi.ψ)
    all(x -> x isa CUDA.CuVector, psi.X) ||
        error("CUDA spectral analysis requires psi.X vectors on device")
    all(k -> k isa CUDA.CuVector, psi.K) ||
        error("CUDA spectral analysis requires psi.K vectors on device")
    return nothing
end

function _grid_metrics(x)
    xh = Array(x)
    dx = xh[2] - xh[1]
    L = xh[end] - xh[begin] + dx
    return dx, L
end

function _fft_differentials(X, K)
    return ntuple(length(X)) do i
        x = Array(X[i])
        k = Array(K[i])
        dx = x[2] - x[1]
        dk = k[2] - k[1]
        RT = promote_type(eltype(x), eltype(k))
        return (RT(dx / sqrt(2π)), RT(length(k) * dk / sqrt(2π)))
    end
end

function _default_radial_grid(psi::Psi, nradial)
    khost = map(Array, psi.K)
    kmax = maximum(map(k -> maximum(abs, k), khost))
    RT = real(eltype(psi.ψ))
    return CUDA.CuArray(collect(RT, LinRange(zero(RT), RT(kmax), nradial)))
end

function _build_bessel_radius_cache(k, nx::Int, ny::Int, dx, dy)
    dx ≈ dy || error("Cached CUDA Bessel reduction currently requires equal grid spacing")
    ix0 = nx ÷ 2
    iy0 = ny ÷ 2
    ids = Vector{Int32}(undef, nx * ny)
    index = Dict{Int,Int32}()
    RT = real(eltype(k))
    radii = RT[]
    next_id = Int32(0)
    for q = 1:ny, p = 1:nx
        ix = p - 1 - ix0
        iy = q - 1 - iy0
        key = ix * ix + iy * iy
        id = get(index, key, Int32(0))
        if id == 0
            push!(radii, RT(dx * sqrt(key)))
            next_id += 1
            id = next_id
            index[key] = id
        end
        ids[p+(q-1)*nx] = id
    end
    radii_gpu = CUDA.CuArray(radii)
    weights = SpecialFunctions.besselj0.(reshape(k, :, 1) .* reshape(radii_gpu, 1, :))
    return RadialWeightCache(CUDA.CuArray(ids), radii_gpu, weights)
end

_sinc_weight(kr) = ifelse(abs(kr) > 1.0e-12, sin(kr) / kr, one(kr) - kr * kr / 6)

function _build_sinc_radius_cache(k, nx::Int, ny::Int, nz::Int, dx, dy, dz)
    (dx ≈ dy && dx ≈ dz) ||
        error("Cached CUDA sinc reduction currently requires equal grid spacing")
    ix0 = nx ÷ 2
    iy0 = ny ÷ 2
    iz0 = nz ÷ 2
    ids = Vector{Int32}(undef, nx * ny * nz)
    index = Dict{Int,Int32}()
    RT = real(eltype(k))
    radii = RT[]
    next_id = Int32(0)
    for r = 1:nz, q = 1:ny, p = 1:nx
        ix = p - 1 - ix0
        iy = q - 1 - iy0
        iz = r - 1 - iz0
        key = ix * ix + iy * iy + iz * iz
        id = get(index, key, Int32(0))
        if id == 0
            push!(radii, RT(dx * sqrt(key)))
            next_id += 1
            id = next_id
            index[key] = id
        end
        ids[p+(q-1)*nx+(r-1)*nx*ny] = id
    end
    radii_gpu = CUDA.CuArray(radii)
    weights = _sinc_weight.(reshape(k, :, 1) .* reshape(radii_gpu, 1, :))
    return RadialWeightCache(CUDA.CuArray(ids), radii_gpu, weights)
end

function spectrum_cache(
    psi::Psi{D};
    backend = CUDADevice(),
    k = nothing,
    nradial = length(psi.K[1]),
    chunk = 2048,
) where {D}
    backend isa CUDADevice ||
        error("Unsupported spectral analysis backend: $(typeof(backend))")
    D in (2, 3) || error("CUDA spectral analysis currently supports 2D and 3D Psi fields")
    _require_device_psi(psi)

    T = eltype(psi.ψ)
    RT = real(T)
    kout = isnothing(k) ? _default_radial_grid(psi, nradial) : CUDA.CuArray(k)
    padsize = ntuple(i -> 2 * size(psi.ψ, i), D)
    DXDK = _fft_differentials(psi.X, psi.K)
    DX = ntuple(i -> RT(DXDK[i][1]), D)
    DK = ntuple(i -> RT(DXDK[i][2]), D)
    nlinear = prod(padsize)
    groups = cld(nlinear, chunk)
    radial = if D == 2
        dx = Array(psi.X[1][2:2] .- psi.X[1][1:1])[1]
        dy = Array(psi.X[2][2:2] .- psi.X[2][1:1])[1]
        _build_bessel_radius_cache(kout, padsize[1], padsize[2], dx, dy)
    elseif D == 3
        dx = Array(psi.X[1][2:2] .- psi.X[1][1:1])[1]
        dy = Array(psi.X[2][2:2] .- psi.X[2][1:1])[1]
        dz = Array(psi.X[3][2:2] .- psi.X[3][1:1])[1]
        _build_sinc_radius_cache(kout, padsize[1], padsize[2], padsize[3], dx, dy, dz)
    else
        nothing
    end

    return CUDASpectrumCache(
        psi.X,
        psi.K,
        kout,
        DX,
        DK,
        CUDA.zeros(T, padsize),
        CUDA.zeros(T, padsize),
        CUDA.zeros(T, padsize),
        CUDA.zeros(RT, length(kout), groups),
        radial,
        CUDA.zeros(RT, length(kout)),
        CUDA.zeros(RT, length(kout)),
        CUDA.zeros(RT, length(kout)),
        CUDA.zeros(RT, length(kout)),
        CUDA.zeros(RT, 3),
        groups,
        chunk,
    )
end

function _zeropad!(B, A::CUDA.CuArray{T,D}) where {T,D}
    S = size(A)
    any(isodd, S) && error("Array dims not divisible by 2")
    fill!(B, zero(T))
    nI = S .÷ 2
    inner = ntuple(i -> (nI[i]+1):(nI[i]+S[i]), D)
    copyto!(view(B, inner...), A)
    return B
end

function _auto_correlate!(out, ψ, cache::CUDASpectrumCache{D}) where {D}
    _zeropad!(out, ψ)
    work = cache.fftwork
    dxscale = prod(cache.DX)^2
    dkscale = prod(cache.DK) * (2π)^(D / 2)
    work .= fft(out)
    @. work = abs2(work) * dxscale
    out .= fftshift(ifft(work) .* dkscale)
    return out
end

function _gradient(psi::Psi{1})
    (; ψ, K) = psi
    kx = K[1]
    ϕ = fft(ψ)
    return ifft(@. im * kx * ϕ)
end

function _gradient(psi::Psi{2})
    (; ψ, K) = psi
    kx, ky = K
    ϕ = fft(ψ)
    ψx = ifft(@. im * kx * ϕ)
    ψy = ifft(@. im * ky' * ϕ)
    return ψx, ψy
end

function _gradient(psi::Psi{3})
    (; ψ, K) = psi
    kx, ky, kz = K
    kzr = reshape(kz, 1, 1, length(kz))
    ϕ = fft(ψ)
    ψx = ifft(@. im * kx * ϕ)
    ψy = ifft(@. im * ky' * ϕ)
    ψz = ifft(@. im * kzr * ϕ)
    return ψx, ψy, ψz
end

function _velocity(psi::Psi{2}, Ω)
    (; ψ, X) = psi
    x, y = X
    ψx, ψy = _gradient(psi)
    ρ = abs2.(ψ)
    threshold = eps(real(eltype(ρ)))
    z = zero(threshold)
    ρsafe = @. ifelse(ρ > threshold, ρ, one(ρ))
    numerx = @. imag(conj(ψ) * ψx)
    numery = @. imag(conj(ψ) * ψy)
    xcol = reshape(x, :, 1)
    yrow = reshape(y, 1, :)
    vx = @. ifelse(ρ > threshold, numerx / ρsafe + Ω * yrow, Ω * yrow)
    vy = @. ifelse(ρ > threshold, numery / ρsafe - Ω * xcol, -Ω * xcol)
    return vx, vy
end

function _velocity(psi::Psi{3}, Ω)
    iszero(Ω) || error("Rotating-frame CUDA spectra are only implemented for 2D")
    (; ψ) = psi
    ψx, ψy, ψz = _gradient(psi)
    ρ = abs2.(ψ)
    threshold = eps(real(eltype(ρ)))
    z = zero(threshold)
    ρsafe = @. ifelse(ρ > threshold, ρ, one(ρ))
    numerx = @. imag(conj(ψ) * ψx)
    numery = @. imag(conj(ψ) * ψy)
    numerz = @. imag(conj(ψ) * ψz)
    vx = @. ifelse(ρ > threshold, numerx / ρsafe, z)
    vy = @. ifelse(ρ > threshold, numery / ρsafe, z)
    vz = @. ifelse(ρ > threshold, numerz / ρsafe, z)
    return vx, vy, vz
end

function _helmholtz(wx, wy, kx, ky)
    wxk = fft(wx)
    wyk = fft(wy)
    kxc = reshape(kx, :, 1)
    kyr = reshape(ky, 1, :)
    den = @. kxc^2 + kyr^2
    zden = zero(eltype(den))
    oden = one(eltype(den))
    zkw = zero(eltype(wxk))
    den_safe = @. ifelse(den > zden, den, oden)
    numer = @. kxc * wxk + kyr * wyk
    kw = @. ifelse(den > zden, numer / den_safe, zkw)
    wxkc = @. kw * kxc
    wykc = @. kw * kyr
    return (ifft(wxk .- wxkc), ifft(wyk .- wykc)), (ifft(wxkc), ifft(wykc))
end

function _helmholtz(wx, wy, wz, kx, ky, kz)
    wxk = fft(wx)
    wyk = fft(wy)
    wzk = fft(wz)
    kxc = reshape(kx, :, 1, 1)
    kyr = reshape(ky, 1, :, 1)
    kzr = reshape(kz, 1, 1, :)
    den = @. kxc^2 + kyr^2 + kzr^2
    zden = zero(eltype(den))
    oden = one(eltype(den))
    zkw = zero(eltype(wxk))
    den_safe = @. ifelse(den > zden, den, oden)
    numer = @. kxc * wxk + kyr * wyk + kzr * wzk
    kw = @. ifelse(den > zden, numer / den_safe, zkw)
    wxkc = @. kw * kxc
    wykc = @. kw * kyr
    wzkc = @. kw * kzr
    return (ifft(wxk .- wxkc), ifft(wyk .- wykc), ifft(wzk .- wzkc)),
    (ifft(wxkc), ifft(wykc), ifft(wzkc))
end

function _bessel_reduce_cached_kernel!(partials, C, rid, weights, nx, ny, chunk)
    i = blockIdx().x
    g = blockIdx().y
    tid = threadIdx().x
    stride = blockDim().x
    start = (g - 1) * chunk + 1
    stop = min(g * chunk, nx * ny)
    s = zero(eltype(partials))
    @inbounds for lin = (start+tid-1):stride:stop
        s += weights[i, rid[lin]] * real(C[lin])
    end
    shared = CuDynamicSharedArray(eltype(partials), blockDim().x)
    @inbounds shared[tid] = s
    sync_threads()

    offset = blockDim().x ÷ 2
    while offset > 0
        if tid <= offset
            @inbounds shared[tid] += shared[tid+offset]
        end
        sync_threads()
        offset ÷= 2
    end

    if tid == 1
        @inbounds partials[i, g] = shared[1]
    end
    return nothing
end

function _sinc_reduce_cached_kernel!(partials, C, rid, weights, nx, ny, nz, chunk)
    i = blockIdx().x
    g = blockIdx().y
    tid = threadIdx().x
    stride = blockDim().x
    start = (g - 1) * chunk + 1
    stop = min(g * chunk, nx * ny * nz)
    s = zero(eltype(partials))
    @inbounds for lin = (start+tid-1):stride:stop
        s += weights[i, rid[lin]] * real(C[lin])
    end
    shared = CuDynamicSharedArray(eltype(partials), blockDim().x)
    @inbounds shared[tid] = s
    sync_threads()

    offset = blockDim().x ÷ 2
    while offset > 0
        if tid <= offset
            @inbounds shared[tid] += shared[tid+offset]
        end
        sync_threads()
        offset ÷= 2
    end

    if tid == 1
        @inbounds partials[i, g] = shared[1]
    end
    return nothing
end

function _reduce!(out, cache::CUDASpectrumCache{2}, C)
    x, y = cache.X
    dx, _ = _grid_metrics(x)
    dy, _ = _grid_metrics(y)
    nx, ny = size(C)
    fill!(cache.partials, zero(eltype(cache.partials)))
    threads = 256
    groups = cld(nx * ny, cache.chunk)
    shmem = threads * sizeof(eltype(cache.partials))
    @cuda threads = threads blocks = (length(cache.k), groups) shmem = shmem _bessel_reduce_cached_kernel!(
        cache.partials,
        C,
        cache.radial.rid,
        cache.radial.weights,
        nx,
        ny,
        cache.chunk,
    )
    sums = vec(sum(view(cache.partials, :, 1:groups); dims = 2))
    out .= sums .* cache.k .* (dx * dy / (2π))
    return out
end

function _reduce!(out, cache::CUDASpectrumCache{3}, C)
    x, y, z = cache.X
    dx, _ = _grid_metrics(x)
    dy, _ = _grid_metrics(y)
    dz, _ = _grid_metrics(z)
    nx, ny, nz = size(C)
    fill!(cache.partials, zero(eltype(cache.partials)))
    threads = 256
    groups = cld(nx * ny * nz, cache.chunk)
    shmem = threads * sizeof(eltype(cache.partials))
    @cuda threads = threads blocks = (length(cache.k), groups) shmem = shmem _sinc_reduce_cached_kernel!(
        cache.partials,
        C,
        cache.radial.rid,
        cache.radial.weights,
        nx,
        ny,
        nz,
        cache.chunk,
    )
    sums = vec(sum(view(cache.partials, :, 1:groups); dims = 2))
    out .= sums .* cache.k .^ 2 .* (dx * dy * dz / (2π^2))
    return out
end

function _energy_total_partials_kernel!(partials, row, ax, ay, nlinear, chunk)
    g = blockIdx().x
    tid = threadIdx().x
    stride = blockDim().x
    start = (g - 1) * chunk + 1
    stop = min(g * chunk, nlinear)
    s = zero(eltype(partials))
    @inbounds for lin = (start+tid-1):stride:stop
        s += (abs2(ax[lin]) + abs2(ay[lin])) / 2
    end
    shared = CuDynamicSharedArray(eltype(partials), blockDim().x)
    @inbounds shared[tid] = s
    sync_threads()

    offset = blockDim().x ÷ 2
    while offset > 0
        if tid <= offset
            @inbounds shared[tid] += shared[tid+offset]
        end
        sync_threads()
        offset ÷= 2
    end

    if tid == 1
        @inbounds partials[row, g] = shared[1]
    end
    return nothing
end

function _energy_total_partials_kernel!(partials, row, ax, ay, az, nlinear, chunk)
    g = blockIdx().x
    tid = threadIdx().x
    stride = blockDim().x
    start = (g - 1) * chunk + 1
    stop = min(g * chunk, nlinear)
    s = zero(eltype(partials))
    @inbounds for lin = (start+tid-1):stride:stop
        s += (abs2(ax[lin]) + abs2(ay[lin]) + abs2(az[lin])) / 2
    end
    shared = CuDynamicSharedArray(eltype(partials), blockDim().x)
    @inbounds shared[tid] = s
    sync_threads()

    offset = blockDim().x ÷ 2
    while offset > 0
        if tid <= offset
            @inbounds shared[tid] += shared[tid+offset]
        end
        sync_threads()
        offset ÷= 2
    end

    if tid == 1
        @inbounds partials[row, g] = shared[1]
    end
    return nothing
end

function _finish_total_kernel!(totals, slot, partials, row, groups)
    tid = threadIdx().x
    stride = blockDim().x
    s = zero(eltype(totals))
    @inbounds for g = tid:stride:groups
        s += partials[row, g]
    end
    shared = CuDynamicSharedArray(eltype(totals), blockDim().x)
    @inbounds shared[tid] = s
    sync_threads()

    offset = blockDim().x ÷ 2
    while offset > 0
        if tid <= offset
            @inbounds shared[tid] += shared[tid+offset]
        end
        sync_threads()
        offset ÷= 2
    end

    if tid == 1
        @inbounds totals[slot] = shared[1]
    end
    return nothing
end

function _add_total_kernel!(totals, total_slot, a_slot, b_slot)
    if threadIdx().x == 1
        @inbounds totals[total_slot] = totals[a_slot] + totals[b_slot]
    end
    return nothing
end

function _finish_total!(cache::CUDASpectrumCache, slot, row, groups)
    threads = 256
    shmem = threads * sizeof(eltype(cache.totals))
    @cuda threads = threads blocks = 1 shmem = shmem _finish_total_kernel!(
        cache.totals,
        slot,
        cache.partials,
        row,
        groups,
    )
    return cache.totals
end

function _update_energy_total!(cache::CUDASpectrumCache, slot, components...)
    fill!(cache.partials, zero(eltype(cache.partials)))
    threads = 256
    nlinear = length(first(components))
    groups = cld(nlinear, cache.chunk)
    shmem = threads * sizeof(eltype(cache.partials))
    @cuda threads = threads blocks = groups shmem = shmem _energy_total_partials_kernel!(
        cache.partials,
        1,
        components...,
        nlinear,
        cache.chunk,
    )
    _finish_total!(cache, slot, 1, groups)
    return cache.totals
end

function _update_kinetic_total!(cache::CUDASpectrumCache)
    @cuda threads = 1 blocks = 1 _add_total_kernel!(
        cache.totals,
        TOTAL_KINETIC,
        TOTAL_INCOMPRESSIBLE,
        TOTAL_COMPRESSIBLE,
    )
    return cache.totals
end

function _density_spectrum!(out, cache::CUDASpectrumCache, psi::Psi)
    _auto_correlate!(cache.corr, abs2.(psi.ψ), cache)
    _reduce!(out, cache, cache.corr)
    return out
end

function _kinetic_density!(out, cache::CUDASpectrumCache{2}, psi::Psi{2})
    ψx, ψy = _gradient(psi)
    _auto_correlate!(cache.corr, ψx, cache)
    _auto_correlate!(cache.work, ψy, cache)
    @. cache.corr = 0.5 * (cache.corr + cache.work)
    _reduce!(out, cache, cache.corr)
    return out
end

function _kinetic_density!(out, cache::CUDASpectrumCache{3}, psi::Psi{3})
    ψx, ψy, ψz = _gradient(psi)
    _auto_correlate!(cache.corr, ψx, cache)
    _auto_correlate!(cache.work, ψy, cache)
    @. cache.corr = cache.corr + cache.work
    _auto_correlate!(cache.work, ψz, cache)
    @. cache.corr = 0.5 * (cache.corr + cache.work)
    _reduce!(out, cache, cache.corr)
    return out
end

function _kinetic_decomposition!(
    incompressible,
    compressible,
    cache::CUDASpectrumCache{2},
    psi::Psi{2},
    Ω,
)
    (; ψ, K) = psi
    vx, vy = _velocity(psi, Ω)
    amp = abs.(ψ)
    wx = @. amp * vx
    wy = @. amp * vy
    Wi, Wc = _helmholtz(wx, wy, K...)
    _update_energy_total!(cache, TOTAL_INCOMPRESSIBLE, Wi...)
    _update_energy_total!(cache, TOTAL_COMPRESSIBLE, Wc...)
    _update_kinetic_total!(cache)

    _auto_correlate!(cache.corr, Wi[1], cache)
    _auto_correlate!(cache.work, Wi[2], cache)
    @. cache.corr = 0.5 * (cache.corr + cache.work)
    _reduce!(incompressible, cache, cache.corr)

    _auto_correlate!(cache.corr, Wc[1], cache)
    _auto_correlate!(cache.work, Wc[2], cache)
    @. cache.corr = 0.5 * (cache.corr + cache.work)
    _reduce!(compressible, cache, cache.corr)
    return incompressible, compressible
end

function _kinetic_decomposition!(
    incompressible,
    compressible,
    cache::CUDASpectrumCache{3},
    psi::Psi{3},
    Ω,
)
    (; ψ, K) = psi
    vx, vy, vz = _velocity(psi, Ω)
    amp = abs.(ψ)
    wx = @. amp * vx
    wy = @. amp * vy
    wz = @. amp * vz
    Wi, Wc = _helmholtz(wx, wy, wz, K...)
    _update_energy_total!(cache, TOTAL_INCOMPRESSIBLE, Wi...)
    _update_energy_total!(cache, TOTAL_COMPRESSIBLE, Wc...)
    _update_kinetic_total!(cache)

    _auto_correlate!(cache.corr, Wi[1], cache)
    _auto_correlate!(cache.work, Wi[2], cache)
    @. cache.corr = cache.corr + cache.work
    _auto_correlate!(cache.work, Wi[3], cache)
    @. cache.corr = 0.5 * (cache.corr + cache.work)
    _reduce!(incompressible, cache, cache.corr)

    _auto_correlate!(cache.corr, Wc[1], cache)
    _auto_correlate!(cache.work, Wc[2], cache)
    @. cache.corr = cache.corr + cache.work
    _auto_correlate!(cache.work, Wc[3], cache)
    @. cache.corr = 0.5 * (cache.corr + cache.work)
    _reduce!(compressible, cache, cache.corr)
    return incompressible, compressible
end

function analyze_spectra!(
    cache::CUDASpectrumCache,
    psi::Psi;
    spectra = (:density, :kinetic),
    Ω = 0,
)
    _require_device_psi(psi)
    psi.X === cache.X || error("Spectrum cache X vectors do not match psi.X")
    psi.K === cache.K || error("Spectrum cache K vectors do not match psi.K")
    if :density in spectra
        _density_spectrum!(cache.density, cache, psi)
    end
    if :kinetic in spectra
        _kinetic_density!(cache.kinetic, cache, psi)
    end
    if (:incompressible in spectra) || (:compressible in spectra)
        _kinetic_decomposition!(
            cache.incompressible,
            cache.compressible,
            cache,
            psi,
            real(eltype(cache.k))(Ω),
        )
    end
    return cache
end

function spectrum_results(cache::CUDASpectrumCache; host = false)
    result = (
        k = cache.k,
        density = cache.density,
        kinetic = cache.kinetic,
        incompressible = cache.incompressible,
        compressible = cache.compressible,
        totals = cache.totals,
    )
    host || return result
    totals = Array(cache.totals)
    return (
        k = Array(cache.k),
        density = Array(cache.density),
        kinetic = Array(cache.kinetic),
        incompressible = Array(cache.incompressible),
        compressible = Array(cache.compressible),
        totals = (
            kinetic = totals[TOTAL_KINETIC],
            incompressible = totals[TOTAL_INCOMPRESSIBLE],
            compressible = totals[TOTAL_COMPRESSIBLE],
        ),
    )
end

end
