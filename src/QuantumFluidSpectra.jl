module QuantumFluidSpectra

using Tullio
using FFTW
using SpecialFunctions
using PaddedViews
using UnPack

# fallback since fast_hypot is 2 argument only
@fastmath hypot(x::Float64, y::Float64, z::Float64)=sqrt(x^2+y^2+z^2)
export hypot

abstract type Field end
struct Psi{
    D,
    T<:Complex,
    AT<:AbstractArray{T,D},
    XT<:NTuple{D,<:AbstractVector},
    KT<:NTuple{D,<:AbstractVector},
} <: Field
    ψ::AT
    X::XT
    K::KT
end

include("arrays.jl")
include("analysis.jl")

export Psi, xvec, kvec, xvecs, kvecs, radial_kgrid
export auto_correlate, cross_correlate
export bessel_reduce, sinc_reduce, gv, gv3
export log10range, convolve

export xk_arrays, fft_differentials
export gradient, velocity, current
export energydecomp, helmholtz, kinetic_density, knumber_density, wave_action
export incompressible_spectrum, compressible_spectrum, qpressure_spectrum
export incompressible_density, compressible_density, qpressure_density
export ic_density, iq_density, cq_density
export density_spectrum, trap_spectrum
export gpe_particle_transfer, gpe_particle_flux

end
