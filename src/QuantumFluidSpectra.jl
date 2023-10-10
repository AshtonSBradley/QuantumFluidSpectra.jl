module QuantumFluidSpectra

using Tullio
using FFTW 
using SpecialFunctions
using PaddedViews
using UnPack
using TensorCast

# fallback since fast_hypot is 2 argument only
@fastmath hypot(x::Float64, y::Float64, z::Float64)=sqrt(x^2+y^2+z^2)
export hypot 

abstract type Field end
struct Psi{D} <: Field
    ψ::Array{Complex{Float64},D}
    X::NTuple{D}
    K::NTuple{D}
end

include("arrays.jl")
include("analysis.jl")

export Psi, xvecs, kvecs
export auto_correlate, cross_correlate
export bessel_reduce, sinc_reduce, gv, gv3
export log10range, convolve

export xk_arrays, fft_differentials
export gradient, velocity, current
export energydecomp, helmholtz, kinetic_density
export incompressible_spectrum, compressible_spectrum, qpressure_spectrum
export incompressible_density, compressible_density, qpressure_density
export ic_density, iq_density, cq_density
export density_spectrum, trap_spectrum

end
