"""
	gradient(psi::Psi{D})

Compute the `D` vector gradient components of a wavefunction `Psi` of spatial dimension `D`.
The `D` gradient components returned are `D`-dimensional arrays.
"""
function gradient(psi::Psi{1})
	@unpack ψ,K = psi; kx = K[1] 
    ϕ = fft(ψ)
	ψx = ifft(im*kx.*ϕ)
    return ψx
end

function gradient(psi::Psi{2})
	@unpack ψ,K = psi; kx,ky = K 
	ϕ = fft(ψ)
	ψx = ifft(im*kx.*ϕ)
	ψy = ifft(im*ky'.*ϕ)
	return ψx,ψy
end

function gradient(psi::Psi{3})
	@unpack ψ,K = psi; kx,ky,kz = K 
	ϕ = fft(ψ)
	ψx = ifft(im*kx.*ϕ)
	ψy = ifft(im*ky'.*ϕ)
	ψz = ifft(im*reshape(kz,1,1,length(kz)).*ϕ)
	return ψx,ψy,ψz
end

"""
	current(psi::Psi{D})

Compute the `D` current components of an `Psi` of spatial dimension `D`.
The `D` cartesian components returned are `D`-dimensional arrays.
"""
function current(psi::Psi{1})
	@unpack ψ = psi 
	ψx = gradient(psi)
	jx = @. imag(conj(ψ)*ψx)
    return jx
end

function current(psi::Psi{2},Ω = 0)
	@unpack ψ,X = psi 
    x,y = X
    ψx,ψy = gradient(psi)
	jx = @. imag(conj(ψ)*ψx) + Ω*abs2(ψ)*y'  
	jy = @. imag(conj(ψ)*ψy) - Ω*abs2(ψ)*x 
	return jx,jy
end

function current(psi::Psi{3})
    @unpack ψ = psi 
    ψx,ψy,ψz = gradient(psi)
	jx = @. imag(conj(ψ)*ψx)
	jy = @. imag(conj(ψ)*ψy)
	jz = @. imag(conj(ψ)*ψz)
	return jx,jy,jz
end

"""
	velocity(psi::Psi{D})

Compute the `D` velocity components of an `Psi` of spatial dimension `D`.
The `D` velocities returned are `D`-dimensional arrays.
"""
_safe_rho_threshold(rho) = eps(float(real(one(eltype(rho)))))

function _safe_velocity(numerator, rho, offset)
    threshold = _safe_rho_threshold(rho)
    return @. ifelse(rho > threshold, numerator/rho + offset, offset)
end

function _safe_radial_divide(spectrum, k, power)
    threshold = eps(float(real(one(eltype(k)))))
    return @. ifelse(abs(k) > threshold, spectrum / (k^power), zero(spectrum))
end

function velocity(psi::Psi{1})
	@unpack ψ = psi
    ψx = gradient(psi)
    rho = abs2.(ψ)
    numer = @. imag(conj(ψ)*ψx)
	vx = _safe_velocity(numer, rho, zero(eltype(rho)))
	return vx
end

function velocity(psi::Psi{2},Ω = 0)
	@unpack ψ,X = psi
    x,y = X
    ψx,ψy = gradient(psi)
    rho = abs2.(ψ)
    numerx = @. imag(conj(ψ)*ψx)
    numery = @. imag(conj(ψ)*ψy)
	vx = _safe_velocity(numerx, rho, Ω*y')
	vy = _safe_velocity(numery, rho, -Ω*x)
	return vx,vy
end

function velocity(psi::Psi{3})
	@unpack ψ = psi
	rho = abs2.(ψ)
    ψx,ψy,ψz = gradient(psi)
    numerx = @. imag(conj(ψ)*ψx)
    numery = @. imag(conj(ψ)*ψy)
    numerz = @. imag(conj(ψ)*ψz)
	vx = _safe_velocity(numerx, rho, zero(eltype(rho)))
	vy = _safe_velocity(numery, rho, zero(eltype(rho)))
	vz = _safe_velocity(numerz, rho, zero(eltype(rho)))
	return vx,vy,vz
end

"""
	Wi,Wc = helmholtz(wx,...,kx,...)

Computes a 2 or 3 dimensional Helmholtz decomposition of the vector field with components `wx`, `wy`, or `wx`, `wy`, `wz`. 
`psi` is passed to provide requisite arrays in `k`-space.
Returned fields `Wi`, `Wc` are tuples of cartesian components of incompressible and compressible respectively.
"""
function helmholtz(wx, wy, kx, ky)
    wxk = fft(wx); wyk = fft(wy)
    kw = @. (kx * wxk + ky' * wyk)/ (kx^2+ky'^2)
    wxkc = @. kw*kx
    wykc = @. kw*ky'
    wxkc[1] = zero(wxkc[1]); wykc[1] = zero(wykc[1])
    wxki = @. wxk - wxkc
    wyki = @. wyk - wykc
    wxc = ifft(wxkc); wyc = ifft(wykc)
  	wxi = ifft(wxki); wyi = ifft(wyki)
  	Wi = (wxi, wyi); Wc = (wxc, wyc)
    return Wi, Wc
end

function helmholtz(wx, wy, wz, kx, ky, kz)
    wxk = fft(wx); wyk = fft(wy); wzk = fft(wz)
    kzr = reshape(kz,1,1,length(kz))
    kw = @. (kx * wxk + ky' * wyk + kzr * wzk)/ (kx^2 + ky'^2 + kzr^2)
    wxkc = @. kw * kx  
    wykc = @. kw * ky'
    wzkc = @. kw * kzr  
    wxkc[1] = zero(wxkc[1]); wykc[1] = zero(wykc[1]); wzkc[1] = zero(wzkc[1])
    wxki = @. wxk - wxkc
    wyki = @. wyk - wykc
    wzki = @. wzk - wzkc
    wxc = ifft(wxkc); wyc = ifft(wykc); wzc = ifft(wzkc)
    wxi = ifft(wxki); wyi = ifft(wyki); wzi = ifft(wzki)
  	Wi = (wxi, wyi, wzi); Wc = (wxc, wyc, wzc)
    return Wi, Wc
end

# function helmholtz(W::NTuple{N,Array{Float64,N}}, psi::Psi{N}) where N
#     return helmholtz(W..., psi)
# end

"""
	et,ei,ec = energydecomp(psi::Xfield{D})

Decomposes the hydrodynamic kinetic energy of `psi`, returning the total `et`, incompressible `ei`,
and compressible `ec` energy densities in position space. `D` can be 2 or 3 dimensions.
"""
function energydecomp(psi::Psi{2})
    @unpack ψ,K = psi; kx,ky = K
    a = abs.(ψ)
    vx, vy = velocity(psi)
    wx = @. a*vx; wy = @. a*vy
    Wi, Wc = helmholtz(wx,wy,kx,ky)
    wxi, wyi = Wi; wxc, wyc = Wc
    et = @. abs2(wx) + abs2(wy); et *= 0.5
    ei = @. abs2(wxi) + abs2(wyi); ei *= 0.5
    ec = @. abs2(wxc) + abs2(wyc); ec *= 0.5
    return et, ei, ec
end

function energydecomp(psi::Psi{3})
	@unpack ψ,K = psi; kx,ky,kz = K
    a = abs.(ψ)
    vx,vy,vz = velocity(psi)
    wx = @. a*vx; wy = @. a*vy; wz = @. a*vz
    Wi, Wc = helmholtz(wx,wy,wz,kx,ky,kz)
    wxi, wyi, wzi = Wi; wxc, wyc, wzc = Wc
    et = @. abs2(wx) + abs2(wy) + abs2(wz); et *= 0.5
    ei = @. abs2(wxi) + abs2(wyi) + abs2(wzi); ei *= 0.5
    ec = @. abs2(wxc) + abs2(wyc) + abs2(wzc); ec *= 0.5
    return et, ei, ec
end

"""
	zeropad(A)

Zero-pad the array `A` to twice the size with the same element type as `A`.
"""
function zeropad(A)
    S = size(A)
    if any(isodd.(S))
        error("Array dims not divisible by 2")
    end
    nO = 2 .* S
    nI = S .÷ 2

    outer = ntuple(i -> 1:nO[i], length(S))
    inner = ntuple(i -> (nI[i] + 1):(nI[i] + 2*nI[i]), length(S))

    return PaddedView(zero(eltype(A)),A,outer,inner) |> collect
end

"""
	log10range(a,b,n)

Create a vector that is linearly spaced in log space, containing `n` values bracketed by `a` and `b`.
"""
function log10range(a,b,n)
	@assert a>0
    x = LinRange(log10(a),log10(b),n)
    return @. 10^x
end

@doc raw"""
	A = convolve(ψ1,ψ2,X,K)

Computes the convolution of two complex fields according to

```math
A(\rho) = \int d^2r\;\psi_1^*(r+\rho)\psi_2(r)
```
using FFTW.
"""
function convolve(ψ1,ψ2,X,K)
    n = length(X)
    DX,DK = fft_differentials(X,K)
	ϕ1 = zeropad(conj.(ψ1))
    ϕ2 = zeropad(ψ2)

	χ1 = fft(ϕ1)*prod(DX)
	χ2 = fft(ϕ2)*prod(DX)
	return ifft(χ1.*χ2)*prod(DK)*(2*pi)^(n/2) |> fftshift
end

@doc raw"""
	auto_correlate(ψ,X,K)

Return the auto-correlation integral of a complex field ``\psi``, ``A``, given by

```
A(\rho)=\int d^2r\;\psi^*(r-\rho)\psi(r)
```

defined on a cartesian grid on a cartesian grid using FFTW.

`X` and `K` are tuples of vectors `x`,`y`,`kx`, `ky`.

This method is useful for evaluating spectra from cartesian data.
"""
function auto_correlate(ψ,X,K)
    n = length(X)
    DX,DK = fft_differentials(X,K)
    ϕ = zeropad(ψ)
	χ = fft(ϕ)*prod(DX)
	return ifft(abs2.(χ))*prod(DK)*(2*pi)^(n/2) |> fftshift
end

auto_correlate(psi::Psi{D}) where D = auto_correlate(psi.ψ,psi.X,psi.K)

@doc raw"""
	cross_correlate(ψ,X,K)

Cross correlation of complex field ``\psi_1``, and ``\psi_2`` given by

```
A(\rho)=\int d^2r\;\psi_1^*(r-\rho)\psi_2(r)
```

evaluated on a cartesian grid using Fourier convolution.

`X` and `K` are tuples of vectors `x`,`y`,`kx`, `ky`.

This method is useful for evaluating spectra from cartesian data.
"""
function cross_correlate(ψ1,ψ2,X,K)
    n = length(X)
    DX,DK = fft_differentials(X,K)
    ϕ1 = zeropad(ψ1)
    ϕ2 = zeropad(ψ2)
	χ1 = fft(ϕ1)*prod(DX)
    χ2 = fft(ϕ2)*prod(DX)
	return ifft(conj(χ1).*χ2)*prod(DK)*(2*pi)^(n/2) |> fftshift
end
cross_correlate(psi1::Psi{D},psi2::Psi{D}) where D = cross_correlate(psi1.ψ,psi2.ψ,psi1.X,psi1.K)

function bessel_reduce(k,x,y,C)
    dx,dy = x[2]-x[1],y[2]-y[1]
    Nx,Ny = 2*length(x),2*length(y)
    Lx = x[end] - x[begin] + dx
    Ly = y[end] - y[begin] + dy
    xp = LinRange(-Lx,Lx,Nx+1)[1:Nx]
    yq = LinRange(-Ly,Ly,Ny+1)[1:Ny]
    E = zero(k)
    @tullio E[i] = real(besselj0(k[i]*hypot(xp[p],yq[q]))*C[p,q])
    @. E *= k*dx*dy/2/pi 
    return E 
end

function _integrated_bessel_reduce(k, x, y, C)
    dx,dy = x[2]-x[1],y[2]-y[1]
    Nx,Ny = 2*length(x),2*length(y)
    Lx = x[end] - x[begin] + dx
    Ly = y[end] - y[begin] + dy
    xp = LinRange(-Lx,Lx,Nx+1)[1:Nx]
    yq = LinRange(-Ly,Ly,Ny+1)[1:Ny]

    out = similar(k, eltype(real(C)))

    @tullio out[i] = begin
        ρ = hypot(xp[p], yq[q])
        if ρ > 1e-12
            real(k[i] * besselj1(k[i] * ρ) / ρ * C[p,q])
        else
            real(0.5 * k[i]^2 * C[p,q])
        end
    end

    @. out *= dx * dy
    return out
end

function sinc_reduce(k,x,y,z,C)
    dx,dy,dz = x[2]-x[1],y[2]-y[1],z[2]-z[1]
    Nx,Ny,Nz = 2*length(x),2*length(y),2*length(z)
    Lx = x[end] - x[begin] + dx
    Ly = y[end] - y[begin] + dy
    Lz = z[end] - z[begin] + dz
    xp = LinRange(-Lx,Lx,Nx+1)[1:Nx]
    yq = LinRange(-Ly,Ly,Ny+1)[1:Ny]
    zr = LinRange(-Lz,Lz,Nz+1)[1:Nz]
    E = zero(k)
    @tullio E[i] = real(π*sinc(k[i]*hypot(xp[p],yq[q],zr[r])/π)*C[p,q,r]) 
    @. E *= k^2*dx*dy*dz/2/pi^2  
    return E 
end

function _integrated_sinc_reduce(k, x, y, z, C)
    dx,dy,dz = x[2]-x[1],y[2]-y[1],z[2]-z[1]
    Nx,Ny,Nz = 2*length(x),2*length(y),2*length(z)
    Lx = x[end] - x[begin] + dx
    Ly = y[end] - y[begin] + dy
    Lz = z[end] - z[begin] + dz
    xp = LinRange(-Lx,Lx,Nx+1)[1:Nx]
    yq = LinRange(-Ly,Ly,Ny+1)[1:Ny]
    zr = LinRange(-Lz,Lz,Nz+1)[1:Nz]

    out = similar(k, eltype(real(C)))

    @tullio out[i] = begin
        r = hypot(xp[p], yq[q], zr[r_])
        if r > 1e-12
            kr = k[i] * r
            real((sin(kr) - kr * cos(kr)) / r^3 * C[p,q,r_])
        else
            real((k[i]^3) / 3 * C[p,q,r_])
        end
    end

    @. out *= dx * dy * dz
    return out
end

"""
	kinetic_density(k,ψ,X,K)

Calculates the kinetic enery spectrum for wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `xk_arrays`.
"""
function kinetic_density(k,psi::Psi{2})
    @unpack ψ,X,K = psi; 
    ψx,ψy = gradient(psi)
	cx = auto_correlate(ψx,X,K)
	cy = auto_correlate(ψy,X,K)
    C = @. 0.5(cx + cy)
    return bessel_reduce(k,X...,C)
end

function kinetic_density(k,psi::Psi{3})
    @unpack ψ,X,K = psi;  
    ψx,ψy,ψz = gradient(psi)
	cx = auto_correlate(ψx,X,K)
    cy = auto_correlate(ψy,X,K)
    cz = auto_correlate(ψz,X,K)
    C = @. 0.5(cx + cy + cz)
    return sinc_reduce(k,X...,C)
end

"""
	knumber_density(k,ψ,X,K)

Calculates the angle integrated momentum density ``|\\phi(k)|^2``, at the
points `k`, with the usual radial weight in `k` space ensuring normalization under ∫dk. Units will be population per wavenumber. Arrays `X`, `K` should be computed using `xk_arrays`.
"""
function knumber_density(k,psi::Psi{2})  
    @unpack ψ,X,K = psi; 
	C = auto_correlate(ψ,X,K)
    return bessel_reduce(k,X...,C)
end

function knumber_density(k,psi::Psi{3})  
    @unpack ψ,X,K = psi; 
	C = auto_correlate(ψ,X,K)
    return sinc_reduce(k,X...,C)
end

"""
	wave_action(k,ψ,X,K)

Calculates the angle integrated wave-action spectrum ``|\\phi(\\mathbf{k})|^2``, at the
points `k`, without the radial weight in `k` space ensuring normalization under ∫dk. Units will be population per wavenumber cubed. Isotropy is not assumed. Arrays `X`, `K` should be computed using `xk_arrays`.
"""
wave_action(k,psi::Psi{2}) = _safe_radial_divide(knumber_density(k,psi::Psi{2}), k, 1)
wave_action(k,psi::Psi{3}) = _safe_radial_divide(knumber_density(k,psi::Psi{3}), k, 2)

"""
	incompressible_spectrum(k,ψ)

Caculate the incompressible velocity correlation spectrum for wavefunction ``\\psi``, via Helmholtz decomposition.
Input arrays `X`, `K` must be computed using `xk_arrays`.
"""
function incompressible_spectrum(k,psi::Psi{2},Ω=0.0)
    @unpack ψ,X,K = psi;  
    vx,vy = velocity(psi,Ω)
    a = abs.(ψ)
    wx = @. a*vx; wy = @. a*vy
    Wi, _ = helmholtz(wx,wy,K...)
    wx,wy = Wi

	cx = auto_correlate(wx,X,K)
	cy = auto_correlate(wy,X,K)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function incompressible_spectrum(k,psi::Psi{3})
    @unpack ψ,X,K = psi; 
    vx,vy,vz = velocity(psi)
    a = abs.(ψ)
    wx = @. a*vx; wy = @. a*vy; wz = @. a*vz
    Wi, _ = helmholtz(wx,wy,wz,K...)
    wx,wy,wz = Wi

	cx = auto_correlate(wx,X,K)
    cy = auto_correlate(wy,X,K)
    cz = auto_correlate(wz,X,K)
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce(k,X...,C)
end

"""
	compressible_spectrum(k,ψ,X,K)

Caculate the compressible kinetic enery spectrum for wavefunction ``\\psi``, via Helmholtz decomposition.
Input arrays `X`, `K` must be computed using `xk_arrays`.
"""
function compressible_spectrum(k,psi::Psi{2})
    @unpack ψ,X,K = psi 
    vx,vy = velocity(psi)
    a = abs.(ψ)
    wx = @. a*vx; wy = @. a*vy
    _, Wc = helmholtz(wx,wy,K...)
    wx,wy = Wc

	cx = auto_correlate(wx,X,K)
	cy = auto_correlate(wy,X,K)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function compressible_spectrum(k,psi::Psi{3})
    @unpack ψ,X,K = psi
    vx,vy,vz = velocity(psi)
    a = abs.(ψ)
    wx = @. a*vx; wy = @. a*vy; wz = @. a*vz
    _, Wc = helmholtz(wx,wy,wz,K...)
    wx,wy,wz = Wc

	cx = auto_correlate(wx,X,K)
    cy = auto_correlate(wy,X,K)
    cz = auto_correlate(wz,X,K)
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce(k,X...,C)
end

"""
	qpressure_spectrum(k,psi::Psi{D})

Caculate the quantum pressure correlation spectrum for wavefunction ``\\psi``.
Input arrays `X`, `K` must be computed using `xk_arrays`.
"""
function qpressure_spectrum(k,psi::Psi{2})
    @unpack ψ,X,K = psi
    psia = Psi(abs.(ψ) |> complex,X,K)
    wx,wy = gradient(psia)

	cx = auto_correlate(wx,X,K)
	cy = auto_correlate(wy,X,K)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function qpressure_spectrum(k,psi::Psi{3})
    @unpack ψ,X,K = psi
    psia = Psi(abs.(ψ) |> complex,X,K )
    wx,wy,wz = gradient(psia)

	cx = auto_correlate(wx,X,K)
    cy = auto_correlate(wy,X,K)
    cz = auto_correlate(wz,X,K)
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce(k,X...,C)
end

"""
    incompressible_density(k,ψ,X,K)

Calculates the kinetic energy density of the incompressible velocity field in the wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `xk_arrays`.
"""
function incompressible_density(k,psi::Psi{2})
    @unpack ψ,X,K = psi 
    vx,vy = velocity(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy 
    Wi, Wc = helmholtz(ux,uy,K...)
    wix,wiy = Wi
    U = @. exp(im*angle(ψ))
    @. wix *= U # restore phase factors
    @. wiy *= U

	cx = auto_correlate(wix,X,K)
	cy = auto_correlate(wiy,X,K)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function incompressible_density(k,psi::Psi{3})
    @unpack ψ,X,K = psi 
    vx,vy,vz = velocity(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy; uz = @. a*vz
    Wi, Wc = helmholtz(ux,uy,uz,K...)
    wix,wiy,wiz = Wi
    U = @. exp(im*angle(ψ))
    @. wix *= U # restore phase factors
    @. wiy *= U
    @. wiz *= U

	cx = auto_correlate(wix,X,K)
    cy = auto_correlate(wiy,X,K)
    cz = auto_correlate(wiz,X,K)
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce(k,X...,C)
end

"""
    compressible_density(k,ψ,X,K)

Calculates the kinetic energy density of the compressible velocity field in the wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `xk_arrays`.
"""
function compressible_density(k,psi::Psi{2})
    @unpack ψ,X,K = psi 
    vx,vy = velocity(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy 
    Wi, Wc = helmholtz(ux,uy,K...)
    wcx,wcy = Wc
    U = @. exp(im*angle(ψ))
    @. wcx *= U # restore phase factors
    @. wcy *= U

	cx = auto_correlate(wcx,X,K)
	cy = auto_correlate(wcy,X,K)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function compressible_density(k,psi::Psi{3})
    @unpack ψ,X,K = psi 
    vx,vy,vz = velocity(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy; uz = @. a*vz
    Wi, Wc = helmholtz(ux,uy,uz,K...)
    wcx,wcy,wcz = Wc
    U = @. exp(im*angle(ψ))
    @. wcx *= U # restore phase factors
    @. wcy *= U
    @. wcz *= U

	cx = auto_correlate(wcx,X,K)
    cy = auto_correlate(wcy,X,K)
    cz = auto_correlate(wcz,X,K)
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce(k,X...,C)
end

"""
    qpressure_density(k,ψ,X,K)

Energy density of the quantum pressure in the wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `xk_arrays`.
"""
function qpressure_density(k,psi::Psi{2})
    @unpack ψ,X,K = psi
    psia = Psi(abs.(ψ) |> complex,X,K )
    rnx,rny = gradient(psia)
    U = @. exp(im*angle(ψ))
    @. rnx *= U # restore phase factors
    @. rny *= U 

	cx = auto_correlate(rnx,X,K)
	cy = auto_correlate(rny,X,K)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function qpressure_density(k,psi::Psi{3})
    @unpack ψ,X,K = psi
    psia = Psi(abs.(ψ) |> complex,X,K )
    rnx,rny,rnz = gradient(psia)
    U = @. exp(im*angle(ψ))
    @. rnx *= U # restore phase factors
    @. rny *= U 
    @. rnz *= U 

	cx = auto_correlate(rnx,X,K)
    cy = auto_correlate(rny,X,K)
    cz = auto_correlate(rnz,X,K)
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce(k,X...,C)
end

## coupling terms

"""
    ic_density(k,ψ,X,K)

Energy density of the incompressible-compressible interaction in the wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `xk_arrays`.
"""
function ic_density(k,psi::Psi{2})
    @unpack ψ,X,K = psi 
    vx,vy = velocity(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy 
    Wi, Wc = helmholtz(ux,uy,K...)
    wix,wiy = Wi; wcx,wcy = Wc
    U = @. exp(im*angle(ψ))
    @. wix *= im*U # restore phase factors and make u -> w fields
    @. wiy *= im*U
    @. wcx *= im*U 
    @. wcy *= im*U

    cicx = convolve(wix,wcx,X,K) 
    ccix = convolve(wcx,wix,X,K)
    cicy = convolve(wiy,wcy,X,K) 
    cciy = convolve(wcy,wiy,X,K)
    C = @. 0.5*(cicx + ccix + cicy + cciy)  
    return bessel_reduce(k,X...,C)
end

function ic_density(k,psi::Psi{3})
    @unpack ψ,X,K = psi 
    vx,vy,vz = velocity(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy; uz = @. a*vz 
    Wi, Wc = helmholtz(ux,uy,uz,K...)
    wix,wiy,wiz = Wi; wcx,wcy,wcz = Wc
    U = @. exp(im*angle(ψ))
    @. wix *= im*U # restore phase factors and make u -> w fields
    @. wiy *= im*U
    @. wiz *= im*U   
    @. wcx *= im*U 
    @. wcy *= im*U
    @. wcz *= im*U

    cicx = convolve(wix,wcx,X,K) 
    ccix = convolve(wcx,wix,X,K)
    cicy = convolve(wiy,wcy,X,K) 
    cciy = convolve(wcy,wiy,X,K)
    cicz = convolve(wiz,wcz,X,K) 
    cciz = convolve(wcz,wiz,X,K)
    C = @. 0.5*(cicx + ccix + cicy + cciy + cicz + cciz)  
    return sinc_reduce(k,X...,C)
end

"""
    iq_density(k,ψ,X,K)

Energy density of the incompressible-quantum pressure interaction in the wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `xk_arrays`.
"""
function iq_density(k,psi::Psi{2})
    @unpack ψ,X,K = psi 
    vx,vy = velocity(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy 
    Wi, Wc = helmholtz(ux,uy,K...)
    wix,wiy = Wi 

    psia = Psi(abs.(ψ) |> complex,X,K )
    wqx,wqy = gradient(psia)

    U = @. exp(im*angle(ψ))
    @. wix *= im*U # phase factors and make u -> w fields
    @. wiy *= im*U
    @. wqx *= U
    @. wqy *= U

    ciqx = convolve(wix,wqx,X,K) 
    cqix = convolve(wqx,wix,X,K) 
    ciqy = convolve(wiy,wqy,X,K) 
    cqiy = convolve(wqy,wiy,X,K) 
    C = @. 0.5*(ciqx + cqix + ciqy + cqiy) 
    return bessel_reduce(k,X...,C)
end

function iq_density(k,psi::Psi{3})
    @unpack ψ,X,K = psi 
    vx,vy,vz = velocity(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy; uz = @. a*vz
    Wi, Wc = helmholtz(ux,uy,uz,K...)
    wix,wiy,wiz = Wi

    psia = Psi(abs.(ψ) |> complex,X,K )
    wqx,wqy,wqz = gradient(psia)

    U = @. exp(im*angle(ψ))
    @. wix *= im*U # phase factors and make u -> w fields
    @. wiy *= im*U
    @. wiz *= im*U
    @. wqx *= U
    @. wqy *= U
    @. wqz *= U

    ciqx = convolve(wix,wqx,X,K) 
    cqix = convolve(wqx,wix,X,K) 
    ciqy = convolve(wiy,wqy,X,K) 
    cqiy = convolve(wqy,wiy,X,K) 
    ciqz = convolve(wiz,wqz,X,K) 
    cqiz = convolve(wqz,wiz,X,K) 
    C = @. 0.5*(ciqx + cqix + ciqy + cqiy + ciqz + cqiz) 
    return sinc_reduce(k,X...,C)
end


"""
    cq_density(k,ψ,X,K)

Energy density of the compressible-quantum pressure interaction in the wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `xk_arrays`.
"""
function cq_density(k,psi::Psi{2})
    @unpack ψ,X,K = psi 
    vx,vy = velocity(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy 
    Wi, Wc = helmholtz(ux,uy,K...)
    wcx,wcy = Wc 

    psia = Psi(abs.(ψ) |> complex,X,K)
    wqx,wqy = gradient(psia)

    U = @. exp(im*angle(ψ))
    @. wcx *= im*U # phase factors and make u -> w fields
    @. wcy *= im*U
    @. wqx *= U
    @. wqy *= U

    ccqx = convolve(wcx,wqx,X,K) 
    cqcx = convolve(wqx,wcx,X,K) 
    ccqy = convolve(wcy,wqy,X,K) 
    cqcy = convolve(wqy,wcy,X,K) 
    C = @. 0.5*(ccqx + cqcx + ccqy + cqcy) 
    return bessel_reduce(k,X...,C)
end

function cq_density(k,psi::Psi{3})
    @unpack ψ,X,K = psi 
    vx,vy,vz = velocity(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy; uz = @. a*vz
    Wi, Wc = helmholtz(ux,uy,uz,K...)
    wcx,wcy,wcz = Wc  

    psia = Psi(abs.(ψ) |> complex,X,K)
    wqx,wqy,wqz = gradient(psia)

    U = @. exp(im*angle(ψ))
    @. wcx *= im*U # phase factors and make u -> w fields
    @. wcy *= im*U
    @. wcz *= im*U
    @. wqx *= U
    @. wqy *= U
    @. wqz *= U

    ccqx = convolve(wcx,wqx,X,K) 
    cqcx = convolve(wqx,wcx,X,K) 
    ccqy = convolve(wcy,wqy,X,K) 
    cqcy = convolve(wqy,wcy,X,K) 
    ccqz = convolve(wcz,wqz,X,K) 
    cqcz = convolve(wqz,wcz,X,K) 
    C = @. 0.5*(ccqx + cqcx + ccqy + cqcy + ccqz + cqcz) 
    return sinc_reduce(k,X...,C)
end

"""
    gv(r,k,ε)

Transform the power spectrum `ε(k)` defined at `k` to position space to give a system averaged velocity two-point correlation function on the spatial points `r`. The vector `r` can be chosen arbitrarily, provided `r ≥ 0`. 
"""
function gv(r,k,ε)
    dk = diff(k)
    push!(dk,last(dk))  # vanishing spectra at high k
    E = sum(@. ε*dk)
    gv = zero(r)
    @tullio gv[i] = ε[j]*besselj0(k[j]*r[i])*dk[j] avx=false
    return gv/E
end

"""
    gv3(r,k,ε)

Transform the power spectrum `ε(k)` defined at `k` to position space to give a system averaged velocity two-point correlation function on the spatial points `r`. The vector `r` can be chosen arbitrarily, provided `r ≥ 0`. 
"""
function gv3(r,k,ε)
    dk = diff(k)
    push!(dk,last(dk))  # vanishing spectra at high k
    E = sum(@. ε*dk)
    gv = zero(r)
    @tullio gv[i] = ε[j]*sinc(k[j]*r[i]/pi)*dk[j] avx=false
    return gv/E
end

function trap_spectrum(k,V,psi::Psi{2})
    @unpack ψ,X,K = psi; x,y = X
    f = @. abs(ψ)*sqrt(V(x,y',0.))
    C = auto_correlate(f,X,K)

    return bessel_reduce(k,X...,C)
end

function trap_spectrum(k,V,psi::Psi{3})
    @unpack ψ,X,K = psi; x,y,z = X
    f = @. abs(ψ)*sqrt(V(x,y',reshape(z,1,1,length(z)),0.))
    C = auto_correlate(f,X,K)

    return sinc_reduce(k,X...,C)
end

function density_spectrum(k,psi::Psi{2}) 
    @unpack ψ,X,K = psi 
    n = abs2.(ψ)
    C = auto_correlate(n,X,K) 

    return bessel_reduce(k,X...,C)
end

function density_spectrum(k,psi::Psi{3}) 
    @unpack ψ,X,K = psi 
    n = abs2.(ψ)
    C = auto_correlate(n,X,K) 

    return sinc_reduce(k,X...,C)
end

function _k2_grid(K::NTuple{2})
    kx,ky = K
    return @. kx^2 + ky'^2
end

function _k2_grid(K::NTuple{3})
    kx,ky,kz = K
    kzr = reshape(kz,1,1,length(kz))
    return @. kx^2 + ky'^2 + kzr^2
end

function _laplacian(psi::Psi{2})
    @unpack ψ,K = psi
    ϕ = fft(ψ)
    return ifft(-_k2_grid(K) .* ϕ)
end

function _laplacian(psi::Psi{3})
    @unpack ψ,K = psi
    ϕ = fft(ψ)
    return ifft(-_k2_grid(K) .* ϕ)
end

function _trap_field(psi::Psi{2},V,t)
    @unpack ψ,X = psi; x,y = X
    return @. V(x,y',t) * ψ
end

function _trap_field(psi::Psi{3},V,t)
    @unpack ψ,X = psi; x,y,z = X
    zr = reshape(z,1,1,length(z))
    return @. V(x,y',zr,t) * ψ
end

function _gpe_rhs(psi::Psi{2}; g=1.0, V=nothing, t=0.0)
    @unpack ψ = psi
    lap = _laplacian(psi)
    rhs = @. -im * (-0.5 * lap + g * abs2(ψ) * ψ)
    if !isnothing(V)
        trap = _trap_field(psi,V,t)
        rhs .+= (-im) .* trap
    end
    return rhs
end

function _gpe_rhs(psi::Psi{3}; g=1.0, V=nothing, t=0.0)
    @unpack ψ = psi
    lap = _laplacian(psi)
    rhs = @. -im * (-0.5 * lap + g * abs2(ψ) * ψ)
    if !isnothing(V)
        trap = _trap_field(psi,V,t)
        rhs .+= (-im) .* trap
    end
    return rhs
end

function _gpe_reduce(k,X,C)
    if length(X) == 2
        return bessel_reduce(k,X...,C)
    else
        return sinc_reduce(k,X...,C)
    end
end

_integrated_gpe_reduce(k, X::Tuple{<:AbstractVector,<:AbstractVector}, C) =
    _integrated_bessel_reduce(k, X[1], X[2], C)

_integrated_gpe_reduce(k, X::Tuple{<:AbstractVector,<:AbstractVector,<:AbstractVector}, C) =
    _integrated_sinc_reduce(k, X[1], X[2], X[3], C)

_shell_area(::Psi{2}) = 2π
_shell_area(::Psi{3}) = 4π

_integrated_gpe_prefactor(::Psi{2}) = 1 / π
_integrated_gpe_prefactor(::Psi{3}) = 1 / π^2

_gradient_fields(psi::Psi) = gradient(psi)

function _cumulative_integral(k,f)
    T = promote_type(eltype(k), eltype(f), Float64)
    I = zeros(T, length(k))
    for i in 2:length(k)
        dk = k[i] - k[i-1]
        I[i] = I[i-1] + 0.5 * dk * (f[i] + f[i-1])
    end
    return I
end

function _gpe_kinetic_correlation(psi::Psi{2},rhs)
    @unpack X,K = psi
    ψx,ψy = gradient(psi)
    rhspsi = Psi(rhs,X,K)
    rhsx,rhsy = gradient(rhspsi)

    cx = cross_correlate(ψx,rhsx,X,K)
    cy = cross_correlate(ψy,rhsy,X,K)
    return @. 0.5 * (cx + cy)
end

function _gpe_kinetic_correlation(psi::Psi{3},rhs)
    @unpack X,K = psi
    ψx,ψy,ψz = gradient(psi)
    rhspsi = Psi(rhs,X,K)
    rhsx,rhsy,rhsz = gradient(rhspsi)

    cx = cross_correlate(ψx,rhsx,X,K)
    cy = cross_correlate(ψy,rhsy,X,K)
    cz = cross_correlate(ψz,rhsz,X,K)
    return @. 0.5 * (cx + cy + cz)
end

function _gpe_interaction_correlation(psi::Psi,rhs; g=1.0)
    @unpack ψ,X,K = psi
    χ = @. g * abs2(ψ) * ψ
    return cross_correlate(χ,rhs,X,K)
end

function _gpe_trap_correlation(psi::Psi; V=nothing, t=0.0, rhs)
    isnothing(V) && return nothing
    @unpack X,K = psi
    η = _trap_field(psi,V,t)
    return cross_correlate(η,rhs,X,K)
end

function _gpe_kinetic_transfer(k,psi::Psi,rhs)
    C = _gpe_kinetic_correlation(psi,rhs)
    return 2 .* real.(_gpe_reduce(k,psi.X,C))
end

function _gpe_interaction_transfer(k,psi::Psi,rhs; g=1.0)
    C = _gpe_interaction_correlation(psi,rhs; g=g)
    return 2 .* real.(_gpe_reduce(k,psi.X,C))
end

function _gpe_trap_transfer(k,psi::Psi,rhs; V=nothing, t=0.0)
    C = _gpe_trap_correlation(psi; V=V, t=t, rhs=rhs)
    isnothing(C) && return zeros(promote_type(eltype(k), Float64), length(k))
    return 2 .* real.(_gpe_reduce(k,psi.X,C))
end

function _gpe_kinetic_flux(k, psi::Psi, rhs)
    C = _gpe_kinetic_correlation(psi, rhs)
    return .-_integrated_gpe_prefactor(psi) .* real.(_integrated_gpe_reduce(k, psi.X, C))
end

function _gpe_interaction_flux(k, psi::Psi, rhs; g=1.0)
    C = _gpe_interaction_correlation(psi, rhs; g=g)
    return .-_integrated_gpe_prefactor(psi) .* real.(_integrated_gpe_reduce(k, psi.X, C))
end

function _gpe_trap_flux(k, psi::Psi, rhs; V=nothing, t=0.0)
    C = _gpe_trap_correlation(psi; V=V, t=t, rhs=rhs)
    isnothing(C) && return zeros(promote_type(eltype(k), Float64), length(k))
    return .-_integrated_gpe_prefactor(psi) .* real.(_integrated_gpe_reduce(k, psi.X, C))
end

"""
    gpe_particle_transfer(k, psi::Psi; g=1.0, V=nothing, t=0.0, components=false)

Return the angle-integrated GPE particle-transfer spectrum on radial wavenumbers `k`.

If `components=true`, also return the kinetic, interaction, and trap contributions.
"""
function gpe_particle_transfer(k,psi::Psi; g=1.0, V=nothing, t=0.0, components=false)
    rhs = _gpe_rhs(psi; g=g, V=V, t=t)
    Tkin = _gpe_kinetic_transfer(k,psi,rhs)
    Tint = _gpe_interaction_transfer(k,psi,rhs; g=g)
    Ttrap = _gpe_trap_transfer(k,psi,rhs; V=V, t=t)
    T = Tkin .+ Tint .+ Ttrap
    return components ? (T,Tkin,Tint,Ttrap) : T
end

"""
    gpe_particle_flux(k, psi::Psi; g=1.0, V=nothing, t=0.0, components=false)

Return the cumulative GPE particle flux `Π(k) = -∫₀ᵏ T(q)dq`.
"""
function gpe_particle_flux(k, psi::Psi; g=1.0, V=nothing, t=0.0, components=false)
    rhs = _gpe_rhs(psi; g=g, V=V, t=t)
    Πkin = _gpe_kinetic_flux(k, psi, rhs)
    Πint = _gpe_interaction_flux(k, psi, rhs; g=g)
    Πtrap = _gpe_trap_flux(k, psi, rhs; V=V, t=t)
    Π = Πkin .+ Πint .+ Πtrap
    return components ? (Π, Πkin, Πint, Πtrap) : Π
end

"""
    gpe_energy_transfer(k, psi::Psi; g=1.0, V=nothing, t=0.0, components=false)

Compatibility alias for [`gpe_particle_transfer`](@ref).
"""
gpe_energy_transfer(k, psi::Psi; g=1.0, V=nothing, t=0.0, components=false) =
    gpe_particle_transfer(k, psi; g=g, V=V, t=t, components=components)

"""
    gpe_energy_flux(k, psi::Psi; g=1.0, V=nothing, t=0.0, components=false)

Compatibility alias for [`gpe_particle_flux`](@ref).
"""
gpe_energy_flux(k, psi::Psi; g=1.0, V=nothing, t=0.0, components=false) =
    gpe_particle_flux(k, psi; g=g, V=V, t=t, components=components)
