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
function velocity(psi::Psi{1})
	@unpack ψ = psi
    ψx = gradient(psi)
	vx = @. imag(conj(ψ)*ψx)/abs2(ψ)
    @. vx[isnan(vx)] = zero(vx[1])
	return vx
end

function velocity(psi::Psi{2},Ω = 0)
	@unpack ψ,X = psi
    x,y = X
    ψx,ψy = gradient(psi)
    rho = abs2.(ψ)
	vx = @. imag(conj(ψ)*ψx)/rho + Ω*y'  
	vy = @. imag(conj(ψ)*ψy)/rho - Ω*x 
    @. vx[isnan(vx)] = zero(vx[1])
    @. vy[isnan(vy)] = zero(vy[1])
	return vx,vy
end

function velocity(psi::Psi{3})
	@unpack ψ = psi
	rho = abs2.(ψ)
    ψx,ψy,ψz = gradient(psi)
	vx = @. imag(conj(ψ)*ψx)/rho
	vy = @. imag(conj(ψ)*ψy)/rho
	vz = @. imag(conj(ψ)*ψz)/rho
    @. vx[isnan(vx)] = zero(vx[1])
    @. vy[isnan(vy)] = zero(vy[1])
    @. vz[isnan(vz)] = zero(vz[1])
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

function helmholtz_incompressible(wx, wy, wz, kx, ky, kz)
    wxk = fft(wx); wyk = fft(wy); wzk = fft(wz)
    kzr = reshape(kz,1,1,length(kz))
    kw = @. (kx * wxk + ky' * wyk + kzr * wzk)/ (kx^2 + ky'^2 + kzr^2)
    wxkc = @. kw * kx  
    wykc = @. kw * ky'
    wzkc = @. kw * kzr  
    wxkc[1] = zero(wxkc[1]); wykc[1] = zero(wykc[1]); wzkc[1] = zero(wzkc[1])
    @. wxk -= wxkc
    @. wyk -= wykc
    @. wzk -= wzkc
    ifft!(wxk)
    ifft!(wyk)
    ifft!(wzk)
    return wxk, wyk, wzk
end

function helmholtz_compressible(wx, wy, wz, kx, ky, kz)
    wxk = fft(wx); wyk = fft(wy); wzk = fft(wz)
    kzr = reshape(kz,1,1,length(kz))
    kw = @. (kx * wxk + ky' * wyk + kzr * wzk)/ (kx^2 + ky'^2 + kzr^2)
    wxkc = @. kw * kx  
    wykc = @. kw * ky'
    wzkc = @. kw * kzr  
    wxkc[1] = zero(wxkc[1]); wykc[1] = zero(wykc[1]); wzkc[1] = zero(wzkc[1])
    wxc = ifft(wxkc); wyc = ifft(wykc); wzc = ifft(wzkc)

    return wxc,wyc,wzc
end

"""
	ei,ec = energydecomp(psi::Xfield{D})

Decomposes the hydrodynamic kinetic energy of `psi`, returning the total incompressible `ei`,
and compressible `ec` energy densities in position space. `D` can be 2 or 3 dimensions.
"""
function energydecomp(psi::Psi{2})
    @unpack ψ,K,X = psi; kx,ky = K
    dx = X[1][2] - X[1][1]
    dy = X[2][2] - X[2][1]
    a = abs.(ψ)
    vx, vy = velocity(psi)
    wx = @. a*vx; wy = @. a*vy
    Wi, Wc = helmholtz(wx,wy,kx,ky)
    wxi, wyi = Wi; wxc, wyc = Wc
    ei = 0.5*sum(@. abs2(wxi) + abs2(wyi))*dx*dy
    ec = 0.5*sum(@. abs2(wxc) + abs2(wyc))*dx*dy
    return ei, ec
end

function energydecomp(psi::Psi{3})
	@unpack ψ,K,X = psi; kx,ky,kz = K
    dx = X[1][2] - X[1][1]
    dy = X[2][2] - X[2][1]
    dz = X[3][2] - X[3][1]
    a = abs.(ψ)
    vx,vy,vz = velocity(psi)
    wx = @. a*vx; wy = @. a*vy; wz = @. a*vz
    Wi, Wc = helmholtz(wx,wy,wz,kx,ky,kz)
    vx = nothing; vy = nothing; vz = nothing; GC.gc()
    wxi, wyi, wzi = Wi; wxc, wyc, wzc = Wc
    Wi = nothing; Wc = nothing; GC.gc()
    ei = 0.5*sum(@. abs2(wxi) + abs2(wyi) + abs2(wzi))*dx*dy*dz
    ec = 0.5*sum(@. abs2(wxc) + abs2(wyc) + abs2(wzc))*dx*dy*dz
    return ei, ec
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

    outer = []
    inner = []

    for no in nO
        push!(outer,(1:no))
    end

    for ni in nI
        push!(inner,(ni+1:ni+2*ni))
    end

    return PaddedView(zero(eltype(A)),A,Tuple(outer),Tuple(inner)) |> collect
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

    fft!(ϕ1); fft!(ϕ2)
    ϕ1 .*= prod(DX); ϕ2 .*= prod(DX)
    @. ϕ1 *= ϕ2 
    ϕ2 = nothing; GC.gc()
    ifft!(ϕ1)
    ϕ1 .*= prod(DK)*(2*pi)^(n/2)

	return ϕ1
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
	fft!(ϕ)
    dμx = prod(DX)
    @. ϕ *= dμx

    Threads.@threads for i in eachindex(ϕ)
        ϕ[i] = abs2(ϕ[i])
    end

    ifft!(ϕ)
    dμk = prod(DK)*(2*π)^(n/2) 
    @. ϕ *= dμk
	return ϕ |> fftshift
end

auto_correlate(psi::Psi{D}) where D = auto_correlate(psi.ψ,psi.X,psi.K)

@doc raw"""
	cross_correlate(ψ1,ψ2,X,K)

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
    ϕ1 = zeropad(conj.(ψ1)); fft!(ϕ1); dμx = prod(DX); @. ϕ1 *= dμx
    ϕ2 = zeropad(ψ2); fft!(ϕ2); @. ϕ2 *= dμx
    @. ϕ1 *= ϕ2
    ifft!(ϕ1)
    dμk = prod(DK)*(2*π)^(n/2)
    ϕ1 .*= dμk
	return ϕ1 |> fftshift
end


cross_correlate(psi1::Psi{D},psi2::Psi{D}) where D = cross_correlate(psi1.ψ,psi2.ψ,psi1.X,psi1.K)

function bessel_reduce(k,x,y,C)
    dx,dy = x[2]-x[1],y[2]-y[1]
    Nx,Ny = 2*length(x),2*length(y)
    Lx = x[end] - x[begin] 
    Ly = y[end] - y[begin] 
    xp = LinRange(-Lx,Lx,Nx+1)[1:Nx] |> fftshift
    yq = LinRange(-Ly,Ly,Ny+1)[1:Ny] |> fftshift
    E = zero(k)
    @tullio E[i] = real(besselj0(k[i]*hypot(xp[p],yq[q]))*C[p,q])
    @. E *= k*dx*dy/2/pi 
    return E 
end

function sinc_reduce(k,x,y,z,C)
    dx,dy,dz = x[2]-x[1],y[2]-y[1],z[2]-z[1]
    Nx,Ny,Nz = 2*length(x),2*length(y),2*length(z)
    Lx = x[end] - x[begin] 
    Ly = y[end] - y[begin] 
    Lz = z[end] - z[begin] 
    xp = LinRange(-Lx,Lx,Nx+1)[1:Nx] |> fftshift
    yq = LinRange(-Ly,Ly,Ny+1)[1:Ny] |> fftshift
    zr = LinRange(-Lz,Lz,Nz+1)[1:Nz] |> fftshift
    E = zero(k)
    @tullio E[i] = real(π*sinc(k[i]*hypot(xp[p],yq[q],zr[r])/π)*C[p,q,r]) 
    @. E *= k^2*dx*dy*dz/2/pi^2  
    return E 
end

"""
	kinetic_density(k,ψ,X,K)

Calculates the kinetic enery spectrum for wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `makearrays`.
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
	C = auto_correlate(ψx,X,K)
    ψx = nothing; GC.gc()
    @. C += $auto_correlate(ψy,X,K)
    ψy = nothing; GC.gc()
    @. C += $auto_correlate(ψz,X,K)
    ψz = nothing; GC.gc()
    @. C *= 0.5 
    return sinc_reduce(k,X...,C)
end

"""
	kdensity(k,ψ,X,K)

Calculates the angle integrated momentum density ``|\\phi(k)|^2``, at the
points `k`, with the usual radial weight in `k` space ensuring normalization under ∫dk. Units will be population per wavenumber. Arrays `X`, `K` should be computed using `makearrays`.
"""
function kdensity(k,psi::Psi{2})  
    @unpack ψ,X,K = psi; 
	C = auto_correlate(ψ,X,K)
    return bessel_reduce(k,X...,C)
end

function kdensity(k,psi::Psi{3})  
    @unpack ψ,X,K = psi; 
	C = auto_correlate(ψ,X,K)
    return sinc_reduce(k,X...,C)
end

"""
	wave_action(k,ψ,X,K)

Calculates the angle integrated wave-action spectrum ``|\\phi(\\mathbf{k})|^2``, at the
points `k`, without the radial weight in `k` space ensuring normalization under ∫dk. Units will be population per wavenumber cubed. Isotropy is not assumed. Arrays `X`, `K` should be computed using `makearrays`.
"""
wave_action(k,psi::Psi{2}) = kdensity(k,psi::Psi{2}) ./k 
wave_action(k,psi::Psi{3}) = kdensity(k,psi::Psi{3})./k^2

"""
	incompressible_spectrum(k,ψ)

Caculate the incompressible velocity correlation spectrum for wavefunction ``\\psi``, via Helmholtz decomposition.
Input arrays `X`, `K` must be computed using `makearrays`.
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
    wx, wy, wz = velocity(psi)
    @. wx *= abs(ψ)
    @. wy *= abs(ψ)
    @. wz *= abs(ψ)

    wx,wy,wz = helmholtz_incompressible(wx,wy,wz,K...)

    C = auto_correlate(wx,X,K)
    wx = nothing; GC.gc()
    @. C += $auto_correlate(wy,X,K)
    wy = nothing; GC.gc()
    @. C += $auto_correlate(wz,X,K)
    wz = nothing; GC.gc()
    @. C *= 0.5
    return sinc_reduce(k,X...,C)
end

"""
	compressible_spectrum(k,ψ,X,K)

Caculate the compressible kinetic enery spectrum for wavefunction ``\\psi``, via Helmholtz decomposition.
Input arrays `X`, `K` must be computed using `makearrays`.
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
    wx, wy, wz = velocity(psi)
    @. wx *= abs(ψ)
    @. wy *= abs(ψ)
    @. wz *= abs(ψ)

    wx,wy,wz = helmholtz_compressible(wx,wy,wz,K...)

    C = auto_correlate(wx,X,K)
    wx = nothing; GC.gc()
    @. C += $auto_correlate(wy,X,K)
    wy = nothing; GC.gc()
    @. C += $auto_correlate(wz,X,K)
    wz = nothing; GC.gc()
    @. C *= 0.5
    return sinc_reduce(k,X...,C)
end

"""
	qpressure_spectrum(k,psi::Psi{D})

Caculate the quantum pressure correlation spectrum for wavefunction ``\\psi``.
Input arrays `X`, `K` must be computed using `makearrays`.
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
    wx,wy,wz = gradient(Psi(abs.(ψ) |> complex,X,K ))

    C = auto_correlate(wx,X,K)
    wx = nothing; GC.gc()
    @. C += $auto_correlate(wy,X,K)
    wy = nothing; GC.gc()
    @. C += $auto_correlate(wz,X,K)
    wz = nothing; GC.gc()
    @. C *= 0.5
    return sinc_reduce(k,X...,C)
end

"""
    incompressible_density(k,ψ,X,K)

Calculates the kinetic energy density of the incompressible velocity field in the wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `makearrays`.
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
    wx,wy,wz = velocity(psi)

    @. wx *= abs(ψ)
    @. wy *= abs(ψ)
    @. wz *= abs(ψ)

    wx,wy,wz = helmholtz_incompressible(wx,wy,wz,K...)

    @. wx *= exp(im*angle(ψ))  
    @. wy *= exp(im*angle(ψ))
    @. wz *= exp(im*angle(ψ))

	C = auto_correlate(wx,X,K)
    wx = nothing; GC.gc()
    @. C += $auto_correlate(wy,X,K)
    wy = nothing; GC.gc()
    @. C += $auto_correlate(wz,X,K)
    wz = nothing; GC.gc()
    @. C *= 0.5
    return sinc_reduce(k,X...,C)
end

"""
    compressible_density(k,ψ,X,K)

Calculates the kinetic energy density of the compressible velocity field in the wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `makearrays`.
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
    wx,wy,wz = velocity(psi)

    @. wx *= abs(ψ)
    @. wy *= abs(ψ)
    @. wz *= abs(ψ)

    wx,wy,wz = helmholtz_compressible(wx,wy,wz,K...)

    @. wx *= exp(im*angle(ψ))
    @. wy *= exp(im*angle(ψ))
    @. wz *= exp(im*angle(ψ))

    C = auto_correlate(wx,X,K)
    wx = nothing; GC.gc()
    @. C += $auto_correlate(wy,X,K)
    wy = nothing; GC.gc()
    @. C += $auto_correlate(wz,X,K)
    wz = nothing; GC.gc()
    @. C *= 0.5
    return sinc_reduce(k,X...,C)
end

"""
    qpressure_density(k,ψ,X,K)

Energy density of the quantum pressure in the wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `makearrays`.
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
    wx,wy,wz = gradient(Psi(abs.(ψ) |> complex,X,K ))

    @. wx *= exp(im*angle(ψ))
    @. wy *= exp(im*angle(ψ))
    @. wz *= exp(im*angle(ψ))

	C = auto_correlate(wx,X,K)
    wx = nothing; GC.gc()
    @. C += $auto_correlate(wy,X,K)
    wy = nothing; GC.gc()
    @. C += $auto_correlate(wz,X,K)
    wz = nothing; GC.gc()
    @. C = 0.5
    return sinc_reduce(k,X...,C)
end

## coupling terms

"""
    ic_density(k,ψ,X,K)

Energy density of the incompressible-compressible interaction in the wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `makearrays`.
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
    wx,wy,wz = velocity(psi)

    @. wx *= abs(ψ)
    @. wy *= abs(ψ)
    @. wz *= abs(ψ)

    Wi, Wc = helmholtz(wx,wy,wz,K...)
    wix,wiy,wiz = Wi; wcx,wcy,wcz = Wc

    wx = nothing; wy = nothing; wz = nothing; GC.gc()

    @. wix *= im*exp(im*angle(ψ))  
    @. wiy *= im*exp(im*angle(ψ))
    @. wiz *= im*exp(im*angle(ψ))  
    @. wcx *= im*exp(im*angle(ψ)) 
    @. wcy *= im*exp(im*angle(ψ))
    @. wcz *= im*exp(im*angle(ψ))

    C = convolve(wix,wcx,X,K); GC.gc()
    C .+= convolve(wcx,wix,X,K); wix = nothing; wcx = nothing; GC.gc() 
    C .+= convolve(wiy,wcy,X,K); GC.gc()
    C .+= convolve(wcy,wiy,X,K); wcy = nothing; wiy = nothing; GC.gc()
    C .+= convolve(wiz,wcz,X,K); GC.gc() 
    C .+= convolve(wcz,wiz,X,K); wcz = nothing; wiz = nothing; GC.gc()
    @. C *= 0.5  
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

    psia = Psi(a |> complex,X,K )
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
    wx,wy,wz = velocity(psi)

    @. wx *= abs(ψ)
    @. wy *= abs(ψ)
    @. wz *= abs(ψ)

    wix,wiy,wiz = helmholtz_incompressible(wx,wy,wz,K...)
    wx,wy,wz = gradient(Psi(abs.(ψ) |> complex,X,K ))

    U = @. exp(im*angle(ψ))
    @. wix *= im*U # phase factors and make u -> w fields
    @. wiy *= im*U
    @. wiz *= im*U
    @. wx *= U
    @. wy *= U
    @. wz *= U

    C = convolve(wix,wx,X,K); GC.gc()
    C .+= convolve(wx,wix,X,K); wix = nothing; wx = nothing; GC.gc() 
    C .+= convolve(wiy,wy,X,K); GC.gc()
    C .+= convolve(wy,wiy,X,K); wy = nothing; wiy = nothing; GC.gc()
    C .+= convolve(wiz,wz,X,K); GC.gc() 
    C .+= convolve(wz,wiz,X,K); wz = nothing; wiz = nothing; GC.gc()
    @. C *= 0.5  
    return sinc_reduce(k,X...,C)
end


"""
    cq_density(k,ψ,X,K)

Energy density of the compressible-quantum pressure interaction in the wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `makearrays`.
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
    wx,wy,wz = velocity(psi)

    @. wx *= abs(ψ)
    @. wy *= abs(ψ)
    @. wz *= abs(ψ)

    wcx,wcy,wcz = helmholtz_compressible(wx,wy,wz,K...) 

    wx,wy,wz = gradient(Psi(abs.(ψ) |> complex,X,K))

    U = @. exp(im*angle(ψ))
    @. wcx *= im*U # phase factors and make u -> w fields
    @. wcy *= im*U
    @. wcz *= im*U
    @. wx *= U
    @. wy *= U
    @. wz *= U

    C = convolve(wcx,wx,X,K); GC.gc()
    C .+= convolve(wx,wcx,X,K); wcx = nothing; wx = nothing; GC.gc() 
    C .+= convolve(wcy,wy,X,K); GC.gc()
    C .+= convolve(wy,wcy,X,K); wy = nothing; wcy = nothing; GC.gc()
    C .+= convolve(wcz,wz,X,K); GC.gc() 
    C .+= convolve(wz,wcz,X,K); wz = nothing; wcz = nothing; GC.gc()
    @. C *= 0.5  
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