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

function current(psi::Psi{2})
	@unpack ψ = psi 
    ψx,ψy = gradient(psi)
	jx = @. imag(conj(ψ)*ψx)
	jy = @. imag(conj(ψ)*ψy)
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

function velocity(psi::Psi{2})
	@unpack ψ = psi
    ψx,ψy = gradient(psi)
    rho = abs2.(ψ)
	vx = @. imag(conj(ψ)*ψx)/rho
	vy = @. imag(conj(ψ)*ψy)/rho
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
    @cast kw[i,j] := kx[i] * wxk[i,j] + ky[j] * wyk[i,j]
    @cast wxkc[i,j] := kw[i,j] * kx[i] / (kx[i]^2+ky[j]^2)
    @cast wykc[i,j] := kw[i,j] * ky[j] / (kx[i]^2+ky[j]^2)
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
    @cast kw[i,j,k] = kx[i] * wxk[i,j,k] + ky[j] * wyk[i,j,k] + kz[k] * wzk[i,j,k]
    @cast wxkc[i,j,k] := kw[i,j,k] * kx[i] / (kx[i]^2 + ky[j]^2 + kz[k]^2) 
    @cast wykc[i,j,k] := kw[i,j,k] * ky[j] / (kx[i]^2 + ky[j]^2 + kz[k]^2)  
    @cast wzkc[i,j,k] := kw[i,j,k] * kz[k] / (kx[i]^2 + ky[j]^2 + kz[k]^2)  
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
    DX,DK = dfftall(X,K)
	ϕ1 = zeropad(conj.(ψ1))
    ϕ2 = zeropad(ψ2)

	χ1 = fft(ϕ1)*prod(DX)
	χ2 = fft(ϕ2)*prod(DX)
	return ifft(χ1.*χ2)*prod(DK)*(2*pi)^(n/2) |> fftshift
end

@doc raw"""
	auto_correlate(ψ,X,K)

Return the autocorrelation integral of a complex field ``\psi``, ``A``, given by

```
A(\rho)=\int d^2r\;\psi^*(r-\rho)\psi(r)
```

defined on a cartesian grid on a cartesian grid using FFTW.

`X` and `K` are tuples of vectors `x`,`y`,`kx`, `ky`.

This method is useful for evaluating spectra from cartesian data.
"""
function auto_correlate(ψ,X,K)
    n = length(X)
    DX,DK = dfftall(X,K)
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
    DX,DK = dfftall(X,K)
    ϕ1 = zeropad(ψ1)
    ϕ2 = zeropad(ψ2)
	χ1 = fft(ϕ1)*prod(DX)
    χ2 = fft(ϕ2)*prod(DX)
	return ifft(conj(χ1).*χ2)*prod(DK)*(2*pi)^(n/2) |> fftshift
end
cross_correlate(psi::Psi{D}) where D = cross_correlate(psi.ψ,psi.X,psi.K)

function bessel_reduce(k,x,y,C)
    dx,dy = x[2]-x[1],y[2]-y[1]
    Nx,Ny = 2*length(x),2*length(y)
    Lx = x[end] - x[begin] + dx
    xp = LinRange(-Lx,Lx,Nx+1)[1:Nx]
    Ly = y[end] - y[begin] + dy
    yq = LinRange(-Ly,Ly,Ny+1)[1:Ny]
    E = zero(k)
    @tullio E[i] = real(besselj0(k[i]*sqrt(xp[p]^2 + yq[q]^2))*C[p,q])
    @. E *= k*dx*dy/2/pi 
    return E 
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
    @tullio E[i] = real(π*sinc(k[i]*sqrt(xp[p]^2 + yq[q]^2 + zr[r]^2)/π)*C[p,q,r]) 
    @. E *= k^2*dx*dy*dz/2/pi^2  
    return E 
end

"""
	kinetic_density(k,ψ,X,K)

Calculates the kinetic enery spectrum for wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `makearrays`.
"""
function kinetic_density(k,psi::Psi{2})
    @unpack ψ,X,K = psi
    ψx,ψy = gradient(psi)
	cx = auto_correlate(ψx,X,K)
	cy = auto_correlate(ψy,X,K)
    C = @. 0.5(cx + cy)
    return bessel_reduce(k,x,y,C)
end

function kinetic_density(k,psi::Psi{3})
    @unpack ψ,X,K = psi
    ψx,ψy,ψz = gradient(psi)
	cx = auto_correlate(ψx,X,K)
    cy = auto_correlate(ψy,X,K)
    cz = auto_correlate(ψz,X,K)
    C = @. 0.5(cx + cy + cz)
    return sinc_reduce(k,x,y,z,C)
end

"""
	incompressible_spectrum(k,ψ,X,K)

Caculate the incompressible velocity correlation spectrum for wavefunction ``\\psi``, via Helmholtz decomposition.
Input arrays `X`, `K` must be computed using `makearrays`.
"""
function incompressible_spectrum(k,psi::Psi{2})
    @unpack ψ,X,K = psi; kx,ky = K
    vx,vy = velocity(psi)
    a = abs.(ψ)
    wx = @. a*vx; wy = @. a*vy
    Wi, Wc = helmholtz(wx,wy,kx,ky)
    wx,wy = Wi

	cx = auto_correlate(wx,X,K)
	cy = auto_correlate(wy,X,K)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,x,y,C)
end

function incompressible_spectrum(k,psi::Psi{3})
    @unpack ψ,X,K = psi; kx,ky,kz = K
    vx,vy,vz = velocity(psi)
    a = abs.(ψ)
    wx = @. a*vx; wy = @. a*vy; wz = @. a*vz
    Wi, Wc = helmholtz(wx,wy,wz,kx,ky,kz)
    wx,wy,wz = Wi

	cx = auto_correlate(wx,X,K)
    cy = auto_correlate(wy,X,K)
    cz = auto_correlate(wz,X,K)
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce(k,x,y,z,C)
end

"""
	compressible_spectrum(k,ψ,X,K)

Caculate the compressible kinetic enery spectrum for wavefunction ``\\psi``, via Helmholtz decomposition.
Input arrays `X`, `K` must be computed using `makearrays`.
"""
function compressible_spectrum(k,psi::Psi{2})
    @unpack ψ,X,K = psi; kx,ky = K
    vx,vy = velocity(psi)
    a = abs(ψ)
    wx = @. abs(ψ)*vx; wy = @. abs(ψ)*vy
    Wi, Wc = helmholtz(wx,wy,kx,ky)
    wx,wy = Wc

	cx = auto_correlate(wx,X,K)
	cy = auto_correlate(wy,X,K)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,x,y,C)
end

function compressible_spectrum(k,psi::Psi{3})
    @unpack ψ,X,K = psi; kx,ky,kz = K
    vx,vy,vz = velocity(psi)
    a = abs(ψ)
    wx = @. a*vx; wy = @. a*vy; wz = @. a*vz
    Wi, Wc = helmholtz(wx,wy,wz,kx,ky,kz)
    wx,wy,wz = Wc

	cx = auto_correlate(wx,X,K)
    cy = auto_correlate(wy,X,K)
    cz = auto_correlate(wz,X,K)
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce(k,x,y,z,C)
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
    return bessel_reduce(k,x,y,C)
end

function qpressure_spectrum(k,psi::Psi{3})
    @unpack ψ,X,K = psi
    psia = Psi(abs.(ψ) |> complex,X,K )
    wx,wy,wz = gradient(psia)

	cx = auto_correlate(wx,X,K)
    cy = auto_correlate(wy,X,K)
    cz = auto_correlate(wz,X,K)
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce(k,x,y,z,C)
end

"""
    incompressible_density(k,ψ,X,K)

Calculates the kinetic energy density of the incompressible velocity field in the wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `makearrays`.
"""
function incompressible_density(k,psi::Psi{2})
    @unpack ψ,X,K = psi; kx,ky = K
    vx,vy = velocity(psi)
    a = abs(ψ)
    ux = @. a*vx; uy = @. a*vy 
    Wi, Wc = helmholtz(ux,uy,kx,ky)
    wix,wiy = Wi
    U = @. exp(im*angle(ψ))
    @. wix *= U # restore phase factors
    @. wiy *= U

	cx = auto_correlate(wix,X,K)
	cy = auto_correlate(wiy,X,K)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,x,y,C)
end

function incompressible_density(k,psi::Psi{3})
    @unpack ψ,X,K = psi; kx,ky,kz = K
    vx,vy,vz = velocity(psi)
    a = abs(ψ)
    ux = @. a*vx; uy = @. a*vy; uz = @. a*vz
    Wi, Wc = helmholtz(ux,uy,uz,kx,ky,kz)
    wix,wiy,wiz = Wi
    U = @. exp(im*angle(ψ))
    @. wix *= U # restore phase factors
    @. wiy *= U
    @. wiz *= U

	cx = auto_correlate(wix,X,K)
    cy = auto_correlate(wiy,X,K)
    cz = auto_correlate(wiz,X,K)
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce(k,x,y,z,C)
end

"""
    compressible_density(k,ψ,X,K)

Calculates the kinetic energy density of the compressible velocity field in the wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `makearrays`.
"""
function compressible_density(k,psi::Psi{2})
    @unpack ψ,X,K = psi; kx,ky = K
    vx,vy = velocity(psi)
    a = abs(ψ)
    ux = @. a*vx; uy = @. a*vy 
    Wi, Wc = helmholtz(ux,uy,kx,ky)
    wcx,wcy = Wc
    U = @. exp(im*angle(ψ))
    @. wcx *= U # restore phase factors
    @. wcy *= U

	cx = auto_correlate(wcx,X,K)
	cy = auto_correlate(wcy,X,K)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,x,y,C)
end

function compressible_density(k,psi::Psi{3})
    @unpack ψ,X,K = psi; kx,ky,kz = K
    vx,vy,vz = velocity(psi)
    a = abs(ψ)
    ux = @. a*vx; uy = @. a*vy; uz = @. a*vz
    Wi, Wc = helmholtz(ux,uy,uz,kx,ky,kz)
    wcx,wcy,wcz = Wc
    U = @. exp(im*angle(ψ))
    @. wcx *= U # restore phase factors
    @. wcy *= U
    @. wcz *= U

	cx = auto_correlate(wcx,X,K)
    cy = auto_correlate(wcy,X,K)
    cz = auto_correlate(wcz,X,K)
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce(k,x,y,z,C)
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
    return bessel_reduce(k,x,y,C)
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
    return sinc_reduce(k,x,y,z,C)
end

## coupling terms

"""
    ic_density(k,ψ,X,K)

Energy density of the incompressible-compressible interaction in the wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `makearrays`.
"""
function ic_density(k,psi::Psi{2})
    @unpack ψ,X,K = psi; kx,ky=K
    vx,vy = velocity(psi)
    a = abs(ψ)
    ux = @. a*vx; uy = @. a*vy 
    Wi, Wc = helmholtz(ux,uy,kx,ky)
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
    return bessel_reduce(k,x,y,C)
end

function ic_density(k,psi::Psi{3})
    @unpack ψ,X,K = psi; kx,ky,kz=K
    vx,vy,vz = velocity(psi)
    a = abs(ψ)
    ux = @. a*vx; uy = @. a*vy; uz = @. a*vz 
    Wi, Wc = helmholtz(ux,uy,uz,kx,ky,kz)
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
    return sinc_reduce(k,x,y,z,C)
end

"""
    iq_density(k,ψ,X,K)

Energy density of the incompressible-quantum pressure interaction in the wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `makearrays`.
"""
function iq_density(k,psi::Psi{2})
    @unpack ψ,X,K = psi; kx,ky=K
    vx,vy = velocity(psi)
    a = abs(ψ)
    ux = @. a*vx; uy = @. a*vy 
    Wi, Wc = helmholtz(ux,uy,kx,ky)
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
    return bessel_reduce(k,x,y,C)
end

function iq_density(k,psi::Psi{3})
    @unpack ψ,X,K = psi; kx,ky,kz=K
    vx,vy,vz = velocity(psi)
    a = abs(ψ)
    ux = @. a*vx; uy = @. a*vy; uz = @. a*vz
    Wi, Wc = helmholtz(ux,uy,uz,kx,ky,kz)
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
    return sinc_reduce(k,x,y,z,C)
end


"""
    cq_density(k,ψ,X,K)

Energy density of the compressible-quantum pressure interaction in the wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `makearrays`.
"""
function cq_density(k,psi::Psi{2})
    @unpack ψ,X,K = psi; kx,ky = K
    vx,vy = velocity(psi)
    a = abs(ψ)
    ux = @. a*vx; uy = @. a*vy 
    Wi, Wc = helmholtz(ux,uy,kx,ky)
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
    return bessel_reduce(k,x,y,C)
end

function cq_density(k,psi::Psi{3})
    @unpack ψ,X,K = psi; kx,ky,kz=K
    vx,vy,vz = velocity(psi)
    a = abs(ψ)
    ux = @. a*vx; uy = @. a*vy; uz = @. a*vz
    Wi, Wc = helmholtz(ux,uy,uz,kx,ky,kz)
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
    return sinc_reduce(k,x,y,z,C)
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

function trap_spectrum(k,V,ψ,X,K)
    x,y = X 
    f = @. abs(ψ)*sqrt(V(x,y',0.))
    C = autocorrelate(f,X,K)

    return bessel_reduce(k,x,y,C)
end

function density_spectrum(k,ψ,g,X,K)
    x,y = X 
    n = abs2.(ψ)
    C = autocorrelate(n,X,K) 

    return bessel_reduce(k,x,y,C)
end