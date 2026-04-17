using QuantumFluidSpectra
using Test

function trapz(x, y)
    T = promote_type(eltype(x), eltype(y), Float64)
    s = zero(T)
    for i in 2:length(x)
        s += 0.5 * (x[i] - x[i-1]) * (y[i] + y[i-1])
    end
    return s
end

include("test_arrays_and_1d.jl")
include("test_exported_spectra.jl")
include("test_internal_helpers.jl")

@testset "2D Analysis" begin 
    # Velocity and Helmholtz tests
    n = 256
    L = (1,1)
    N = (n,n)
    X,K,dX,dK = xk_arrays(L,N)
    kx,ky = K

    ##
    ktest = K[1][2]
    ψ = @. exp(im*ktest*X[1]*one.(X[2]'))
    psi = Psi(ψ,X,K)

    ## flow only in x direction, of correct value
    vx,vy = velocity(psi)
    @test vx ≈ ktest*one.(vx)
    @test vy ≈ zero.(vy)

    ## helmholtz decomposition
    Vi,Vc = helmholtz(vx,vy,kx,ky)

    ## Orthogonality
    vidotvc = Vi[1].*Vc[1] .+ Vi[2].*Vc[2]
    @test maximum(abs.(vidotvc)) < 1e-8

    ## Projective
    @test Vi[1] .+ Vc[1] ≈ vx
    @test Vi[2] .+ Vc[2] ≈ vy

    ## Energy decomposition is additive
    et, ei, ec = energydecomp(psi)
    @test sum(et) ≈ sum(ei) + sum(ec)

    ## autocorrelation
    x,y = X; kx,ky = K
    dx = x[2]-x[1]
    
    Natoms = sum(abs2.(ψ))*dx^2
    AC = auto_correlate(ψ,X,K)
    Nr,Nim = AC[n+1,n+1] .|> (real,imag) 

    @test Natoms ≈ Nr
    @test Nim ≈ 0.0 

    ## cross correlated reduces to autocorrelate correctly
    CC = cross_correlate(ψ,ψ,X,K)
    @test CC ≈ AC

    ## Zero-density points have a defined velocity
    ψ0 = zeros(ComplexF64, size(ψ))
    psi0 = Psi(ψ0,X,K)
    vx0,vy0 = velocity(psi0)
    @test all(iszero, vx0)
    @test all(iszero, vy0)

    ## k = 0 is handled in wave-action output
    wa = wave_action(kx,psi)
    @test wa[1] == 0
    @test all(isfinite, wa)
    @test radial_kgrid(psi, 32) == collect(LinRange(0.0, maximum(abs.(kx)), 32))

    ## GPE energy transfer vanishes for a plane wave and components add up
    kr = LinRange(0.0, maximum(abs.(kx)), 64)
    T,Tkin,Tint,Ttrap = gpe_particle_transfer(kr, psi; g=1.0, components=true)
    Π,Πkin,Πint,Πtrap = gpe_particle_flux(kr, psi; g=1.0, components=true)
    @test maximum(abs.(T)) < 1e-4
    @test maximum(abs.(Π)) < 1e-4
    @test T ≈ Tkin .+ Tint .+ Ttrap
    @test Π ≈ Πkin .+ Πint .+ Πtrap
    @test all(iszero, Ttrap)

    ## Flux is finite and vanishes for the zero field
    ψzero = copy(psi.ψ)
    ψzero .= 0
    psizero = Psi(ψzero, psi.X, psi.K)
    @test gpe_particle_flux(kr, psizero) ≈ zeros(length(kr))
    @test all(isfinite.(gpe_particle_flux(kr, psi)))

    ## Trap contribution is finite and bookkeeping stays consistent
    V2(x,y,t) = 0.5 .* (x.^2 .+ y.^2)
    ψg = @. exp(-((X[1] - 0.13)^2 + 1.2 * (X[2]' + 0.07)^2))
    psig = Psi(complex.(ψg), X, K)
    Tg,Tkg,Tig,Ttg = gpe_particle_transfer(kr, psig; g=1.0, V=V2, components=true)
    Πg = gpe_particle_flux(kr, psig; g=1.0, V=V2)
    @test Tg ≈ Tkg .+ Tig .+ Ttg
    @test all(isfinite, Tg)
    @test all(isfinite, Πg)
    @test maximum(abs.(Ttg)) > 0

    ## Direct flux stays finite on a nonlinear field
    nsmall = 128
    Xs,Ks,_,_ = xk_arrays(L,(nsmall,nsmall))
    xs,ys = Xs
    yrs = reshape(ys, 1, :)
    ψnl = @. (1 + 0.15*cos(2π*xs) + 0.1*sin(4π*yrs)) * exp(im*(2π*xs + 4π*yrs))
    psinl = Psi(complex.(ψnl), Xs, Ks)
    knl = collect(LinRange(0.0, maximum(abs.(Ks[1])), 1200))
    Πnl = gpe_particle_flux(knl, psinl; g=1.0)
    @test all(isfinite.(Πnl))

    ## Psi accepts generic complex arrays, including views
    ψ32 = ComplexF32.(ψ)
    psi32 = Psi(@view(ψ32[:,:]), X, K)
    vx32,vy32 = velocity(psi32)
    @test psi32.ψ isa SubArray
    @test maximum(abs.(vx32 .- ktest)) < 1e-4
    @test maximum(abs.(vy32)) < 1e-4
 
end

@testset "3D Analysis" begin 
    # Velocity and Helmholtz tests
    n = 32
    L = (1,1,1)
    N = (n,n,n)
    X,K,dX,dK = xk_arrays(L,N)
    kx,ky,kz = K

    ##
    ktest = K[1][2]
    ψ = exp.(im*ktest*X[1].*one.(X[2]').*one.(reshape(X[3],(1,1,n))))
    psi = Psi(ψ,X,K)

    ## flow only in x direction, of correct value
    vx,vy,vz = velocity(psi)
    @test vx ≈ ktest*one.(vx)
    @test vy ≈ zero.(vy)
    @test vz ≈ zero.(vz) 

    ## helmholtz decomposition
    Vi,Vc = helmholtz(vx,vy,vz,kx,ky,kz)

    ## Orthogonality
    vidotvc = Vi[1].*Vc[1] .+ Vi[2].*Vc[2] .+ Vi[3].*Vc[3]
    @test maximum(abs.(vidotvc)) < 1e-10

    ## Projective
    @test Vi[1] .+ Vc[1] ≈ vx
    @test Vi[2] .+ Vc[2] ≈ vy
    @test Vi[3] .+ Vc[3] ≈ vz

    ## Energy decomposition is additive
    et, ei, ec = energydecomp(psi)
    @test sum(et) ≈ sum(ei) + sum(ec)

    ## autocorrelation
    x,y,z = X; kx,ky,kz = K
    dx = x[2]-x[1];dk = kx[2]-kx[1] # (isotropic grid)
    
    Natoms = sum(abs2.(ψ))*dx^3
    AC = auto_correlate(ψ,X,K)
    Nr,Nim = AC[n+1,n+1,n+1] .|> (real,imag) 

    @test Natoms ≈ Nr
    @test Nim ≈ 0.0 

    ## cross correlated reduces to autocorrelate correctly
    CC = cross_correlate(ψ,ψ,X,K)
    @test CC ≈ AC

    ## Zero-density points have a defined velocity
    ψ0 = zeros(ComplexF64, size(ψ))
    psi0 = Psi(ψ0,X,K)
    vx0,vy0,vz0 = velocity(psi0)
    @test all(iszero, vx0)
    @test all(iszero, vy0)
    @test all(iszero, vz0)

    ## Spectra stay finite at k = 0
    εi = incompressible_spectrum(kx,psi)
    wa = wave_action(kx,psi)
    @test length(εi) == length(kx)
    @test all(isfinite, εi)
    @test wa[1] == 0
    @test all(isfinite, wa)
    @test radial_kgrid(psi, 24) == collect(LinRange(0.0, maximum(abs.(kx)), 24))

    ## GPE energy transfer vanishes for a plane wave and components add up
    kr = LinRange(0.0, maximum(abs.(kx)), 24)
    T,Tkin,Tint,Ttrap = gpe_particle_transfer(kr, psi; g=1.0, components=true)
    Π,Πkin,Πint,Πtrap = gpe_particle_flux(kr, psi; g=1.0, components=true)
    @test maximum(abs.(T)) < 1e-8
    @test maximum(abs.(Π)) < 1e-8
    @test T ≈ Tkin .+ Tint .+ Ttrap
    @test Π ≈ Πkin .+ Πint .+ Πtrap
    @test all(iszero, Ttrap)

    ## Flux is finite and vanishes for the zero field
    ψzero = copy(psi.ψ)
    ψzero .= 0
    psizero = Psi(ψzero, psi.X, psi.K)
    @test gpe_particle_flux(kr, psizero) ≈ zeros(length(kr))
    @test all(isfinite.(gpe_particle_flux(kr, psi)))

    ## Trap contribution is finite and bookkeeping stays consistent
    z3 = reshape(X[3], (1,1,n))
    ψg = @. exp(-((X[1] - 0.11)^2 + 1.1 * (X[2]' + 0.05)^2 + 0.9 * (z3 - 0.08)^2))
    psig = Psi(complex.(ψg), X, K)
    V3(x,y,z,t) = 0.5 .* (x.^2 .+ y.^2 .+ z.^2)
    Tg,Tkg,Tig,Ttg = gpe_particle_transfer(kr, psig; g=1.0, V=V3, components=true)
    Πg = gpe_particle_flux(kr, psig; g=1.0, V=V3)
    Tg0 = gpe_particle_transfer(kr, psig; g=1.0)
    Πg0 = gpe_particle_flux(kr, psig; g=1.0)
    @test Tg ≈ Tkg .+ Tig .+ Ttg
    @test all(isfinite, Tg)
    @test all(isfinite, Πg)
    @test maximum(abs.(Tg .- Tg0)) > 0
    @test maximum(abs.(Πg .- Πg0)) > 0

    ## Direct flux stays finite on a nonlinear field
    z3 = reshape(X[3], (1,1,n))
    ψnl = @. (1 + 0.1*cos(2π*X[1]) + 0.08*sin(2π*X[2]') + 0.06*cos(2π*z3)) *
        exp(im*(2π*X[1] + 2π*X[2]' + 2π*z3))
    psinl = Psi(complex.(ψnl), X, K)
    knl = collect(LinRange(0.0, maximum(abs.(kx)), 120))
    Πnl = gpe_particle_flux(knl, psinl; g=1.0)
    @test all(isfinite.(Πnl))
end
