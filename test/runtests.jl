using QuantumFluidSpectra
using Test

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
    n = 64
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
end
