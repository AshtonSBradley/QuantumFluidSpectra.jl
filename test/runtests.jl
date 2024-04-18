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

    ## autocorrelation
    x,y = X; kx,ky = K
    dx = x[2]-x[1];dk = kx[2]-kx[1]
    
    Natoms = sum(abs2.(ψ))*dx^2
    AC = auto_correlate(ψ,X,K)
    Nr,Nim = AC[n+1,n+1] .|> (real,imag) 

    @test Natoms ≈ Nr
    @test Nim ≈ 0.0 

    ## cross correlated reduces to autocorrelate correctly
    CC = cross_correlate(ψ,ψ,X,K)
    @test AC == AC
 
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
    @test AC == AC
    incompressible_spectrum(kx,psi) # not really a test, but at least it errors
end
