using QuantumFluidSpectra
using Test

@testset "Analysis" begin 
    # Velocity and Helmholtz tests
    n = 100
    L = (1,1)
    N = (n,n)
    X,K,dX,dK = makearrays(L,N)
    kx,ky = K
    k² = k2(K)

    ##
    ktest = K[1][2]
    ψ = @. exp(im*ktest*X[1]*one.(X[2]'))
    psi = XField(ψ,X,K)

    ## flow only in x direction, of correct value
    vx,vy = velocity(psi)
    @test vx ≈ ktest*one.(vx)
    @test vy ≈ zero.(vy)

    ## helmholtz decomposition
    Vi,Vc = helmholtz(vx,vy,kx,ky)

    ## Orthogonality
    vidotvc = Vi[1].*Vc[1] .+ Vi[2].*Vc[2]
    @test maximum(abs.(vidotvc)) < 1e-10

    ## Projective
    @test Vi[1] .+ Vc[1] ≈ vx
    @test Vi[2] .+ Vc[2] ≈ vy

    ## autocorrelation
    x,y = X; kx,ky = K
    dx = x[2]-x[1];dk = kx[2]-kx[1]
    
    Natoms = sum(abs2.(ψ))*dx^2
    AC = auto_correlate(ψ,X,K)
    Nr,Nim = AC[101,101] .|> (real,imag) 

    @test Natoms ≈ Nr
    @test Nim ≈ 0.0 

    ## cross correlated reduces to autocorrelate correctly
    CC = cross_correlate(ψ,ψ,X,K)
    @test AC == AC
 
end
