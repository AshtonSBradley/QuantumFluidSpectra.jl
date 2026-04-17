@testset "Array Helpers" begin
    x = xvec(2.0, 4)
    @test x ≈ [-0.5, 0.0, 0.5, 1.0]

    k = kvec(2.0, 4)
    @test k == π .* [0.0, 1.0, -2.0, -1.0]

    X = xvecs((2.0, 3.0), (4, 6))
    K = kvecs((2.0, 3.0), (4, 6))
    @test length(X) == 2
    @test length(K) == 2
    @test length(X[1]) == 4
    @test length(K[2]) == 6

    X2, K2, dX2, dK2 = xk_arrays((2.0, 3.0), (4, 6))
    @test X2 == X
    @test K2 == K
    @test all(dX2 .≈ (0.5, 0.5))
    @test all(dK2 .≈ (π, 2π / 3))

    Dx, Dk = QuantumFluidSpectra.dfft(X[1], K[1])
    @test Dx * Dk ≈ 1.0

    DX, DK = fft_differentials(X, K)
    @test DX[1] * DK[1] ≈ 1.0
    @test DX[2] * DK[2] ≈ 1.0

    ksq = QuantumFluidSpectra.k2(([-1.0, 0.0], [2.0, 3.0]))
    @test collect(ksq) == [5.0 10.0; 4.0 9.0]

    @test radial_kgrid(3.0, 4) == [0.0, 1.0, 2.0, 3.0]
    @test log10range(1e-2, 1e2, 5) ≈ [1e-2, 1e-1, 1.0, 1e1, 1e2]
    @test_throws AssertionError log10range(0.0, 1.0, 5)

    A = reshape(collect(1.0:4.0), 2, 2)
    padded = QuantumFluidSpectra.zeropad(A)
    @test size(padded) == (4, 4)
    @test padded[2:3, 2:3] == A
    @test_throws ErrorException QuantumFluidSpectra.zeropad(ones(3, 2))
end

@testset "1D Analysis" begin
    n = 128
    X, K, dX, _ = xk_arrays((1.0,), (n,))
    x = X[1]
    k = K[1]
    ktest = k[3]
    ψ = @. exp(im * ktest * x)
    psi = Psi(ψ, X, K)

    ψx = gradient(psi)
    @test ψx ≈ @.(im * ktest * ψ)

    jx = current(psi)
    vx = velocity(psi)
    @test jx ≈ fill(ktest, size(jx))
    @test vx ≈ fill(ktest, size(vx))

    AC = auto_correlate(ψ, X, K)
    CC = cross_correlate(ψ, ψ, X, K)
    @test CC ≈ AC
    @test imag(AC[n + 1]) ≈ 0.0 atol = 1e-10
    @test real(AC[n + 1]) ≈ sum(abs2, ψ) * dX[1]

    ψ0 = zeros(ComplexF64, size(ψ))
    psi0 = Psi(ψ0, X, K)
    @test all(iszero, velocity(psi0))
    @test all(iszero, current(psi0))
end
