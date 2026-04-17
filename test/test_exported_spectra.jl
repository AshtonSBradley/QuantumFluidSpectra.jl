function nonlinear_field_2d(X)
    x, y = X
    yr = reshape(y, 1, :)
    amp = @. 1 + 0.12 * cos(2π * x) + 0.08 * sin(4π * yr)
    phase = @. 2π * x + 4π * yr + 0.2 * sin(2π * x) * cos(2π * yr)
    return complex.(amp .* exp.(im .* phase))
end

function nonlinear_field_3d(X)
    x, y, z = X
    yr = reshape(y, 1, :, 1)
    zr = reshape(z, 1, 1, :)
    amp = @. 1 + 0.1 * cos(2π * x) + 0.08 * sin(2π * yr) + 0.06 * cos(2π * zr)
    phase = @. 2π * x + 2π * yr + 2π * zr + 0.15 * sin(2π * x) * cos(2π * yr)
    return complex.(amp .* exp.(im .* phase))
end

@testset "2D Exported Spectra" begin
    n = 64
    X, K, _, _ = xk_arrays((1.0, 1.0), (n, n))
    psi = Psi(nonlinear_field_2d(X), X, K)
    k = collect(LinRange(0.0, maximum(abs.(K[1])), 48))
    r = collect(LinRange(0.0, 0.5, 20))

    vx, vy = velocity(psi, 0.3)
    jx, jy = current(psi, 0.3)
    @test all(isfinite, vx)
    @test all(isfinite, vy)
    @test all(isfinite, jx)
    @test all(isfinite, jy)

    for f in (
        kinetic_density,
        knumber_density,
        wave_action,
        incompressible_spectrum,
        compressible_spectrum,
        qpressure_spectrum,
        incompressible_density,
        compressible_density,
        qpressure_density,
        ic_density,
        iq_density,
        cq_density,
        density_spectrum,
    )
        out = f(k, psi)
        @test length(out) == length(k)
        @test all(isfinite, out)
    end

    V2(x, y, t) = 0.5 * (x^2 + y^2)
    trap = trap_spectrum(k, V2, psi)
    @test length(trap) == length(k)
    @test all(isfinite, trap)

    εi = incompressible_spectrum(k, psi)
    gvi = gv(r, k, εi)
    @test length(gvi) == length(r)
    @test all(isfinite, gvi)
    @test gvi[1] ≈ 1.0 atol = 1e-8

    conv = convolve(psi.ψ, psi.ψ, X, K)
    @test size(conv) == (2n, 2n)
    @test all(isfinite, conv)
end

@testset "3D Exported Spectra" begin
    n = 24
    X, K, _, _ = xk_arrays((1.0, 1.0, 1.0), (n, n, n))
    psi = Psi(nonlinear_field_3d(X), X, K)
    k = collect(LinRange(0.0, maximum(abs.(K[1])), 32))
    r = collect(LinRange(0.0, 0.5, 16))

    jx, jy, jz = current(psi)
    @test all(isfinite, jx)
    @test all(isfinite, jy)
    @test all(isfinite, jz)

    for f in (
        kinetic_density,
        knumber_density,
        wave_action,
        incompressible_spectrum,
        compressible_spectrum,
        qpressure_spectrum,
        incompressible_density,
        compressible_density,
        qpressure_density,
        ic_density,
        iq_density,
        cq_density,
        density_spectrum,
    )
        out = f(k, psi)
        @test length(out) == length(k)
        @test all(isfinite, out)
    end

    V3(x, y, z, t) = 0.5 * (x^2 + y^2 + z^2)
    trap = trap_spectrum(k, V3, psi)
    @test length(trap) == length(k)
    @test all(isfinite, trap)

    εi = incompressible_spectrum(k, psi)
    gvi = gv3(r, k, εi)
    @test length(gvi) == length(r)
    @test all(isfinite, gvi)
    @test gvi[1] ≈ 1.0 atol = 1e-8

    conv = convolve(psi.ψ, psi.ψ, X, K)
    @test size(conv) == (2n, 2n, 2n)
    @test all(isfinite, conv)
end
