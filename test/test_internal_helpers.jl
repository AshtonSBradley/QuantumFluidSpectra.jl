@testset "Internal Helpers and Aliases" begin
    n2 = 32
    X2, K2, _, _ = xk_arrays((1.0, 1.0), (n2, n2))
    x2, y2 = X2
    yr2 = reshape(y2, 1, :)
    ψ2 = @. (1 + 0.1 * cos(2π * x2) + 0.05 * sin(2π * yr2)) * exp(im * (2π * x2 + 2π * yr2))
    psi2 = Psi(complex.(ψ2), X2, K2)
    k2r = collect(LinRange(0.0, maximum(abs.(K2[1])), 24))

    ac2 = auto_correlate(psi2)
    cc2 = cross_correlate(psi2, psi2)
    @test ac2 ≈ cc2

    jxΩ, jyΩ = current(psi2, 0.25)
    vxΩ, vyΩ = velocity(psi2, 0.25)
    @test all(isfinite, jxΩ)
    @test all(isfinite, jyΩ)
    @test all(isfinite, vxΩ)
    @test all(isfinite, vyΩ)

    zc2 = zeros(ComplexF64, 2n2, 2n2)
    @test QuantumFluidSpectra.bessel_reduce(k2r, X2[1], X2[2], zc2) ≈ zeros(length(k2r))
    @test QuantumFluidSpectra._integrated_bessel_reduce(k2r, X2[1], X2[2], zc2) ≈
          zeros(length(k2r))

    p2 = collect(0:(size(zc2, 1)-1))
    q2 = collect(0:(size(zc2, 2)-1))
    c2 = @. complex(cos(2π * p2 / size(zc2, 1))) + 0.2im * sin(2π * q2' / size(zc2, 2))
    ids2, radii2, dx2, _ = QuantumFluidSpectra._cached_radial_ids_2d(X2[1], X2[2])
    @test minimum(ids2) == 1
    @test maximum(ids2) == length(radii2)
    cached_bessel_weights =
        QuantumFluidSpectra.besselj0.(reshape(k2r, :, 1) .* reshape(radii2, 1, :))
    serial_bessel = zeros(Float64, length(k2r))
    QuantumFluidSpectra._radial_reduce_partial!(
        serial_bessel,
        c2,
        ids2,
        cached_bessel_weights,
        eachindex(c2),
    )
    @test QuantumFluidSpectra._threaded_radial_weight_reduce(
        k2r,
        c2,
        ids2,
        cached_bessel_weights,
        Float64,
        ntasks = 2,
    ) ≈ serial_bessel
    ix2 = reshape(collect(0:(size(zc2, 1)-1)) .- size(zc2, 1) ÷ 2, :, 1)
    iy2 = reshape(collect(0:(size(zc2, 2)-1)) .- size(zc2, 2) ÷ 2, 1, :)
    full_radii2 = dx2 .* sqrt.(ix2 .^ 2 .+ iy2 .^ 2)
    full_bessel_weights =
        QuantumFluidSpectra.besselj0.(
            reshape(k2r, :, 1, 1) .* reshape(full_radii2, 1, size(zc2)...),
        )
    lookup_bessel_weights = similar(full_bessel_weights)
    for q in axes(ids2, 2), p in axes(ids2, 1), i in eachindex(k2r)
        lookup_bessel_weights[i, p, q] = cached_bessel_weights[i, ids2[p, q]]
    end
    @test lookup_bessel_weights == full_bessel_weights
    direct_bessel =
        vec(sum(full_bessel_weights .* reshape(real.(c2), 1, size(c2)...); dims = (2, 3)))
    @. direct_bessel *= k2r * dx2^2 / 2 / pi

    cached2 = QuantumFluidSpectra._cached_bessel_reduce(k2r, X2[1], X2[2], c2)
    @test !isnothing(cached2)
    @test cached2 ≈ QuantumFluidSpectra.bessel_reduce(k2r, X2[1], X2[2], c2)
    radial_cache2 = radial_reduction_cache(k2r, X2)
    @test radial_cache2 isa RadialReductionCache{2}
    @test bessel_reduce(radial_cache2, c2) ≈ cached2
    @test bessel_reduce(radial_cache2, c2) ≈ direct_bessel
    @test density_spectrum(radial_cache2, psi2) ≈ density_spectrum(k2r, psi2)
    @test wave_action(radial_cache2, psi2) ≈ wave_action(k2r, psi2)
    @test isnothing(QuantumFluidSpectra._cached_bessel_reduce(k2r, 2 .* X2[1], X2[2], c2))
    @test isnothing(radial_reduction_cache(k2r, 2 .* X2[1], X2[2]))
    @test all(isfinite, QuantumFluidSpectra.bessel_reduce(k2r, 2 .* X2[1], X2[2], c2))
    X2f = map(x -> Float32.(x), X2)
    k2f = Float32.(k2r)
    c2f = ComplexF32.(c2)
    _, radii2f, _, _ = QuantumFluidSpectra._cached_radial_ids_2d(X2f[1], X2f[2])
    @test eltype(radii2f) === Float32
    @test eltype(QuantumFluidSpectra._cached_bessel_reduce(k2f, X2f[1], X2f[2], c2f)) ===
          Float32

    V2(x, y, t) = 0.5 * (x^2 + y^2)
    trap2 = QuantumFluidSpectra._trap_field(psi2, V2, 0.0)
    rhs2 = QuantumFluidSpectra._gpe_rhs(psi2; g = 1.0, V = V2, t = 0.0)
    rhs2_notrap = QuantumFluidSpectra._gpe_rhs(psi2; g = 1.0)
    @test all(isfinite, trap2)
    @test all(isfinite, rhs2)
    @test maximum(abs.(rhs2 .- rhs2_notrap)) > 0

    T2 = gpe_particle_transfer(k2r, psi2; g = 1.0, V = V2)
    Π2 = gpe_particle_flux(k2r, psi2; g = 1.0, V = V2)
    @test QuantumFluidSpectra.gpe_energy_transfer(k2r, psi2; g = 1.0, V = V2) ≈ T2
    @test QuantumFluidSpectra.gpe_energy_flux(k2r, psi2; g = 1.0, V = V2) ≈ Π2

    @test QuantumFluidSpectra._gpe_reduce(k2r, X2, zc2) ≈ zeros(length(k2r))
    @test QuantumFluidSpectra._integrated_gpe_reduce(k2r, X2, zc2) ≈ zeros(length(k2r))
    @test QuantumFluidSpectra._shell_area(psi2) == 2π
    @test QuantumFluidSpectra._integrated_gpe_prefactor(psi2) == 1 / π
    @test QuantumFluidSpectra._gradient_fields(psi2) == gradient(psi2)
    @test QuantumFluidSpectra._cumulative_integral([0.0, 1.0, 2.0], [0.0, 1.0, 1.0]) ≈
          [0.0, 0.5, 1.5]

    n3 = 16
    X3, K3, _, _ = xk_arrays((1.0, 1.0, 1.0), (n3, n3, n3))
    x3, y3, z3 = X3
    yr3 = reshape(y3, 1, :, 1)
    zr3 = reshape(z3, 1, 1, :)
    ψ3 = @. (1 + 0.1 * cos(2π * x3) + 0.05 * sin(2π * yr3) + 0.03 * cos(2π * zr3)) *
       exp(im * (2π * x3 + 2π * yr3 + 2π * zr3))
    psi3 = Psi(complex.(ψ3), X3, K3)
    k3r = collect(LinRange(0.0, maximum(abs.(K3[1])), 20))

    ac3 = auto_correlate(psi3)
    cc3 = cross_correlate(psi3, psi3)
    @test ac3 ≈ cc3

    zc3 = zeros(ComplexF64, 2n3, 2n3, 2n3)
    @test QuantumFluidSpectra.sinc_reduce(k3r, X3[1], X3[2], X3[3], zc3) ≈
          zeros(length(k3r))
    @test QuantumFluidSpectra._integrated_sinc_reduce(k3r, X3[1], X3[2], X3[3], zc3) ≈
          zeros(length(k3r))
    p3 = collect(0:(size(zc3, 1)-1))
    q3 = collect(0:(size(zc3, 2)-1))
    r3 = collect(0:(size(zc3, 3)-1))
    p3r = reshape(p3, :, 1, 1)
    q3r = reshape(q3, 1, :, 1)
    r3r = reshape(r3, 1, 1, :)
    c3 = @. complex(cos(2π * p3r / size(zc3, 1))) +
       0.2im * sin(2π * q3r / size(zc3, 2)) +
       0.1 * cos(2π * r3r / size(zc3, 3))
    ids3, radii3, dx3, _, _ = QuantumFluidSpectra._cached_radial_ids_3d(X3[1], X3[2], X3[3])
    @test minimum(ids3) == 1
    @test maximum(ids3) == length(radii3)
    cached_sinc_weights =
        QuantumFluidSpectra._sinc_times_pi.(reshape(k3r, :, 1) .* reshape(radii3, 1, :))
    serial_sinc = zeros(Float64, length(k3r))
    QuantumFluidSpectra._radial_reduce_partial!(
        serial_sinc,
        c3,
        ids3,
        cached_sinc_weights,
        eachindex(c3),
    )
    @test QuantumFluidSpectra._threaded_radial_weight_reduce(
        k3r,
        c3,
        ids3,
        cached_sinc_weights,
        Float64,
        ntasks = 2,
    ) ≈ serial_sinc
    ix3 = reshape(collect(0:(size(zc3, 1)-1)) .- size(zc3, 1) ÷ 2, :, 1, 1)
    iy3 = reshape(collect(0:(size(zc3, 2)-1)) .- size(zc3, 2) ÷ 2, 1, :, 1)
    iz3 = reshape(collect(0:(size(zc3, 3)-1)) .- size(zc3, 3) ÷ 2, 1, 1, :)
    full_radii3 = dx3 .* sqrt.(ix3 .^ 2 .+ iy3 .^ 2 .+ iz3 .^ 2)
    full_sinc_weights =
        QuantumFluidSpectra._sinc_times_pi.(
            reshape(k3r, :, 1, 1, 1) .* reshape(full_radii3, 1, size(zc3)...),
        )
    lookup_sinc_weights = similar(full_sinc_weights)
    for r in axes(ids3, 3), q in axes(ids3, 2), p in axes(ids3, 1), i in eachindex(k3r)
        lookup_sinc_weights[i, p, q, r] = cached_sinc_weights[i, ids3[p, q, r]]
    end
    @test lookup_sinc_weights == full_sinc_weights
    direct_sinc =
        vec(sum(full_sinc_weights .* reshape(real.(c3), 1, size(c3)...); dims = (2, 3, 4)))
    @. direct_sinc *= k3r^2 * dx3^3 / 2 / pi^2

    cached3 = QuantumFluidSpectra._cached_sinc_reduce(k3r, X3[1], X3[2], X3[3], c3)
    @test !isnothing(cached3)
    @test cached3 ≈ QuantumFluidSpectra.sinc_reduce(k3r, X3[1], X3[2], X3[3], c3)
    radial_cache3 = radial_reduction_cache(k3r, X3)
    @test radial_cache3 isa RadialReductionCache{3}
    @test sinc_reduce(radial_cache3, c3) ≈ cached3
    @test sinc_reduce(radial_cache3, c3) ≈ direct_sinc
    @test density_spectrum(radial_cache3, psi3) ≈ density_spectrum(k3r, psi3)
    @test wave_action(radial_cache3, psi3) ≈ wave_action(k3r, psi3)
    @test isnothing(
        QuantumFluidSpectra._cached_sinc_reduce(k3r, 2 .* X3[1], X3[2], X3[3], c3),
    )
    @test isnothing(radial_reduction_cache(k3r, 2 .* X3[1], X3[2], X3[3]))
    @test all(isfinite, QuantumFluidSpectra.sinc_reduce(k3r, 2 .* X3[1], X3[2], X3[3], c3))
    X3f = map(x -> Float32.(x), X3)
    k3f = Float32.(k3r)
    c3f = ComplexF32.(c3)
    _, radii3f, _, _, _ = QuantumFluidSpectra._cached_radial_ids_3d(X3f[1], X3f[2], X3f[3])
    @test eltype(radii3f) === Float32
    @test eltype(
        QuantumFluidSpectra._cached_sinc_reduce(k3f, X3f[1], X3f[2], X3f[3], c3f),
    ) === Float32
    @test QuantumFluidSpectra._sinc_times_pi(0.0) == 1.0
    @test QuantumFluidSpectra._sinc_times_pi(1e-3) ≈ sin(1e-3) / 1e-3
    @test QuantumFluidSpectra._integrated_sinc_kernel(2.0, 0.0) ≈ 8 / 3

    V3(x, y, z, t) = 0.5 * (x^2 + y^2 + z^2)
    trap3 = QuantumFluidSpectra._trap_field(psi3, V3, 0.0)
    rhs3 = QuantumFluidSpectra._gpe_rhs(psi3; g = 1.0, V = V3, t = 0.0)
    rhs3_notrap = QuantumFluidSpectra._gpe_rhs(psi3; g = 1.0)
    @test all(isfinite, trap3)
    @test all(isfinite, rhs3)
    @test maximum(abs.(rhs3 .- rhs3_notrap)) > 0

    T3 = gpe_particle_transfer(k3r, psi3; g = 1.0, V = V3)
    Π3 = gpe_particle_flux(k3r, psi3; g = 1.0, V = V3)
    @test QuantumFluidSpectra.gpe_energy_transfer(k3r, psi3; g = 1.0, V = V3) ≈ T3
    @test QuantumFluidSpectra.gpe_energy_flux(k3r, psi3; g = 1.0, V = V3) ≈ Π3

    @test QuantumFluidSpectra._gpe_reduce(k3r, X3, zc3) ≈ zeros(length(k3r))
    @test QuantumFluidSpectra._integrated_gpe_reduce(k3r, X3, zc3) ≈ zeros(length(k3r))
    @test QuantumFluidSpectra._shell_area(psi3) == 4π
    @test QuantumFluidSpectra._integrated_gpe_prefactor(psi3) == 1 / π^2
    @test QuantumFluidSpectra._gradient_fields(psi3) == gradient(psi3)
end

@testset "CUDA spectral-analysis stubs without CUDA.jl" begin
    @test CUDADevice() isa CUDADevice
    @test MetalDevice() isa AbstractSpectrumBackend
    @test OneAPIDevice() isa AbstractSpectrumBackend
    @test_throws ErrorException gpu("not a psi"; copy = false)
    @test_throws ErrorException gpu(CUDADevice(), "not a psi"; copy = false)
    @test_throws ErrorException gpu(MetalDevice(), "not a psi"; copy = false)
    @test_throws ErrorException gpu(OneAPIDevice(), "not a psi"; copy = false)
    @test_throws ErrorException cpu("not a psi"; copy = false)
    @test_throws ErrorException cpu(CUDADevice(), "not a psi"; copy = false)
    @test_throws ErrorException cpu(MetalDevice(), "not a psi"; copy = false)
    @test_throws ErrorException cpu(OneAPIDevice(), "not a psi"; copy = false)
    err = try
        spectrum_cache("not a psi"; nradial = 4)
    catch err
        err
    end
    @test err isa ErrorException
    @test occursin("CUDA spectral analysis requires CUDA.jl", sprint(showerror, err))
    err = try
        spectrum_cache("not a psi"; backend = MetalDevice(), nradial = 4)
    catch err
        err
    end
    @test err isa ErrorException
    @test occursin("Metal spectral analysis requires Metal.jl", sprint(showerror, err))
    err = try
        spectrum_cache("not a psi"; backend = OneAPIDevice(), nradial = 4)
    catch err
        err
    end
    @test err isa ErrorException
    @test occursin("oneAPI spectral analysis requires oneAPI.jl", sprint(showerror, err))
    err = try
        spectrum_cache("not a psi"; backend = :cpu)
    catch err
        err
    end
    @test err isa ErrorException
    @test occursin("Unsupported spectral analysis backend", sprint(showerror, err))
    @test_throws ErrorException analyze_spectra!(nothing, nothing; spectra = (:density,))
    @test_throws ErrorException spectrum_results(nothing; host = true)
end
