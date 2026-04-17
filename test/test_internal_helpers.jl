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
    @test QuantumFluidSpectra._integrated_bessel_reduce(k2r, X2[1], X2[2], zc2) ≈ zeros(length(k2r))

    V2(x, y, t) = 0.5 * (x^2 + y^2)
    trap2 = QuantumFluidSpectra._trap_field(psi2, V2, 0.0)
    rhs2 = QuantumFluidSpectra._gpe_rhs(psi2; g=1.0, V=V2, t=0.0)
    rhs2_notrap = QuantumFluidSpectra._gpe_rhs(psi2; g=1.0)
    @test all(isfinite, trap2)
    @test all(isfinite, rhs2)
    @test maximum(abs.(rhs2 .- rhs2_notrap)) > 0

    T2 = gpe_particle_transfer(k2r, psi2; g=1.0, V=V2)
    Π2 = gpe_particle_flux(k2r, psi2; g=1.0, V=V2)
    @test QuantumFluidSpectra.gpe_energy_transfer(k2r, psi2; g=1.0, V=V2) ≈ T2
    @test QuantumFluidSpectra.gpe_energy_flux(k2r, psi2; g=1.0, V=V2) ≈ Π2

    @test QuantumFluidSpectra._gpe_reduce(k2r, X2, zc2) ≈ zeros(length(k2r))
    @test QuantumFluidSpectra._integrated_gpe_reduce(k2r, X2, zc2) ≈ zeros(length(k2r))
    @test QuantumFluidSpectra._shell_area(psi2) == 2π
    @test QuantumFluidSpectra._integrated_gpe_prefactor(psi2) == 1 / π
    @test QuantumFluidSpectra._gradient_fields(psi2) == gradient(psi2)
    @test QuantumFluidSpectra._cumulative_integral([0.0, 1.0, 2.0], [0.0, 1.0, 1.0]) ≈ [0.0, 0.5, 1.5]

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
    @test QuantumFluidSpectra.sinc_reduce(k3r, X3[1], X3[2], X3[3], zc3) ≈ zeros(length(k3r))
    @test QuantumFluidSpectra._integrated_sinc_reduce(k3r, X3[1], X3[2], X3[3], zc3) ≈ zeros(length(k3r))
    @test QuantumFluidSpectra._sinc_times_pi(0.0) == 1.0
    @test QuantumFluidSpectra._sinc_times_pi(1e-3) ≈ sin(1e-3) / 1e-3
    @test QuantumFluidSpectra._integrated_sinc_kernel(2.0, 0.0) ≈ 8 / 3

    V3(x, y, z, t) = 0.5 * (x^2 + y^2 + z^2)
    trap3 = QuantumFluidSpectra._trap_field(psi3, V3, 0.0)
    rhs3 = QuantumFluidSpectra._gpe_rhs(psi3; g=1.0, V=V3, t=0.0)
    rhs3_notrap = QuantumFluidSpectra._gpe_rhs(psi3; g=1.0)
    @test all(isfinite, trap3)
    @test all(isfinite, rhs3)
    @test maximum(abs.(rhs3 .- rhs3_notrap)) > 0

    T3 = gpe_particle_transfer(k3r, psi3; g=1.0, V=V3)
    Π3 = gpe_particle_flux(k3r, psi3; g=1.0, V=V3)
    @test QuantumFluidSpectra.gpe_energy_transfer(k3r, psi3; g=1.0, V=V3) ≈ T3
    @test QuantumFluidSpectra.gpe_energy_flux(k3r, psi3; g=1.0, V=V3) ≈ Π3

    @test QuantumFluidSpectra._gpe_reduce(k3r, X3, zc3) ≈ zeros(length(k3r))
    @test QuantumFluidSpectra._integrated_gpe_reduce(k3r, X3, zc3) ≈ zeros(length(k3r))
    @test QuantumFluidSpectra._shell_area(psi3) == 4π
    @test QuantumFluidSpectra._integrated_gpe_prefactor(psi3) == 1 / π^2
    @test QuantumFluidSpectra._gradient_fields(psi3) == gradient(psi3)
end
