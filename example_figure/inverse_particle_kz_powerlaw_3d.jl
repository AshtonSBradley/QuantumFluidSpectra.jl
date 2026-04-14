#
# 3D inverse-particle Kolmogorov-Zakharov test state for the GPE.
# This example compares `knumber_density` directly against a k^(-7/3) law,
# following the requested angle-averaged momentum-space number-density view.
#
using QuantumFluidSpectra
using FFTW
using Plots
using Plots.Measures
using Printf
using Random

gr()

const G = 0.1
const NTOTAL = 80_000.0
const LBOX = 40.0
const NX = 128
const KZ_OCCUPATION_EXPONENT = 7 / 3
const WALL_STEEPNESS = 8
const WALL_HEIGHT_FACTOR = 20.0

particle_number(ψ, dV) = sum(abs2, ψ) * dV

function kz_band_limits(L, ξ)
    return π / L, π / ξ
end

function wall_profile(s, halfwidth, wallwidth; power=WALL_STEEPNESS)
    d = max(halfwidth - abs(s), 0.0)
    return exp(-((d / wallwidth)^power))
end

function box_trap(x, y, z, t; L=LBOX, ξ=sqrt(8.0), μ=1.0)
    halfwidth = L / 2
    wallwidth = ξ
    V0 = WALL_HEIGHT_FACTOR * μ
    wx = wall_profile(x, halfwidth, wallwidth)
    wy = wall_profile(y, halfwidth, wallwidth)
    wz = wall_profile(z, halfwidth, wallwidth)
    return V0 * (wx + wy + wz)
end

function confinement_envelope(X; L=LBOX, ξ=sqrt(8.0))
    halfwidth = L / 2
    wallwidth = ξ
    x, y, z = X
    yr = reshape(y, 1, :, 1)
    zr = reshape(z, 1, 1, :)
    fx = @. 1 - wall_profile(x, halfwidth, wallwidth)
    fy = @. 1 - wall_profile(yr, halfwidth, wallwidth)
    fz = @. 1 - wall_profile(zr, halfwidth, wallwidth)
    return fx .* fy .* fz
end

function build_inverse_particle_kz_state_3d(; g=G, Ntotal=NTOTAL, L=LBOX, nx=NX, seed=1234)
    X, K, dX, _ = xk_arrays((L, L, L), (nx, nx, nx))
    dx = dX[1]
    dV = prod(dX)

    n0 = Ntotal / L^3
    ξ = 1 / sqrt(g * n0)
    kmin, kmax = kz_band_limits(L, ξ)
    knyquist = π / dx

    @assert dx < ξ / 4 "Need dx < ξ/4, but found dx=$(dx) and ξ/4=$(ξ / 4)."
    @assert kmax < knyquist "Need π/ξ < π/dx, but found kmax=$(kmax) and Nyquist=$(knyquist)."

    rng = MersenneTwister(seed)
    kx, ky, kz = K
    kzr = reshape(kz, 1, 1, :)
    kmag = @. sqrt(kx^2 + ky'^2 + kzr^2)

    ϕ = zeros(ComplexF64, nx, nx, nx)
    edges = collect(range(kmin, kmax, length=160))
    for j in 1:(length(edges) - 1)
        klo = edges[j]
        khi = edges[j + 1]
        kshell = 0.5 * (klo + khi)
        Δk = khi - klo
        mask = @. (kmag >= klo) & (kmag < khi)
        nmode = count(mask)
        nmode == 0 && continue

        shell_population = kshell^(-KZ_OCCUPATION_EXPONENT) * Δk
        shell_weights = randexp(rng, nmode)
        shell_weights ./= sum(shell_weights)
        shell_phases = cis.(2π .* rand(rng, nmode))
        ϕ[mask] .= sqrt.(shell_population .* shell_weights) .* shell_phases
    end
    ϕ[1, 1, 1] = 0.0

    ψ = ifft(ϕ)
    ψ .*= confinement_envelope(X; L=L, ξ=ξ)
    ψ .*= sqrt(Ntotal / particle_number(ψ, dV))

    psi = Psi(ψ, X, K)
    return (; psi, ξ, dx, dV, kmin, kmax, knyquist, n0)
end

function kz_reference(k, kn)
    ref = @. k^(-KZ_OCCUPATION_EXPONENT)
    scale = kn[1] / ref[1]
    return scale .* ref
end

function summarize_state(; g=G, Ntotal=NTOTAL, L=LBOX, nx=NX)
    state = build_inverse_particle_kz_state_3d(; g=g, Ntotal=Ntotal, L=L, nx=nx)
    psi = state.psi
    X = psi.X
    dX = (X[1][2] - X[1][1], X[2][2] - X[2][1], X[3][2] - X[3][1])
    dV = prod(dX)
    Ncheck = particle_number(psi.ψ, dV)
    kplot_max = state.knyquist * 0.8
    k = collect(range(state.kmin, kplot_max, length=240))
    trap = (x, y, z, t) -> box_trap(x, y, z, t; L=L, ξ=state.ξ, μ=g * state.n0)
    Π = gpe_particle_flux(k, psi; g=g, V=trap)
    nk = knumber_density(k, psi)

    iz = argmin(abs.(X[3]))
    zslice = abs2.(psi.ψ[:, :, iz])

    return (; state..., k, Π, nk, Ncheck, iz, zslice, trap, kplot_max)
end

function make_plot(data)
    psi = data.psi
    X = psi.X
    z0 = X[3][data.iz]

    prefactor = data.nk[1] * data.k[1]^KZ_OCCUPATION_EXPONENT
    p1 = heatmap(
        X[1],
        X[2],
        data.zslice',
        xlabel="x",
        ylabel="y",
        title=@sprintf("|ψ(x,y,z≈0)|² at z = %.3f", z0),
        colorbar_title="density",
        aspect_ratio=:equal,
        frame=:box,
        left_margin=6mm,
        bottom_margin=6mm,
    )

    p2 = plot(
        data.k,
        data.Π,
        xlabel="k",
        ylabel="Π(k)",
        title="Particle flux",
        frame=:box,
        lw=2.5,
        legend=false,
        left_margin=6mm,
        bottom_margin=6mm,
    )
    hline!(p2, [0.0], color=:black, linestyle=:dash, alpha=0.5, lw=1.5)

    p3 = plot(
        data.k,
        data.nk,
        xscale=:log10,
        yscale=:log10,
        xlabel="k",
        ylabel="knumber_density(k)",
        title="Angle-averaged k-space number density",
        frame=:box,
        lw=2.5,
        label="measured",
        left_margin=6mm,
        bottom_margin=6mm,
    )
    plot!(p3, data.k, prefactor .* data.k .^ (-KZ_OCCUPATION_EXPONENT), linestyle=:dash, lw=2, label="k^(-7/3) guide")
    vline!(p3, [data.kmax], linestyle=:dot, lw=1.5, color=:black, alpha=0.7, label="π/ξ")

    return plot(
        p1,
        p2,
        p3,
        layout=(1, 3),
        size=(1500, 460),
    )
end

function main()
    data = summarize_state()

    println("3D inverse-particle KZ test state")
    @printf("  g                         = %.6f\n", G)
    @printf("  total particle number N   = %.6f\n", NTOTAL)
    @printf("  box size L                = %.6f\n", LBOX)
    @printf("  mean density n0           = %.6f\n", data.n0)
    @printf("  healing length ξ          = %.6f\n", data.ξ)
    @printf("  dx                        = %.6f\n", data.dx)
    @printf("  ξ/4                       = %.6f\n", data.ξ / 4)
    @printf("  2π/ξ                      = %.6f\n", 2π / data.ξ)
    @printf("  KZ construction band      = [%.6f, %.6f]\n", data.kmin, data.kmax)
    @printf("  plotted k range           = [%.6f, %.6f]\n", data.k[1], data.k[end])
    @printf("  Nyquist π/dx              = %.6f\n", data.knyquist)
    @printf("  particle number check     = %.6f\n", data.Ncheck)

    fig = make_plot(data)
    outfile = joinpath(@__DIR__, "inverse_particle_kz_powerlaw_3d.png")
    savefig(fig, outfile)
    println("Saved plot to " * outfile)
end

main()
