using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=joinpath(@__DIR__, ".."))

using QuantumFluidSpectra
using Plots
using Plots.Measures
using Printf
using Random

gr()

"""
Construct a 3D random-phase box-mode cascade state

    ψ(r) = √n0 + Σ a_n φ_n(r),

with hard-wall box eigenmodes

    φ_n(r) = (2/L)^(3/2)
             sin(nx π x / L) sin(ny π y / L) sin(nz π z / L),

and amplitudes

    A(k) ∝ k^(-7/6)  for a particle-flux-like state
    A(k) ∝ k^(-3/2)  for an energy-flux-like state.

This example evaluates the package's current `gpe_particle_flux` path as a particle-flux
diagnostic on a hard-wall box cascade state, including an explicit box trap in
the time-evolution call.
"""

"""
Hard-wall box trap.

Returns zero inside `|x|, |y|, |z| < L/2` and a large barrier outside.
"""
function box_trap(L; barrier=1e6)
    halfL = L / 2
    return (x, y, z, t) -> begin
        inside = ((abs.(x) .< halfL) .& (abs.(y) .< halfL)) .& (abs.(z) .< halfL)
        ifelse.(inside, zero.(x .+ y .+ z), barrier .* one.(x .+ y .+ z))
    end
end

function boxmode_random_phase_state(;
    L=1.0,
    Nxyz=(64, 64, 64),
    n0=1.0,
    g=1.0,
    kmin=nothing,
    kmax=nothing,
    cascade=:particle,
    amplitude_scale=0.03,
    seed=7,
)
    cascade in (:particle, :energy) || error("cascade must be :particle or :energy")
    rng = MersenneTwister(seed)

    Nx, Ny, Nz = Nxyz
    x = xvec(L, Nx)
    y = xvec(L, Ny)
    z = xvec(L, Nz)
    X = (x, y, z)
    K = kvecs((L, L, L), Nxyz)
    V = box_trap(L)
    ξ = 1 / sqrt(g * n0)
    halfL = L / 2
    kmin_eff = isnothing(kmin) ? 2π / L : kmin
    kmax_eff = isnothing(kmax) ? 20π / ξ : kmax

    p = cascade == :particle ? 7 / 6 : 3 / 2
    pref = (2 / L)^(3 / 2)
    nmax = floor(Int, L * kmax_eff / π)

    sx = [sin.(nx * π .* (x .+ halfL) ./ L) for nx in 1:nmax]
    sy = [sin.(ny * π .* (y .+ halfL) ./ L) for ny in 1:nmax]
    sz = [sin.(nz * π .* (z .+ halfL) ./ L) for nz in 1:nmax]

    ψ = complex.(fill(sqrt(n0), Nx, Ny, Nz))
    kmodes = Float64[]
    weights = Float64[]

    for nx in 1:nmax, ny in 1:nmax, nz in 1:nmax
        kn = (π / L) * sqrt(nx^2 + ny^2 + nz^2)
        if kmin_eff <= kn <= kmax_eff
            θ = 2π * rand(rng)
            A = amplitude_scale * kn^(-p)
            an = A * cis(θ)
            mode = pref .* reshape(sx[nx], Nx, 1, 1) .* reshape(sy[ny], 1, Ny, 1) .* reshape(sz[nz], 1, 1, Nz)
            ψ .+= an .* mode
            push!(kmodes, kn)
            push!(weights, abs2(an))
        end
    end

    return (; psi=Psi(ψ, X, K), V, ξ, g, kmodes, weights, cascade, kmin=kmin_eff, kmax=kmax_eff, n0, L, Nxyz)
end

function representative_kgrid(state; n=100)
    return collect(log10range(2π / state.L, max(state.kmax, 2*2π / state.ξ), n))
end

function make_plot(state)
    kplot = representative_kgrid(state)
    Π = gpe_particle_flux(kplot, state.psi; g=state.g, V=state.V)
    waveaction_k = wave_action(kplot, state.psi)
    ρ_k = density_spectrum(kplot, state.psi)

    pexp = state.cascade == :particle ? 7 / 3 : 3
    ref = ρ_k[1] .* (kplot ./ kplot[1]).^(-pexp)

    p1 = plot(
        xscale=:log10,
        yscale=:log10,
        xlabel="k",
        ylabel="density spectrum",
        frame=:box,
        legend=:bottomleft,
        title="Density spectrum diagnostic",
        left_margin=8mm,
        bottom_margin=6mm,
    )
    plot!(p1, kplot, ρ_k; label="density_spectrum", linewidth=2)
    plot!(p1, kplot, ref; label="k^(-$(pexp)) reference", linewidth=2, linestyle=:dash)

    p2 = plot(
        kplot,
        Π,
        xscale=:log10,
        xlabel="k",
        ylabel="Π(k)",
        frame=:box,
        legend=false,
        title="Current Π(k) particle-flux diagnostic",
        left_margin=8mm,
        bottom_margin=6mm,
    )

    p3 = plot(
        kplot,
        waveaction_k,
        xscale=:log10,
        yscale=:log10,
        xlabel="k",
        ylabel="wave action",
        frame=:box,
        legend=false,
        title="wave_action(k, ψ)",
        left_margin=8mm,
        bottom_margin=6mm,
    )

    p = plot(p1, p2, p3, layout=(1, 3), size=(1650, 440))
    outfile = joinpath(@__DIR__, "boxmode_particle_flux_cascade.png")
    savefig(p, outfile)
    println("Saved plot to " * outfile)
    return p, Π, waveaction_k
end

state = boxmode_random_phase_state(
    L=1.0,
    Nxyz=(64, 64, 64),
    n0=1.0,
    g=1.0,
    cascade=:particle,
    amplitude_scale=0.03,
    seed=7,
)

@printf("cascade                = %s\n", String(state.cascade))
@printf("box size L             = %.3f\n", state.L)
@printf("grid                   = %d × %d × %d\n", state.Nxyz...)
@printf("inertial band          = [%.6f, %.6f]\n", state.kmin, state.kmax)
@printf("healing length ξ       = %.6f\n", state.ξ)
@printf("UV target 20π/ξ        = %.6f\n", 20π / state.ξ)
@printf("occupied modes         = %d\n", length(state.kmodes))
@printf("min/max occupied k     = [%.6f, %.6f]\n", minimum(state.kmodes), maximum(state.kmodes))
@printf("Σ |a_n|²               = %.12e\n", sum(state.weights))
@printf("max |a_n|²             = %.12e\n", maximum(state.weights))
println("trap                   = hard-wall box potential")
println("diagnostic             = current Π(k) path interpreted as particle flux")

make_plot(state)
