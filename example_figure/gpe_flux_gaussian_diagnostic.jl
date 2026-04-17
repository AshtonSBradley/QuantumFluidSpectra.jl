# Diagnostic script for direct GPE particle flux on trapped states.
#
# Run from the package root with, for example:
#   julia --project=. example_figure/gpe_flux_gaussian_diagnostic.jl
#
# This example uses Plots.jl for visualization. If Plots is not installed in
# your current environment, install it first with:
#   julia --project=. -e 'using Pkg; Pkg.add("Plots")'

using QuantumFluidSpectra
using Plots
using Plots.Measures
using Printf

gr()

function trapz(x, y)
    s = 0.0
    for i = 2:length(x)
        s += 0.5 * (x[i] - x[i-1]) * (y[i] + y[i-1])
    end
    return s
end

function normalize_wavefunction(ψ, X)
    if length(X) == 2
        dx = X[1][2] - X[1][1]
        dy = X[2][2] - X[2][1]
        norm2 = sum(abs2, ψ) * dx * dy
    else
        dx = X[1][2] - X[1][1]
        dy = X[2][2] - X[2][1]
        dz = X[3][2] - X[3][1]
        norm2 = sum(abs2, ψ) * dx * dy * dz
    end
    return ψ ./ sqrt(norm2)
end

function thomas_fermi_field_2d(X; g = 1.0, μ = 1.0, ω = 1.0)
    V(x, y) = 0.5 * ω^2 * (x^2 + y^2)
    ψ = @. sqrt(complex(max((μ - V(X[1], X[2]')) / g, 0.0)))
    return normalize_wavefunction(ψ, X)
end

function thomas_fermi_field_3d(X; g = 1.0, μ = 1.0, ω = 1.0)
    z3 = reshape(X[3], (1, 1, length(X[3])))
    V(x, y, z) = 0.5 * ω^2 * (x^2 + y^2 + z^2)
    ψ = @. sqrt(complex(max((μ - V(X[1], X[2]', z3)) / g, 0.0)))
    return normalize_wavefunction(ψ, X)
end

function representative_kgrid(x; n = 50)
    dx = x[2] - x[1]
    L = x[end] - x[begin] + dx
    k_from_box = π / L
    k_from_spacing = (2π / dx) / 3
    if k_from_spacing <= k_from_box
        @warn "Requested k-band is empty for this spatial grid; using sorted endpoints instead." L dx k_from_box k_from_spacing
    end
    klo = min(k_from_box, k_from_spacing)
    khi = max(k_from_box, k_from_spacing)
    return collect(log10range(klo, khi, n))
end

function summarize_case(name, k, psi; g = 1.0, V = nothing, t = 0.0)
    T = gpe_particle_transfer(k, psi; g = g, V = V, t = t)
    Πnew = gpe_particle_flux(k, psi; g = g, V = V, t = t)
    Πold = -QuantumFluidSpectra._cumulative_integral(k, T)
    dΠ = diff(Πnew) ./ diff(k)
    absdiff = abs.(Πnew .- Πold)

    println("CASE: " * name)
    @printf("  k-range                     = [%.6e, %.6e]\n", first(k), last(k))
    @printf("  Π_new[end]                  = %.12e\n", Πnew[end])
    @printf("  -trapz(T)                   = %.12e\n", -trapz(k, T))
    @printf("  Π_old[end]                  = %.12e\n", Πold[end])
    @printf("  max|Π_new - Π_old|          = %.12e\n", maximum(absdiff))
    @printf("  mean|Π_new - Π_old|         = %.12e\n", sum(absdiff) / length(absdiff))
    @printf("  max|dΠ/dk + T(right)|       = %.12e\n", maximum(abs.(dΠ .+ T[2:end])))
    @printf(
        "  rms|dΠ/dk + T(right)|       = %.12e\n",
        sqrt(sum(abs2.(dΠ .+ T[2:end])) / length(dΠ))
    )
    println()

    return Πnew
end

function diagnostic_plot()
    p2 = plot(
        xlabel = "k",
        ylabel = "Π(k)",
        xscale = :log10,
        legend = :bottomleft,
        frame = :box,
        title = "2D trapped states",
        left_margin = 8mm,
        right_margin = 6mm,
        bottom_margin = 6mm,
        size = (1250, 420),
    )
    p3 = plot(
        xlabel = "k",
        ylabel = "Π(k)",
        xscale = :log10,
        legend = :bottomleft,
        frame = :box,
        title = "3D trapped states",
        left_margin = 8mm,
        right_margin = 6mm,
        bottom_margin = 6mm,
        size = (1250, 420),
    )

    V2(x, y, t) = 0.5 .* (x .^ 2 .+ y .^ 2)
    for Lval in (8.0, 12.0)
        n = 64
        X, K, _, _ = xk_arrays((Lval, Lval), (n, n))
        x = X[1]
        k = representative_kgrid(x)
        ψg = @. exp(-(X[1]^2 + X[2]'^2))
        ψg = normalize_wavefunction(ψg, X)
        psi = Psi(complex.(ψg), X, K)
        Π = summarize_case("2D trapped gaussian, L=$(Lval)", k, psi; V = V2)
        plot!(p2, k, Π, label = "Gaussian L=$(Lval)")

        ψtf = thomas_fermi_field_2d(X; g = 1.0, μ = 1.0, ω = 1.0)
        psitf = Psi(complex.(ψtf), X, K)
        Πtf = summarize_case("2D Thomas-Fermi, L=$(Lval)", k, psitf; V = V2)
        plot!(p2, k, Πtf, linestyle = :dash, label = "TF L=$(Lval)")
    end

    V3(x, y, z, t) = 0.5 .* (x .^ 2 .+ y .^ 2 .+ z .^ 2)
    for Lval in (8.0, 12.0)
        n = 64
        X, K, _, _ = xk_arrays((Lval, Lval, Lval), (n, n, n))
        x = X[1]
        z3 = reshape(X[3], (1, 1, n))
        k = representative_kgrid(x)
        ψg = @. exp(-(X[1]^2 + X[2]'^2 + z3^2))
        ψg = normalize_wavefunction(ψg, X)
        psi = Psi(complex.(ψg), X, K)
        Π = summarize_case("3D trapped gaussian, L=$(Lval), n=$(n)", k, psi; V = V3)
        plot!(p3, k, Π, label = "Gaussian L=$(Lval)")

        ψtf = thomas_fermi_field_3d(X; g = 1.0, μ = 1.0, ω = 1.0)
        psitf = Psi(complex.(ψtf), X, K)
        Πtf = summarize_case("3D Thomas-Fermi, L=$(Lval), n=$(n)", k, psitf; V = V3)
        plot!(p3, k, Πtf, linestyle = :dash, label = "TF L=$(Lval)")
    end

    p = plot(p2, p3, layout = (1, 2), size = (1380, 460))
    outfile = joinpath(@__DIR__, "gpe_flux_gaussian_diagnostic.png")
    savefig(p, outfile)
    println("Saved plot to " * outfile)
    return p
end

diagnostic_plot()
