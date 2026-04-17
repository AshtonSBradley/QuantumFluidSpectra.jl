# Script to create the test wavefunction
# Dynamics not required: imprint an exact GPE vortex core onto a Thomas-Fermi state

using SpecialFunctions,
    Plots, Plots.Measures, JLD2, LaTeXStrings, QuadGK, VortexDistributions
using QuantumFluidSpectra
import VortexDistributions: Λ, ψa
gr()

function new_plot(; size = (400, 200))
    fs, tfs, lw = 12, 8, 1.5
    p = plot(
        legend = :topright,
        lw = lw,
        size = size,
        frame = :box,
        foreground_color_legend = nothing,
        xtickfontsize = tfs,
        ytickfontsize = tfs,
        xguidefontsize = fs,
        tickfont = font(fs, "Times"),
        yguidefontsize = fs,
        legendfontsize = tfs,
    )
    return p
end

## square domain
L = 22
N = 256
μ = 30.0
g = 0.1
w = 1

## useful methods
V(x, y) = 0.5*w^2*(x^2 + y^2)
ψ0(x, y, μ, g) = sqrt(μ/g)*sqrt(max(1.0-V(x, y)/μ, 0.0)+im*0.0)
healinglength(x, y, μ, g) = 1/sqrt(g*abs2(ψ0(x, y, μ, g)))
R(w) = sqrt(2*μ/w^2)
Rtf = R(1)
ξ = healinglength(0, 0.0, μ, g)
x = LinRange(-L/2, L/2, N);
y = x
X = (x, y)
n0 = abs2.(ψ0(0, 0, μ, g))
ek_unit = pi*n0*Rtf^3/(2μ)

## 
ψ = ψ0.(x, y', μ, g)
# heatmap(x,y,abs.(ψ),aspect_ratio=1.0)


## Imprint charge 1 vortex at the origin
pv = PointVortex(0, 0.0, 1)
# v1 = ScalarVortex(ξ,pv)

# check ansatz core agrees exactly with TFv
v1 = ScalarVortex(Ansatz(ψa, ξ, Λ), pv)
psi = Torus(copy(ψ), x, y)
vortex!(psi, v1)
gr()
heatmap(x, y, abs.(psi.ψ), aspect_ratio = 1.0)

## k grid for spectra, and reference values 
kx, ky = kvecs((L, L), (N, N))
K = (kx, ky)
kR = 2*pi/Rtf
kxi = 2*pi/ξ

kmin = 0.1kR
kmax = 1.2kxi
Np = 1000
k = LinRange(kmin, kmax, Np)

εki = incompressible_spectrum(k, Psi(psi.ψ, X, K))
# calc spec and save 
# ψv = psi.ψ
# psiv = Psi(ψv,X,K)
# εki = incompressible_spectrum(k,psiv)
# @save joinpath(@__DIR__,"test_psi.jld2") ψv x y εki

#@load joinpath(@__DIR__,"test_psi.jld2") ψv x y εki


## Fig 3 (a) ui power spectrum plot
pgfplotsx()
ep3a = new_plot()

# Analytic form homog. [PRX]
f(x) = x*(besseli(0, x)*besselk(1, x)-besseli(1, x)*besselk(0, x))
FΛ(x) = f(x/2/Λ)^2/x
plot!(k*ξ, FΛ.(k*ξ), line = (1, :blue, 0.8), label = false)

# analytic spectra: vortex in trap
Tint(x, a, b) = x*sqrt((1-x^2)/(x^2+b^2))*besselj1(a*x)
Tv(a, b) = quadgk(x->Tint(x, a, b), 0.0, 1.0)[1]
Mv(x, y) = x*abs2(Tv(x, y))
ekin_v_a(k) = Mv(k*Rtf, ξ/Λ/Rtf)*ek_unit
plot!(k*ξ, ekin_v_a.(k)*ξ/μ, line = (4, :pink), label = false)


plot!(
    k*ξ,
    εki*ξ/μ,
    scale = :log10,
    line = (1, :red),
    legend = :topright,
    label = false,
    minorticks = true,
    grid = false,
)
xlims!(extrema(k*ξ)[1], ξ*kxi*1.15)
ylims!(1e-3, 1e2)
xlabel!(L"k \xi")
ylabel!(L"\varepsilon^i_k(k)\xi/\mu", grid = false)
vline!([1.0], label = false, line = (1, :gray))
vline!([2π], label = false, line = (1, :gray))
vline!([kR*ξ], label = false, line = (1, :gray))
ya = 6e-2
annotate!(0.115, ya, text(L"\frac{2\pi}{R}", 10))
annotate!(0.83, ya, text(L"\frac{1}{\xi}", 10))
annotate!(4.9, ya, text(L"\frac{2\pi}{\xi}", 10))
annotate!(5, 35, text(L"(a)", 10))
# joinpath(@__DIR__,"test_data/central_vortex_εik.pdf") |>  savefig
# gr()
plot!()

# εki 
# ekin_v_a.(k)
# maximum relative error as function of grid size 
# @256 => 1.65979 
# @512 => 0.07953
# @1024 => 0.01276 
# @2048 => 0.00337 
maximum(abs.((εki .- ekin_v_a.(k)) ./ ekin_v_a.(k)))

errk = abs.((εki .- ekin_v_a.(k)) ./ ekin_v_a.(k))
plot(k*ξ, errk, yscale = :log10)

## Two point velocity correlation [check units!]
# pgfplotsx()
ep3b = new_plot()
r = LinRange(0, 2*Rtf, 1000)
gi = gv(r, k, ekin_v_a.(k))
plot!(r/ξ, gi, label = false, grid = false)
vline!([Rtf/ξ], line = (1, :gray), label = false)
ylabel!(L"g_k^i(r)")
plot!(yticks = 0:0.5:1.0)
xlabel!(L"r/\xi")
xlims!(extrema(r/ξ)...)
annotate!(45, 0.75, text(L"R", 10))
annotate!(82, 0.92, text(L"(b)", 10))

## combined plot
l = @layout [a; b]
pc = plot(ep3a, ep3b, layout = l, size = (430, 350))
# "2d_trapvtf_combined.pdf" |> savefig
plot!()
