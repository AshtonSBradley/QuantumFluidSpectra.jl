# Script to create the test wavefunction
# Dynamics not required: imprint an exact GPE vortex core onto a Thomas-Fermi state

using VortexDistributions, Plots, Plots.Measures, JLD2, LaTeXStrings
using SpecialFunctions
using QuantumFluidSpectra

pgfplotsx()
function new_plot(;size=(400,200))
    fs,tfs,lw = 12,8,1.5
    p = plot(legend=:topright,lw=lw,size=size,frame=:box,
    foreground_color_legend = nothing,
    xtickfontsize=tfs,ytickfontsize=tfs,xguidefontsize=fs,
    tickfont=font(fs, "Times"),
    yguidefontsize=fs,legendfontsize=tfs)
    return p
end


#square domain
L = 22
N = 1024
μ = 30.0  
g = 0.1
w = 1

## useful methods
V(x,y) = 0.5*w^2*(x^2 + y^2)
ψ0(x,y,μ,g) = sqrt(μ/g)*sqrt(max(1.0-V(x,y)/μ,0.0)+im*0.0)
healinglength(x,y,μ,g) = 1/sqrt(g*abs2(ψ0(x,y,μ,g)))
R(w) = sqrt(2*μ/w^2)
Rtf = R(1)
ξ = healinglength(0,0.,μ,g)
x = LinRange(-L/2,L/2,N)
y = x
X = (x,y)

## 
ψ = ψ0.(x,y',μ,g)
heatmap(x,y,abs.(ψ),aspect_ratio=1.0)

## Imprint vortex
import VortexDistributions:Λ

pv = PointVortex(0,0.,1)
v1 = ScalarVortex(ξ,pv)
psi = Torus(copy(ψ),x,y)
vortex!(psi,v1)

heatmap(x,y,abs.(psi.ψ),aspect_ratio=1.0)

# @save joinpath(@__DIR__,"test_psi.jld2") ψv x y 
@load joinpath(@__DIR__,"test_psi.jld2") ψv x y 

## spectral analysis 
psiv = Psi(psi.ψ,X,K)
kx,ky = kvecs((L,L),(N,N))
K = (kx,ky)
kR = 2*pi/Rtf
kxi = 2*pi/ξ

kmin = 0.1kR
kmax = 1.2kxi; #kxi/5
Np = 1000
k = LinRange(kmin,kmax,Np)

## calc spec
# εki = incompressible_spectrum(k,psiv)
# @save joinpath(@__DIR__,"test_psi.jld2") ψv x y εki

@load joinpath(@__DIR__,"test_psi.jld2") ψv x y εki

## Fig 3 (a) ui power spectrum plot
ep3a = new_plot()
plot!(k*ξ,εki*ξ/μ,scale=:log10,lw=1,legend=:topright,label=false,minorticks=true,grid=false)
xlims!(extrema(k*ξ)...)
ylims!(1e-3,1e2)
xlabel!(L"k \xi")
ylabel!(L"\varepsilon^i_k(k)\xi/\mu",grid=false)
vline!([1.0],label=false,line=(1,:gray))
vline!([2π],label=false,line=(1,:gray))
vline!([kR*ξ],label=false,line=(1,:gray))
annotate!(0.12,4e-2,text(L"\frac{2\pi}{R}",10))
annotate!(.83,4e-2,text(L"\frac{1}{\xi}",10))
annotate!(5.1,4e-2,text(L"\frac{2\pi}{\xi}",10))

# Analytic form [PRX]
f(x) = x*(besseli(0,x)*besselk(1,x)-besseli(1,x)*besselk(0,x))
FΛ(x) = f(x/2/Λ)^2/x

plot!(k*ξ,FΛ.(k*ξ),line=(3,:red,.3),label=false)

# joinpath(@__DIR__,"central_vortex_εik.pdf") |>  savefig

