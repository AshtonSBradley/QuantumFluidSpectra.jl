# QuantumFluidSpectra.jl

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://AshtonSBradley.github.io/QuantumFluidSpectra.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://AshtonSBradley.github.io/QuantumFluidSpectra.jl/dev) -->
[![Build Status](https://github.com/AshtonSBradley/QuantumFluidSpectra.jl/workflows/CI/badge.svg)](https://github.com/AshtonSBradley/QuantumFluidSpectra.jl/actions)
[![Coverage](https://codecov.io/gh/AshtonSBradley/QuantumFluidSpectra.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/AshtonSBradley/QuantumFluidSpectra.jl)

This package provides methods to calculate energy spectra of compressible quantum fluids described by a wavefunction, including dilute-gas Bose-Einstein condensates, polariton BEC, and quantum fluids of light. 


Fast, accurate, and flexible spectral analysis provides a wealth of information about nonlinear quantum fluid dynamics. 

We rely on Fourier spectral methods throughout. The user provides a wavefunction and minimal information about the spatial domain. 

## Install

```julia
julia> ]add QuantumFluidSpectra
```
The setup is described below. 

<details><summary><b>Create Field</b></summary>

```julia
# Create arrays including `x` and `k` grids

    n = 100
    L = (1,1)
    N = (n,n)
    X,K,dX,dK = xk_arrays(L,N) # setup domain
```
```julia
# make a test field
    ktest = K[1][2] # pick one of the `k` values
    ψ = @. exp(im*ktest*X[1]*one.(X[2]'))
    psi = Psi(ψ,X,K) # make field object with required arrays.
```
</details>
<details><summary><b>Power spectra and correlations</b></summary>
To evaluate the incompressible power spectral density on a particular k grid:
    
```julia 
k = LinRange(0.05,10,300) # can be anything
εki = incompressible_spectrum(k,psi)
```
    
The (angle-averaged) two-point correlator of the incompressible velocity field may then be calculated by 
```
r = LinRange(0,10,300) # can be anything
gi = gv(r,k,εki) # pass k vals on which εki is defined
```
See the citation below for details. 
</details>

## Example: central vortex in a 2D Bose-Einstein condensate
For creation script, see `/example_figure/test_2Dtrap_vortex.jl`.

<img src="/example_figure/central_vortex.png" width="600">

to reproduce Figure 3(a) of [https://arxiv.org/abs/2112.04012](https://arxiv.org/abs/2112.04012}).

# Citation
If you use `QuantumFluidSpectra.jl` please cite the paper

```bib
@article{PhysRevA.106.043322,
  title = {Spectral analysis for compressible quantum fluids},
  author = {Bradley, Ashton S. and Kumar, R. Kishor and Pal, Sukla and Yu, Xiaoquan},
  journal = {Phys. Rev. A},
  volume = {106},
  issue = {4},
  pages = {043322},
  numpages = {15},
  year = {2022},
  month = {Oct},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevA.106.043322},
  url = {https://link.aps.org/doi/10.1103/PhysRevA.106.043322}
}
```
