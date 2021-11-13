# QuantumFluidSpectra.jl

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://AshtonSBradley.github.io/QuantumFluidSpectra.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://AshtonSBradley.github.io/QuantumFluidSpectra.jl/dev) -->
[![Build Status](https://github.com/AshtonSBradley/QuantumFluidSpectra.jl/workflows/CI/badge.svg)](https://github.com/AshtonSBradley/QuantumFluidSpectra.jl/actions)
<!-- [![Coverage](https://codecov.io/gh/AshtonSBradley/QuantumFluidSpectra.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/AshtonSBradley/QuantumFluidSpectra.jl) -->

This package provides methods to calculate energy spectra of quantum fluids described by a wavefunction, including dilute-gas Bose-Einstein condensates, polariton BEC, and quantum fluids of light. 

Fast, accurate spectral analysis provides a wealth of information about nonlinear quantum fluid dynamics. 

We rely on Fourier spectral methods throughout. The user provides a wavefunction and minimal information about the spatial domain. 

<details><summary><b>Create Field</b></summary>

```julia
# Create arrays including `x` and `k` grids

    n = 100
    L = (1,1)
    N = (n,n)
    X,K,dX,dK = makearrays(L,N) # setup domain
```
```julia
# make a test field
    ktest = K[1][2] # pick one of the `k` values
    ψ = @. exp(im*ktest*X[1]*one.(X[2]'))
    psi = Psi(ψ,X,K) # make field object with required arrays.
```
</details>
<details><summary><b>Basic Usage</b></summary>

</details>