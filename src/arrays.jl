"""
    x = xvec(λ,N)

Create `x` values with periodicity for box specified by length `λ`, using `N` points.
"""
xvec(L,N) = LinRange(-L/2,L/2,N+1)[2:end] |> collect

"""
    k = kvec(L,N)

Create `k` values with correct periodicity for box specified by length `λ` for number of points `N`.
"""
kvec(L,N) = fftfreq(N)*N*2*π/L |> Vector


"""
    X = xvecs(L,N)

Create a tuple containing the spatial coordinate array for each spatial dimension.
"""
function xvecs(L,N)
    return ntuple(i -> xvec(L[i],N[i]), length(L))
end

"""
    K = kvecs(L,N)

Create a tuple containing the spatial coordinate array for each spatial dimension.
"""
function kvecs(L,N)
    return ntuple(i -> kvec(L[i],N[i]), length(L))
end

"""
    k² = k2(K)

Create the kinetic energy array `k²` on the `k`-grid defined by the tuple `K`.
"""
function k2(K)
    kind = Iterators.product(K...)
    return map(k-> sum(abs2.(k)),kind)
end

"""
    X,K,dX,dK = xk_arrays(L,N)

Create all `x` and `k` arrays for box specified by tuples `L=(Lx,...)` and `N=(Nx,...)`.
For convenience, differentials `dX`, `dK` are also reaturned. `L` and `N` must be tuples of equal length.
"""
function xk_arrays(L,N)
    @assert length(L) == length(N)
    X = xvecs(L,N)
    K = kvecs(L,N)
    dX = ntuple(j -> X[j][2] - X[j][1], length(X))
    dK = ntuple(j -> K[j][2] - K[j][1], length(K))
    return X,K,dX,dK
end

"""
    Dx,Dk = dfft(x,k)

Measures that make `fft`, `ifft` 2-norm preserving.
Correct measures for mapping between `x`- and `k`-space.
"""
function dfft(x,k)
    dx = x[2]-x[1]; dk = k[2]-k[1]
    Dx = dx/sqrt(2*pi)
    Dk = length(k)*dk/sqrt(2*pi)
    return Dx, Dk
end

"""
    DX,DK = fft_differentials(X,K)

Evalutes tuple of measures that make `fft`, `ifft` 2-norm preserving for each
`x` or `k` dimension.
"""
function fft_differentials(X,K)
    DX = ntuple(i -> dfft(X[i],K[i])[1], length(X))
    DK = ntuple(i -> dfft(X[i],K[i])[2], length(X))
    return DX,DK
end
