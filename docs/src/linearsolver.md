# Linear solvers

The linear solvers are subtypes of `LinearSolver`. Basically, one must provide a way of inverting the Jacobian `J` or computing `J \ x`.

Here is an example of the simplest of them (see `src/LinearSolver.jl`) to give you an idea, the backslash operator:

```julia
struct Default <: LinearSolver end

# solves the equation J * out = x
function (l::Default)(J, x)
    return J \ x, true, 1
end
```

Note that for `newton` to work, you must return 3 arguments. The first one is the result, the second one is whether the computation was successful and the third is the number of iterations required to perform the computation.

You can call it like (and it will be called like this in [`newton`](@ref))

```julia
ls = Default()
ls(rand(2,2), rand(2))
```

You can instead define `struct myLinearSolver <: LinearSolver end` and write `(l::myLinearSolver)(J, x)` where this function would implement GMRES or whatever you prefer.

# Eigen solvers

The eigen solvers are subtypes of `EigenSolver`. Basically, one must provide a way of computing the eigen elements of the Jacobian `J`.

Here is an example of the simplest of them (see `src/EigSolver.jl`) to give you an idea:

```julia
@with_kw struct Default_eig <: EigenSolver
    dim  = 200
    maxiter = 100
end

function (l::Default_eig)(J, nev::Int64)
    F = eigen(Array(J))
    I = sortperm(F.values, by = x-> real(x), rev = true)
    return F.values[I[1:nev]], F.vectors[:, I[1:nev]]
end
```

!!! warning "Eigenvalues"
    The eigenvalues must be ordered by increasing real part for the detection of bifurcations to work properly.
	
!!! note "Eigenvectors"
    The eigenvectors must be a 2d array for the simplified calls `newtonHopf` and `newtonFold` to work properly.
