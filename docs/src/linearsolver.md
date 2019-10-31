# Linear solvers

The linear solvers are subtypes of `AbstractLinearSolver`. Basically, one must provide a way of inverting the Jacobian `J` or computing `J \ x`.

Here is an example of the simplest of them (see `src/LinearSolver.jl`) to give you an idea, the backslash operator:

```julia
struct DefaultLS <: AbstractLinearSolver end

function (l::DefaultLS)(J, rhs)
	return J \ rhs, true, 1
end
```

Note that for `newton` to work, you must return 3 arguments. The first one is the result, the second one is whether the computation was successful and the third is the number of iterations required to perform the computation.

You can call it like (and it will be called like this in [`newton`](@ref))

```julia
ls = DefaultLS()
ls(rand(2,2), rand(2))
```

You can instead define `struct myLinearSolver <: AbstractLinearSolver end` and write `(l::myLinearSolver)(J, x)` where this function would implement GMRES or whatever you prefer.

## List of implemented solvers
- GMRES from `IterativeSolvers.jl`. You can call it via `linsolver = GMRES_IterativeSolvers()` and pass appropriate options.
- GMRES from `KrylovKit.jl`. You can call it via `linsolver = GMRES_KrylovKit{Float64}()` and pass appropriate options.

# Eigen solvers

The eigen solvers are subtypes of `AbstractEigenSolver`. Basically, one must provide a way of computing the eigen elements of the Jacobian `J`.

Here is an example of the simplest of them (see `src/EigSolver.jl`) to give you an idea:

```julia
struct DefaultEig <: AbstractEigenSolver end

function (l::DefaultEig)(J, nev::Int64)
	# I put Array so we can call it on small sparse matrices
	F = eigen(Array(J))
	I = sortperm(F.values, by = x-> real(x), rev = true)
	nev2 = min(nev, length(I))
	return F.values[I[1:nev2]], F.vectors[:, I[1:nev2]], 1
end
```

!!! warning "Eigenvalues"
    The eigenvalues must be ordered by increasing real part for the detection of bifurcations to work properly.
	
!!! note "Eigenvectors"
    The eigenvectors must be a 2d array for the simplified calls `newtonHopf` and `newtonFold` to work properly.

## List of implemented solvers
- Solver from `KrylovKit.jl`. You can call it via `eigsolver = eig_KrylovKit{Float64}()` and pass appropriate options.
- Matrix-Free Solver from `KrylovKit.jl`. You can call it via `eigsolver = eig_MF_KrylovKit{Float64, typeof(u0)}(x₀ = u0)` and pass appropriate options.