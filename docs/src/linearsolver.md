# Linear solvers

> If you provide your own linear solver, it must be a subtype of `AbstractLinearSolver` otherwise `BifurcationKit.jl` will not recognize it. See example just below. 

The linear solvers provide a way of inverting the Jacobian `J` or solving `J * x = rhs`. Such linear solver `linsolve` will be called like `sol, success, itnumber = linsolve(J, rhs)` throughout the package.

Here is an example of the simplest of them (see `src/LinearSolver.jl` for the true implementation) to give you an idea, the backslash operator:

```julia
struct DefaultLS <: AbstractLinearSolver end

function (l::DefaultLS)(J, rhs)
	return J \ rhs, true, 1
end
```

Note that for `newton` to work, the linear solver must return 3 arguments. The first one is the result, the second one is whether the computation was successful and the third is the number of iterations required to perform the computation.

You can then call it as follows (and it will be called like this in [`newton`](@ref))

```julia
ls = DefaultLS()
ls(rand(2,2), rand(2))
```

## List of implemented linear solvers
- Default `\` solver based on `LU` or `Cholesky` depending on the type of the Jacobian. This works for sparse matrices as well. You can create one via `linsolver = DefaultLS()`.
- GMRES from `IterativeSolvers.jl`. You can create one via `linsolver = GMRESIterativeSolvers()` and pass appropriate options.
- GMRES from `KrylovKit.jl`. You can create one via `linsolver = GMRESKrylovKit()` and pass appropriate options.

!!! note "Other solvers"
    It is very straightforward to implement the Conjugate Gradients from [IterativeSolvers.jl](https://juliamath.github.io/IterativeSolvers.jl/dev/linear_systems/cg/) by copying the interface done for `gmres`. Same goes for `minres`,... Not needing them, I did not implement this.

## Preconditioner

 Preconditioners should be considered when using Matrix Free methods such as GMRES. `GMRESIterativeSolvers` provides a very simple interface for using them. For `GMRESKrylovKit`, we implemented a left preconditioner. Note that, for `GMRESKrylovKit`, you are not restricted to use `Vector`s anymore. Finally, here are some packages to use preconditioners

- [IncompleteLU.jl](https://github.com/haampie/IncompleteLU.jl) an ILU like preconditioner
- [AlgebraicMultigrid.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl) Algebraic Multigrid (AMG) preconditioners. This works especially well for symmetric positive definite matrices.
- [Preconditioners.jl](https://github.com/mohamed82008/Preconditioners.jl) A convenient interface to conveniently called most of the above preconditioners using a single syntax.
- We provide a preconditioner based on deflation of eigenvalues (also called preconditioner based on Leading Invariant Subspaces) using a partial Schur decomposition. There are two ways to define one *i.e.* [`PrecPartialSchurKrylovKit`](@ref) and [`PrecPartialSchurArnoldiMethod`](@ref). 

!!! tip "Using Preconditioners"
    Apart from setting a preconditioner for a linear solver, it can be advantageous to change the preconditioner during computations, *e.g.* during a call to `continuation` or `newton`. This can be achieved by taking advantage of the callbacks to these methods. See the example [Complex Ginzburg-Landau 2d](@ref).


# Eigen solvers

> The eigen solvers must be subtypes of `AbstractEigenSolver`. 

They provide a way of computing the eigen elements of the Jacobian `J`. Such eigen solver `eigsolve` will be called like `ev, evecs, itnumber = eigsolve(J, nev; kwargs...)` throughout the package, `nev` being the number of requested eigen elements of largest real part and `kwargs` being used to send information about the algorithm (perform bisection,...).

 
Here is an example of the simplest of them (see `src/EigSolver.jl` for the true implementation) to give you an idea:

```julia
struct DefaultEig <: AbstractEigenSolver end

function (l::DefaultEig)(J, nev; kwargs...)
	# I put Array so we can call it on small sparse matrices
	F = eigen(Array(J))
	I = sortperm(F.values, by = x-> real(x), rev = true)
	nev2 = min(nev, length(I))
	return F.values[I[1:nev2]], F.vectors[:, I[1:nev2]], true, 1
end
```

!!! warning "Eigenvalues"
    The eigenvalues must be ordered by increasing real part for the detection of bifurcations to work properly.

!!! warning "Eigenvectors"
    You have to implement the method `geteigenvector(eigsolver, eigenvectors, i::Int)` for `newtonHopf` to work properly.

## Methods for computing eigenvalues
Like for the linear solvers, computing the spectrum of operators $A$ associated to PDE is a highly non trivial task because of the clustering of eigenvalues. Most methods are based on the so-called [power method](https://en.wikipedia.org/wiki/Power_iteration) but this only yields the eigenvalues with largest modulus. In case of the Laplacian operator, this can be disastrous and it is better to apply the power method to $(\sigma I-A)^{-1}$ instead. 

This method, called **Shift-invert**, is readily available for the solver `EigArpack `, see below. It is mostly used to compute interior eigenvalues. For the solver `EigKrylovKit`, one must implement its own shift invert operator, using for example `GMRESKrylovKit`.

In some cases, it may be advantageous to consider the **Cayley transform** $(\sigma I-A)^{-1}(\tau I+A)$ to focus on a specific part of the spectrum. As it is mathematically equivalent to the Shift-invert method, we did not implement it.


## List of implemented eigen solvers
- Default Julia eigensolver for matrices. You can create it via `eig = DefaultEig()`. Note that you can also specify how the eigenvalues are ordered (by decreasing real part by default). You can then compute 3 eigenelements of `J` like `eig(J, 3)`.
- Eigensolver from `Arpack.jl`. You can create one via `eigsolver = EigArpack()` and pass appropriate options (see [Arpack.jl](https://github.com/JuliaLinearAlgebra/Arpack.jl)). For example, you can compute eigenvalues using Shift-Invert method with shift `σ` by using `EigArpack(σ, :LR)`. Note that you can specify how the eigenvalues are ordered (by decreasing real part by default). Finally, this method can be used for (sparse) matrix or Matrix-Free formulation. For a matrix `J`, you can compute 3 eigen-elements using `eig(J, 3)`. In the case of a Matrix-Free jacobian `dx -> J(dx)`, you need to tell the eigensolver the dimension of the state space by giving an example of vector: `eig = EigArpack(v0 = zeros(10))`. You can then compute 3 eigen-elements using `eig(dx -> J(dx), 3)`. 
- Eigensolver from `KrylovKit.jl`. You create one via `eigsolver = EigKrylovKit()` and pass appropriate options (see [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl)). This method can be used for (sparse) matrix or Matrix-Free formulation. In the case of a matrix `J`, you can create a solver like this `eig = EigKrylovKit()`. Then, you compute 3 eigen-elements using `eig(J, 3)`. In the case of a Matrix-Free jacobian `dx -> J(dx)`, you need to tell the eigensolver the dimension of the state space by giving an example of vector: `eig = EigKrylovKit(x₀ = zeros(10))`. You can then compute 3 eigen-elements using `eig(dx -> J(dx), 3)`.
- Eigensolver from `ArnoldiMethod.jl`. You can create one via `eigsolver = EigArnoldiMethod()` and pass appropriate options (see [ArnoldiMethod.jl](https://github.com/haampie/ArnoldiMethod.jl)). For example, you can compute eigenvalues using the Shift-Invert method with shift `σ` by using `EigArpack(σ, LR())`. Note that you can also specify how the eigenvalues are ordered (by decreasing real part by default). In the case of a matrix `J`, you can create a solver like `eig = EigArpack()`. Then, you compute 3 eigen-elements using `eig(J, 3)`. In the case of a Matrix-Free jacobian `dx -> J(dx)`, you need to tell the eigensolver the dimension of the state space by giving an example of vector: `eig = EigArpack(x₀ = zeros(10))`. You can then compute 3 eigen-elements using `eig(dx -> J(dx), 3)`. 
