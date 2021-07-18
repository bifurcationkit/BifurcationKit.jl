# Linear solvers (LS)

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
    
!!! tip "Different linear solvers"
    By tuning the options of `GMRESKrylovKit`, you can select CG, GMRES... see [KrylovKit.jl](https://jutho.github.io/KrylovKit.jl/stable/man/linear/#KrylovKit.linsolve).
    
!!! note "Other solvers"
    It is very straightforward to implement the Conjugate Gradients from [IterativeSolvers.jl](https://juliamath.github.io/IterativeSolvers.jl/dev/linear_systems/cg/) by copying the interface done for `gmres`. Same goes for `minres`,... Not needing them, I did not implement this.

## Preconditioner

 Preconditioners should be considered when using Matrix Free methods such as GMRES. `GMRESIterativeSolvers` provides a very simple interface for using them. For `GMRESKrylovKit`, we implemented a left preconditioner. Note that, for `GMRESKrylovKit`, you are not restricted to use `Vector`s anymore. Finally, here are some packages to use preconditioners

- [IncompleteLU.jl](https://github.com/haampie/IncompleteLU.jl) an ILU like preconditioner
- [AlgebraicMultigrid.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl) Algebraic Multigrid (AMG) preconditioners. This works especially well for symmetric positive definite matrices.
- [Preconditioners.jl](https://github.com/mohamed82008/Preconditioners.jl) A convenient interface to conveniently called most of the above preconditioners using a single syntax.
- We provide a preconditioner based on deflation of eigenvalues (also called preconditioner based on Leading Invariant Subspaces) using a partial Schur decomposition. There are two ways to define one *i.e.* [`PrecPartialSchurKrylovKit`](@ref) and [`PrecPartialSchurArnoldiMethod`](@ref). 

!!! tip "Using Preconditioners"
    Apart from setting a preconditioner for a linear solver, it can be advantageous to change the preconditioner during computations, *e.g.* during a call to `continuation` or `newton`. This can be achieved by taking advantage of the callbacks to these methods. See the example [2d Ginzburg-Landau equation (finite differences)](@ref).