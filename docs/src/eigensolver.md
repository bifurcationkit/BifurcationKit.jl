# Eigen solvers (Eig)

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
- Default `eigen` Julia eigensolver for matrices. You can create it via `eig = DefaultEig()`. Note that you can also specify how the eigenvalues are ordered (by decreasing real part by default). You can then compute 3 eigenelements of `J` like `eig(J, 3)`.
- Eigensolver from `Arpack.jl`. You can create one via `eigsolver = EigArpack()` and pass appropriate options (see [Arpack.jl](https://github.com/JuliaLinearAlgebra/Arpack.jl)). For example, you can compute eigenvalues using Shift-Invert method with shift `σ` by using `EigArpack(σ, :LR)`. Note that you can specify how the eigenvalues are ordered (by decreasing real part by default). Finally, this method can be used for (sparse) matrix or Matrix-Free formulation. For a matrix `J`, you can compute 3 eigen-elements using `eig(J, 3)`. In the case of a Matrix-Free jacobian `dx -> J(dx)`, you need to tell the eigensolver the dimension of the state space by giving an example of vector: `eig = EigArpack(v0 = zeros(10))`. You can then compute 3 eigen-elements using `eig(dx -> J(dx), 3)`. 
- Eigensolver from `KrylovKit.jl`. You create one via `eigsolver = EigKrylovKit()` and pass appropriate options (see [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl)). This method can be used for (sparse) matrix or Matrix-Free formulation. In the case of a matrix `J`, you can create a solver like this `eig = EigKrylovKit()`. Then, you compute 3 eigen-elements using `eig(J, 3)`. In the case of a Matrix-Free jacobian `dx -> J(dx)`, you need to tell the eigensolver the dimension of the state space by giving an example of vector: `eig = EigKrylovKit(x₀ = zeros(10))`. You can then compute 3 eigen-elements using `eig(dx -> J(dx), 3)`.
- Eigensolver from `ArnoldiMethod.jl`. You can create one via `eigsolver = EigArnoldiMethod()` and pass appropriate options (see [ArnoldiMethod.jl](https://github.com/haampie/ArnoldiMethod.jl)). For example, you can compute eigenvalues using the Shift-Invert method with shift `σ` by using `EigArpack(σ, LR())`. Note that you can also specify how the eigenvalues are ordered (by decreasing real part by default). In the case of a matrix `J`, you can create a solver like `eig = EigArpack()`. Then, you compute 3 eigen-elements using `eig(J, 3)`. In the case of a Matrix-Free jacobian `dx -> J(dx)`, you need to tell the eigensolver the dimension of the state space by giving an example of vector: `eig = EigArpack(x₀ = zeros(10))`. You can then compute 3 eigen-elements using `eig(dx -> J(dx), 3)`. 
