# FAQ

## How can I save a solution every n steps, or at specific parameter values?

You can use the callback `finaliseSolution` in the function call `continuation`. For example, you can use something like this to save all steps

```julia
function mySave(u, tau, step, contResult, personaldata)
	push!(personaldata, u)
end
```
and pass it like `continuation(F, J, u, p0, finaliseSolution = (z, tau, step, contResult) -> mySave(z, tau, step, contResult, myData))`

## The Fold / Hopf Continuation does not work, why?

This requires some precise computations. Have you tried passing the expression of the Jacobian instead of relying on finite differences.

## What is the parameter `theta` about in `ContinuationPar`?

See the description of `continuation` on the page Library.

## How can I change the preconditioner during computations?

The easiest way to achieve this is by using the callbacks provided by `newton` and `continuation`. See the documentation about these two methods. See also the example [Complex Ginzburg-Landau 2d](@ref)

## How can I implement my own bifurcation detection method?

You can use the callback `finaliseSolution` but the best way is probably to use the [Iterator Interface](@ref) to inject your code anywhere in the continuation procedure. 

## How do I dissociate the computation of eigenvalues from the jacobian that I passed?

Sometimes, for example when implementing boundary conditions, you pass a jacobian `J` but the eigenvalues, and the bifurcation points are not simply related to `J`. One way to do bypass this issue is to define a new eigensolver `<: AbstractEigenSolver` and pass it to `NewtonPar` field `eigsolver`. This is done for example in `example/SH2d-fronts-cuda.jl`.

## How can I print the eigenvalues?

You can print the eigenvalues using the following callback:

```julia
finaliseSolution = (z, tau, step, contResult) -> 
		(Base.display(contResult.eig[end].eigenvals) ;true)
```