# FAQ

### How can I save a solution every n steps, or at specific parameter values?

You can use the callback `finaliseSolution` in the function call `continuation`. For example, you can use something like this to save all steps

```julia
function mySave(u, tau, step, contResult, personaldata)
	push!(personaldata, u)
end
```
and pass it like `continuation(F, J, u, p0, finaliseSolution = (z, tau, step, contResult; k...) -> mySave(z, tau, step, contResult, myData))`

### The Fold / Hopf Continuation does not work, why?

This requires some precise computations. Have you tried passing the expression of the Jacobian instead of relying on finite differences.

### What is the parameter `theta` about in `ContinuationPar`?

See the description of [continuation](https://rveltz.github.io/BifurcationKit.jl/dev/library/#Continuation-1) on the page [Library](https://rveltz.github.io/BifurcationKit.jl/dev/library/).

### How can I change the preconditioner during computations?

The easiest way to achieve this is by using the callbacks provided by `newton` and `continuation`. See the documentation about these two methods. See also the example [2d Ginzburg-Landau equation (finite differences)](@ref)

### How can I implement my own bifurcation detection method?

You can use the callback `finaliseSolution` but the best way is probably to use the [Iterator Interface](@ref) to inject your code anywhere in the continuation procedure. 

### How do I dissociate the computation of eigenvalues from the jacobian that I passed?

Sometimes, for example when implementing boundary conditions, you pass a jacobian `J` but the eigenvalues, and the bifurcation points are not simply related to `J`. One way to bypass this issue is to define a new eigensolver `<: AbstractEigenSolver` and pass it to the `NewtonPar` field `eigsolver`. This is done for example in `example/SH2d-fronts-cuda.jl`.

### How can I print the eigenvalues during `continuation`?

You can print the eigenvalues using the following callback:

```juliaw
finaliseSolution = (z, tau, step, contResult; k...) -> begin
		BK.haseigenvalues(contResult) && Base.display(contResult.eig[end].eigenvals)
		return true
	end,
```

### How can I reject a Newton Step?

You can reject a newton step by passing to `continuation` the argument `callbackN` 

```julia
function mycallback(x, f, J, res, iteration, itlinear, options; kwargs...)
	# stop Newton algo if residual too large
	if res > 1e2
		@warn "Reject Newton step!!"
		return false
	end
	return true
end
```

### How do I stop `continuation`?

Using the argument `finaliseSolution` in `continuation`. Simply make this function `finaliseSolution` return false.

### How do I compute both sides of a branch?

Instead of using two calls to `continuation`, you can pass the keyword `bothside=true` to `continuation`