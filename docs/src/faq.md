# FAQ

## How can I save a solution every n steps, or at specific parameter values?

You can use the callback `finaliseSolution` in the function call `continuation`. For example, you can use something like this to save all steps

```julia
mySave(u, tau, step, contResult, personaldata)
	push!(personaldata, u)
end
```
and pass it like `continuation(F,J,u,p0, finaliseSolution = (z, tau, step, contResult) -> mySave(z, tau, step, contResult, myData))`

## The Fold / Hopf Continuation does not work, why?

This requires some precise computations. Have you tried passing the expression of the Jacobian instead of relying on finite differences.

## What is the parameter `theta` about in `ContinuationPar`?

See the description of `continuation` on the page Library.