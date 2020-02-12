# Iterator Interface

The iterator interface gives the possibility of stepping through the numerical steps of the continuation procedure. It thus allows to inject custom monitoring function (saving, plotting, bifurcation detection, ...) at will and during the continuation run. In short, it allows to completely re-write the continuation algorithm as one sees fit and this, in a straightforward manner.

The general method `continuation` is built upon this iterator interface and we refer to the source code for a complete example of use.

## Initialization

> More information about **iterators** can be found on the [page](https://docs.julialang.org/en/v1/base/collections/#Collections-and-Data-Structures-1) of [julialang](https://docs.julialang.org/en/v1/).

The interface is set by defining an iterator, pretty much in the same way one calls [`continuation`](@ref):

```julia
iter = PALCIterable(F, J, x0, p0, opts; kwargs...)
```

## Stepping

Once an iterator `iter` has been defined, one can step through the numerical continuation using a for loop:

```julia
for state in iter
	println("Continuation step = ", state.step)
end
```

The `state::PALCStateVariables` has the following description. It is a mutable object which holds the current state of the continuation procedure from which one can step to the next state.

The for loop stops when `done(iter, state)` returns `false`. The condition which is implemented is basically that the number of iterations should be smaller than `maxIter`, that the parameters should be in `(pMin, pMax)`...

```@docs
PALCStateVariables
```

!!! tip "continuation"
    You can also call `continuation(iter)` to have access to the regular continuation method used throughout the tutorials.

## Example

We show a quick and simple example of use. Note that it is not very optimized because of the use of global variables.

```julia
using PseudoArcLengthContinuation, SparseArrays, LinearAlgebra, Plots, Setfield
const PALC = PseudoArcLengthContinuation

# define a norm
normInf = x -> norm(x, Inf)

k = 2

# functional we want to study
F = (x, p) -> (@. p + x - x^(k+1)/(k+1))

# Jacobian for the fonctional
Jac_m = (x, p) -> diagm(0 => 1  .- x.^k)


# parameters for the continuation
opts = PALC.ContinuationPar(dsmax = 0.1, dsmin = 1e-3, ds = -0.001, maxSteps = 130, pMin = -3., pMax = 3., saveSolEveryNsteps = 0, computeEigenValues = true, detectBifurcation = true, newtonOptions = NewtonPar(tol = 1e-8, verbose = true))

# we define an iterator to hold the continuation routine
iter = PALC.PALCIterable(F, Jac_m, [0.8], 1., opts; verbosity = 2)

resp = Float64[]
resu = Float64[]

# this is the PALC algorithm
for state in iter
	# we save the current solution on the branch
	push!(resu, getu(state)[1])
	push!(resp, getp(state))
end

# plot the result
plot(resp, resu; label = "", xlabel = "p")
```

and you should see:

![](iterator.png)
