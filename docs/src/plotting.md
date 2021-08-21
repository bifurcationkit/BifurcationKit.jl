# Plotting

## Plotting branches

Plotting is provided by calling `Plots.jl`. It means that to plot a branch `br`, you just need to call

```julia
plot(br)
```

where `br` is a branch computed after a call to `br, = continuation(...)`. You can use the keywords provided by `Plots.jl` and the different backends. You can thus call

```julia
scatter(br)
plot!(br, label = "continuous line")
```

The available arguments specific to our plotting methods are

- `plotfold = true`: plot the fold points with black dots
- `putspecialptlegend = true`: display the legend corresponding to the bifurcation points
- `vars = nothing`: see below
- `plotstability = true`: display the stability of the branch
- `plotspecialpoints = true`: plot the special (bifurcation) points on the branch
- `branchlabel = "fold branch"`: assign label to a branch which is printed in the legend
- `linewidthunstable`: set the linewidth for the unstable part of the branch
- `linewidthstable`: set the linewidth for the stable part of the branch
- `plotcirclesbif = false` use circles to plot bifurcation points
- `applytoX = identity` apply transformation `applytoX` to x-axis
- `applytoY = identity` apply transformation `applytoY` to y-axis

If you have severals branches `br1, br2`, you can plot them in the same figure by doing

```julia
plot(br1, br2)
```

in place of

```julia
plot(br1)
plot!(br2)
```

!!! warn "Plot of bifurcation points"
    The bifurcation points for which the bisection was successful are indicated with circles and with squares otherwise.

### Choosing Variables

You can select which variables to plot using the keyword argument `vars`:

```julia
plot(br, vars = (:param, :x))
```
The available symbols are `:param, :sol, :itnewton, :ds, :theta, :step` and:

- `x` if `recordFromSolution` (see [`continuation`](@ref)) returns a `Number`.
- `x1, x2,...` if `recordFromSolution` returns a `Tuple`.
- the keys of the `NamedTuple` returned by `recordFromSolution`.

### Plotting directly using the field names

You can define your own plotting functions using the internal fields of `br` which is of type [`ContResult`](@ref). For example, the previous plot can be done as follows:

```julia
plot(br.branch.param, br.branch.x)
```

You can also plot the spectrum at a specific continuation `step::Int` by calling

```julia
# get the eigenvalues
eigvals = br.eig[step].eigenvals

# plot them in the complex plane
scatter(real.(eigvals), imag.(eigvals))
```

## Plotting bifurcation diagrams

To do this, you just need to call

```julia
plot(diagram)
```

where `diagram` is a branch computed after a call to `diagram, = bifurcationdiagram(...)`. You can use the keywords provided by `Plots.jl` and the different backends. You can thus call `scatter(diagram)`. In addition to the options for plotting branches (see above), there are specific arguments available for bifurcation diagrams

- `code` specify the part of the bifurcation diagram to plot. For example `code = (1,1,)` plots the part after the first branch of the first branch of the root branch.
- `level = (-Inf, Inf)` restrict the branching level for plotting.
