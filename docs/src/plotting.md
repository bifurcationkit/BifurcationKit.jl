# Plotting 

Plotting is provided by calling `Plots.jl`. It means that to plot a branch `br`, you just need to call 

```
plot(br)
```

where `br` is a branch computed after a call to `br,_ = continuation(...)`. You can use the keywords provided by `Plots.jl` and the different backends. You can thus call 

```
scatter(br)
plot!(br, label = "continuous line")
```

The available arguments specific to our plotting methods are 

- `plotfold = true`: plot the fold points with black dots
- `putbifptlegend = true`: display the legend corresponding to the bifurcation points
- `plotstability = true`: display the stability of the branch
- `plotbifpoints = true`: plot the bifurcation points on the branch
- `branchlabel = "fold branch"`: assign label to a branch which is printed in the legend
- `vars = nothing`: see below

If you have severals branches `br1, br2`, you can plot them in the same figure by doing 

```
plot([br1, br2])
```

in place of 

```
plot(br1)
plot!(br2)
```

## Choosing Variables

You can select which variables to plot using the keyword argument vars:

```
plot(br, vars = (:p, :sol))
```
The available symbols are `:p, :sol, :itnewton, :ds, :theta, :step`.

## Plotting directly using the field names

You can define your own plotting functions using the internal fields of `br` which is of type [`ContResult`](@ref). For example, the previous plot can be done as follows:

```
plot(br.branch[1, :], br.branch[2, :])
```

You can also plot the spectrum at a specific continuation `step::Int` by calling 

```
# get the eigenvalues
eigvals = br.eig[step].eigenvals

# plot them in the complex plane
scatter(real.(eigvals), imag.(eigvals))
```