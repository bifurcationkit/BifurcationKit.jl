# Automatic Bifurcation diagram computation

!!! info "Experimental"
    This feature is still experimental. It has not been tested thoroughly.
    
Thanks to the functionality presented in this part, we can compute the bifurcation diagram of a system recursively and **fully automatically**. More precisely, the function `bifurcationdiagram` allows to:

- compute a branch $\gamma$ of equilibria
- detect all bifurcations on the branch
- recursively compute the branches emanating from the **simple** branch points on $\gamma$.

## Pitfalls
 
 For now, there is no way to decide if two branches $\gamma_1,\gamma_2$ are the same. As a consequence:

- there is no loop detection. Hence, if the branch $\gamma$ has a component akin to a circle, you may experience a large number of branches
- if the bifurcation diagram itself has loops (see example below), you may experience a large number of branches

!!! warning "Memory"
    The whole diagram is stored in RAM and you might be careful computing it on GPU. We'll add a file system for this in the future. 

## Basic example with simple branch points

```julia
using Revise, Plots
using BifurcationKit, Setfield, ForwardDiff
const BK = BifurcationKit

function FbpSecBif(u, p)
	return @. -u * (p + u * (2-5u)) * (p -.15 - u * (2+20u))
end

dFbpSecBif(x,p)         =  ForwardDiff.jacobian( z-> FbpSecBif(z,p), x)
d1FbpSecBif(x,p,dx1)         = D((z, p0) -> FbpSecBif(z, p0), x, p, dx1)
d2FbpSecBif(x,p,dx1,dx2)     = D((z, p0) -> d1FbpSecBif(z, p0, dx1), x, p, dx2)
d3FbpSecBif(x,p,dx1,dx2,dx3) = D((z, p0) -> d2FbpSecBif(z, p0, dx1, dx2), x, p, dx3)
jet = (FbpSecBif, dFbpSecBif, d2FbpSecBif, d3FbpSecBif)

diagram = bifurcationdiagram(jet..., 
	# initial point and parameter
	[0.0], -0.2, 
	# specify the continuation parameter
	(@lens _), 
	# very important parameter. This specifies the maximum amount of recursion
	# when computing the bifurcation diagram. It means we allow computing branches of branches 
	# at most in the present case.
	2,
	(args...) -> setproperties(opts_br; pMin = -1.0, pMax = .3, ds = 0.001, dsmax = 0.005, nInversion = 8, detectBifurcation = 3,dsminBisection =1e-18, tolBisectionEigenvalue=1e-11, maxBisectionSteps=20, newtonOptions = (@set opt_newton.verbose=false));
	printSolution = (x, p) -> x[1])
```

This gives

```julia
julia> diagram
Bifurcation diagram. Root branch (level 1) has 4 children and is such that:
Branch number of points: 76
Branch of Equilibrium
Bifurcation points:
 (ind_ev = index of the bifurcating eigenvalue e.g. `br.eig[idx].eigenvals[ind_ev]`)
- #  1,      bp point around p ≈ 0.00000281, step =  31, eigenelements in eig[ 32], ind_ev =   1 [converged], δ = ( 1,  0)
- #  2,      bp point around p ≈ 0.15000005, step =  53, eigenelements in eig[ 54], ind_ev =   1 [converged], δ = (-1,  0)
```

You can plot the diagram like `plot(bdiag; putbifptlegend=false, markersize=2, plotfold=false, title = "#branches = $(size(bdiag))")` and it gives:

![](diagram1d.png)

## Additional information
