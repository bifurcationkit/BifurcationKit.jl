# 1d Swift-Hohenberg equation (Automatic)

```@contents
Pages = ["Swift-Hohenberg1d.md"]
Depth = 3
```

In this tutorial, we will see how to compute automatically the bifurcation diagram of the 1d Swift-Hohenberg equation

$$-(I+\Delta)^2 u+l\cdot u +\nu u^3-u^5 = 0\tag{E}$$

with Dirichlet boundary conditions. We use a Sparse Matrix to express the operator $L_1=(I+\Delta)^2$. We start by loading the packages:

```julia
using Revise
using SparseArrays, LinearAlgebra, DiffEqOperators, Setfield, Parameters
using BifurcationKit
using Plots
const BK = BifurcationKit
```

We then define a discretization of the problem

```julia
# define a norm
norminf(x) = norm(x, Inf64)

# discretisation
Nx = 200; Lx = 6.;
X = -Lx .+ 2Lx/Nx*(0:Nx-1) |> collect
hx = X[2]-X[1]

# boundary condition
Q = Dirichlet0BC(hx |> typeof)
Dxx = sparse(CenteredDifference(2, 2, hx, Nx) * Q)[1]
Lsh = -(I + Dxx)^2

# functional of the problem
function R_SH(u, par)
	@unpack l, ν, L1 = par
	out = similar(u)
	out .= L1 * u .+ l .* u .+ ν .* u.^3 - u.^5
end

# Jacobian of the function
Jac_sp(u, par) = par.L1 + spdiagm(0 => par.l .+ 3 .* par.ν .* u.^2 .- 5 .* u.^4)

# second derivative
d2R(u,p,dx1,dx2) = @. p.ν * 6u*dx1*dx2 - 5*4u^3*dx1*dx2

# third derivative
d3R(u,p,dx1,dx2,dx3) = @. p.ν * 6dx3*dx1*dx2 - 5*4*3u^2*dx1*dx2*dx3

# jet associated with the functional
jet = (R_SH, Jac_sp, d2R, d3R)

# parameters associated with the equation
parSH = (l = -0.7, ν = 2., L1 = Lsh)
```

We then choose the parameters for [`continuation`](@ref) with precise detection of bifurcation points by bisection:

```julia
optnew = NewtonPar(verbose = true, tol = 1e-12)
opts = ContinuationPar(dsmin = 0.0001, dsmax = 0.01, ds = 0.01, pMax = 1.,
	newtonOptions = setproperties(optnew; maxIter = 30, tol = 1e-8), 
	maxSteps = 300, plotEveryStep = 40, 
	detectBifurcation = 3, nInversion = 4, tolBisectionEigenvalue = 1e-17, dsminBisection = 1e-7)
```

Before we continue, it is useful to define a callback (see [`continuation`](@ref)) for [`newton`](@ref) to avoid spurious branch switching. It is not strictly necessary for what follows. 

```julia
function cb(x,f,J,res,it,itl,optN; kwargs...)
	_x = get(kwargs, :z0, nothing)
	fromNewton = get(kwargs, :fromNewton, false)
	if ~fromNewton
		# if the residual is too large or if the parameter jump
		# is too big, abord continuation step
		return norm(_x.u - x) < 20.5 && abs(_x.p - kwargs[:p]) < 0.05
	end
	true
end
```

Next, we specify the arguments to be used during continuation, such as plotting function, tangent predictors, callbacks...

```julia
args = (verbosity = 3,
	plot = true,
	linearAlgo  = MatrixBLS(),
	plotSolution = (x, p;kwargs...)->(plot!(X, x; ylabel="solution", label="", kwargs...)),
	callbackN = cb, halfbranch = true,
	)
```

Depending on the level of recursion in the bifurcation diagram, we change a bit the options as follows

```julia
function optrec(x, p, l; opt = opts)
	level =  l
	if level <= 2
		return setproperties(opt; maxSteps = 300, detectBifurcation = 3, 
			nev = Nx, detectLoop = false)
	else
		return setproperties(opt; maxSteps = 250, detectBifurcation = 3, 
			nev = Nx, detectLoop = true)
	end
end
```

!!! tip "Tuning"
    The function `optrec` modifies the continuation options `opts` as function of the branching `level`. It can be used to alter the continuation parameters inside the bifurcation diagram.
    
We are now in position to compute the bifurcation diagram

```julia
# initial condition
sol0 = zeros(Nx)

diagram = @time bifurcationdiagram(jet..., 
	sol0, (@set parSH.l = -1.), (@lens _.l), 
	# here we specify a maximum branching level of 4
	4, optrec; args...)
```  

After ~700s, you can plot the result  

```julia
plot(diagram;  plotfold = false,  
	markersize = 2, putspecialptlegend = false, xlims=(-1,1))
title!("#branches = $(size(diagram))")
```	

![](BDSH1d.png)

Et voilà!

## Exploration of the diagram

The bifurcation diagram `diagram` is stored as tree:

```julia
julia> diagram
Bifurcation diagram. Root branch (level 1) has 5 children and is such that:
Branch number of points: 146
Branch of Equilibrium
Parameters l from -1.0 to 1.0
Special points:
 (ind_ev = index of the bifurcating eigenvalue e.g. `br.eig[idx].eigenvals[ind_ev]`)
- #  1,    bp at l ≈ +0.00729225 ∈ (+0.00728880, +0.00729225), |δp|=3e-06, [converged], δ = ( 1,  0), step =  72, eigenelements in eig[ 73], ind_ev =   1
- #  2,    bp at l ≈ +0.15169672 ∈ (+0.15158623, +0.15169672), |δp|=1e-04, [converged], δ = ( 1,  0), step =  83, eigenelements in eig[ 84], ind_ev =   2
- #  3,    bp at l ≈ +0.48386427 ∈ (+0.48385737, +0.48386427), |δp|=7e-06, [converged], δ = ( 1,  0), step = 107, eigenelements in eig[108], ind_ev =   3
- #  4,    bp at l ≈ +0.53115204 ∈ (+0.53071010, +0.53115204), |δp|=4e-04, [converged], δ = ( 1,  0), step = 111, eigenelements in eig[112], ind_ev =   4
- #  5,    bp at l ≈ +0.86889220 ∈ (+0.86887839, +0.86889220), |δp|=1e-05, [converged], δ = ( 1,  0), step = 135, eigenelements in eig[136], ind_ev =   5
```

We can access the different branches with `BK.getBranch(diagram, (1,))`. Alternatively, you can plot a specific branch:

```julia
plot(diagram; code = (1,), plotfold = false,  markersize = 2, putspecialptlegend = false, xlims=(-1,1))
```

![](BDSH1d-1.png)
