# Snaking in the 2d Swift-Hohenberg equation
We study the following PDE

$$-(I+\Delta)^2 u+l\cdot u +\nu u^2-u^3 = 0$$

with Neumann boundary conditions. This full example is in the file `example/SH2d-fronts.jl`. This example is also treated in the MATLAB package [pde2path](http://www.staff.uni-oldenburg.de/hannes.uecker/pde2path/). We use a Sparse Matrix to express the operator $L_1=(I+\Delta)^2$

```julia
using DiffEqOperators, Setfield, Parameters
using PseudoArcLengthContinuation, LinearAlgebra, Plots, SparseArrays
const PALC = PseudoArcLengthContinuation

# helper function to plot solution
heatmapsol(x) = heatmap(reshape(x,Nx,Ny)',color=:viridis)

Nx = 151
Ny = 100
lx = 4*2pi
ly = 2*2pi/sqrt(3)

# we use DiffEqOperators to compute the Laplacian operator
function Laplacian2D(Nx, Ny, lx, ly)
	hx = 2lx/Nx
	hy = 2ly/Ny
	D2x = CenteredDifference(2, 2, hx, Nx)
	D2y = CenteredDifference(2, 2, hy, Ny)
	Qx = Neumann0BC(hx)
	Qy = Neumann0BC(hy)
	
	A = kron(sparse(I, Ny, Ny), sparse(D2x * Qx)[1]) + kron(sparse(D2y * Qy)[1], sparse(I, Nx, Nx))
	return A, D2x
end
```
We also write the functional and its Jacobian which is a Sparse Matrix

```julia
function F_sh(u, p)
	@unpack l, ν, L1 = p
	return -L1 * u .+ (l .* u .+ ν .* u.^2 .- u.^3)
end

function dF_sh(u, p)
	@unpack l, ν, L1 = p
	return -L1 .+ spdiagm(0 => l .+ 2 .* ν .* u .- 3 .* u.^2)
end
```

We first look for hexagonal patterns. This is done with

```julia
X = -lx .+ 2lx/(Nx) * collect(0:Nx-1)
Y = -ly .+ 2ly/(Ny) * collect(0:Ny-1)

# initial guess for hexagons
sol0 = [(cos(x) + cos(x/2) * cos(sqrt(3) * y/2) ) for x in X, y in Y]
	sol0 .= sol0 .- minimum(vec(sol0))
	sol0 ./= maximum(vec(sol0))
	sol0 = sol0 .- 0.25
	sol0 .*= 1.7
	heatmap(sol0',color=:viridis)

# define parameters for the PDE
Δ, D2x = Laplacian2D(Nx, Ny, lx, ly, :Neumann)
L1 = (I + Δ)^2
par = (l = -0.1, ν = 1.3, L1 = L1)

# newton corrections of the initial guess
optnewton = NewtonPar(verbose = true, tol = 1e-8, maxIter = 20)
	sol_hexa, _, _ = @time newton(
		x ->  F_sh(x, par),
		u -> dF_sh(u, par),
		vec(sol0),
		optnewton)
	println("--> norm(sol) = ",norm(sol_hexa,Inf64))
	heatmapsol(sol_hexa)
```
which produces the results

```julia
 Newton Iterations 
   Iterations      Func-count      f(x)      Linear-Iterations

        0                1     1.7391e+02         0
        1                2     5.0465e+03         1
        2                3     1.4878e+03         1
        3                4     4.3529e+02         1
        4                5     1.2560e+02         1
        5                6     3.5512e+01         1
        6                7     9.5447e+00         1
        7                8     2.1763e+00         1
        8                9     3.3503e-01         1
        9               10     7.7259e-02         1
       10               11     7.4767e-03         1
       11               12     7.9505e-05         1
       12               13     8.8395e-09         1
  1.442878 seconds (43.22 k allocations: 664.210 MiB, 1.45% gc time)
```

with `sol_hexa` being

![](sh2dhexa.png)

## Continuation and bifurcation points

We can now continue this solution as follows. We want to detect bifurcations along the branches. We thus need an eigensolver. However, if we use an iterative eigensolver, like `eig = EigArpack()`, it has trouble computing the eigenvalues. One can see that using 

```julia
# compute the jacobian
J0 = dF_sh(sol_hexa, par)

# compute 10 eigenvalues
eig(J0, 10)
```

The reason is that the jacobian operator is not very well conditioned unlike its inverse. We thus opt for the *shift-invert* method (see [Eigen solvers](@ref) for more information) with shift `0.1`:

```julia
EigArpack(0.1, :LM)
```

If we want to compute the bifurcation points along the branches, we have to tell the solver by setting `detectBifurcation = 1`. However, this won't be very precise and each bifurcation point will be located at best at the step size precision. We can use bisection to locate this points more precisely using the option `detectBifurcation = 2` (see [Detection of bifurcation points](@ref) for more information).

We are now ready to compute the branches:

```julia
optcont = ContinuationPar(dsmin = 0.0001, dsmax = 0.005, ds= -0.001, pMax = 0.00, pMin = -1.0,
	newtonOptions = setproperties(optnew; tol = 1e-9, maxIter = 15), maxSteps = 125,
	detectBifurcation = 2, nev = 40, detectFold = false, 
	dsminBisection =1e-7, saveSolEveryNsteps = 4)
	optcont = @set optcont.newtonOptions.eigsolver = EigArpack(0.1, :LM)

	br, u1 = @time PALC.continuation(
		(x, p) ->  F_sh(x, @set par.l = p),
		(x, p) -> dF_sh(x, @set par.l = p),
		sol_hexa, -0.1, optcont;
		plot = true, verbosity = 3,
		tangentAlgo = BorderedPred(),
		plotSolution = (x; kwargs...) -> (heatmap!(X, Y, reshape(x, Nx, Ny)'; color=:viridis, label="", kwargs...);ylims!(-1,1,subplot=4);xlims!(-.5,.3,subplot=4)),
		printSolution = (x, p) -> norm(x),
		normC = x -> norm(x, Inf))
```

Note that we can get some information about the branch:

```julia
julia> br
Branch number of points: 91
Bifurcation points:
-   1,      bp point around p ≈ -0.21554703, step =  27, idx =  28, ind_bif =   1 [converged]
-   2,      bp point around p ≈ -0.21551270, step =  28, idx =  29, ind_bif =   2 [converged]
-   3,      bp point around p ≈ -0.21502386, step =  29, idx =  30, ind_bif =   3 [converged]
-   4,      bp point around p ≈ -0.21290012, step =  31, idx =  32, ind_bif =   4 [converged]
-   5,      bp point around p ≈ -0.21092914, step =  32, idx =  33, ind_bif =   5 [converged]
-   6,      bp point around p ≈ -0.21008215, step =  33, idx =  34, ind_bif =   6 [converged]
-   7,      bp point around p ≈ -0.20682609, step =  35, idx =  36, ind_bif =   8 [converged]
-   8,      bp point around p ≈ -0.19985956, step =  37, idx =  38, ind_bif =   9 [converged]
-   9,      bp point around p ≈ -0.18887677, step =  40, idx =  41, ind_bif =  10 [converged]
-  10,      bp point around p ≈ -0.18104915, step =  42, idx =  43, ind_bif =  11 [converged]
```

We get the following plot during computation:

![](sh2dbrhexa.png)

!!! tip "Tip"
    We don't need to call `newton` first in order to use `continuation`.

## Snaking computed with deflation

We know that there is snaking near the left fold. Let us look for other solutions like fronts. The problem is that if the guess is not precise enough, the newton iterations will converge to the solution with hexagons `sol_hexa`. We appeal to the technique initiated by P. Farrell and use a **deflated problem** (see [`DeflationOperator`](@ref) and [`DeflatedProblem`](@ref) for more information). More precisely, we apply the newton iterations to the following functional $$u\to \frac{F_{sh}(u)}{\Pi_{i=1}^{n_s} \|u-sol_{hexa,i}\|^p + \sigma}$$
which penalizes `sol_hexa`.

```julia
# this define the above penalizing factor with p=2, sigma=1, norm associated to dot
# and the set of sol_{hexa} is of length ns=1
deflationOp = DeflationOperator(2.0,(x,y) -> dot(x,y),1.0,[sol_hexa])
optnewton = @set optnewton.maxIter = 250
outdef, _, flag, _ = @time newton(
				x ->  F_sh(x, par),
				x -> dF_sh(x, par),
				0.2vec(sol_hexa) .* vec([exp.(-(x+lx)^2/25) for x in X, y in Y]),
				optnewton,deflationOp, normN = x -> norm(x,Inf64))
		heatmapsol(outdef) |> display
		flag && push!(deflationOp, outdef)
```
which gives:

![](sh2dfrontleft.png)

Note that `push!(deflationOp, outdef)` deflates the newly found solution so that by repeating the process we find another one:

```julia
outdef, _, flag, _ = @time newton(
				x ->  F_sh(x, par),
				x -> dF_sh(x, par),
				0.2vec(sol_hexa) .* vec([exp.(-(x)^2/25) for x in X, y in Y]),
				optnewton,deflationOp, normN = x -> norm(x,Inf64))
		heatmapsol(outdef) |> display
		flag && push!(deflationOp, outdef)
```

![](sh2dfrontmiddle.png)

Again, repeating this from random guesses, we find several more solutions, like for example

![](sh2dsol4.png)

![](sh2dsol5.png)

We can now continue the solutions located in `deflationOp.roots`

```julia
br, _ = @time continuation(
	(x, p) ->  F_sh(x, @set par.l = p),
	(x, p) -> dF_sh(x, @set par.l = p),,
	deflationOp[2], -0.1, optcont;
	plot = true, 
	plotSolution = (x; kwargs...) -> (heatmap!(X,Y,reshape(x,Nx,Ny)'; color=:viridis, label="", kwargs...)))
```

and using `plotBranch(br)`, we obtain:

![](sh2dbranches.png)

Note that the plot provides the stability of solutions and bifurcation points. Interested readers should consult the associated file `example/SH2d-fronts.jl` in the `example` folder.