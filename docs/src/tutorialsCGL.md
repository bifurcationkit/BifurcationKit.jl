# Complex Ginzburg-Landau 2d

> This example is also treated in the MATLAB library [pde2path](http://www.staff.uni-oldenburg.de/hannes.uecker/pde2path/).

We look at the Ginzburg-Landau equations in 2d. The code is very similar to the Brusselator example except that some special care has to be taken in order to cope with the "high" dimensionality of the problem.

Note that we try to be pedagogical here. Hence, we may write "bad" code that we improve later. Finally, we could use all sort of tricks to take advantage of the specificity of the problem. Rather, we stay quite close to the example in the MATLAB library [pde2path](http://www.staff.uni-oldenburg.de/hannes.uecker/pde2path/) (and discussed in **Hopf Bifurcation and Time Periodic Orbits with Pde2path – Algorithms and Applications.**, Uecker, Hannes, Communications in Computational Physics 25, no. 3 (2019)) for fair comparison.


The equations are as follows

$$\partial_{t} u=\Delta u+(r+\mathrm{i} v) u-\left(c_{3}+\mathrm{i} \mu\right)|u|^{2} u-c_{5}|u|^{4} u, \quad u=u(t, x) \in \mathbb{C}$$

with Dirichlet boundary conditions. We discretize the square $\Omega = (0,L_x)\times(0,L_y)$ with $2N_xN_y$ points. We start by writing the Laplacian:

```julia
using Revise
using DiffEqOperators, ForwardDiff
using PseudoArcLengthContinuation, LinearAlgebra, Plots, SparseArrays, Parameters, Setfield
const PALC = PseudoArcLengthContinuation

norminf = x -> norm(x, Inf)

function Laplacian2D(Nx, Ny, lx, ly, bc = :Dirichlet)
	hx = 2lx/Nx
	hy = 2ly/Ny
	D2x = CenteredDifference(2, 2, hx, Nx)
	D2y = CenteredDifference(2, 2, hy, Ny)
	if bc == :Neumann
		Qx = Neumann0BC(hx)
		Qy = Neumann0BC(hy)
	elseif  bc == :Dirichlet
		Qx = Dirichlet0BC(typeof(hx))
		Qy = Dirichlet0BC(typeof(hy))
	end
	
	D2xsp = sparse(D2x * Qx)[1]
	D2ysp = sparse(D2y * Qy)[1]
	
	A = kron(sparse(I, Ny, Ny), D2xsp) + kron(D2ysp, sparse(I, Nx, Nx))
	return A, D2x
end
```

It is then straightforward to write the vector field

```julia
# this encodes the nonlinearity
function NL(u, p)
	@unpack r, μ, ν, c3, c5 = p
	n = div(length(u), 2)
	u1 = @view u[1:n]
	u2 = @view u[n+1:2n]

	ua = u1.^2 .+ u2.^2

	f = similar(u)
	f1 = @view f[1:n]
	f2 = @view f[n+1:2n]

	@. f1 .= r * u1 - ν * u2 - ua * (c3 * u1 - μ * u2) - c5 * ua^2 * u1
	@. f2 .= r * u2 + ν * u1 - ua * (c3 * u2 + μ * u1) - c5 * ua^2 * u2

	return f
end

function Fcgl(u, p)
	f = similar(u)
	mul!(f, p.Δ, u)
	f .= f .+ NL(u, p)
end
```

and its jacobian:

```julia
function Jcgl(u, p)
	@unpack r, μ, ν, c3, c5, Δ = p

	n = div(length(u), 2)
	u1 = @view u[1:n]
	u2 = @view u[n+1:2n]

	ua = u1.^2 .+ u2.^2

	f1u = zero(u1)
	f2u = zero(u1)
	f1v = zero(u1)
	f2v = zero(u1)

	@. f1u =  r - 2 * u1 * (c3 * u1 - μ * u2) - c3 * ua - 4 * c5 * ua * u1^2 - c5 * ua^2
	@. f1v = -ν - 2 * u2 * (c3 * u1 - μ * u2)  + μ * ua - 4 * c5 * ua * u1 * u2
	@. f2u =  ν - 2 * u1 * (c3 * u2 + μ * u1)  - μ * ua - 4 * c5 * ua * u1 * u2
	@. f2v =  r - 2 * u2 * (c3 * u2 + μ * u1) - c3 * ua - 4 * c5 * ua * u2 ^2 - c5 * ua^2

	jacdiag = vcat(f1u, f2v)

	Δ + spdiagm(0 => jacdiag, n => f1v, -n => f2u)
end
```

We now define the parameters and the stationary solution:

```julia
Nx = 41
Ny = 21
n = Nx * Ny
lx = pi
ly = pi/2

Δ = Laplacian2D(Nx, Ny, lx, ly)[1]
par_cgl = (r = 0.5, μ = 0.1, ν = 1.0, c3 = -1.0, c5 = 1.0, Δ = blockdiag(Δ, Δ))
sol0 = zeros(2Nx, Ny)
```

and we continue it to find the Hopf bifurcation points. We use a Shift-Invert eigensolver.

```julia
# Shift-Invert eigensolver
eigls = EigArpack(1.0, :LM)
opt_newton = NewtonPar(tol = 1e-10, verbose = true, eigsolver = eigls)
opts_br = ContinuationPar(dsmin = 0.001, dsmax = 0.005, ds = 0.001, pMax = 2., detectBifurcation = 1, nev = 5, plotEveryNsteps = 50, newtonOptions = opt_newton, maxSteps = 1060)

br, _ = @time continuation(
	(x, p) -> Fcgl(x, @set par_cgl.r = p),
	(x, p) -> Jcgl(x, @set par_cgl.r = p),
	vec(sol0), par_cgl.r, opts_br, verbosity = 0)
```

![](cgl2d-bif.png)

## Periodic orbits continuation with stability
Having found two Hopf bifurcation points, we aim at computing the periodic orbits branching from them. Like for the Brusselator example, we need to find some educated guess for the periodic orbits in order to have a successful Newton call.

The following code is very close to the one explained in the tutorial [Brusselator 1d](@ref) so we won't give too much details here.

We focus on the first Hopf bifurcation point. Note that, we do not improve the guess for the Hopf bifurcation point, *e.g.* by calling `newtonHopf`, as this is not really needed.

```julia
# index of the Hopf point we want to branch from
ind_hopf = 1

# number of time slices in the periodic orbit
M = 30

# periodic orbit initial guess
r_hopf, Th, orbitguess2, hopfpt, vec_hopf = guessFromHopf(br, ind_hopf, opt_newton.eigsolver, M, 22*sqrt(0.1); phase = 0.25)

# flatten the initial guess
orbitguess_f2 = reduce(hcat, orbitguess2)
orbitguess_f = vcat(vec(orbitguess_f2), Th) |> vec
```

Like in the [Brusselator 1d](@ref) example, we create a problem to hold the functional and find periodic orbits based on Finite Differences

```julia
poTrap = p -> PeriodicOrbitTrapProblem(
# vector field
	x ->  Fcgl(x, p),
# sparse representation of the Jacobian	
	x ->  Jcgl(x, p),
# parameters for the phase condition
	real.(vec_hopf),
	hopfpt.u,
# number of time slices	
	M)
```

We can use this (family) problem `poTrap` with `newton` on our periodic orbit guess to find a periodic orbit. Hence, one can be tempted to use

	
!!! danger "Don't run this!!"
    It uses too much memory 
    
    ```julia
    opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.03, ds= 0.001, pMax = 2.5, 	 maxSteps = 250, plotEveryNsteps = 3, newtonOptions = (@set opt_po.linsolver = DefaultLS()))
    br_po, upo , _= @time continuationPOTrap(p -> poTrap(@set par_cgl.r = p),
     orbitguess_f, r_hopf - 0.01, opts_po_cont)
    ```	


**However, the linear system associated to the newton iterations will be solved by forming the sparse jacobian of size $(2N_xN_yM+1)^2$ and the use of `\` (based on LU decomposition). It takes way too much time and memory.**

Instead, we use a preconditioner. We build the jacobian once, compute its **incomplete LU decomposition** (ILU) and use it as a preconditioner.

```julia
using IncompleteLU

# Sparse matrix representation of the jacobian of the periodic orbit functional
Jpo = poTrap(@set par_cgl.r = r_hopf - 0.01)(Val(:JacFullSparse), orbitguess_f)

# incomplete LU factorization with threshold
Precilu = @time ilu(Jpo, τ = 0.005)

# we define the linear solver with left preconditioner Precilu
ls = GMRESIterativeSolvers(verbose = false, tol = 1e-3, N = size(Jpo,1), restart = 40, maxiter = 50, Pl = Precilu, log=true)

# we try the linear solver
ls(Jpo, rand(ls.N))
```

This converges in `7` iterations whereas, without the preconditioner, it does not converge after `100` iterations. 

We set the parameters for the `newton` solve.

```julia
opt_po = @set opt_newton.verbose = true
outpo_f, _, flag = @time newton(poTrap(@set par_cgl.r = r_hopf - 0.01),
	orbitguess_f, (@set opt_po.linsolver = ls), 
	:FullMatrixFree; normN = norminf)
flag && printstyled(color=:red, "--> T = ", outpo_f[end], ", amplitude = ", PALC.amplitude(outpo_f, Nx*Ny, M; ratio = 2),"\n")
PALC.plotPeriodicPOTrap(outpo_f, M, Nx, Ny; ratio = 2);
```

which gives 

```julia
 Newton Iterations 
   Iterations      Func-count      f(x)      Linear-Iterations

        0                1     6.5509e-03         0
        1                2     1.4311e-03         9
        2                3     3.6948e-04         9
        3                4     6.5156e-05        10
        4                5     4.3270e-06        11
        5                6     3.9205e-08        12
        6                7     1.0685e-10        13
        7                8     1.0492e-13        14
  1.896905 seconds (165.04 k allocations: 1.330 GiB, 12.03% gc time)
--> T = 6.5367097374070315, amplitude = 0.3507182067194716
```

and

![](cgl2d-po-newton.png)

At this point, we are still wasting a lot of resources, because the matrix-free version of the jacobian of the functional uses the jacobian of the vector field `x ->  Jcgl(x, p)`. Hence, it builds `M` sparse matrices for each evaluation!! Let us create a problem which is fully Matrix Free:

```julia
# computation of the first derivative using automatic differentiation
d1Fcgl(x, p, dx) = ForwardDiff.derivative(t -> Fcgl(x .+ t .* dx, p), 0.)

# linear solver for solving Jcgl*x = rhs. Needed for Floquet multipliers computation
ls0 = GMRESIterativeSolvers(N = 2Nx*Ny, tol = 1e-9, Pl = lu(I + par_cgl.Δ))

# matrix-free problem
poTrapMF = p -> PeriodicOrbitTrapProblem(
	x ->  Fcgl(x, p),
	x ->  (dx -> d1Fcgl(x, p, dx)),
	real.(vec_hopf),
	hopfpt.u,
	M, ls0)
```

We can now use newton

```julia
outpo_f, _, flag = @time newton(poTrapMF(@set par_cgl.r = r_hopf - 0.01),
	orbitguess_f, (@set opt_po.linsolver = ls), 
	:FullMatrixFree; normN = norminf)
flag && printstyled(color=:red, "--> T = ", outpo_f[end], ", amplitude = ", PALC.amplitude(outpo_f, Nx*Ny, M; ratio = 2),"\n")
```

which gives 

```julia
 Newton Iterations 
   Iterations      Func-count      f(x)      Linear-Iterations

        0                1     6.5509e-03         0
        1                2     1.4311e-03         9
        2                3     3.6948e-04         9
        3                4     6.5156e-05        10
        4                5     4.3270e-06        11
        5                6     3.9205e-08        12
        6                7     1.0685e-10        13
        7                8     1.0495e-13        14
  1.251035 seconds (69.10 k allocations: 488.773 MiB, 3.95% gc time)
--> T = 6.53670973740703, amplitude = 0.3507182067194715
```

The speedup will increase a lot for larger $N_x, N_y$. Also, for Floquet multipliers computation, the speedup will be substantial.

### Removing most allocations (Advanced and Experimental)

We show here how to remove most allocations and speed up the computations. This is an **experimental** feature as the Floquet multipliers computation is not yet readily available in this case. To this end, we rewrite the functional using *inplace* formulation and trying to avoid allocations. This can be done as follows:

```julia
# compute just the nonlinearity
function NL!(f, u, p, t = 0.)
	@unpack r, μ, ν, c3, c5 = p
	n = div(length(u), 2)
	u1v = @view u[1:n]
	u2v = @view u[n+1:2n]

	f1 = @view f[1:n]
	f2 = @view f[n+1:2n]

	@inbounds for ii = 1:n
		u1 = u1v[ii]
		u2 = u2v[ii]
		ua = u1^2+u2^2
		f1[ii] = r * u1 - ν * u2 - ua * (c3 * u1 - μ * u2) - c5 * ua^2 * u1
		f2[ii] = r * u2 + ν * u1 - ua * (c3 * u2 + μ * u1) - c5 * ua^2 * u2
	end
	return f
end

# derivative of the nonlinearity
function dNL!(f, u, p, du)
	@unpack r, μ, ν, c3, c5 = p
	n = div(length(u), 2)
	u1v = @view u[1:n]
	u2v = @view u[n+1:2n]

	du1v = @view du[1:n]
	du2v = @view du[n+1:2n]

	f1 = @view f[1:n]
	f2 = @view f[n+1:2n]

	@inbounds for ii = 1:n
		u1 = u1v[ii]
		u2 = u2v[ii]
		du1 = du1v[ii]
		du2 = du2v[ii]
		ua = u1^2+u2^2
		f1[ii] = (-5*c5*u1^4 + (-6*c5*u2^2 - 3*c3)*u1^2 + 2*μ*u1*u2 - c5*u2^4 - c3*u2^2 + r) * du1 +
		(-4*c5*u2*u1^3 + μ*u1^2 + (-4*c5*u2^3 - 2*c3*u2)*u1 + 3*u2^2*μ - ν) * du2

		f2[ii] = (-4*c5*u2*u1^3 - 3*μ*u1^2 + (-4*c5*u2^3 - 2*c3*u2)*u1 - u2^2*μ + ν) * du1 + (-c5*u1^4 + (-6*c5*u2^2 - c3)*u1^2 - 2*μ*u1*u2 - 5*c5*u2^4 - 3*c3*u2^2 + r) * du2
	end

	return f
end

# inplace vector field
function Fcgl!(f, u, p, t = 0.)
	NL!(f, u, p)
	mul!(f, p.Δ, u, 1., 1.)
end

# inplace derivative of the vector field
function dFcgl!(f, x, p, dx)
	dNL!(f, x, p, dx)
	mul!(f, p.Δ, dx, 1., 1.)
end
```

We can now define an inplace functional

```julia
ls0 = GMRESIterativeSolvers(N = 2Nx*Ny, tol = 1e-9)#, Pl = lu(I + par_cgl.Δ))
poTrapMFi = p -> PeriodicOrbitTrapProblem(
	(o, x) ->  Fcgl!(o, x, p),
	(o, x, dx) -> dFcgl!(o, x, p, dx),
	real.(vec_hopf),
	hopfpt.u,
	M, ls0; isinplace = true)
```
and run the `newton` method:

```julia
outpo_f, _, flag = @time newton(poTrapMFi(@set par_cgl.r = r_hopf - 0.01),
	orbitguess_f, (@set opt_po.linsolver = ls),
	:FullMatrixFree; normN = norminf)
```
It gives	

```julia
 Newton Iterations 
   Iterations      Func-count      f(x)      Linear-Iterations

        0                1     6.5509e-03         0
        1                2     1.4311e-03         9
        2                3     3.6948e-04         9
        3                4     6.5156e-05        10
        4                5     4.3270e-06        11
        5                6     3.9205e-08        12
        6                7     1.0685e-10        13
        7                8     1.0592e-13        14
  1.157987 seconds (23.44 k allocations: 154.468 MiB, 3.39% gc time)
```

Notice the small speed boost but the reduced allocations. At this stage, further improvements could target the use of `BlockBandedMatrices.jl` for the Laplacian operator, etc.


### Other linear formulation

We could use another way to "invert" jacobian of the functional based on bordered technics. We try to use an ILU preconditioner on the cyclic matrix $J_c$ (see [Periodic orbits based on finite differences](@ref)) which has a smaller memory footprint:

```julia
Jpo2 = poTrap(@set par_cgl.r = r_hopf - 0.1)(Val(:JacCyclicSparse), orbitguess_f)
Precilu = @time ilu(Jpo2, τ = 0.005)
ls2 = GMRESIterativeSolvers(verbose = false, tol = 1e-3, N = size(Jpo2,1), restart = 30, maxiter = 50, Pl = Precilu, log=true)

opt_po = @set opt_newton.verbose = true
outpo_f, hist, flag = @time newton(
	poTrapMF(@set par_cgl.r = r_hopf - 0.1),
	orbitguess_f, (@set opt_po.linsolver = ls2), :BorderedMatrixFree;
	normN = norminf)
```

but it gives:

```julia
 Newton Iterations 
   Iterations      Func-count      f(x)      Linear-Iterations

        0                1     3.3281e-03         0
        1                2     9.4520e-03        34
        2                3     1.2632e-03        26
        3                4     6.7022e-05        29
        4                5     4.2398e-07        34
        5                6     1.4380e-09        43
        6                7     6.7513e-13        60
  4.139557 seconds (143.13 k allocations: 1.007 GiB, 3.67% gc time)
```

**Hence, it seems better to use the previous preconditioner.**

## Continuation of periodic solutions

We can now perform continuation of the newly found periodic orbit and compute the Floquet multipliers using Matrix-Free methods.

```julia
# set the eigensolver for the computation of the Floquet multipliers
opt_po = @set opt_po.eigsolver = EigKrylovKit(tol = 1e-3, x₀ = rand(2n), verbose = 2, dim = 25)

# parameters for the continuation
opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.02, ds = 0.001, pMax = 2.2, maxSteps = 250, plotEveryNsteps = 3, newtonOptions = (@set opt_po.linsolver = ls), 
	nev = 5, precisionStability = 1e-7, detectBifurcation = 0)

br_po, _ , _= @time continuationPOTrap(
	p -> poTrapMF(@set par_cgl.r = p),
	outpo_f, r_hopf - 0.01,
	opts_po_cont, linearPO = :FullMatrixFree;
	verbosity = 2,	plot = true,
	plotSolution = (x, p; kwargs...) -> PALC.plotPeriodicPOTrap(x, M, Nx, Ny; ratio = 2, kwargs...),
	printSolution = (u, p) -> PALC.amplitude(u, Nx*Ny, M; ratio = 2), normC = norminf)
```

This gives the following bifurcation diagram:

![](cgl2d-po-cont.png)

!!! tip "Improved performances"
    Although it would be "cheating" for fair comparisons with existing packages, there is a trick to compute the bifurcation diagram without using preconditionners. We will not detail it here but it allows to handle the case `Nx = 200; Ny = 110; M = 30` and above.

We did not change the preconditioner in the previous example as it does not seem needed. Let us show how to do this nevertheless:

```julia
# callback which will be sent to newton. 
# `iteration` in the arguments refers to newton iterations
function callbackPO(x, f, J, res, itlinear, iteration, linsolver = ls, prob = poTrap, p = par_cgl; kwargs...)
	# we update the preconditioner every 10 continuation steps
	if mod(kwargs[:iterationC], 10) == 9 && iteration == 1
		@info "update Preconditioner"
		Jpo = poTrap(@set p.r = kwargs[:p])(Val(:JacCyclicSparse), x)
		Precilu = @time ilu(Jpo, τ = 0.003)
		ls.Pl = Precilu
	end
	true
end

br_po, _ , _= @time continuationPOTrap(
	p -> poTrapMF(@set par_cgl.r = p),
	outpo_f, r_hopf - 0.01,
	opts_po_cont, linearPO = :FullMatrixFree;
	verbosity = 2,	plot = true,
	callbackN = callbackPO,
	plotSolution = (x, p; kwargs...) -> PALC.plotPeriodicPOTrap(x, M, Nx, Ny; ratio = 2, kwargs...),
	printSolution = (u, p) -> PALC.amplitude(u, Nx*Ny, M; ratio = 2), normC = norminf)
```

## Continuation of Fold of periodic orbits

We continue the Fold point of the first branch of the previous bifurcation diagram in the parameter plane $(r, c_5)$. To this end, we need to be able to compute the Hessian of the periodic orbit functional. This is not yet readily available so we turn to automatic differentiation.

```julia
using ForwardDiff

# computation of the second derivative of a function f
function d2Fcglpb(f, x, dx1, dx2)
   return ForwardDiff.derivative(t2 -> ForwardDiff.derivative( t1-> f(x .+ t1 .* dx1 .+ t2 .* dx2,), 0.), 0.)
end
```

We select the Fold point from the branch `br_po` and redefine our linear solver to get the ILU preconditioner tuned close to the Fold point.

```julia
indfold = 2
foldpt = FoldPoint(br_po, indfold)

Jpo = poTrap(@set par_cgl.r = r_hopf - 0.1)(Val(:JacFullSparse), orbitguess_f)
Precilu = @time ilu(Jpo, τ = 0.005)
ls = GMRESIterativeSolvers(verbose = false, tol = 1e-4, N = size(Jpo, 1), restart = 40, maxiter = 60, Pl = Precilu)
```

We can then use our functional to call `newtonFold` like for a regular function (see Tutorial 1)

```julia
outfold, hist, flag = @time PALC.newtonFold(
	(x, p) -> poTrap(@set par_cgl.r = p)(x),
	(x, p) -> poTrap(@set par_cgl.r = p)(Val(:JacFullSparse), x),
	br_po , indfold, #index of the fold point
	@set opt_po.linsolver = ls;
	d2F = (x, p, dx1, dx2) -> d2Fcglpb(poTrap(@set par_cgl.r = p), x, dx1, dx2))
flag && printstyled(color=:red, "--> We found a Fold Point at α = ", outfold.p," from ", br_po.foldpoint[indfold][3],"\n")
```

and this gives

```julia
 Newton Iterations 
   Iterations      Func-count      f(x)      Linear-Iterations

        0                1     4.6366e-01         0
        1                2     5.6561e-01        20
        2                3     3.2592e-02        24
        3                4     3.2054e-05        32
        4                5     2.3656e-07        37
        5                6     1.2573e-10        43
        6                7     1.9629e-13        49
 27.289005 seconds (1.07 M allocations: 24.444 GiB, 10.12% gc time)
--> We found a Fold Point at α = 0.9470569704262517 from 0.9481896723164748
```

Finally, one can perform continuation of the Fold bifurcation point as follows

```julia
optcontfold = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, pMax = 40.1, pMin = -10., newtonOptions = (@set opt_po.linsolver = ls), maxSteps = 20)

outfoldco, hist, flag = @time PALC.continuationFold(
	(x, r, p) -> poTrap(setproperties(par_cgl, (r=r, c5=p)))(x),
	(x, r, p) -> poTrap(setproperties(par_cgl, (r=r, c5=p)))(Val(:JacFullSparse), x),
	br_po, indfold, par_cgl.c5, optcontfold;
	d2F = p -> ((x, r, dx1, dx2) -> d2Fcglpb(poTrap(setproperties(par_cgl, (r=r, c5=p))), x, dx1, dx2)),
	plot = true, verbosity = 2)
```

which yields:

![](cgl2d-po-foldcont.png)

There is still room for a lot of improvements here. Basically, the idea would be to use full Matrix-Free the jacobian functional and its transpose.

## Continuation of periodic orbits on the GPU (Advanced)

!!! tip ""
    This is a very neat example **all done** on the GPU using the following ingredients: Matrix-Free computation of periodic orbits using preconditioners.

We now take advantage of the computing power of GPUs. The section is run on an NVIDIA Tesla V100. Given the small number of unknowns, we can (only) expect significant speedup in the application of the **big** preconditioner. 

> Note that we use the parameters `Nx = 82; Ny = 42; M=30`.

```julia
# computation of the first derivative
d1Fcgl(x, p, dx) = ForwardDiff.derivative(t -> Fcgl(x .+ t .* dx, p), 0.)

d1NL(x, p, dx) = ForwardDiff.derivative(t -> NL(x .+ t .* dx, p), 0.)

function dFcgl(x, p, dx)
	f = similar(dx)
	mul!(f, p.Δ, dx)
	nl = d1NL(x, p, dx)
	f .= f .+ nl
end
```

We first load `CuArrays`

```julia
using CuArrays
CuArrays.allowscalar(false)
import LinearAlgebra: mul!, axpby!
mul!(x::CuArray, y::CuArray, α::T) where {T <: Number} = (x .= α .* y)
mul!(x::CuArray, α::T, y::CuArray) where {T <: Number} = (x .= α .* y)
axpby!(a::T, X::CuArray, b::T, Y::CuArray) where {T <: Number} = (Y .= a .* X .+ b .* Y)
```

and update the parameters

```julia
par_cgl_gpu = @set par_cgl.Δ = CuArrays.CUSPARSE.CuSparseMatrixCSC(par_cgl.Δ);
```

Then, we precompute the preconditioner on the CPU:

```julia
Jpo = poTrap(@set par_cgl.r = r_hopf - 0.01)(Val(:JacFullSparse), orbitguess_f)
Precilu = @time ilu(Jpo, τ = 0.003)
```

To invert `Precilu` on the GPU, we need to define a few functions which are not in `CuArrays` and which are related to LU decomposition:

```julia
struct LUperso
	L
	Ut	# transpose of U in LU decomposition
end

import Base: ldiv!
function LinearAlgebra.ldiv!(_lu::LUperso, rhs::CuArrays.CuArray)
	_x = UpperTriangular(_lu.Ut) \ (LowerTriangular(_lu.L) \ rhs)
	rhs .= vec(_x)
	CuArrays.unsafe_free!(_x)
	rhs
end
```

Finally, for the methods in `PeriodicOrbitTrapProblem` to work, we need to redefine the following method. Indeed, we disable the use of scalar on the GPU to increase the speed.

```julia
import PseudoArcLengthContinuation: extractPeriodFDTrap
extractPeriodFDTrap(x::CuArray) = x[end:end]
```

We can now define our functional:

```julia
# matrix-free problem on the gpu
ls0gpu = GMRESKrylovKit(rtol = 1e-9)
poTrapMFGPU = p -> PeriodicOrbitTrapProblem(
	x ->  Fcgl(x, p),
	x ->  (dx -> dFcgl(x, p, dx)),
	CuArray(real.(vec_hopf)),
	CuArray(hopfpt.u),
	M, ls0gpu;
	ongpu = true) # this is required to alter the way the constraint is handled
```

Let us have a look at the linear solvers and compare the speed on CPU and GPU:

```julia
ls = GMRESKrylovKit(verbose = 2, Pl = Precilu, rtol = 1e-3, dim  = 20)
   # runs in 	2.990495 seconds (785 allocations: 31.564 MiB, 0.98% gc time)
	outh, _, _ = @time ls((Jpo), orbitguess_f)

Precilu_gpu = LUperso(LowerTriangular(CuArrays.CUSPARSE.CuSparseMatrixCSR(I+Precilu.L)), UpperTriangular(CuArrays.CUSPARSE.CuSparseMatrixCSR(sparse(Precilu.U'))));
lsgpu = GMRESKrylovKit(verbose = 2, Pl = Precilu_gpu, rtol = 1e-3, dim  = 20)
	Jpo_gpu = CuArrays.CUSPARSE.CuSparseMatrixCSR(Jpo);
	orbitguess_cu = CuArray(orbitguess_f)
	# runs in 1.751230 seconds (6.54 k allocations: 188.500 KiB, 0.43% gc time)
	outd, _, _ = @time lsgpu(Jpo_gpu, orbitguess_cu)
```	 

So we can expect a pretty descent x2 speed up in computing the periodic orbits. We can thus call newton:

```julia
opt_po = @set opt_newton.verbose = true
	outpo_f, hist, flag = @time newton(
			poTrapMFGPU(@set par_cgl_gpu.r = r_hopf - 0.01),
			orbitguess_cu,
			(@set opt_po.linsolver = lsgpu), :FullMatrixFree;
			normN = x->maximum(abs.(x))) 
```
The computing time is `6.914367 seconds (2.94 M allocations: 130.348 MiB, 1.10% gc time)`. The same computation on the CPU, runs in `13.972836 seconds (551.41 k allocations: 1.300 GiB, 1.05% gc time)`.

You can also perform continuation, here is a simple example:

```julia
opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.02, ds= 0.001, pMax = 2.2, maxSteps = 250, plotEveryNsteps = 3, newtonOptions = (@set opt_po.linsolver = lsgpu))
br_po, upo , _= @time continuationPOTrap(
   p -> poTrapMFGPU(@set par_cgl_gpu.r = p),
   orbitguess_cu, r_hopf - 0.01,
   opts_po_cont, linearPO = :FullMatrixFree;
   verbosity = 2,
   printSolution = (u,p) -> amplitude(u, Nx*Ny, M), normC = x->maximum(abs.(x)))
```

!!! info "Preconditioner update"
    For now, the preconditioner has been precomputed on the CPU which forbids its (efficient) update during continuation of a branch of periodic orbits. This could be improved using `ilu0!` and friends in `CuArrays`.


