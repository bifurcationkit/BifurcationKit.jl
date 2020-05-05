# Complex Ginzburg-Landau 2d (shooting)

```@contents
Pages = ["tutorialsCGLShoot.md"]
Depth = 3
```

In this tutorial, we re-visit the example [Complex Ginzburg-Landau 2d](@ref) using a Standard Simple Shooting method. In the tutorial [Brusselator 1d](@ref), we used the implicit solver `Rodas4P` for the shooting. We will use the exponential-RK scheme `ETDRK2` ODE solver to compute the solution of cGL equations. This method is convenient for solving semilinear problems of the form 

$$\dot x = Ax+g(x)$$ 

where $A$ is the infinitesimal generator of a $C_0$-semigroup. We use the same beginning as in [Complex Ginzburg-Landau 2d](@ref):

```julia
using Revise
	using DiffEqOperators, DifferentialEquations
	using PseudoArcLengthContinuation, LinearAlgebra, Plots, SparseArrays, Parameters, Setfield
	const PALC = PseudoArcLengthContinuation

norminf = x -> norm(x, Inf)

function Laplacian2D(Nx, Ny, lx, ly)
	hx = 2lx/Nx
	hy = 2ly/Ny
	D2x = CenteredDifference(2, 2, hx, Nx)
	D2y = CenteredDifference(2, 2, hy, Ny)
	
	Qx = Dirichlet0BC(typeof(hx))
	Qy = Dirichlet0BC(typeof(hy))
	
	D2xsp = sparse(D2x * Qx)[1]
	D2ysp = sparse(D2y * Qy)[1]

	A = kron(sparse(I, Ny, Ny), D2xsp) + kron(D2ysp, sparse(I, Nx, Nx))
	return A, D2x
end
```

We then encode the PDE:

```julia
function NL!(f, u, p, t = 0.)
	@unpack r, μ, ν, c3, c5 = p
	n = div(length(u), 2)
	u1 = @view u[1:n]
	u2 = @view u[n+1:2n]

	ua = u1.^2 .+ u2.^2

	f1 = @view f[1:n]
	f2 = @view f[n+1:2n]

	@. f1 .= r * u1 - ν * u2 - ua * (c3 * u1 - μ * u2) - c5 * ua^2 * u1
	@. f2 .= r * u2 + ν * u1 - ua * (c3 * u2 + μ * u1) - c5 * ua^2 * u2

	return f
end

function NL(u, p)
	out = similar(u)
	NL!(out, u, p)
end

function Fcgl!(f, u, p, t = 0.)
	mul!(f, p.Δ, u)
	f .= f .+ NL(u, p)
end

function Fcgl(u, p, t = 0.)
	f = similar(u)
	Fcgl!(f, u, p, t)
end
```

with parameters 

```julia
Nx = 41
Ny = 21
n = Nx*Ny
lx = pi
ly = pi/2

Δ = Laplacian2D(Nx, Ny, lx, ly)[1]
par_cgl = (r = 0.5, μ = 0.1, ν = 1.0, c3 = -1.0, c5 = 1.0, Δ = blockdiag(Δ, Δ))
sol0 = 0.1rand(2Nx, Ny)
sol0_f = vec(sol0)
```

and the ODE problem

```julia
f1 = DiffEqArrayOperator(par_cgl.Δ)
f2 = NL!
prob_sp = SplitODEProblem(f1, f2, sol0_f, (0.0, 120.0), @set par_cgl.r = 1.2)
# we solve the PDE!!!
sol = @time solve(prob_sp, ETDRK2(krylov=true); abstol=1e-14, reltol=1e-14, dt = 0.1)
```

We now define the Shooting functional

```julia
probSh = p -> ShootingProblem(
	# pass the vector field and parameter (to be passed to the vector field)
	u -> Fcgl(u, p), p,

	# we pass the ODEProblem encoding the flow and the time stepper
	prob_sp, ETDRK2(krylov = true),

	# this is the phase condition
	[sol[:, end]];

	# these are options passed to the ODE time stepper
	atol = 1e-14, rtol = 1e-14, dt = 0.1)
```

## Computation of the first branch of periodic orbits

We use the solution from the ODE solver as a starting guess for the shooting method.

```julia
# initial guess with period 6.9 using solution at time t = 116
initpo = vcat(sol(116.), 6.9) |> vec

# linear solver for shooting functional
ls = GMRESIterativeSolvers(tol = 1e-4, N = 2Nx * Ny + 1, maxiter=50, verbose = false)

# newton parameters
optn = NewtonPar(verbose = true, tol = 1e-9,  maxIter = 20, linsolver = ls)

# continuation parameters
eig = EigKrylovKit(tol=1e-7, x₀ = rand(2Nx*Ny), verbose = 2, dim = 40)
opts_po_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.01, ds= -0.01, pMax = 1.5, maxSteps = 60, newtonOptions = (@set optn.eigsolver = eig), nev = 5, precisionStability = 1e-3, detectBifurcation = 2)

br_po, _ , _= @time continuationPOShooting(
	p -> probSh(@set par_cgl.r = p),
	initpo, 1.2, opts_po_cont;
	verbosity = 3,
	plot = true,
	plotSolution = (x, p; kwargs...) -> heatmap!(reshape(x[1:Nx*Ny], Nx, Ny); color=:viridis, kwargs...),
	printSolution = (u, p) -> PALC.getAmplitude(probSh(@set par_cgl.r = p), u; ratio = 2), normC = norminf)
```

![](cgl-sh-br.png)
