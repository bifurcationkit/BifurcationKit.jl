# Colpitts–type Oscillator (Intermediate)

```@contents
Pages = ["Colpitts.md"]
Depth = 3
```

In this tutorial, we show how to study parametrized quasilinear DAEs like:

$$A(\mu,x)\dot x = G(\mu,x).$$

In particular, we detect a Hopf bifurcation and compute the periodic orbit branching from it using a multiple standard method.

The following DAE model is taken from [^Rabier]:

$$\left(\begin{array}{cccc}
-\left(C_{1}+C_{2}\right) & C_{2} & 0 & 0 \\
C_{2} & -C_{2} & 0 & 0 \\
C_{1} & 0 & 0 & 0 \\
0 & 0 & L & 0
\end{array}\right)\left(\begin{array}{c}
\dot{x}_{1} \\
\dot{x}_{2} \\
\dot{x}_{3} \\
\dot{x}_{4}
\end{array}\right)=\left(\begin{array}{c}
R^{-1}\left(x_{1}-V\right)+I E\left(x_{1}, x_{2}\right) \\
x_{3}+I C\left(x_{1}, x_{2}\right) \\
-x_{3}-x_{4} \\
-\mu+x_{2}
\end{array}\right)$$

It is easy to encode the DAE as follows. The mass matrix is defined next.

```@example TUTDAE1
using Revise, Parameters, Setfield, Plots, LinearAlgebra
using BifurcationKit, Test
const BK = BifurcationKit

# sup norm
norminf(x) = norm(x, Inf)

# function to record information from the soluton
recordFromSolution(x, p) = (u1 = norminf(x), x1 = x[1], x2 = x[2], x3 = x[3], x4 = x[4])

# vector field
f(x, p) = p.Is * (exp(p.q * x) - 1)
IE(x1, x2, p) = -f(x2, p) + f(x1, p) / p.αF
IC(x1, x2, p) = f(x2, p)/ p.αR - f(x1, p)

function Colpitts!(dz, z, p, t=0)
	@unpack C1, C2, L, R, Is, q, αF, αR, V, μ = p
	x1, x2, x3, x4 = z
	dz[1] = (x1 - V) / R + IE(x1, x2, p)
	dz[2] =	x3 + IC(x1, x2, p)
	dz[3] = -x3-x4
	dz[4] = -μ+x2
	dz
end

# we put the option t = 0 in order to use the function with DifferentialEquations
Colpitts(z, p, t = 0) = Colpitts!(similar(z), z, p, t)

# we group the differentials together
jet = BK.getJet(Colpitts; matrixfree=false)

# parameter values
par_Colpitts = (C1 = 1.0, C2 = 1.0, L = 1.0, R = 1/4., Is = 1e-16, q = 40., αF = 0.99, αR = 0.5, μ = 0.5, V = 6.)

# initial condition
z0 = [0.9957,0.7650,19.81,-19.81]

# mass matrix
Be = [-(par_Colpitts.C1+par_Colpitts.C2) par_Colpitts.C2 0 0;par_Colpitts.C2 -par_Colpitts.C2 0 0;par_Colpitts.C1 0 0 0; 0 0 par_Colpitts.L 0]

nothing #hide
```

We first compute the branch of equilibria. But we need  a generalized eigenvalue solver for this.

```@example TUTDAE1
# we need  a specific eigensolver with mass matrix B
struct EigenDAE{Tb} <: BK.AbstractEigenSolver
	B::Tb
end

# compute the eigen elements
function (eig::EigenDAE)(Jac, nev; k...)
	F = eigen(Jac, eig.B)
	I = sortperm(F.values, by = real, rev = true)
	return Complex.(F.values[I]), Complex.(F.vectors[:, I]), true, 1
end

# continuation options
optn = NewtonPar(tol = 1e-13, verbose = true, maxIter = 10, eigsolver = EigenDAE(Be))
opts_br = ContinuationPar(pMin = -0.4, pMax = 0.8, ds = 0.01, dsmax = 0.01, nInversion = 8, detectBifurcation = 3, maxBisectionSteps = 25, nev = 4, plotEveryStep = 3, maxSteps = 1000, newtonOptions = optn)
	# opts_br = @set opts_br.newtonOptions.verbose = false
	br, = continuation(jet[1], jet[2], z0, par_Colpitts, (@lens _.μ), opts_br;
	recordFromSolution = recordFromSolution,
	verbosity = 0,
	normC = norminf)

scene = plot(br, vars = (:param, :x1))
```


## Periodic orbits with Multiple Standard Shooting

We use shooting to compute periodic orbits: we rely on a fixed point of the flow. To compute the flow, we use `DifferentialEquations.jl`.

Thanks to [^Lamour], we can  just compute the Floquet coefficients to get the nonlinear stability of the periodic orbit. Two period doubling bifurcations are detected.

Note that we use Automatic Branch Switching from a Hopf bifurcation despite the fact the normal form implemented in `BifurcationKit.jl` is not valid for DAE. For example, it predicts a subciritical Hopf point whereas we see below that it is supercritical. Nevertheless, it provides a

```@example TUTDAE1
using DifferentialEquations

# this is the ODEProblem used with `DiffEqBase.solve`
# we  set  the initial conditions
prob_dae = ODEFunction{false}(jet[1]; mass_matrix = Be)
probFreez_ode = ODEProblem(prob_dae, z0, (0., 1.), par_Colpitts)

# we lower the tolerance of newton for the periodic orbits
optnpo = @set optn.tol = 1e-9

opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.005, ds= -0.0001, pMin = 0.2, maxSteps = 150, newtonOptions = optnpo, nev = 4, precisionStability = 1e-3, detectBifurcation = 0, plotEveryStep = 5)

# we use a regular eigensolver for the Floquet coefficients
@set! opts_po_cont.newtonOptions.eigsolver = DefaultEig()
@set! opts_po_cont.detectBifurcation = 2

# Shooting functional. Note the  stringent tolerances used to cope with
# the extreme parameters of the model
probSH = ShootingProblem(5, probFreez_ode, Rodas4(); reltol = 1e-10, abstol = 1e-13)

# automatic branching from the Hopf point
br_po, = continuation(jet..., br, 1, opts_po_cont, probSH; plot = true, verbosity = 3,
	linearAlgo = MatrixBLS(),
	linearPO = :autodiffDense,
	# δp is use to parametrise the first parameter point on the
	# branch of periodic orbits
	δp = 0.001,
	recordFromSolution = (u, p) -> begin
		outt = BK.getTrajectory(p.prob, u, (@set  par_Colpitts.μ=p.p))
		m = maximum(outt[1,:])
		return (s = m, period = u[end])
	end,
	updateSectionEveryStep = 1,
	# plotting of a solution
	plotSolution = (x, p; k...) -> begin
		outt = BK.getTrajectory(p.prob, x, (@set  par_Colpitts.μ=p.p))
		plot!(outt, vars = [2], subplot = 3)
		plot!(br, vars = (:param, :x1), subplot = 1)
	end,
	# the newton Callback is used to reject residual > 1
	# this is to avoid numerical instabilities from DE.jl
	callbackN = BK.cbMaxNorm(1.0),
	normC = norminf)
```

![](Colpitts1.png)

with detailed information

```julia
 ┌─ Branch number of points: 125
 ├─ Branch of PeriodicOrbit from Hopf bifurcation point.
 ├─ Type of vectors: Vector{Float64}
 ├─ Parameter μ starts at 0.7640482828951152, ends at 0.2
 └─ Special points:

 (ind_ev = index of the bifurcating eigenvalue e.g. `br.eig[idx].eigenvals[ind_ev]`)

- #  1,    pd at μ ≈ +0.72603451 ∈ (+0.72603451, +0.73184970), |δp|=6e-03, [    guess], δ = ( 1,  1), step =  18, eigenelements in eig[ 19], ind_ev =   1
- #  2,    pd at μ ≈ +0.67290724 ∈ (+0.67290724, +0.67883206), |δp|=6e-03, [    guess], δ = (-1, -1), step =  27, eigenelements in eig[ 28], ind_ev =   1
- #  3,    bp at μ ≈ +0.62294985 ∈ (+0.62237946, +0.62294985), |δp|=6e-04, [    guess], δ = ( 1,  0), step =  37, eigenelements in eig[ 38], ind_ev =   1
- #  4,    bp at μ ≈ +0.63758469 ∈ (+0.63758469, +0.63761073), |δp|=3e-05, [    guess], δ = (-1,  0), step =  43, eigenelements in eig[ 44], ind_ev =   1
```

Let us show that this bifurcation diagram is valid by showing evidences for the period doubling bifurcation.

```@example TUTDAE1
probFreez_ode = ODEProblem(prob_dae, br.specialpoint[1].x .+ 0.01rand(4), (0., 200.), @set par_Colpitts.μ = 0.733)

solFreez = @time solve(probFreez_ode, Rodas4(), progress = true;reltol = 1e-10, abstol = 1e-13)

scene = plot(solFreez, vars = [2], xlims=(195,200), title="μ = $(probFreez_ode.p.μ)")
```

and after the bifurcation

```@example TUTDAE1
probFreez_ode = ODEProblem(prob_dae, br.specialpoint[1].x .+ 0.01rand(4), (0., 200.), @set par_Colpitts.μ = 0.72)

solFreez = @time solve(probFreez_ode, Rodas4(), progress = true;reltol = 1e-10, abstol = 1e-13)

scene = plot(solFreez, vars = [2], xlims=(195,200), title="μ = $(probFreez_ode.p.μ)")
```


## References

[^Rabier]:> Rabier, Patrick J. “The Hopf Bifurcation Theorem for Quasilinear Differential-Algebraic Equations.” Computer Methods in Applied Mechanics and Engineering 170, no. 3–4 (March 1999): 355–71. https://doi.org/10.1016/S0045-7825(98)00203-5.

[^Lamour]:> Lamour, René, Roswitha März, and Renate Winkler. “How Floquet Theory Applies to Index 1 Differential Algebraic Equations.” Journal of Mathematical Analysis and Applications 217, no. 2 (January 1998): 372–94. https://doi.org/10.1006/jmaa.1997.5714.
