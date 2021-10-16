# Period doubling in Lur'e problem (PD aBS)

```@contents
Pages = ["tutorialsODE-PD.md"]
Depth = 3
```

The following model is an adaptive control system of Lur’e type. It is an example from the MatCont library.

$$\left\{\begin{array}{l}
\dot{x}=y \\
\dot{y}=z \\
\dot{z}=-\alpha z-\beta y-x+x^{2}
\end{array}\right.$$


The model is interesting because there is a period doubling bifurcation and we want to show the branch switching capabilities of `BifurcationKit.jl` in this case. We provide two different ways to compute this periodic orbits and highlight their pro / cons.

It is easy to encode the ODE as follows

```@example TUTLURE
using Revise, Parameters, Setfield, Plots, LinearAlgebra
using BifurcationKit, Test
const BK = BifurcationKit

norminf(x) = norm(x, Inf)
recordFromSolution(x, p) = (u1 = x[1], u2 = x[2])
####################################################################################################
function lur!(dz, z, p, t)
	@unpack α, β = p
	x, y, z = z
	dz[1] = y
	dz[2] =	z
	dz[3] = -α * z - β * y - x + x^2
	dz
end

lur(z, p) = lur!(similar(z), z, p, 0)

# we collect the diffferentials
jet = BK.getJet(lur; matrixfree=false)

# parameters
par_lur = (α = 1.0, β = 0.)

# initial guess
z0 = zeros(3)
nothing #hide
```

We first compute the branch of equilibria

```@example TUTLURE
# continuation options
opts_br = ContinuationPar(pMin = -0.4, pMax = 1.8, ds = -0.01, dsmax = 0.01, nInversion = 8, detectBifurcation = 3, maxBisectionSteps = 25, nev = 3, plotEveryStep = 20, maxSteps = 1000, theta = 0.3)
# turn off Newton display
opts_br = @set opts_br.newtonOptions.verbose = false

# computation of the branch
br, = continuation(jet[1], jet[2], z0, par_lur, (@lens _.β), opts_br;
	recordFromSolution = recordFromSolution,
	bothside = true,
	plot = false, verbosity = 0, normC = norminf)

scene = plot(br)
```

With detailed information:

```@example TUTLURE
br
```

We note the Hopf bifurcation point which we shall investigate now.

## Branch of periodic orbits with finite differences

We compute the branch of periodic orbits from the Hopf bifurcation point. We use finite differences to discretize the problem of finding periodic orbits. We appeal to automatic branch switching as follows

```@example TUTLURE
# newton parameters
optn_po = NewtonPar(verbose = true, tol = 1e-8,  maxIter = 25)

# continuation parameters
opts_po_cont = ContinuationPar(dsmax = 0.03, ds= 0.0001, dsmin = 1e-4, pMax = 1.8, pMin=-5., maxSteps = 130, newtonOptions = (@set optn_po.tol = 1e-8), nev = 3, precisionStability = 1e-4, detectBifurcation = 3, plotEveryStep = 20, saveSolEveryStep=1, nInversion = 6)

Mt = 90 # number of time sections
	br_po, = continuation(
	jet..., br, 1, opts_po_cont,
	PeriodicOrbitTrapProblem(M = Mt);
	ampfactor = 1., δp = 0.01,
	updateSectionEveryStep = 1,
	linearPO = :Dense,
	verbosity = 2,	plot = true,
	recordFromSolution = (x, p) -> (xtt=reshape(x[1:end-1],3,Mt); return (max = maximum(xtt[1,:]), min = minimum(xtt[1,:]), period = x[end])),
	plotSolution = (x, p; k...) -> begin
		xtt = BK.getTrajectory(p.prob, x, p.p)
		plot!(xtt.t, xtt.u[1,:]; markersize = 2, k...)
		plot!(xtt.t, xtt.u[2,:]; k...)
		plot!(xtt.t, xtt.u[3,:]; legend = false, k...)
		plot!(br, subplot=1, putbifptlegend = false)
		end,
	finaliseSolution = (z, tau, step, contResult; prob = nothing, kwargs...) -> begin
		# limit the period
			return z.u[end] < 30
			true
		end,
	normC = norminf)

scene = plot(br, br_po)
```

Two period doubling bifurcations were detected. We shall now compute the branch of periodic orbits from these PD points. We do not provide Automatic Branch Switching as we do not have the PD normal form computed in `BifurcationKit`. Hence, it takes some trial and error to find the `ampfactor` of the PD branch.

```@example TUTLURE
# aBS from PD
br_po_pd, = continuation(br_po, 1, setproperties(br_po.contparams, detectBifurcation = 3, maxSteps = 51, ds = 0.01, dsmax = 0.01, plotEveryStep = 10);
	verbosity = 3, plot = true,
	ampfactor = .1, δp = -0.005,
	usedeflation = false,
	linearPO = :Dense,
	updateSectionEveryStep = 1,
	plotSolution = (x, p; k...) -> begin
		xtt = BK.getTrajectory(br_po.functional, x, (@set par_lur.β = p))
		plot!(xtt.t, xtt.u[1,:]; markersize = 2, k...)
		plot!(xtt.t, xtt.u[2,:]; k...)
		plot!(xtt.t, xtt.u[3,:]; legend = false, k...)
		plot!(br_po; legend=false, subplot=1)
	end,

	recordFromSolution = (x, p) -> (xtt=reshape(x[1:end-1],3,Mt); return (max = maximum(xtt[1,:]), min = minimum(xtt[1,:]), period = x[end])),
	normC = norminf
	)
Scene = title!("")
```

```@example TUTLURE
plot(br, br_po, br_po_pd, xlims=(0.5,0.65))
```

## Periodic orbits with Parallel Standard Shooting

We use a different method to compute periodic orbits: we rely on a fixed point of the flow. To compute the flow, we use `DifferentialEquations.jl`. This way of computing periodic orbits should be more precise than the previous one. We use a particular instance called multiple shooting which is computed in parallel. This is an additional advantage compared to the previous method. Finally, please note the close similarity to the code of the previous part. As before, we first rely on Hopf **aBS**.

```@example TUTLURE
using DifferentialEquations

# plotting function
plotSH = (x, p; k...) -> begin
	xtt = BK.getTrajectory(p.prob, x, @set par_lur.β = p.p)
	plot!(xtt.t, xtt[1,:]; markersize = 2, k...)
	plot!(xtt.t, xtt[2,:]; k...)
	plot!(xtt.t, xtt[3,:]; legend = false, k...)
	plot!(br, subplot=1, putbifptlegend = false)
end

# ODE problem for using DifferentialEquations
probsh = ODEProblem(lur!, copy(z0), (0., 1000.), par_lur; atol = 1e-10, rtol = 1e-7)

# newton parameters
optn_po = NewtonPar(verbose = true, tol = 1e-8, maxIter = 25)

# continuation parameters
opts_po_cont = ContinuationPar(dsmax = 0.01, ds= -0.001, dsmin = 1e-4, maxSteps = 130, newtonOptions = optn_po, precisionStability = 1e-5, detectBifurcation = 3, plotEveryStep = 10, saveSolEveryStep = 1, nInversion = 6, nev = 2)

br_po, = continuation(
	jet..., br, 1, opts_po_cont,
	# parallel shooting functional with 10 sections
	ShootingProblem(15, probsh, Rodas4P(); parallel = true, reltol = 1e-9);
	# first parameter value on the branch
	δp = 0.0051,
	# method for solving newton linear system
	# specific to ODE
	linearPO = :autodiffDense,
	verbosity = 3,	plot = true,
	recordFromSolution = (x, p) -> (return (max = getMaximum(p.prob, x, @set par_lur.β = p.p), period = getPeriod(p.prob, x, @set par_lur.β = p.p))),
	plotSolution = plotSH,
	# limit the residual, useful to help DifferentialEquations
	callbackN = BK.cbMaxNorm(10),
	normC = norminf)

scene = title!("")
```

We do not provide Automatic Branch Switching as we do not have the PD normal form computed in `BifurcationKit`. Hence, it takes some trial and error to find the `ampfactor` of the PD branch.

```@example TUTLURE
# aBS from PD
br_po_pd, = BK.continuation(br_po, 1, setproperties(br_po.contparams, detectBifurcation = 3, maxSteps = 50, ds = 0.01, plotEveryStep = 1);
	verbosity = 3, plot = true,
	ampfactor = .3, δp = -0.005,
	linearPO = :autodiffDense,
	plotSolution = (x, p; k...) -> begin
		plotSH(x, p; k...)
		plot!(br_po; subplot = 1)
	end,
	recordFromSolution = (x, p) -> (return (max = getMaximum(p.prob, x, @set par_lur.β = p.p), period = getPeriod(p.prob, x, @set par_lur.β = p.p))),
	normC = norminf,
	callbackN = BK.cbMaxNorm(10),
	)

scene = plot(br_po, br_po_pd)
```
