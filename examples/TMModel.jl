using Revise, Test, ForwardDiff, Parameters, Setfield, Plots, LinearAlgebra
using BifurcationKit
const BK = BifurcationKit

D(f, x, p, dx) = ForwardDiff.derivative(t->f(x .+ t .* dx, p), 0.)
norminf(x) = norm(x, Inf)
####################################################################################################
function TMvf!(dz, z, p, t)
	@unpack J, α, E0, τ, τD, τF, U0 = p
	E, x, u = z
	SS0 = J * u * x * E + E0
	SS1 = α * log(1 + exp(SS0 / α))
	dz[1] = (-E + SS1) / τ
	dz[2] =	(1.0 - x) / τD - u * x * E
	dz[3] = (U0 - u) / τF +  U0 * (1.0 - u) * E
	dz
end

TMvf(z, p) = TMvf!(similar(z), z, p, 0)
dTMvf(z,p) = ForwardDiff.jacobian(x-> TMvf(x,p), z)

# we group the differentials together
jet  = BK.getJet(TMvf, dTMvf)

par_tm = (α = 1.5, τ = 0.013, J = 3.07, E0 = -2.0, τD = 0.200, U0 = 0.3, τF = 1.5, τS = 0.007) #2.87
z0 = [0.238616, 0.982747, 0.367876 ]

opts_br = ContinuationPar(pMin = -10.0, pMax = -0.9, ds = 0.04, dsmax = 0.125, nInversion = 8, detectBifurcation = 3, maxBisectionSteps = 25, nev = 3)
	opts_br = @set opts_br.newtonOptions.verbose = false
	br, = continuation(TMvf, dTMvf, z0, par_tm, (@lens _.E0), opts_br;
	printSolution = (x, p) -> (E = x[1], x = x[2], u = x[3]),
	tangentAlgo = BorderedPred(),
	plot = true, verbosity = 0, normC = norminf)

plot(br, plotfold=false, markersize=4, legend=:topleft)
####################################################################################################
hopfpt = computeNormalForm(jet..., br, 4)

# newton parameters
optn_po = NewtonPar(verbose = true, tol = 1e-8,  maxIter = 8)

# continuation parameters
opts_po_cont = ContinuationPar(dsmax = 0.1, ds= -0.0001, dsmin = 1e-4, pMax = 0., pMin=-5., maxSteps = 110, newtonOptions = (@set optn_po.tol = 1e-7), nev = 2, precisionStability = 1e-8, detectBifurcation = 3, plotEveryStep = 10, saveSolEveryStep=1)

Mt = 200 # number of sections
	br_potrap, utrap = continuation(
	jet..., br, 4, opts_po_cont,
	PeriodicOrbitTrapProblem(M = Mt);
	linearPO = :Dense,
	verbosity = 2,	plot = true,
	recordFromSolution = (x, p) -> (xtt=reshape(x[1:end-1],3,Mt); return (max = maximum(xtt[1,:]), min = minimum(xtt[1,:]), period = x[end])),
	plotSolution = (x, p; k...) -> begin
		xtt = BK.getTrajectory(p.prob, x, p.p)
		plot!(xtt.t, xtt.u[1,:]; label = "E", k...)
		plot!(xtt.t, xtt.u[2,:]; label = "x", k...)
		plot!(xtt.t, xtt.u[3,:]; label = "u", k...)
		plot!(br,subplot=1, putbifptlegend = false)
		end,
	normC = norminf)

plot(br, br_potrap, markersize = 3)
	plot!(br_potrap.param, br_potrap.min, label = "")

####################################################################################################
# idem with Standard shooting
using DifferentialEquations

# this is the ODEProblem used with `DiffEqBase.solve`
probsh = ODEProblem(TMvf!, copy(z0), (0., 1000.), par_tm; atol = 1e-10, rtol = 1e-9)

opts_po_cont = ContinuationPar(dsmax = 0.05, ds= -0.0001, dsmin = 1e-4, pMax = 0., pMin=-5., maxSteps = 210, newtonOptions = (@set optn_po.tol = 1e-6), nev = 25, precisionStability = 1e-8, detectBifurcation = 0, plotEveryStep = 10, saveSolEveryStep=0)

br_posh, = @time continuation(
	jet..., br, 4,
	# arguments for continuation
	opts_po_cont,
	# this is where we tell that we want Standard Shooting
	ShootingProblem(15, par_tm, probsh, Rodas4(), parallel = true);
	ampfactor = 1.0, δp = 0.0005,
	usedeflation = true,
	linearPO = :autodiffDense,
	updateSectionEveryStep = 2,
	verbosity = 2,	plot = true,
	recordFromSolution = (x, p) -> (return (max = getMaximum(p.prob, x, @set par_tm.E0 = p.p), period = getPeriod(p.prob, x, @set par_tm.E0 = p.p))),
	plotSolution = (x, p; k...) ->
		begin
			xtt = BK.getTrajectory(p.prob, x, @set par_tm.E0 = p.p)
			plot!(xtt; legend = false, k...);
			plot!(br, subplot=1, putspecialptlegend = false)
		end,
	normC = norminf)

plot(br_posh, br, markersize=3)
plot(br, br_potrap, br_posh, markersize=3)
####################################################################################################
# idem with Poincaré shooting

function TMvfExtended!(dz, z, p, t)
	# we write the first part
	TMvf!(dz, z, p, t)
	dz[4:end] .= @views ForwardDiff.derivative(t -> TMvf(z[1:3] .+ t .* z[4:end], p), 0)
end

probmono = ODEProblem(TMvfExtended!, vcat(z0, z0), (0., 1000.), par_tm; atol = 1e-10, rtol = 1e-9)

opts_po_cont = ContinuationPar(dsmax = 0.02, ds= -0.001, dsmin = 1e-5, pMax = 0., pMin=-5., maxSteps = 200, newtonOptions = NewtonPar(optn_po;tol = 1e-6, maxIter=15), nev = 25, precisionStability = 1e-8, detectBifurcation = 0, plotEveryStep = 5, saveSolEveryStep = 2)

br_popsh, = @time continuation(
	jet..., br, 4,
	# arguments for continuation
	opts_po_cont,
	# this is where we tell that we want Poincaré Shooting
	PoincareShootingProblem(9, par_tm, probsh, Rodas4P(), probmono, Rosenbrock23(), parallel = false);
	ampfactor = 1.0, δp = 0.0005,
	usedeflation = false,
	linearPO = :autodiffDense,
	updateSectionEveryStep = 2,
	verbosity = 2,	plot = true,
	recordFromSolution = (x, p) -> (return (max = getMaximum(p.prob, x, @set par_tm.E0 = p.p), period = getPeriod(p.prob, x, @set par_tm.E0 = p.p))),
	plotSolution = (x, p; k...) ->
		begin
			xtt = BK.getTrajectory(p.prob, x, @set par_tm.E0 = p.p)
			plot!(xtt; legend = false, k...);
			plot!(br,subplot=1, putspecialptlegend = false)
			# plot!(br_potrap,subplot=1, putspecialptlegend = false)
		end,
	callbackN = (x, f, J, res, iteration, itlinear, options; kwargs...) -> (return res<1e16),
	finaliseSolution = (z, tau, step, contResult; prob=nothing, k...) ->
		begin
			isnothing(prob) && return true
			T = getPeriod(prob, z.u, @set par_tm.E0 = z.p)
			@show T
			T < 20.
		end,
	normC = norminf)

plot(br, br_popsh, markersize=3)
