using Revise, Test, ForwardDiff, Parameters, Setfield, Plots, LinearAlgebra
using BifurcationKit
const BK = BifurcationKit

norminf(x) = norm(x, Inf)
####################################################################################################
function TMvf!(dz, z, p, t)
	@unpack J, α, E0, τ, τD, τF, U0 = p
	E, x, u = z
	SS0 = J * u * x * E + E0
	SS1 = α * log(1 + exp(SS0 / α))
	[
		(-E + SS1) / τ
		(1.0 - x) / τD - u * x * E
		(U0 - u) / τF +  U0 * (1.0 - u) * E
	]
end

TMvf(z, p) = TMvf!(similar(z), z, p, 0)
dTMvf(z,p) = ForwardDiff.jacobian(x-> TMvf(x,p), z)

# we group the differentials together
jet  = BK.getJet(TMvf, matrixfree=false)

par_tm = (α = 1.5, τ = 0.013, J = 3.07, E0 = -2.0, τD = 0.200, U0 = 0.3, τF = 1.5, τS = 0.007) #2.87
z0 = [0.238616, 0.982747, 0.367876 ]

opts_br = ContinuationPar(pMin = -10.0, pMax = -0.9, ds = 0.04, dsmax = 0.125, nInversion = 8, detectBifurcation = 3, maxBisectionSteps = 25, nev = 3)
	opts_br = @set opts_br.newtonOptions.verbose = false
	br, = continuation(TMvf, dTMvf, z0, par_tm, (@lens _.E0), opts_br;
	printSolution = (x, p) -> (E = x[1], x = x[2], u = x[3]),
	tangentAlgo = BorderedPred(),
	plot = true, normC = norminf)

plot(br, plotfold=false, markersize=4, legend=:topleft)
####################################################################################################
hopfpt = computeNormalForm(jet..., br, 4)

# newton parameters
optn_po = NewtonPar(verbose = true, tol = 1e-8,  maxIter = 8)

# continuation parameters
opts_po_cont = ContinuationPar(dsmax = 0.1, ds= 0.0001, dsmin = 1e-4, pMax = 0., pMin=-5., maxSteps = 110, newtonOptions = (@set optn_po.tol = 1e-7), nev = 3, precisionStability = 1e-8, detectBifurcation = 0, plotEveryStep = 20, saveSolEveryStep=1)

# arguments for periodic orbits
args_po = (	recordFromSolution = (x, p) -> begin
		xtt = BK.getPeriodicOrbit(p.prob, x, @set par_tm.E0 = p.p)
		return (max = maximum(xtt[1,:]),
				min = minimum(xtt[1,:]),
				period = getPeriod(p.prob, x, @set par_tm.E0 = p.p))
	end,
	plotSolution = (x, p; k...) -> begin
		xtt = BK.getPeriodicOrbit(p.prob, x, @set par_tm.E0 = p.p)
		@show size(xtt[:,:]) maximum(xtt[1,:])
		plot!(xtt.t, xtt[1,:]; label = "E", k...)
		plot!(xtt.t, xtt[2,:]; label = "x", k...)
		plot!(xtt.t, xtt[3,:]; label = "u", k...)
		plot!(br; subplot = 1, putspecialptlegend = false)
		end,
	normC = norminf)

Mt = 200 # number of sections
	br_potrap, utrap = continuation(
	jet..., br, 4, opts_po_cont,
	PeriodicOrbitTrapProblem(M = Mt);
	jacobianPO = :Dense,
	verbosity = 2,	plot = true,
	args_po...,
	)

plot(br, br_potrap, markersize = 3)
	plot!(br_potrap.param, br_potrap.min, label = "")
####################################################################################################
# based on collocation
hopfpt = computeNormalForm(jet..., br, 4)

# newton parameters
optn_po = NewtonPar(verbose = true, tol = 1e-8,  maxIter = 25)

# continuation parameters
opts_po_cont = ContinuationPar(dsmax = 0.1, ds= -0.001, dsmin = 1e-4, pMax = 0., pMin=-5., maxSteps = 150, newtonOptions = (@set optn_po.tol = 1e-7), nev = 3, precisionStability = 1e-5, detectBifurcation = 0, plotEveryStep = 40, saveSolEveryStep=1)

br_pocoll, ucoll, = @time continuation(
	jet..., br, 4, opts_po_cont,
	PeriodicOrbitOCollProblem(20, 5);
	tangentAlgo = BorderedPred(),
	updateSectionEveryStep = 1,
	verbosity = 2,	plot = false,
	args_po...,
	plotSolution = (x, p; k...) -> begin
		xtt = BK.getPeriodicOrbit(p.prob, x, p.p)
		plot!(xtt.t, xtt[1,:]; label = "", marker =:d, markersize = 1.5, k...)
		plot!(br; subplot = 1, putspecialptlegend = false)

	end,
	callbackN = BK.cbMaxNorm(1000.),
	finaliseSolution = (z, tau, step, contResult; k...) -> begin
		prob = k[:prob]
		newt, err = BK.computeError(prob, z.u; verbosity = 3, par = BK.setParam(contResult, z.p))#, K = 100)
		return true
	end,

	)

plot(br, br_pocoll, markersize = 3)
	# plot!(br_pocoll.param, br_pocoll.min, label = "")
	# plot!(br, br_potrap, markersize = 3)
	# plot!(br_potrap.param, br_potrap.min, label = "", marker = :d)

####################################################################################################
# idem with Standard shooting
using DifferentialEquations#, TaylorIntegration

# this is the ODEProblem used with `DiffEqBase.solve`
probsh = ODEProblem(TMvf!, copy(z0), (0., 1000.), par_tm; abstol = 1e-10, reltol = 1e-9)

opts_po_cont = ContinuationPar(dsmax = 0.1, ds= -0.0001, dsmin = 1e-4, pMax = 0., pMin=-5., maxSteps = 120, newtonOptions = (@set optn_po.tol = 1e-6), nev = 3, precisionStability = 1e-8, detectBifurcation = 0, plotEveryStep = 10, saveSolEveryStep=0)

br_posh, = @time continuation(jet...,
	br, 4, opts_po_cont,
	# this is where we tell that we want Standard Shooting
	# with 15 time sections
	ShootingProblem(15, probsh, Rodas4(), parallel = true);
	# this to help branching
	δp = 0.0005,
	# deflation helps not converging to an equilibrium instead of a PO
	usedeflation = true,
	# this linear solver is specific to ODEs
	# it is computed using AD of the flow and
	# updated inplace
	jacobianPO = :autodiffDense,
	# we update the section along the branches
	updateSectionEveryStep = 2,
	# regular continuation parameters
	verbosity = 2,	plot = true,
	args_po...)

plot(br_posh, br, markersize=3)
	# plot(br, br_potrap, br_posh, markersize=3)
####################################################################################################
# idem with Poincaré shooting
@assert 1==0 "There is a PB with DE!!! I am getting NaN :("

function TMvfExtended!(dz, z, p, t)
	# we write the first part
	TMvf!(dz, z, p, t)
	dz[4:end] .= @views ForwardDiff.derivative(t -> TMvf(z[1:3] .+ t .* z[4:end], p), 0)
end

probmono = ODEProblem(TMvfExtended!, vcat(z0, z0), (0., 1000.), par_tm; abstol = 1e-10, reltol = 1e-9)

opts_po_cont = ContinuationPar(dsmax = 0.02, ds= 0.001, dsmin = 1e-6, pMax = 0., pMin=-5., maxSteps = 200, newtonOptions = NewtonPar(optn_po;tol = 1e-6, maxIter=15), nev = 3, precisionStability = 1e-8, detectBifurcation = 0, plotEveryStep = 1, saveSolEveryStep = 1)

br_popsh, = @time continuation(
	jet..., br, 4,
	# arguments for continuation
	opts_po_cont,
	# this is where we tell that we want Poincaré Shooting
	# PoincareShootingProblem(Mt, probsh, Rodas5(), probmono, Rodas4P(), parallel = false);
	PoincareShootingProblem(10, probsh, Rodas4P2(), parallel = false);
	ampfactor = 1.0, δp = 0.005,
	# usedeflation = true,
	jacobianPO = :autodiffDenseAnalytical,
	# jacobianPO = :autodiffDense,
	linearAlgo = MatrixBLS(),
	updateSectionEveryStep = 2,
	verbosity = 2,	plot = true,
	args_po...,
	callbackN = BK.cbMaxNorm(1e2),
	recordFromSolution = (x, p) -> (return (max = getMaximum(p.prob, x, @set par_tm.E0 = p.p), period = getPeriod(p.prob, x, @set par_tm.E0 = p.p))),
	normC = norminf)

plot(br, br_popsh, markersize=3)
