# using Revise, Plots
using Parameters, Setfield, LinearAlgebra, Test
using BifurcationKit, Test
const BK = BifurcationKit

norminf = (x) -> norm(x, Inf)
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
jet = BK.getJet(lur; matrixfree=false)

par_lur = (α = 1.0, β = 0.)
z0 = zeros(3)

opts_br = ContinuationPar(pMin = -0.4, pMax = 1.8, ds = -0.01, dsmax = 0.01, nInversion = 8, detectBifurcation = 3, maxBisectionSteps = 25, nev = 3, plotEveryStep = 20, maxSteps = 1000, theta = 0.3)
	opts_br = @set opts_br.newtonOptions.verbose = false
	br, = continuation(jet[1], jet[2], z0, par_lur, (@lens _.β), opts_br;
	recordFromSolution = recordFromSolution,
	tangentAlgo = BorderedPred(),
	bothside = true, normC = norminf)

# plot(br)
####################################################################################################
# newton parameters
optn_po = NewtonPar(tol = 1e-8,  maxIter = 25)

# continuation parameters
opts_po_cont = ContinuationPar(dsmax = 0.03, ds= 0.0001, dsmin = 1e-4, pMax = 1.8, pMin=-5., maxSteps = 130, newtonOptions = (@set optn_po.tol = 1e-8), nev = 3, precisionStability = 1e-4, detectBifurcation = 3, plotEveryStep = 20, saveSolEveryStep=1, nInversion = 6)

Mt = 90 # number of time sections
	br_po, = continuation(
	jet..., br, 1, opts_po_cont,
	PeriodicOrbitTrapProblem(M = Mt);
	ampfactor = 1., δp = 0.01,
	updateSectionEveryStep = 1,
	linearPO = :Dense,
	verbosity = 0,	plot = false,
	recordFromSolution = (x, p) -> (xtt=reshape(x[1:end-1],3,Mt); return (max = maximum(xtt[1,:]), min = minimum(xtt[1,:]), period = x[end])),
	finaliseSolution = (z, tau, step, contResult; prob = nothing, kwargs...) -> begin
			return z.u[end] < 40
			true
		end,
	normC = norminf)

# plot(br, br_po)
# plot(br_po, vars=(:param, :period))

# aBS from PD
br_po_pd, = BK.continuation(br_po, 1, setproperties(br_po.contparams, detectBifurcation = 3, maxSteps = 51, ds = 0.01, dsmax = 0.01, plotEveryStep = 10);
	verbosity = 0, plot = false,
	ampfactor = .1, δp = -0.005,
	usedeflation = false,
	linearPO = :Dense,
	updateSectionEveryStep = 1,
	recordFromSolution = (x, p) -> (xtt=reshape(x[1:end-1],3,Mt); return (max = maximum(xtt[1,:]), min = minimum(xtt[1,:]), period = x[end])),
	normC = norminf
	)

# plot(br, br_po, br_po_pd, xlims=(0.5,0.65))
####################################################################################################
using OrdinaryDiffEq

probsh = ODEProblem(lur!, copy(z0), (0., 1000.), par_lur; atol = 1e-10, rtol = 1e-7)

optn_po = NewtonPar(tol = 1e-7, maxIter = 25)

# continuation parameters
opts_po_cont = ContinuationPar(dsmax = 0.01, ds= -0.001, dsmin = 1e-4, maxSteps = 130, newtonOptions = (@set optn_po.tol = 1e-8), precisionStability = 1e-5, detectBifurcation = 3, plotEveryStep = 10, saveSolEveryStep = 0, nInversion = 6, nev = 3)

br_po, = continuation(
	jet..., br, 1, opts_po_cont,
	ShootingProblem(15, par_lur, probsh, Rodas4P(); parallel = true, reltol = 1e-9);
	ampfactor = 1., δp = 0.0051,
	updateSectionEveryStep = 1,
	linearPO = :autodiffDense,
	# verbosity = 3,	plot = true,
	# recordFromSolution = (x, p) -> (return (max = getMaximum(p.prob, x, @set par_lur.β = p.p), period = getPeriod(p.prob, x, @set par_lur.β = p.p))),
	# plotSolution = plotSH,
	# finaliseSolution = (z, tau, step, contResult; prob = nothing, kwargs...) -> begin
	# 		BK.haseigenvalues(contResult) && Base.display(contResult.eig[end].eigenvals)
	# 		return z.u[end] < 30 && length(contResult.specialpoint) < 3
	# 		true
	# 	end,
	callbackN = BK.cbMaxNorm(10),
	normC = norminf)

# plot(br, br_po)
# plot(br_po, vars=(:param, :period))

@test br_po.specialpoint[1].param ≈ 0.6273246 rtol = 1e-4
@test br_po.specialpoint[2].param ≈ 0.5417461 rtol = 1e-4

# aBS from PD
br_po_pd, = BK.continuation(br_po, 1, setproperties(br_po.contparams, detectBifurcation = 3, maxSteps = 50, ds = 0.01, plotEveryStep = 1);
	# verbosity = 3, plot = true,
	ampfactor = .3, δp = -0.005,
	linearPO = :autodiffDense,
	normC = norminf,
	callbackN = BK.cbMaxNorm(10),
	)

# plot(br_po, br_po_pd)
#######################################
opts_po_cont_ps = @set opts_po_cont.newtonOptions.tol = 1e-7
@set opts_po_cont_ps.dsmax = 0.0025
br_po, = continuation(jet..., br, 1, opts_po_cont_ps,
	PoincareShootingProblem(2, par_lur, probsh, Rodas4P(); parallel = true, reltol = 1e-6);
	ampfactor = 1., δp = 0.0051, #verbosity = 3,plot=true,
	updateSectionEveryStep = 1,
	linearPO = :autodiffDenseAnalytical,
	callbackN = BK.cbMaxNorm(10),
	normC = norminf)

# aBS from PD
br_po_pd, = BK.continuation(br_po, 1, setproperties(br_po.contparams, detectBifurcation = 3, maxSteps = 50, ds = 0.01, plotEveryStep = 1);
	# verbosity = 3, plot = true,
	ampfactor = .3, δp = -0.005,
	linearPO = :autodiffDenseAnalytical,
	normC = norminf,
	callbackN = BK.cbMaxNorm(10),
	)
