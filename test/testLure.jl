# using Revise, Plots
using Parameters, Setfield, LinearAlgebra, Test
using BifurcationKit, Test
const BK = BifurcationKit

norminf(x) = norm(x, Inf)
recordFromSolution(x, p) = (u1 = x[1], u2 = x[2])
####################################################################################################
function lur!(dz, u, p, t)
	@unpack α, β = p
	x, y, z = u
	dz[1] = y
	dz[2] =	z
	dz[3] = -α * z - β * y - x + x^2
	dz
end

lur(z, p) = lur!(similar(z), z, p, 0)
par_lur = (α = 1.0, β = 0.)
z0 = zeros(3)
prob = BK.BifurcationProblem(lur, z0, par_lur, (@lens _.β); recordFromSolution = recordFromSolution)

opts_br = ContinuationPar(pMin = -0.4, pMax = 1.8, ds = -0.01, dsmax = 0.01, nInversion = 8, detectBifurcation = 3, maxBisectionSteps = 25, nev = 3, plotEveryStep = 20, maxSteps = 1000, θ = 0.3)
	opts_br = @set opts_br.newtonOptions.verbose = false
	br = continuation(prob, PALC(tangent = Bordered()), opts_br;
	bothside = true, normC = norminf)

# plot(br)
####################################################################################################
# newton parameters
optn_po = NewtonPar(tol = 1e-8,  maxIter = 25)

# continuation parameters
opts_po_cont = ContinuationPar(dsmax = 0.03, ds= 0.0001, dsmin = 1e-4, pMax = 1.8, pMin=-5., maxSteps = 130, newtonOptions = (@set optn_po.tol = 1e-8), nev = 3, tolStability = 1e-4, detectBifurcation = 3, plotEveryStep = 20, saveSolEveryStep=1, nInversion = 6)

Mt = 90 # number of time sections
	br_po = continuation(
	br, 2, opts_po_cont,
	PeriodicOrbitTrapProblem(M = Mt; updateSectionEveryStep = 1,
	jacobian = :Dense);
	ampfactor = 1., δp = 0.01,
	verbosity = 0,	plot = false,
	recordFromSolution = (x, p) -> (xtt=reshape(x[1:end-1],3,Mt); return (max = maximum(xtt[1,:]), min = minimum(xtt[1,:]), period = x[end])),
	finaliseSolution = (z, tau, step, contResult; prob = nothing, kwargs...) -> begin
			return z.u[end] < 40
			true
		end,
	normC = norminf)

# plot(br, br_po)
# plot(br_po, vars=(:param, :period))

# test normal forms
for _ind in (1,3,16)
	if br_po.specialpoint[_ind].type ∈ (:bp, :pd, :ns)
		println("")
		pt = getNormalForm(br_po, _ind; verbose = true)
		predictor(pt, 0.1, 1.)
		show(pt)
	end
end

# aBS from PD
br_po_pd = continuation(br_po, 1, setproperties(br_po.contparams, detectBifurcation = 3, maxSteps = 51, ds = 0.01, dsmax = 0.01, plotEveryStep = 10);
	verbosity = 0, plot = false,
	ampfactor = .1, δp = -0.005,
	usedeflation = false,
	recordFromSolution = (x, p) -> (xtt=reshape(x[1:end-1],3,Mt); return (max = maximum(xtt[1,:]), min = minimum(xtt[1,:]), period = x[end])),
	normC = norminf
	)

# plot(br, br_po, br_po_pd, xlims=(0.5,0.65))
####################################################################################################
# continuation parameters
opts_po_cont = ContinuationPar(dsmax = 0.03, ds= -0.0001, dsmin = 1e-4, pMax = 1.8, pMin=-5., maxSteps = 120, newtonOptions = (@set optn_po.tol = 1e-8), nev = 3, tolStability = 1e-4, detectBifurcation = 3, plotEveryStep = 20, saveSolEveryStep = 1, nInversion = 6)

br_po = continuation(
	br, 2, opts_po_cont,
	BK.PeriodicOrbitOCollProblem(20, 4);
	alg = PALC(tangent = Bordered()),
	ampfactor = 1., δp = 0.01,
	# usedeflation = true,
	# verbosity = 2,	plot = true,
	normC = norminf)

# test normal forms
for _ind in (1,)
	if br_po.specialpoint[_ind].type ∈ (:bp, :pd, :ns)
		println("")
		pt = getNormalForm(br_po, _ind; verbose = true)
		predictor(pt, 0.1, 1.)
		show(pt)
	end
end
####################################################################################################
using OrdinaryDiffEq

probsh = ODEProblem(lur!, copy(z0), (0., 1000.), par_lur; abstol = 1e-10, reltol = 1e-7)

optn_po = NewtonPar(tol = 1e-7, maxIter = 25)

# continuation parameters
opts_po_cont = ContinuationPar(dsmax = 0.02, ds= -0.001, dsmin = 1e-4, maxSteps = 122, newtonOptions = (@set optn_po.tol = 1e-8), tolStability = 1e-5, detectBifurcation = 3, plotEveryStep = 10, nInversion = 6, nev = 3)

br_po = continuation(
	br, 2, opts_po_cont,
	ShootingProblem(15, probsh, Rodas5P(); parallel = true, reltol = 1e-9, updateSectionEveryStep = 1, jacobian = :autodiffDense);
	ampfactor = 1., δp = 0.0051,
	# verbosity = 3,	plot = true,
	recordFromSolution = (x, p) -> (return (max = getMaximum(p.prob, x, @set par_lur.β = p.p), period = getPeriod(p.prob, x, @set par_lur.β = p.p))),
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

# test showing normal form
for _ind in (1,3)
	if br_po.specialpoint[_ind].type ∈ (:bp, :pd, :ns)
		println("")
		pt = getNormalForm(br_po, _ind; verbose = true)
		predictor(pt, 0.1, 1.)
		show(pt)
	end
end

# aBS from PD
br_po_pd = continuation(br_po, 1, setproperties(br_po.contparams, detectBifurcation = 3, maxSteps = 50, ds = 0.01, plotEveryStep = 1, saveSolEveryStep = 1);
	verbosity = 0, plot = false,
	usedeflation = false,
	ampfactor = .3, δp = -0.005,
	recordFromSolution = (x, p) -> (return (max = getMaximum(p.prob, x, @set par_lur.β = p.p), period = getPeriod(p.prob, x, @set par_lur.β = p.p))),
	normC = norminf,
	callbackN = BK.cbMaxNorm(10),
	)
@test br_po_pd.sol[1].x[end] ≈ 16.956 rtol = 1e-4

# plot(br_po, br_po_pd)
#######################################
@info "testLure Poincare"
opts_po_cont_ps = @set opts_po_cont.newtonOptions.tol = 1e-7
@set opts_po_cont_ps.dsmax = 0.0025
br_po = continuation(br, 2, opts_po_cont_ps,
	PoincareShootingProblem(2, probsh, Rodas4P(); parallel = true, reltol = 1e-6, updateSectionEveryStep = 1, jacobian = :autodiffDenseAnalytical);
	ampfactor = 1., δp = 0.0051, #verbosity = 3,plot=true,
	callbackN = BK.cbMaxNorm(10),
	recordFromSolution = (x, p) -> (return (max = getMaximum(p.prob, x, @set par_lur.β = p.p), period = getPeriod(p.prob, x, @set par_lur.β = p.p))),
	normC = norminf)

# plot(br_po, br)

# test showing normal form
for _ind in (1,)
	if br_po.specialpoint[_ind].type ∈ (:bp, :pd, :ns)
		println("")
		pt = getNormalForm(br_po, _ind; verbose = true)
		predictor(pt, 0.1, 1.)
		show(pt)
	end
end

# aBS from PD
br_po_pd = BK.continuation(br_po, 1, setproperties(br_po.contparams, detectBifurcation = 3, maxSteps = 50, ds = 0.01, plotEveryStep = 1);
	# verbosity = 3, plot = true,
	ampfactor = .3, δp = -0.005,
	normC = norminf,
	callbackN = BK.cbMaxNorm(10),
	recordFromSolution = (x, p) -> (return (max = getMaximum(p.prob, x, @set par_lur.β = p.p), period = getPeriod(p.prob, x, @set par_lur.β = p.p))),
	)

# plot(br_po_pd, br_po)
