# using Revise, Plots
using Test, ForwardDiff, Parameters, LinearAlgebra
using BifurcationKit, Test
const BK = BifurcationKit

norminf(x) = norm(x, Inf)
###################################################################################################
function Pop!(du, X, p, t = 0)
	@unpack r,K,a,ϵ,b0,e,d, = p
	x, y, u, v = X
	p = a * x / (b0 * (1 + ϵ * u) + x)
	du[1] = r * (1 - x/K) * x - p * y
	du[2] = e * p * y - d * y
	s = u^2 + v^2
	du[3] = u-2pi * v - s * u
	du[4] = 2pi * u + v - s * v
	du
end
Pop(u,p) = Pop!(similar(u),u,p,0)

par_pop = ( K = 1., r = 2π, a = 4π, b0 = 0.25, e = 1., d = 2π, ϵ = 0.2, )

z0 = [0.1,0.1,1,0]

prob = BK.BifurcationProblem(Pop, z0, par_pop, (@lens _.b0); recordFromSolution = (x, p) -> (x = x[1], y = x[2], u = x[3]))

opts_br = ContinuationPar(pMin = 0., pMax = 20.0, ds = 0.002, dsmax = 0.01, nInversion = 6, detectBifurcation = 3, maxBisectionSteps = 25, nev = 4, maxSteps = 20000)
@set! opts_br.newtonOptions.verbose = false

################################################################################
using DifferentialEquations
prob_de = ODEProblem(Pop!, z0, (0,600.), par_pop)
alg = Rodas5()
# alg = Vern9()
sol = solve(prob_de, alg)
prob_de = ODEProblem(Pop!, sol.u[end], (0,5.), par_pop, reltol = 1e-8, abstol = 1e-10)
sol = solve(prob_de, Rodas5())
################################################################################
argspo = (recordFromSolution = (x, p) -> begin
		xtt = BK.getPeriodicOrbit(p.prob, x, p.p)
		return (max = maximum(xtt[1,:]),
				min = minimum(xtt[1,:]),
				period = getPeriod(p.prob, x, p.p))
	end,
	plotSolution = (x, p; k...) -> begin
		xtt = BK.getPeriodicOrbit(p.prob, x, p.p)
		plot!(xtt.t, xtt[1,:]; label = "x", k...)
		plot!(xtt.t, xtt[2,:]; label = "y", k...)
		# plot!(br; subplot = 1, putspecialptlegend = false)
	end)
################################################################################
probcoll, ci = generateCIProblem(PeriodicOrbitOCollProblem(26, 3; updateSectionEveryStep = 0), prob, sol, 2.)

solpo = newton(probcoll, ci, NewtonPar(verbose = false))
@test BK.converged(solpo)

_sol = BK.getPeriodicOrbit(probcoll, solpo.u,1)

opts_po_cont = setproperties(opts_br, maxSteps = 50, saveEigenvectors = true, tolStability = 1e-8)
@set! opts_po_cont.newtonOptions.verbose = false
brpo_fold = continuation(probcoll, ci, PALC(), opts_po_cont;
	verbosity = 0, plot = false,
	argspo...
	)
# pt = getNormalForm(brpo_fold, 1)

prob2 = @set probcoll.prob_vf.lens = @lens _.ϵ
brpo_pd = continuation(prob2, ci, PALC(), ContinuationPar(opts_po_cont, dsmax = 5e-3);
	verbosity = 0, plot = false,
	argspo...
	)
# pt = getNormalForm(brpo_pd, 1, prm = false)

# codim 2 Fold
opts_pocoll_fold = ContinuationPar(brpo_fold.contparams, detectBifurcation = 3, maxSteps = 120, pMin = 0., pMax=1.2, nInversion = 4, plotEveryStep = 10)
@set! opts_pocoll_fold.newtonOptions.tol = 1e-12
fold_po_coll1 = continuation(brpo_fold, 1, (@lens _.ϵ), opts_pocoll_fold;
		verbosity = 0, plot = false,
		detectCodim2Bifurcation = 0,
		startWithEigen = false,
		bothside = true,
		jacobian_ma = :minaug,
		bdlinsolver = BorderingBLS(solver = DefaultLS(), checkPrecision = false),
		)
@test fold_po_coll1.kind isa BK.FoldPeriodicOrbitCont

# codim 2 PD
opts_pocoll_pd = ContinuationPar(brpo_pd.contparams, detectBifurcation = 3, maxSteps = 20, pMin = -1., plotEveryStep = 10, dsmax = 1e-2, ds = 1e-3)
@set! opts_pocoll_pd.newtonOptions.tol = 1e-9
pd_po_coll = continuation(brpo_pd, 1, (@lens _.b0), opts_pocoll_pd;
		verbosity = 0, plot = false,
		detectCodim2Bifurcation = 0,
		startWithEigen = false,
		usehessian = false,
		jacobian_ma = :minaug,
		# jacobian_ma = :autodiff,
		# jacobian_ma = :finiteDifferences,
		normN = norminf,
		callbackN = BK.cbMaxNorm(10),
		bothside = true,
		# bdlinsolver = BorderingBLS(solver = DefaultLS(), checkPrecision = false),
		)

@test pd_po_coll.kind isa BK.PDPeriodicOrbitCont


#####
# find the NS case
par_pop2 = @set par_pop.b0 = 0.4
sol2 = solve(remake(prob_de, p = par_pop2, u0 = [0.1,0.1,1,0], tspan=(0,1000)), Rodas5())
sol2 = solve(remake(sol2.prob, tspan = (0,10), u0 = sol2[end]), Rodas5())
# plot(sol2, xlims= (8,10))

probcoll, ci = generateCIProblem(PeriodicOrbitOCollProblem(26, 3; updateSectionEveryStep = 0), reMake(prob, params = sol2.prob.p), sol2, 1.2)

brpo_ns = continuation(probcoll, ci, PALC(), ContinuationPar(opts_po_cont; maxSteps = 50, ds = -0.001);
	verbosity = 0, plot = false,
	argspo...,
	# bothside = true,
	)

getNormalForm(brpo_ns, 1)

prob2 = @set probcoll.prob_vf.lens = @lens _.ϵ
brpo_pd = continuation(prob2, ci, PALC(), ContinuationPar(opts_po_cont, dsmax = 5e-3);
	verbosity = 0, plot = false,
	argspo...,
	bothside = true,
	)
# getNormalForm(brpo_pd, 2, prm = true)

# codim 2 PD
opts_pocoll_pd = ContinuationPar(brpo_pd.contparams, detectBifurcation = 3, maxSteps = 40, pMin = 1.e-2, plotEveryStep = 1, dsmax = 1e-2, ds = 1e-3)
@set! opts_pocoll_pd.newtonOptions.tol = 1e-10
pd_po_coll2 = continuation(brpo_pd, 2, (@lens _.b0), opts_pocoll_pd;
		verbosity = 0, plot = false,
		detectCodim2Bifurcation = 0,
		startWithEigen = false,
		usehessian = false,
		jacobian_ma = :minaug,
		# jacobian_ma = :autodiff,
		# jacobian_ma = :finiteDifferences,
		normN = norminf,
		callbackN = BK.cbMaxNorm(1),
		bothside = true,
		# bdlinsolver = BorderingBLS(solver = DefaultLS(), checkPrecision = false),
		)

# ns = getNormalForm(brpo_ns, 1)

opts_pocoll_ns = ContinuationPar(brpo_pd.contparams, detectBifurcation = 0, maxSteps = 20, pMin = 0., plotEveryStep = 1, dsmax = 1e-2, ds = 1e-3)
ns_po_coll = continuation(brpo_ns, 1, (@lens _.ϵ), opts_pocoll_ns;
		verbosity = 0, plot = false,
		detectCodim2Bifurcation = 0,
		startWithEigen = false,
		usehessian = false,
		jacobian_ma = :minaug,
		# jacobian_ma = :autodiff,
		# jacobian_ma = :finiteDifferences,
		normN = norminf,
		callbackN = BK.cbMaxNorm(10),
		bothside = true,
		# bdlinsolver = BorderingBLS(solver = DefaultLS(), checkPrecision = false),
		)

#####
# find the PD case
par_pop2 = @set par_pop.b0 = 0.45
sol2 = solve(remake(prob_de, p = par_pop2, u0 = [0.1,0.1,1,0], tspan=(0,1000)), Rodas5())
sol2 = solve(remake(sol2.prob, tspan = (0,10), u0 = sol2[end]), Rodas5())
# plot(sol2, xlims= (8,10))

probcoll, ci = generateCIProblem(PeriodicOrbitOCollProblem(26, 3; updateSectionEveryStep = 0), reMake(prob, params = sol2.prob.p), sol2, 1.2)

prob2 = @set probcoll.prob_vf.lens = @lens _.ϵ
brpo_pd = continuation(prob2, ci, PALC(), ContinuationPar(opts_po_cont, dsmax = 5e-3);
	verbosity = 0, plot = false,
	argspo...,
	bothside = true,
	)

# codim 2 PD
opts_pocoll_pd = ContinuationPar(brpo_pd.contparams, detectBifurcation = 3, maxSteps = 40, pMin = 1.e-2, plotEveryStep = 1, dsmax = 1e-2, ds = -1e-3)
@set! opts_pocoll_pd.newtonOptions.tol = 1e-10
pd_po_coll2 = continuation(brpo_pd, 2, (@lens _.b0), opts_pocoll_pd;
		verbosity = 0, plot = false,
		detectCodim2Bifurcation = 0,
		startWithEigen = false,
		usehessian = false,
		jacobian_ma = :minaug,
		# jacobian_ma = :autodiff,
		# jacobian_ma = :finiteDifferences,
		normN = norminf,
		callbackN = BK.cbMaxNorm(10),
		bothside = true,
		# bdlinsolver = BorderingBLS(solver = DefaultLS(), checkPrecision = false),
		)