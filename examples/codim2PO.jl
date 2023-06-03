using Revise, AbbreviatedStackTraces
using Test, ForwardDiff, Parameters, Plots, LinearAlgebra
using BifurcationKit, Test
const BK = BifurcationKit

norminf(x) = norm(x, Inf)
###################################################################################################
function Pop!(du, X, p, t = 0)
	@unpack r,K,a,ϵ,b0,e,d = p
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
@set! opts_br.newtonOptions.verbose = true

################################################################################
using DifferentialEquations
prob_de = ODEProblem(Pop!, z0, (0,600.), par_pop)
alg = Rodas5()
# alg = Vern9()
sol = solve(prob_de, alg)
prob_de = ODEProblem(Pop!, sol.u[end], (0,5.), par_pop, reltol = 1e-8, abstol = 1e-10)
sol = solve(prob_de, Rodas5())

plot(sol)
################################################################################
function recordFromSolution(x, p)
	xtt = BK.getPeriodicOrbit(p.prob, x, p.p)
	return (max = maximum(xtt[1,:]),
			min = minimum(xtt[1,:]),
			period = getPeriod(p.prob, x, p.p))
end

function plotSolution(X, p; k...)
	x = X isa BorderedArray ? X.u : X
	xtt = BK.getPeriodicOrbit(p.prob, x, p.p)
	plot!(xtt.t, xtt[1,:]; label = "x", k...)
	plot!(xtt.t, xtt[2,:]; label = "y", k...)
	# plot!(br; subplot = 1, putspecialptlegend = false)
end

argspo = (recordFromSolution = recordFromSolution,
	plotSolution = plotSolution
	)
################################################################################
probtrap, ci = generateCIProblem(PeriodicOrbitTrapProblem(M = 150;  jacobian = :DenseAD, updateSectionEveryStep = 0), prob, sol, 2.)

plot(sol)
probtrap(ci, prob.params) |> plot

solpo = newton(probtrap, ci, NewtonPar(verbose = true))

_sol = BK.getPeriodicOrbit(probtrap, solpo.u,1)
plot(_sol.t, _sol[1:2,:]')

opts_po_cont = setproperties(opts_br, maxSteps = 50, saveEigenvectors = true, tolStability = 1e-8)
@set! opts_po_cont.newtonOptions.verbose = true
brpo_fold = continuation(probtrap, ci, PALC(), opts_po_cont;
	verbosity = 3, plot = true,
	argspo...
	)

pt = getNormalForm(brpo_fold, 1)

prob2 = @set probtrap.prob_vf.lens = @lens _.ϵ
brpo_pd = continuation(prob2, ci, PALC(), opts_po_cont;
	verbosity = 3, plot = true,
	argspo...
	)
pt = getNormalForm(brpo_pd, 1)

# codim 2 Fold
opts_potrap_fold = ContinuationPar(brpo_fold.contparams, detectBifurcation = 3, maxSteps = 100, pMin = 0., pMax=1.2, nInversion = 4, plotEveryStep = 2)
@set! opts_potrap_fold.newtonOptions.tol = 1e-9
fold_po_trap1 = continuation(brpo_fold, 2, (@lens _.ϵ), opts_potrap_fold;
		verbosity = 3, plot = true,
		detectCodim2Bifurcation = 0,
		startWithEigen = false,
		bothside = true,
		jacobian_ma = :minaug,
		bdlinsolver = BorderingBLS(solver = DefaultLS(), checkPrecision = false),
		)

@test fold_po_trap1.kind isa BK.FoldPeriodicOrbitCont
plot(fold_po_trap1)

# codim 2 PD
opts_potrap_pd = ContinuationPar(brpo_pd.contparams, detectBifurcation = 0, maxSteps = 10, pMin = -1., plotEveryStep = 1, dsmax = 1e-2, ds = -1e-3)
@set! opts_potrap_pd.newtonOptions.tol = 1e-9
pd_po_trap = continuation(brpo_pd, 1, (@lens _.b0), opts_potrap_pd;
		verbosity = 3, plot = true,
		detectCodim2Bifurcation = 0,
		startWithEigen = false,
		jacobian_ma = :finiteDifferences,
		normN = norminf,
		callbackN = BK.cbMaxNorm(1),
		# bdlinsolver = BorderingBLS(solver = DefaultLS(), checkPrecision = false),
		)

@test pd_po_trap.kind isa BK.PDPeriodicOrbitCont

plot(fold_po_trap, pd_po_trap)

#####
fold_po_trap2 = continuation(brpo_fold, 2, (@lens _.ϵ), opts_potrap_fold;
		verbosity = 3, plot = true,
		detectCodim2Bifurcation = 0,
		startWithEigen = false,
		bothside = true,
		jacobian_ma = :minaug,
		bdlinsolver = BorderingBLS(solver = DefaultLS(), checkPrecision = false),
		)
plot(fold_po_trap1, fold_po_trap2, ylims = (0, 0.49))
	# plot!(pd_po_trap.branch.ϵ, pd_po_trap.branch.b0)
################################################################################
probcoll, ci = generateCIProblem(PeriodicOrbitOCollProblem(26, 3; updateSectionEveryStep = 0), prob, sol, 2.)

plot(sol)
probcoll(ci, prob.params) |> plot

solpo = newton(probcoll, ci, NewtonPar(verbose = true))

_sol = BK.getPeriodicOrbit(probcoll, solpo.u,1)
plot(_sol.t, _sol[1:2,:]')

opts_po_cont = setproperties(opts_br, maxSteps = 50, saveEigenvectors = true, tolStability = 1e-8)
@set! opts_po_cont.newtonOptions.verbose = true
brpo_fold = continuation(probcoll, ci, PALC(), opts_po_cont;
	verbosity = 3, plot = true,
	argspo...
	)
pt = getNormalForm(brpo_fold, 1)

prob2 = @set probcoll.prob_vf.lens = @lens _.ϵ
brpo_pd = continuation(prob2, ci, PALC(), ContinuationPar(opts_po_cont, dsmax = 5e-3);
	verbosity = 3, plot = true,
	argspo...
	)
pt = getNormalForm(brpo_pd, 1, prm = true)

# codim 2 Fold
opts_pocoll_fold = ContinuationPar(brpo_fold.contparams, detectBifurcation = 3, maxSteps = 120, pMin = 0., pMax=1.2, nInversion = 4, plotEveryStep = 10)
@set! opts_pocoll_fold.newtonOptions.tol = 1e-12
fold_po_coll1 = continuation(brpo_fold, 1, (@lens _.ϵ), opts_pocoll_fold;
		verbosity = 3, plot = true,
		detectCodim2Bifurcation = 0,
		startWithEigen = false,
		bothside = true,
		jacobian_ma = :minaug,
		bdlinsolver = BorderingBLS(solver = DefaultLS(), checkPrecision = false),
		)

fold_po_coll2 = continuation(brpo_fold, 2, (@lens _.ϵ), opts_pocoll_fold;
		verbosity = 3, plot = true,
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
		verbosity = 3, plot = true,
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

plot(fold_po_coll1, pd_po_coll)

#####
fold_po_coll2 = continuation(brpo_fold, 2, (@lens _.ϵ), opts_pocoll_fold;
		verbosity = 3, plot = true,
		detectCodim2Bifurcation = 0,
		startWithEigen = false,
		bothside = true,
		jacobian_ma = :minaug,
		bdlinsolver = BorderingBLS(solver = DefaultLS(), checkPrecision = false),
		)
plot(fold_po_coll1, fold_po_coll2, ylims = (0, 0.49))
	plot!(pd_po_coll, vars = (:ϵ, :b0))


#####
# find the NS case
par_pop2 = @set par_pop.b0 = 0.4
sol2 = solve(remake(prob_de, p = par_pop2, u0 = [0.1,0.1,1,0], tspan=(0,1000)), Rodas5())
sol2 = solve(remake(sol2.prob, tspan = (0,10), u0 = sol2[end]), Rodas5())
plot(sol2, xlims= (8,10))

probcoll, ci = generateCIProblem(PeriodicOrbitOCollProblem(26, 3; updateSectionEveryStep = 0), reMake(prob, params = sol2.prob.p), sol2, 1.2)

brpo_ns = continuation(probcoll, ci, PALC(), ContinuationPar(opts_po_cont; maxSteps = 50, ds = -0.001);
	verbosity = 3, plot = true,
	argspo...,
	# bothside = true,
	)

getNormalForm(brpo_ns, 1)

prob2 = @set probcoll.prob_vf.lens = @lens _.ϵ
brpo_pd = continuation(prob2, ci, PALC(), ContinuationPar(opts_po_cont, dsmax = 5e-3);
	verbosity = 3, plot = true,
	argspo...,
	bothside = true,
	)
getNormalForm(brpo_pd, 2, prm = true)

# codim 2 PD
opts_pocoll_pd = ContinuationPar(brpo_pd.contparams, detectBifurcation = 3, maxSteps = 40, pMin = 1.e-2, plotEveryStep = 1, dsmax = 1e-2, ds = 1e-3)
@set! opts_pocoll_pd.newtonOptions.tol = 1e-10
pd_po_coll2 = continuation(brpo_pd, 2, (@lens _.b0), opts_pocoll_pd;
		verbosity = 3, plot = true,
		detectCodim2Bifurcation = 2,
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

plot(fold_po_coll1, ylims = (0, 0.49))
	plot!(fold_po_coll2)
	# plot!(pd_po_coll, vars = (:ϵ, :b0))
	plot!(pd_po_coll2, vars = (:ϵ, :b0))

ns = getNormalForm(brpo_ns, 1)

opts_pocoll_ns = ContinuationPar(brpo_pd.contparams, detectBifurcation = 0, maxSteps = 20, pMin = 0., plotEveryStep = 1, dsmax = 1e-2, ds = 1e-3)
ns_po_coll = continuation(brpo_ns, 1, (@lens _.ϵ), opts_pocoll_ns;
		verbosity = 3, plot = true,
		detectCodim2Bifurcation = 1,
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

plot!(ns_po_coll, vars = (:ϵ, :b0))
	plot!(pd_po_coll2, vars = (:ϵ, :b0))

#####
# find the PD case
par_pop2 = @set par_pop.b0 = 0.45
sol2 = solve(remake(prob_de, p = par_pop2, u0 = [0.1,0.1,1,0], tspan=(0,1000)), Rodas5())
sol2 = solve(remake(sol2.prob, tspan = (0,10), u0 = sol2[end]), Rodas5())
plot(sol2, xlims= (8,10))

probcoll, ci = generateCIProblem(PeriodicOrbitOCollProblem(26, 3; updateSectionEveryStep = 0), reMake(prob, params = sol2.prob.p), sol2, 1.2)

prob2 = @set probcoll.prob_vf.lens = @lens _.ϵ
brpo_pd = continuation(prob2, ci, PALC(), ContinuationPar(opts_po_cont, dsmax = 5e-3);
	verbosity = 3, plot = true,
	argspo...,
	bothside = true,
	)

# codim 2 PD
opts_pocoll_pd = ContinuationPar(brpo_pd.contparams, detectBifurcation = 3, maxSteps = 40, pMin = 1.e-2, plotEveryStep = 1, dsmax = 1e-2, ds = -1e-3)
@set! opts_pocoll_pd.newtonOptions.tol = 1e-10
pd_po_coll2 = continuation(brpo_pd, 2, (@lens _.b0), opts_pocoll_pd;
		verbosity = 3, plot = true,
		detectCodim2Bifurcation = 2,
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

plot(fold_po_coll1, ylims = (0, 0.49))
	plot!(fold_po_coll2)
	# plot!(pd_po_coll, vars = (:ϵ, :b0))
	plot!(pd_po_coll2, vars = (:ϵ, :b0))
################################################################################
######    Shooting ########
probsh, cish = generateCIProblem( ShootingProblem(M=3), prob, prob_de, sol, 2.; alg = Rodas5(), parallel = true)

solpo = newton(probsh, cish, NewtonPar(verbose = true))

_sol = BK.getPeriodicOrbit(probsh, solpo.u, sol.prob.p)
plot(_sol.t, _sol[1:2,:]')

opts_po_cont = setproperties(opts_br, maxSteps = 50, saveEigenvectors = true, detectLoop = true, tolStability = 1e-3)
@set! opts_po_cont.newtonOptions.verbose = false
br_fold_sh = continuation(probsh, cish, PALC(tangent = Bordered()), opts_po_cont;
	verbosity = 3, plot = true,
	argspo...)
pt = getNormalForm(br_fold_sh, 1)

probsh2 = @set probsh.lens = @lens _.ϵ
brpo_pd_sh = continuation(probsh2, cish, PALC(), opts_po_cont;
	verbosity = 3, plot = true,
	argspo...
	)
pt = getNormalForm(brpo_pd_sh, 1)

# codim 2 Fold
opts_posh_fold = ContinuationPar(br_fold_sh.contparams, detectBifurcation = 3, maxSteps = 200, pMin = 0.01, pMax = 1.2)
@error "ce foire la precision si cette tol est trop petite"
@set! opts_posh_fold.newtonOptions.tol = 1e-12
fold_po_sh1 = continuation(br_fold_sh, 2, (@lens _.ϵ), opts_posh_fold;
		verbosity = 2, plot = true,
		detectCodim2Bifurcation = 2,
		jacobian_ma = :minaug,
		startWithEigen = false,
		bothside = true,
		callbackN = BK.cbMaxNorm(1),
		)

fold_po_sh2 = continuation(br_fold_sh, 1, (@lens _.ϵ), opts_posh_fold;
		verbosity = 2, plot = true,
		detectCodim2Bifurcation = 2,
		jacobian_ma = :minaug,
		startWithEigen = false,
		bothside = true,
		callbackN = BK.cbMaxNorm(1),
		)

@test fold_po_sh1.kind isa BK.FoldPeriodicOrbitCont

# codim 2 PD
opts_posh_pd = ContinuationPar(brpo_pd_sh.contparams, detectBifurcation = 3, maxSteps = 40, pMin = -1.)
@set! opts_posh_pd.newtonOptions.tol = 1e-12
@error "ce foire la precision si cette tol est trop petite"
@set! opts_posh_pd.newtonOptions.verbose = true
pd_po_sh = continuation(brpo_pd_sh, 1, (@lens _.b0), opts_posh_pd;
		verbosity = 3, plot = true,
		detectCodim2Bifurcation = 2,
		jacobian_ma = :minaug,
		# jacobian_ma = :autodiff,
		usehessian = false,
		# jacobian_ma = :finiteDifferences,
		startWithEigen = false,
		bothside = true,
		callbackN = BK.cbMaxNorm(1),
		)

plot(pd_po_sh)

plot(fold_po_sh1, fold_po_sh2, branchlabel = ["FOLD", "FOLD"])
	plot!(pd_po_sh, vars = (:ϵ, :b0), branchlabel = "PD")

#####
# find the NS case
par_pop2 = @set par_pop.b0 = 0.45
sol2 = solve(remake(prob_de, p = par_pop2, u0 = [0.1,0.1,1,0], tspan=(0,1000)), Rodas5())
sol2 = solve(remake(sol2.prob, tspan = (0,10), u0 = sol2[end]), Rodas5())
plot(sol2, xlims= (8,10))

probshns, ci = generateCIProblem(ShootingProblem(M=3), reMake(prob, params = sol2.prob.p), remake(prob_de, p = par_pop2), sol2, 1.; alg = Rodas5())

brpo_ns = continuation(probshns, ci, PALC(), ContinuationPar(opts_po_cont; maxSteps = 50, ds = -0.001);
	verbosity = 3, plot = true,
	argspo...,
	# bothside = true,
	callbackN = BK.cbMaxNorm(1),
	)

ns = getNormalForm(brpo_ns, 1)

# codim 2 NS
opts_posh_ns = ContinuationPar(brpo_ns.contparams, detectBifurcation = 0, maxSteps = 10, pMin = -0., pMax = 1.2)
@set! opts_posh_ns.newtonOptions.tol = 1e-12
@set! opts_posh_ns.newtonOptions.verbose = true
ns_po_sh = continuation(brpo_ns, 1, (@lens _.ϵ), opts_posh_ns;
		verbosity = 2, plot = true,
		detectCodim2Bifurcation = 0,
		startWithEigen = false,
		usehessian = false,
		jacobian_ma = :minaug,
		# jacobian_ma = :autodiff,
		# jacobian_ma = :finiteDifferences,
		normN = norminf,
		bothside = false,
		callbackN = BK.cbMaxNorm(1),
		)

plot(ns_po_sh, vars = (:ϵ, :b0), branchlabel = "NS")
	plot!(pd_po_sh, vars = (:ϵ, :b0), branchlabel = "PD")
	plot!(fold_po_sh1, vars = (:ϵ, :b0), branchlabel = "FOLD")
	plot!(fold_po_sh2, vars = (:ϵ, :b0), branchlabel = "FOLD")

#########
# find the PD case
par_pop2 = @set par_pop.b0 = 0.45
sol2 = solve(remake(prob_de, p = par_pop2, u0 = [0.1,0.1,1,0], tspan=(0,1000)), Rodas5())
sol2 = solve(remake(sol2.prob, tspan = (0,10), u0 = sol2[end]), Rodas5())
plot(sol2, xlims= (8,10))

probshpd, ci = generateCIProblem(ShootingProblem(M=3), reMake(prob, params = sol2.prob.p), remake(prob_de, p = par_pop2), sol2, 1.; alg = Rodas5())

prob2 = @set probshpd.lens = @lens _.ϵ
brpo_pd = continuation(prob2, ci, PALC(), ContinuationPar(opts_po_cont, dsmax = 5e-3);
	verbosity = 3, plot = true,
	argspo...,
	bothside = true,
	)

getNormalForm(brpo_pd, 2)
# codim 2 PD
opts_pocoll_pd = ContinuationPar(brpo_pd.contparams, detectBifurcation = 3, maxSteps = 40, pMin = 1.e-2, plotEveryStep = 10, dsmax = 1e-2, ds = -1e-3)
@set! opts_pocoll_pd.newtonOptions.tol = 1e-12
pd_po_sh2 = continuation(brpo_pd, 2, (@lens _.b0), opts_pocoll_pd;
		verbosity = 3, plot = true,
		detectCodim2Bifurcation = 2,
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

plot(fold_po_sh1, fold_po_sh2, branchlabel = ["FOLD", "FOLD"])
	# plot!(ns_po_sh, vars = (:ϵ, :b0), branchlabel = "NS")
	# plot!(pd_po_sh, vars = (:ϵ, :b0), branchlabel = "PD")
	plot!(pd_po_sh2, vars = (:ϵ, :b0), branchlabel = "PD2")
################################################################################
###### Poincare Shooting ########
probpsh, cipsh = generateCIProblem( PoincareShootingProblem( M=1 ), prob, prob_de, sol, 2.; alg = Rodas5(), updateSectionEveryStep = 0)

# je me demande si la section n'intersecte pas deux fois l'hyperplan
hyper = probpsh.section
plot(sol)
	plot!(sol.t, [hyper(zeros(1),u)[1] for u in sol.u])

# solpo = newton(probpsh, cipsh, NewtonPar(verbose = true))
BK.getPeriod(probpsh, cipsh, sol.prob.p)
_sol = BK.getPeriodicOrbit(probpsh, cipsh, sol.prob.p)
plot(_sol.t, _sol[1:2,:]')


######
JE PENSE QU IL FAUT N CALLBACK ET N FLOW

cisl = BK.getTimeSlices(probpsh, cipsh)
hyper = probpsh.section
ind = 1
	println("\n\n\n*********\n ind = $ind -> $(ind+1)")
	_x = cisl[:,ind]
	_xsol = BK.evolve(probpsh.flow, _x, sol.prob.p, Inf).u
	display(hcat(_xsol, cisl[:, ind]))
	printstyled(color=:green, "--> sol hyper\n")
	hyper(zeros(probpsh.M), _xsol) |> display
	printstyled(color=:green, "--> ci hyper\n")
	hyper(zeros(probpsh.M), cisl[:, ind]) |> display
###### plotting

plot(sol[1,:], sol[2,:], sol[3,:])
cisl = BK.getTimeSlices(probpsh, cipsh)
for ii=1:probpsh.M
	scatter!([cisl[1,ii]],[cisl[2,ii]],[cisl[3,ii]], label="") |> display
end
######

_sol = BK.getPeriodicOrbit(probcoll, solpo.u,1)
plot(_sol.t, _sol[1:2,:]')

cisl = BK.getTimeSlices(probpsh, cipsh)

probpsh.flow

hyper = probpsh.section
hyper(zeros(3),cisl[:,3])

plot(sol,)
	M = probpsh.M
	ts = LinRange(0, 2.1, M+1)[1:end-1]
	scatter!(ts, cisl', label = "")

probpsh(cipsh, sol.prob.p)

getPeriod(probpsh, cipsh, probpsh.par)

@assert 1==0 "Ca foire voilamment"

_sol = getPeriodicOrbit(probpsh, cipsh, sol.prob.p)
plot(sol)

record_sh = recordFromSolution = (x, p) -> begin
		xtt = BK.getPeriodicOrbit(p.prob, x, set(par_pop, p.prob.lens, p.p))
		return (max = maximum(xtt[1,:]),
				min = minimum(xtt[1,:]),
				period = getPeriod(p.prob, x, set(par_pop, p.prob.lens, p.p)))
	end
plot_sh  = (x, p; k...) -> begin
	xtt = BK.getPeriodicOrbit(p.prob, x, set(par_pop, p.prob.lens, p.p))
	plot!(xtt.t, xtt[1,:]; label = "x", k...)
	plot!(xtt.t, xtt[2,:]; label = "y", k...)
	# plot!(xtt.t, xtt[3,:]; label = "u", k...)
	# plot!(xtt.t, xtt[4,:]; label = "v", k...)
	# plot!(br; subplot = 1, putspecialptlegend = false)
	end

probpsh(cipsh, prob_de.p)

opts_po_cont = setproperties(opts_br, maxSteps = 40, saveEigenvectors = true, detectLoop = true, tolStability = 1e-3)
@set! opts_po_cont.newtonOptions.verbose = true
@set! opts_po_cont.newtonOptions.tol = 1e-10
br_fold_psh = continuation(probpsh, cipsh, PALC(), opts_po_cont;
	verbosity = 3, plot = true,
	recordFromSolution = record_sh,
	callbackN = BK.cbMaxNorm(10),
	plotSolution = plot_sh,)

probpsh2 = @set probpsh.lens = @lens _.ϵ
brpo_pd_psh = continuation(probpsh2, cipsh, PALC(), opts_po_cont;
	verbosity = 3, plot = true,
	recordFromSolution = record_sh,
	plotSolution = plot_sh,
	callbackN = BK.cbMaxNorm(10),
	# bothside = true,
	)

getNormalForm(brpo_pd_psh, 1)

# codim 2 ns
opts_posh_fold = ContinuationPar(br_fold_psh.contparams, detectBifurcation = 3, maxSteps = 60, pMin = 0., pMax = 1.2)
@set! opts_posh_fold.newtonOptions.tol = 1e-9
ns_po_sh = continuation(br_fold_psh, 1, (@lens _.ϵ), opts_posh_fold;
		verbosity = 2, plot = true,
		detectCodim2Bifurcation = 0,
		jacobian_ma = :minaug,
		startWithEigen = false,
		bothside = true,
		callbackN = BK.cbMaxNorm(1),
		)

@test fold_po_sh.kind isa BK.FoldPeriodicOrbitCont

# codim 2 PD
opts_posh_pd = ContinuationPar(brpo_pd_sh.contparams, detectBifurcation = 3, maxSteps = 30, pMin = 0., pMax = 1.2)
@set! opts_posh_pd.newtonOptions.tol = 1e-9
pd_po_sh = continuation(brpo_pd_psh, 1, (@lens _.b0), opts_posh_pd;
		verbosity = 2, plot = true,
		detectCodim2Bifurcation = 0,
		usehessian = false,
		jacobian_ma = :minaug,
		startWithEigen = false,
		bothside = true,
		)

plot(pd_po_sh)


proj(a,c,x) = x .- dot(a, x .- c) .* a
function findPointOnHyp(a, c; radius = 1, nd = 0)
	n = size(a,1)
	nd = max(nd, n+1)
	[proj(a,c,rand(n)) for i=1:nd]
end


_normal = probpsh.section.normals[1]
	_center = probpsh.section.centers[1]
	res = findPointOnHyp(_normal, _center; radius = 0.1, nd = 200)

	plot([r[1] for r in res],[r[2] for r in res],[r[3] for r in res])


plot(sol[1,:], sol[2,:], sol[3,:])
	cisl = BK.getTimeSlices(probpsh, cipsh)
	for ii=1:probpsh.M
		scatter!([cisl[1,ii]],[cisl[2,ii]],[cisl[3,ii]], label="")
	end
	plot!([r[1] for r in res],[r[2] for r in res],[r[3] for r in res]) |> display
