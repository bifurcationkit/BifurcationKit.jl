using Revise
using Test, ForwardDiff, Parameters, Plots, LinearAlgebra
using BifurcationKit, Test
const BK = BifurcationKit

using ComponentArrays # this is for SciMLSensitivity and adjoint of flow
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

par_pop = ComponentArray( K = 1., r = 2π, a = 4π, b0 = 0.25, e = 1., d = 2π, ϵ = 0.2, )

z0 = [0.1,0.1,1,0]

prob = BifurcationProblem(Pop, z0, par_pop, (@lens _.b0); recordFromSolution = (x, p) -> (x = x[1], y = x[2], u = x[3]))

opts_br = ContinuationPar(pMin = 0., pMax = 20.0, ds = 0.002, dsmax = 0.01, nInversion = 6, detectBifurcation = 3, maxBisectionSteps = 25, nev = 4, maxSteps = 20000)
@set! opts_br.newtonOptions.verbose = true

################################################################################
using DifferentialEquations
prob_de = ODEProblem(Pop!, z0, (0,600.), par_pop)
alg = Rodas5()
# alg = Vern9()
sol = solve(prob_de, alg)
prob_de = ODEProblem(Pop!, sol.u[end], (0,5.), par_pop, reltol = 1e-10, abstol = 1e-12)
sol = solve(prob_de, Rodas5())

plot(sol)
################################################################################
argspo = (recordFromSolution = (x, p) -> begin
		xtt = BK.getPeriodicOrbit(p.prob, x, set(getParams(p.prob), BK.getLens(p.prob), p.p))
		return (max = maximum(xtt[1,:]),
				min = minimum(xtt[1,:]),
				period = getPeriod(p.prob, x, set(getParams(p.prob), BK.getLens(p.prob), p.p)))
	end,
	plotSolution = (x, p; k...) -> begin
		xtt = BK.getPeriodicOrbit(p.prob, x, set(getParams(p.prob), BK.getLens(p.prob), p.p))
		plot!(xtt.t, xtt[1,:]; label = "x", k...)
		plot!(xtt.t, xtt[2,:]; label = "y", k...)
		# plot!(br; subplot = 1, putspecialptlegend = false)
	end)
################################################################################
using Test, Zygote, SciMLSensitivity, FiniteDifferences
import AbstractDifferentiation as AD

probsh0 = ShootingProblem(M=1)

probshMatrix, = generateCIProblem( ShootingProblem(M=3), prob, prob_de, sol, 2.; alg = Rodas5())
probsh, cish = generateCIProblem( ShootingProblem(M=3), prob, prob_de, sol, 2.; alg = Rodas5(),
			jacobian = BK.AutoDiffMF()
			# jacobian = BK.FiniteDifferencesMF()
			)
######
# il faut un jvp precis pour la monodromy sinon ca ne marche pas bien
@error "c'est pas bon ca car flowDE n'as pas de structure pour les fonctions"
######
function flow(x0, prob0, tm, p = prob0.p)
	prob = remake(prob0, u0 = x0, tspan = (0, tm), p = p)
	sol = solve(prob, Rodas5())
	return sol[end]
end

sol0_f = rand(4)
flow(rand(4), prob_de, 1)

dϕ = ForwardDiff.jacobian(x->flow(x, prob_de, 1), sol0_f)
# jvp
res1 = ForwardDiff.derivative(t->flow(sol0_f .+ t .* sol0_f, prob_de, 1), zero(eltype(sol0_f)))
@test norm(res1 - dϕ * sol0_f, Inf) < 1e-8

res1, = AD.pushforward_function(AD.ForwardDiffBackend(), x->flow(x, prob_de, 1), sol0_f)(sol0_f)
@test norm(res1 - dϕ * sol0_f, Inf) < 1e-7

# vjp
res1, = Zygote.pullback(x->flow(x,prob_de,1), sol0_f)[2](sol0_f)
@test norm(res1 - dϕ' * sol0_f, Inf) < 1e-8

res1 = AD.pullback_function(AD.ZygoteBackend(), x->flow(x, prob_de,1), sol0_f)(sol0_f)[1]
@test norm(res1 - dϕ' * sol0_f, Inf) < 1e-8
######

AD.pullback_function(AD.FiniteDifferencesBackend(), z -> probsh(z, getParams(probsh)), cish)(cish)[1]

@set! probsh.flow.vjp = (x,p,dx,tm) -> AD.pullback_function(AD.ZygoteBackend(), z->flow(z, prob_de,tm,p), x)(dx)[1]

lspo = GMRESIterativeSolvers(verbose = false, N = length(cish), abstol = 1e-12, reltol = 1e-10)
	eigpo = EigKrylovKit(x₀ = rand(4))
	optnpo = NewtonPar(verbose = true, linsolver = lspo, eigsolver = eigpo)
	solpo = newton(probsh, cish, optnpo)

_sol = BK.getPeriodicOrbit(probsh, solpo.u, sol.prob.p)
plot(_sol.t, _sol[1:2,:]')

opts_po_cont = setproperties(opts_br, maxSteps = 50, saveEigenvectors = true, detectLoop = true, tolStability = 1e-3, newtonOptions = optnpo)
@set! opts_po_cont.newtonOptions.verbose = true
br_fold_sh = continuation(probsh, cish, PALC(tangent = Bordered()), opts_po_cont;
	verbosity = 3, plot = true,
	linearAlgo = MatrixFreeBLS(@set lspo.N = lspo.N+1),
	argspo...)
pt = getNormalForm(br_fold_sh, 1)

probsh2 = @set probsh.lens = @lens _.ϵ
brpo_pd_sh = continuation(probsh2, cish, PALC(), opts_po_cont;
	verbosity = 3, plot = true,
	linearAlgo = MatrixFreeBLS(@set lspo.N = lspo.N+1),
	argspo...
	)

# pt = getNormalForm(brpo_pd_sh, 1)

# codim 2 Fold
opts_posh_fold = ContinuationPar(br_fold_sh.contparams, detectBifurcation = 0, maxSteps = 20, pMin = 0.01, pMax = 1.2)
	@set! opts_posh_fold.newtonOptions.tol = 1e-9

	# use this option for jacobian_ma = :finiteDifferencesMF, otherwise do not
	@set! opts_posh_fold.newtonOptions.linsolver.solver.N = opts_posh_fold.newtonOptions.linsolver.solver.N+1
	fold_po_sh1 = continuation(br_fold_sh, 2, (@lens _.ϵ), opts_posh_fold;
		verbosity = 2, plot = true,
		detectCodim2Bifurcation = 0,
		jacobian_ma = :finiteDifferencesMF,

		# jacobian_ma = :minaug,
		bdlinsolver = MatrixFreeBLS(@set lspo.N = lspo.N+1),
		# linearAlgo = MatrixFreeBLS(@set lspo.N = lspo.N+2),
		# Jᵗ = (x,p) -> (dx -> AD.pullback_function(AD.FiniteDifferencesBackend(), z -> probsh(z, p), x)(dx)[1]),
		# # usehessian = false,

		startWithEigen = false,
		bothside = true,
		callbackN = BK.cbMaxNorm(1),
		)

fold_po_sh2 = continuation(br_fold_sh, 1, (@lens _.ϵ), opts_posh_fold;
		verbosity = 2, plot = true,
		detectCodim2Bifurcation = 0,
		# jacobian_ma = :minaug,
		jacobian_ma = :finiteDifferencesMF,
		bdlinsolver = MatrixFreeBLS(@set lspo.N = lspo.N+1),
		startWithEigen = false,
		bothside = true,
		callbackN = BK.cbMaxNorm(1),
		)

@test fold_po_sh1.kind isa BK.FoldPeriodicOrbitCont
plot(fold_po_sh1, fold_po_sh2, branchlabel = ["FOLD", "FOLD"])

# codim 2 PD
opts_posh_pd = ContinuationPar(brpo_pd_sh.contparams, detectBifurcation = 3, maxSteps = 40, pMin = -1.)
	@set! opts_posh_pd.newtonOptions.tol = 1e-8
	# use this option for jacobian_ma = :finiteDifferencesMF, otherwise do not
	# @set! opts_posh_pd.newtonOptions.linsolver.solver.N = opts_posh_pd.newtonOptions.linsolver.solver.N+1
	pd_po_sh = continuation(brpo_pd_sh, 1, (@lens _.b0), opts_posh_pd;
		verbosity = 2, plot = true,
		detectCodim2Bifurcation = 0,
		usehessian = false,
		jacobian_ma = :minaug,
		# jacobian_ma = :finiteDifferencesMF,
		bdlinsolver = MatrixFreeBLS(@set lspo.N = lspo.N+1),
		startWithEigen = false,
		bothside = true,
		callbackN = BK.cbMaxNorm(1),
		)

plot(pd_po_sh)

plot(fold_po_sh1, fold_po_sh2, branchlabel = ["FOLD", "FOLD"])
	plot!(pd_po_sh, vars = (:ϵ, :b0), branchlabel = "PD")


#####
# find the PD NS case
par_pop2 = @set par_pop.b0 = 0.45
sol2 = solve(remake(prob_de, p = par_pop2, u0 = [0.1,0.1,1,0], tspan=(0,1000)), Rodas5())
sol2 = solve(remake(sol2.prob, tspan = (0,10), u0 = sol2[end]), Rodas5())
plot(sol2, xlims= (8,10))

probshns, ci = generateCIProblem( ShootingProblem(M=3), reMake(prob, params = sol2.prob.p), remake(prob_de, p = par_pop2), sol2, 1.; alg = Rodas5(),
			jacobian = BK.AutoDiffMF()
			# jacobian = BK.FiniteDifferencesMF()
			)

@set! probshns.flow.vjp = (x,p,dx,tm) -> AD.pullback_function(AD.ZygoteBackend(), z->flow(z, prob_de,tm,p), x)(dx)[1]

brpo_ns = continuation(probshns, ci, PALC(), ContinuationPar(opts_po_cont; maxSteps = 50, ds = -0.001);
	verbosity = 3, plot = true,
	argspo...,
	# bothside = true,
	callbackN = BK.cbMaxNorm(1),
	linearAlgo = MatrixFreeBLS(@set lspo.N = lspo.N+1),
	)

ns = getNormalForm(brpo_ns, 1)

# codim 2 NS
using AbbreviatedStackTraces
opts_posh_ns = ContinuationPar(brpo_ns.contparams, detectBifurcation = 0, maxSteps = 100, pMin = -0., pMax = 1.2)
@set! opts_posh_ns.newtonOptions.tol = 1e-9
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
		bdlinsolver = MatrixFreeBLS(@set lspo.N = lspo.N+2),
		# bdlinsolver = BorderingBLS(@set lspo.N = lspo.N+2),
		)

plot(ns_po_sh, vars = (:ϵ, :b0), branchlabel = "NS")
	plot!(pd_po_sh, vars = (:ϵ, :b0), branchlabel = "PD")
	plot!(fold_po_sh1, vars = (:ϵ, :b0), branchlabel = "FOLD")
	plot!(fold_po_sh2, vars = (:ϵ, :b0), branchlabel = "FOLD")
