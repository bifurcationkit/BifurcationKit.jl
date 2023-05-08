# using Revise, Plots
using Test, ForwardDiff, Parameters, LinearAlgebra
using BifurcationKit, Test
const BK = BifurcationKit

using ComponentArrays # this is for SciMLSensitivity and adjoint of flow

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

par_pop = ComponentArray( K = 1., r = 2π, a = 4π, b0 = 0.25, e = 1., d = 2π, ϵ = 0.2, )

z0 = [0.1,0.1,1,0]

prob = BK.BifurcationProblem(Pop, z0, par_pop, (@lens _.b0); recordFromSolution = (x, p) -> (x = x[1], y = x[2], u = x[3]))

opts_br = ContinuationPar(pMin = 0., pMax = 20.0, ds = 0.002, dsmax = 0.01, nInversion = 6, detectBifurcation = 3, maxBisectionSteps = 25, nev = 4)
@set! opts_br.newtonOptions.verbose = true

################################################################################
using OrdinaryDiffEq
prob_de = ODEProblem(Pop!, z0, (0, 600.), par_pop)
alg = Rodas5()
sol = solve(prob_de, alg)
prob_de = ODEProblem(Pop!, sol.u[end], (0,5.), par_pop, reltol = 1e-10, abstol = 1e-12)
sol = solve(prob_de, Rodas5())
################################################################################
@info "plotting function"
argspo = (recordFromSolution = (x, p) -> begin
		xtt = BK.getPeriodicOrbit(p.prob, x, set(getParams(p.prob), BK.getLens(p.prob), p.p))
		return (max = maximum(xtt[1,:]),
				min = minimum(xtt[1,:]),
				period = getPeriod(p.prob, x, set(getParams(p.prob), BK.getLens(p.prob), p.p)))
	end,)
################################################################################
@info "import AD"
# import AbstractDifferentiation as AD
@info "import Zygote"
# using Zygote, SciMLSensitivity

@info "generate shooting problem"

probsh, cish = generateCIProblem( ShootingProblem(M=3), deepcopy(prob), deepcopy(prob_de), deepcopy(sol), 2.; alg = Rodas5(),
	jacobian = BK.AutoDiffMF()
	# jacobian = BK.FiniteDifferencesMF()
	)

function flow(x0, prob0, tm, p = prob0.p)
	prob = remake(prob0, u0 = x0, tspan = (0, tm), p = p)
	sol = solve(prob, Rodas5())
	return sol[end]
end


@info "set AD"
@set! probsh.flow.vjp = (x,p,dx,tm) -> AD.pullback_function(AD.ZygoteBackend(), z->flow(z, prob_de,tm,p), x)(dx)[1]

@info "Newton"
lspo = GMRESIterativeSolvers(verbose = false, N = length(cish), abstol = 1e-12, reltol = 1e-10)
	eigpo = EigKrylovKit(x₀ = rand(4))
	optnpo = NewtonPar(verbose = true, linsolver = lspo, eigsolver = eigpo)
	solpo = newton(probsh, cish, optnpo)

_sol = BK.getPeriodicOrbit(probsh, solpo.u, sol.prob.p)
# plot(_sol.t, _sol[1:2,:]')

@info "PO cont1"
opts_po_cont = setproperties(opts_br, maxSteps = 50, saveEigenvectors = true, detectLoop = true, tolStability = 1e-3, newtonOptions = optnpo)
@set! opts_po_cont.newtonOptions.verbose = false
br_fold_sh = continuation(probsh, cish, PALC(tangent = Bordered()), opts_po_cont;
	# verbosity = 3, plot = true,
	linearAlgo = MatrixFreeBLS(@set lspo.N = lspo.N+1),
	argspo...)
# pt = getNormalForm(br_fold_sh, 1)

@info "PO cont2"
probsh2 = @set probsh.lens = @lens _.ϵ
brpo_pd_sh = continuation(probsh2, cish, PALC(), opts_po_cont;
	# verbosity = 3, plot = true,
	linearAlgo = MatrixFreeBLS(@set lspo.N = lspo.N+1),
	argspo...
	)
# pt = getNormalForm(brpo_pd_sh, 1)

# codim 2 Fold
@info "--> Fold curve"
opts_posh_fold = ContinuationPar(br_fold_sh.contparams, detectBifurcation = 0, maxSteps = 3, pMin = 0.01, pMax = 1.2)
	@set! opts_posh_fold.newtonOptions.tol = 1e-9
	@set! opts_posh_fold.newtonOptions.linsolver.solver.N = opts_posh_fold.newtonOptions.linsolver.solver.N+1
	fold_po_sh1 = continuation(br_fold_sh, 2, (@lens _.ϵ), opts_posh_fold;
		# verbosity = 2, plot = true,
		detectCodim2Bifurcation = 0,
		jacobian_ma = :finiteDifferencesMF,
		bdlinsolver = MatrixFreeBLS(@set lspo.N = lspo.N+1),
		startWithEigen = false,
		bothside = true,
		callbackN = BK.cbMaxNorm(1),
		)

@test fold_po_sh1.kind isa BK.FoldPeriodicOrbitCont

# codim 2 PD
@info "--> PD curve"
opts_posh_pd = ContinuationPar(brpo_pd_sh.contparams, detectBifurcation = 0, maxSteps = 4, pMin = -1.)
	@set! opts_posh_pd.newtonOptions.tol = 1e-8
	pd_po_sh = continuation(brpo_pd_sh, 1, (@lens _.b0), opts_posh_pd;
		# verbosity = 2, plot = true,
		detectCodim2Bifurcation = 0,
		usehessian = false,
		# jacobian_ma = :minaug,
		jacobian_ma = :finiteDifferencesMF,
		bdlinsolver = MatrixFreeBLS(@set lspo.N = lspo.N+1),
		startWithEigen = false,
		callbackN = BK.cbMaxNorm(1),
		)

# plot(pd_po_sh)
# plot(fold_po_sh1, fold_po_sh2, branchlabel = ["FOLD", "FOLD"])
# 	plot!(pd_po_sh, vars = (:ϵ, :b0), branchlabel = "PD")


#####
# find the PD NS case
par_pop2 = @set par_pop.b0 = 0.45
sol2 = solve(remake(prob_de, p = par_pop2, u0 = [0.1,0.1,1,0], tspan=(0,1000)), Rodas5())
sol2 = solve(remake(sol2.prob, tspan = (0,10), u0 = sol2[end]), Rodas5())
# plot(sol2, xlims= (8,10))

probshns, ci = generateCIProblem( ShootingProblem(M=3), reMake(prob, params = sol2.prob.p), remake(prob_de, p = par_pop2), sol2, 1.; alg = Rodas5(),
			jacobian = BK.AutoDiffMF()
			)

@set! probshns.flow.vjp = (x,p,dx,tm) -> AD.pullback_function(AD.ZygoteBackend(), z->flow(z, prob_de,tm,p), x)(dx)[1]

brpo_ns = continuation(probshns, ci, PALC(), ContinuationPar(opts_po_cont; maxSteps = 50, ds = -0.001);
	# verbosity = 3, plot = true,
	argspo...,
	callbackN = BK.cbMaxNorm(1),
	linearAlgo = MatrixFreeBLS(@set lspo.N = lspo.N+1),
	)

# ns = getNormalForm(brpo_ns, 1)

# codim 2 NS
@info "--> NS curve"
opts_posh_ns = ContinuationPar(brpo_ns.contparams, detectBifurcation = 0, maxSteps = 4, pMin = -0., pMax = 1.2)
@set! opts_posh_ns.newtonOptions.tol = 1e-9
ns_po_sh = continuation(brpo_ns, 1, (@lens _.ϵ), opts_posh_ns;
		# verbosity = 3, plot = true,
		detectCodim2Bifurcation = 0,
		startWithEigen = false,
		usehessian = false,
		# jacobian_ma = :minaug,
		jacobian_ma = :finiteDifferencesMF,
		normN = norminf,
		callbackN = BK.cbMaxNorm(1),
		bdlinsolver = MatrixFreeBLS(@set lspo.N = lspo.N+2),
		)
@test ns_po_sh.kind isa BK.NSPeriodicOrbitCont		

# plot(ns_po_sh, vars = (:ϵ, :b0), branchlabel = "NS")
# 	plot!(pd_po_sh, vars = (:ϵ, :b0), branchlabel = "PD")
# 	plot!(fold_po_sh1, vars = (:ϵ, :b0), branchlabel = "FOLD")
# 	plot!(fold_po_sh2, vars = (:ϵ, :b0), branchlabel = "FOLD")
