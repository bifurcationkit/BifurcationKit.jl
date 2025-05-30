# using Revise, Plots
using Test, ForwardDiff, LinearAlgebra
using BifurcationKit, Test
const BK = BifurcationKit
###################################################################################################
function Pop!(du, X, p, t = 0)
    (;r,K,a,ϵ,b0,e,d) = p
    x, y, u, v = X
    p = a * x / (b0 * (1 + ϵ * u) + x)
    du[1] = r * (1 - x/K) * x - p * y
    du[2] = e * p * y - d * y
    s = u^2 + v^2
    du[3] = u-2pi * v - s * u
    du[4] = 2pi * u + v - s * v
    du
end

par_pop = ( K = 1., r = 2π, a = 4π, b0 = 0.25, e = 1., d = 2π, ϵ = 0.2, )

z0 = [0.1,0.1,1,0]

prob = BifurcationProblem(Pop!, z0, par_pop, (@optic _.b0); record_from_solution = (x, p; k...) -> (x = x[1], y = x[2], u = x[3]))

opts_br = ContinuationPar(p_min = 0., p_max = 20.0, ds = 0.002, dsmax = 0.01, n_inversion = 6, detect_bifurcation = 3, max_bisection_steps = 25, nev = 4, max_steps = 200)
################################################################################
using OrdinaryDiffEq
prob_de = ODEProblem(Pop!, prob.u0, (0,600.), prob.params)
alg = Rodas5()
sol = OrdinaryDiffEq.solve(prob_de, alg)
prob_de = ODEProblem(Pop!, sol.u[end], (0,5.), prob_de.p, reltol = 1e-8, abstol = 1e-10)
sol = OrdinaryDiffEq.solve(prob_de, Rodas5())
################################################################################
argspo = (record_from_solution = (x, p; k...) -> begin
        xtt = BK.get_periodic_orbit(p.prob, x, p.p)
        return (max = maximum(xtt[1,:]),
                min = minimum(xtt[1,:]),
                period = getperiod(p.prob, x, p.p))
    end,
    plot_solution = (x, p; k...) -> begin
        xtt = BK.get_periodic_orbit(p.prob, x, p.p)
        plot!(xtt.t, xtt[1,:]; label = "x", k...)
        plot!(xtt.t, xtt[2,:]; label = "y", k...)
        # plot!(br; subplot = 1, putspecialptlegend = false)
    end)
################################################################################
######    Shooting ########
probsh, cish = generate_ci_problem( ShootingProblem(M=3), prob, prob_de, sol, 2.; alg = Rodas5(), parallel = true)

solpo = newton(probsh, cish, NewtonPar(verbose = false))
@test BK.converged(solpo)

_sol = BK.get_periodic_orbit(probsh, solpo.u, sol.prob.p)
# plot(_sol.t, _sol[1:2,:]')

opts_po_cont = setproperties(opts_br, max_steps = 40, save_eigenvectors = true, tol_stability = 1e-3)
@reset opts_po_cont.newton_options.verbose = false
br_fold_sh = continuation(probsh, cish, PALC(tangent = Bordered()), opts_po_cont;
    verbosity = 0, plot = false,
    argspo...)
pt = get_normal_form(br_fold_sh, 1)
show(pt)
@test pt isa BK.BranchPointPO

probsh2 = @set probsh.lens = @optic _.ϵ
brpo_pd_sh = continuation(probsh2, cish, PALC(), opts_po_cont;
    verbosity = 0, plot = false,
    argspo...
    )
pt = get_normal_form(brpo_pd_sh, 1)
show(pt)
BK.type(pt)
@test pt isa BifurcationKit.PeriodDoublingPO

# codim 2 Fold
opts_posh_fold = ContinuationPar(br_fold_sh.contparams, detect_bifurcation = 3, max_steps = 2, p_min = 0.01, p_max = 1.2)
@reset opts_posh_fold.newton_options.tol = 1e-12
fold_po_sh1 = continuation(br_fold_sh, 2, (@optic _.ϵ), opts_posh_fold;
        verbosity = 0, plot = false,
        detect_codim2_bifurcation = 1,
        update_minaug_every_step = 1,
        jacobian_ma = BK.MinAug(),
        start_with_eigen = false,
        bothside = true,
        callback_newton = BK.cbMaxNorm(1),
        )
@test fold_po_sh1.kind isa BK.FoldPeriodicOrbitCont

# codim 2 PD
opts_posh_pd = ContinuationPar(brpo_pd_sh.contparams, detect_bifurcation = 3, max_steps = 2, p_min = -1.)
@reset opts_posh_pd.newton_options.tol = 1e-12
@reset opts_posh_pd.newton_options.verbose = false
pd_po_sh = continuation(brpo_pd_sh, 1, (@optic _.b0), opts_posh_pd;
        verbosity = 0, plot = false,
        detect_codim2_bifurcation = 1,
        update_minaug_every_step = 1,
        jacobian_ma = BK.MinAug(),
        usehessian = false,
        start_with_eigen = false,
        bothside = true,
        callback_newton = BK.cbMaxNorm(1),
        )

_pdma = pd_po_sh.prob
BK.is_symmetric(_pdma)
BK.has_adjoint(_pdma.prob)
BK.isinplace(_pdma)

# plot(pd_po_sh)
# plot(fold_po_sh1, fold_po_sh2, branchlabel = ["FOLD", "FOLD"])
#     plot!(pd_po_sh, vars = (:ϵ, :b0), branchlabel = "PD")

#####
# find the PD NS case
par_pop2 = @set par_pop.b0 = 0.45
sol2 = OrdinaryDiffEq.solve(remake(prob_de, p = par_pop2, u0 = [0.1,0.1,1,0], tspan=(0,1000)), Rodas5())
sol2 = OrdinaryDiffEq.solve(remake(sol2.prob, tspan = (0,10), u0 = sol2[end]), Rodas5())
# plot(sol2, xlims= (8,10))

probshns, ci = generate_ci_problem(ShootingProblem(M=3), re_make(prob, params = sol2.prob.p), remake(prob_de, p = par_pop2), sol2, 1.; alg = Rodas5())

brpo_ns = continuation(probshns, ci, PALC(), ContinuationPar(opts_po_cont; max_steps = 20, ds = -0.001);
    verbosity = 0, plot = false,
    argspo...,
    # bothside = true,
    callback_newton = BK.cbMaxNorm(1),
    )

ns = get_normal_form(brpo_ns, 1)
show(ns)
BK.type(ns)
@test ns isa BifurcationKit.NeimarkSackerPO

# codim 2 NS
opts_posh_ns = ContinuationPar(brpo_ns.contparams, detect_bifurcation = 1, max_steps = 5, p_min = -0., p_max = 1.2)
@reset opts_posh_ns.newton_options.tol = 1e-12
@reset opts_posh_ns.newton_options.verbose = false
ns_po_sh = continuation(brpo_ns, 1, (@optic _.ϵ), opts_posh_ns;
        verbosity = 0, plot = false,
        detect_codim2_bifurcation = 1,
        update_minaug_every_step = 1,
        start_with_eigen = false,
        usehessian = false,
        jacobian_ma = BK.MinAug(),
        normC = norminf,
        bothside = false,
        callback_newton = BK.cbMaxNorm(1),
        )
@test ns_po_sh.kind isa BK.NSPeriodicOrbitCont
BK.getprob(ns_po_sh)

# plot(ns_po_sh, vars = (:ϵ, :b0), branchlabel = "NS")
#     plot!(pd_po_sh, vars = (:ϵ, :b0), branchlabel = "PD")
#     plot!(fold_po_sh1, vars = (:ϵ, :b0), branchlabel = "FOLD")
#     plot!(fold_po_sh2, vars = (:ϵ, :b0), branchlabel = "FOLD")

#########
# test of the implementation of the jacobian for the NS case
using ForwardDiff
_probns = ns_po_sh.prob
_x = ns_po_sh.sol[end].x
_solpo = ns_po_sh.sol[end].x.u
_p1 = ns_po_sh.sol[end].x.p
_p2 = ns_po_sh.sol[end].p
_param= BK.setparam(ns_po_sh, _p1[1])
_param = @set _param.ϵ = _p2

_Jnsad = ForwardDiff.jacobian(x -> BK.residual(_probns, x, _param), vcat(_x.u, _x.p))

BK.NSMALinearSolver(_solpo, _p1[1], _p1[2], _probns.prob, _param, copy(_x.u), 1., 1.)

_probns_matrix = @set _probns.jacobian = BK.MinAugMatrixBased()
J_ns_mat = BK.jacobian(_probns_matrix, vcat(_solpo, _p1), _param)
@test norminf(_Jnsad - J_ns_mat) < 1e-6
#########

# find the PD case
par_pop2 = @set par_pop.b0 = 0.45
sol2 = OrdinaryDiffEq.solve(remake(prob_de, p = par_pop2, u0 = [0.1,0.1,1,0], tspan=(0, 1000)), Rodas5())
sol2 = OrdinaryDiffEq.solve(remake(sol2.prob, tspan = (0, 10), u0 = sol2[end]), Rodas5())
# plot(sol2, xlims= (8,10))

probshpd, ci = generate_ci_problem(ShootingProblem(M=3), re_make(prob, params = sol2.prob.p), remake(prob_de, p = par_pop2), sol2, 1.; alg = Rodas5())

prob2 = @set probshpd.lens = @optic _.ϵ
brpo_pd = continuation(prob2, ci, PALC(), ContinuationPar(opts_po_cont, dsmax = 5e-3);
    verbosity = 0, plot = false,
    argspo...,
    bothside = true,
    )

get_normal_form(brpo_pd, 2)

# codim 2 PD
opts_pocoll_pd = ContinuationPar(brpo_pd.contparams, detect_bifurcation = 3, max_steps = 2, p_min = 1.e-2, plot_every_step = 10, dsmax = 1e-2, ds = -1e-3)
@reset opts_pocoll_pd.newton_options.tol = 1e-12
pd_po_sh2 = continuation(brpo_pd, 2, (@optic _.b0), opts_pocoll_pd;
        verbosity = 0, plot = false,
        detect_codim2_bifurcation = 0,
        start_with_eigen = false,
        usehessian = false,
        jacobian_ma = BK.MinAug(),
        normC = norminf,
        callback_newton = BK.cbMaxNorm(10),
        bothside = true,
        # bdlinsolver = BorderingBLS(solver = DefaultLS(), check_precision = false),
        )
BK.getprob(pd_po_sh2)

# plot(fold_po_sh1, fold_po_sh2, branchlabel = ["FOLD", "FOLD"])
#     plot!(ns_po_sh, vars = (:ϵ, :b0), branchlabel = "NS")
#     plot!(pd_po_sh, vars = (:ϵ, :b0), branchlabel = "PD")
#     plot!(pd_po_sh2, vars = (:ϵ, :b0), branchlabel = "PD2")

#####
# test of the implementation of the jacobian for the PD case
_probpd = pd_po_sh2.prob
_x = pd_po_sh2.sol[end].x
_solpo = pd_po_sh2.sol[end].x.u
_p1 = pd_po_sh2.sol[end].x.p
_p2 = pd_po_sh2.sol[end].p
_param= BK.setparam(pd_po_sh2, _p1)
_param = @set _param.ϵ = _p2

_Jpdad = ForwardDiff.jacobian(x -> BK.residual(_probpd, x, _param), vcat(_x.u, _x.p))

_probpd_matrix = @set _probpd.jacobian = BK.MinAugMatrixBased()
J_pd_mat = BK.jacobian(_probpd_matrix, vcat(_x.u, _x.p), _param)

@test norm(_Jpdad - J_pd_mat, Inf) < 1e-6