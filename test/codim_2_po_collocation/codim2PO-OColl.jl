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

opts_br = ContinuationPar(p_min = 0., p_max = 20.0, ds = 0.002, dsmax = 0.01, n_inversion = 6, detect_bifurcation = 3, max_bisection_steps = 25, nev = 4, max_steps = 20000)
################################################################################
using OrdinaryDiffEq
prob_de = ODEProblem(Pop!, z0, (0,600.), par_pop)
alg = Rodas5()
# alg = Vern9()
sol = OrdinaryDiffEq.solve(prob_de, alg)
prob_de = ODEProblem(Pop!, sol.u[end], (0,5.), par_pop, reltol = 1e-8, abstol = 1e-10)
sol = OrdinaryDiffEq.solve(prob_de, Rodas5())
################################################################################
argspo = (record_from_solution = (x, p; k...) -> begin
        xtt = BK.get_periodic_orbit(p.prob, x, p.p)
        return (max = maximum(xtt[1,:]),
                min = minimum(xtt[1,:]),
                period = getperiod(p.prob, x, p.p))
    end,)
################################################################################
probcoll, ci = generate_ci_problem(PeriodicOrbitOCollProblem(26, 3), prob, sol, 2.; use_adapted_mesh = true)
probcoll, ci = generate_ci_problem(PeriodicOrbitOCollProblem(26, 3), prob, sol, 2.)

solpo = newton(probcoll, ci, NewtonPar(verbose = false))
@test BK.converged(solpo)

_sol = BK.get_periodic_orbit(probcoll, solpo.u, 1)

opts_po_cont = setproperties(opts_br, max_steps = 40, save_eigenvectors = true, tol_stability = 1e-8)
@reset opts_po_cont.newton_options.verbose = false
brpo_fold = continuation(probcoll, deepcopy(ci), PALC(), opts_po_cont;
    verbosity = 0, plot = false,
    argspo...
    )
# pt = get_normal_form(brpo_fold, 1)

prob2 = @set probcoll.prob_vf.lens = @optic _.ϵ
brpo_pd = continuation(prob2, deepcopy(ci), PALC(), ContinuationPar(opts_po_cont, dsmax = 5e-3);
    verbosity = 0, plot = false,
    argspo...
    )

get_normal_form(brpo_pd, 1, prm = Val(true))
# test PD normal form computation using Iooss method
get_normal_form(brpo_pd, 1, prm = Val(false))
################################################################################
# codim 2 Fold
opts_pocoll_fold = ContinuationPar(brpo_fold.contparams, detect_bifurcation = 3, max_steps = 3, p_min = 0., p_max=1.2, n_inversion = 4)
@reset opts_pocoll_fold.newton_options.tol = 1e-12
fold_po_coll1 = @time continuation(deepcopy(brpo_fold), 1, (@optic _.ϵ), opts_pocoll_fold;
        verbosity = 0, plot = false,
        detect_codim2_bifurcation = 0,
        start_with_eigen = false,
        bothside = true,
        jacobian_ma = BK.MinAug(),
        bdlinsolver = BorderingBLS(solver = DefaultLS(), check_precision = false),
        )
@test fold_po_coll1.kind isa BK.FoldPeriodicOrbitCont

# codim 2 PD
opts_pocoll_pd = ContinuationPar(brpo_pd.contparams, detect_bifurcation = 3, max_steps = 20, p_min = -1., dsmax = 1e-2, ds = 1e-3)
@reset opts_pocoll_pd.newton_options.tol = 1e-12

pd_po_colls = [ continuation(deepcopy(brpo_pd), 1, (@optic _.b0), 
                    ContinuationPar(opts_pocoll_pd; detect_bifurcation = 3);
                    # verbosity = 3, plot = true,
                    detect_codim2_bifurcation = jma == BK.MinAug() ? 1 : 0,
                    start_with_eigen = false,
                    usehessian = false,
                    jacobian_ma = jma,
                    normC = norminf,
                    callback_newton = BK.cbMaxNorm(10),
                    bothside = true,
                    ) for jma in (BK.MinAug(), BK.MinAugMatrixBased(), )]
@test all(x -> x.kind isa BK.PDPeriodicOrbitCont, pd_po_colls)
get_normal_form(pd_po_colls[1], 2)
################################################################################
# find the NS case
par_pop2 = @set par_pop.b0 = 0.4
sol2 = OrdinaryDiffEq.solve(remake(prob_de, p = par_pop2, u0 = [0.1,0.1,1,0], tspan=(0,1000)), Rodas5())
sol2 = OrdinaryDiffEq.solve(remake(sol2.prob, tspan = (0, 10), u0 = sol2[end]), Rodas5())

probcoll, ci = generate_ci_problem(PeriodicOrbitOCollProblem(26, 3), re_make(prob, params = sol2.prob.p), sol2, 1.2)

brpo_ns = continuation(probcoll, ci, PALC(), ContinuationPar(opts_po_cont; max_steps = 20, ds = -0.001);
    verbosity = 0, plot = false,
    argspo...,
    )

# compute NS normal form using Poincare return map
get_normal_form(brpo_ns, 1; prm = Val(true))
# compute NS normal form using Iooss method
get_normal_form(brpo_ns, 1; prm = Val(false))

opts_pocoll_ns = ContinuationPar(brpo_ns.contparams, detect_bifurcation = 2, max_steps = 20, p_min = 0., dsmax = 7e-3, ds = -1e-3)

ns_po_colls = [continuation(brpo_ns, 1, (@optic _.ϵ), opts_pocoll_ns;
            verbosity = 0, plot = false,
            detect_codim2_bifurcation = 1,
            update_minaug_every_step = 1,
            start_with_eigen = false,
            usehessian = false,

            jacobian_ma = jma,
            normC = norminf,
            callback_newton = BK.cbMaxNorm(10),
            bothside = true,
            ) for jma in (BK.MinAug(), BK.MinAugMatrixBased(), )]
@test all(x -> x.kind isa BK.NSPeriodicOrbitCont, ns_po_colls)
get_normal_form(ns_po_colls[1], 2)
################################################################################
# test of the implementation of the jacobian for the PD case
using ForwardDiff
pd_po_coll2 = pd_po_colls[1]
_probpd = pd_po_coll2.prob
_x = pd_po_coll2.sol[end].x
_solpo = pd_po_coll2.sol[end].x.u
_p1 = pd_po_coll2.sol[end].x.p
_p2 = pd_po_coll2.sol[end].p
_param = BK.setparam(pd_po_coll2, _p1)
_param = @set _param.ϵ = _p2

_Jpdad = ForwardDiff.jacobian(x -> BK.residual(_probpd, x, _param), vcat(_x.u, _x.p))
# _Jpdad = BK.finite_differences(x -> BK.residual(_probpd, x, _param), vcat(_x.u, _x.p))

_duu = rand(length(_x.u))
𝐏𝐝 = _probpd.prob
_sol = BK.PDMALinearSolver(_solpo, _p1, 𝐏𝐝, _param, _duu, 1.)
_solfd = _Jpdad \ vcat(_duu, 1)

@test norminf(_solfd[1:end-1] - _sol[1]) < 1e-3 # it comes from FD in σₓ
@test abs(_solfd[end] - _sol[2]) < 5e-3

_probpd_matrix = @set _probpd.jacobian = BK.MinAugMatrixBased()
J_pd_mat = BK.jacobian(_probpd_matrix, vcat(_solpo, _p1), _param)
@test norminf(_Jpdad - J_pd_mat) < 1e-7

#########
# test of the implementation of the jacobian for the NS case
ns_po_coll = ns_po_colls[1]
_probns = ns_po_coll.prob
_x = ns_po_coll.sol[end].x
_solpo = ns_po_coll.sol[end].x.u
_p1 = ns_po_coll.sol[end].x.p
_p2 = ns_po_coll.sol[end].p
_param = BK.setparam(ns_po_coll, _p1[1])
_param = @set _param.ϵ = _p2

_Jnsad = ForwardDiff.jacobian(x -> BK.residual(_probns, x, _param), vcat(_x.u, _x.p))
# _Jnsad = BK.finite_differences(x -> BK.residual(_probns, x, _param), vcat(_x.u, _x.p))

_duu = rand(317)
_dp = rand()
_sol = BK.NSMALinearSolver(_solpo, _p1[1], _p1[2], _probns.prob, _param, _duu, _dp, 1.)
_solfd = _Jnsad \ vcat(_duu, _dp, 1)

@test norminf(_solfd[1:end-2] - _sol[1]) < 1e-2
@test abs(_solfd[end-1] - _sol[2]) < 1e-2
@test abs(_solfd[end] - _sol[3]) < 1e-2

_probpd_matrix = @set _probns.jacobian = BK.MinAugMatrixBased()
J_ns_mat = BK.jacobian(_probpd_matrix, vcat(_solpo, _p1), _param)
@test norminf(_Jnsad - J_ns_mat) < 1e-7