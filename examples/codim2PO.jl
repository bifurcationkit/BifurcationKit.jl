using Revise
using Plots
# using GLMakie; Makie.inline!(true)
using ForwardDiff, LinearAlgebra
using BifurcationKit
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
sol = OrdinaryDiffEq.solve(prob_de, alg)
prob_de = ODEProblem(Pop!, sol.u[end], (0,5.), par_pop, reltol = 1e-8, abstol = 1e-10)
sol = OrdinaryDiffEq.solve(prob_de, Rodas5())

plot(sol)
################################################################################
function recordFromSolution(x, p; k...)
    xtt = BK.get_periodic_orbit(p.prob, x, p.p)
    _max = maximum(xtt[1,:])
    _min = minimum(xtt[1,:])
    return (max = _max,
            min = _min,
            amplitude = _max - _min,
            period = getperiod(p.prob, x, p.p))
end

function plotSolution(X, p; k...)
    x = X isa BorderedArray ? X.u : X
    xtt = BK.get_periodic_orbit(p.prob, x, p.p)
    plot!(xtt.t, xtt[1,:]; label = "x", k...)
    plot!(xtt.t, xtt[2,:]; label = "y", k...)
    # plot!(br; subplot = 1, putspecialptlegend = false)
end

function plotSolution(ax, X, p; k...)
    x = X isa BorderedArray ? X.u : X
    xtt = BK.get_periodic_orbit(p.prob, x, p.p)
    lines!(ax, xtt.t, xtt[1,:]; label = "x")
    lines!(ax, xtt.t, xtt[2,:]; label = "y")
    axislegend(ax)
    # plot!(br; subplot = 1, putspecialptlegend = false)
end

argspo = (record_from_solution = recordFromSolution,
    plot_solution = plotSolution
    )
################################################################################
probtrap, ci = generate_ci_problem(PeriodicOrbitTrapProblem(M = 150;  jacobian = :DenseAD, update_section_every_step = 0), prob, sol, 2.)

plot(sol)

solpo = BK.newton(probtrap, ci, NewtonPar(verbose = true))

_sol = BK.get_periodic_orbit(probtrap, solpo.u,1)
plot(_sol.t, _sol[1:2,:]')

opts_po_cont = ContinuationPar(opts_br, max_steps = 50, save_eigenvectors = true, tol_stability = 1e-8)
brpo_fold = continuation(probtrap, ci, PALC(), opts_po_cont;
    verbosity = 3, plot = true,
    argspo...,
    )

pt = get_normal_form(brpo_fold, 1)

prob2 = @set probtrap.prob_vf.lens = @optic _.ϵ
brpo_pd = continuation(prob2, ci, PALC(), opts_po_cont;
    verbosity = 3, plot = true,
    argspo...
    )
pt = get_normal_form(brpo_pd, 1)

# codim 2 Fold
opts_potrap_fold = ContinuationPar(brpo_fold.contparams, detect_bifurcation = 0, max_steps = 100, p_min = 0., p_max=1.2, n_inversion = 4, plot_every_step = 2)
@reset opts_potrap_fold.newton_options.tol = 1e-9
# fold_po_trap1 = continuation(brpo_fold, 2, (@optic _.ϵ), opts_potrap_fold;
#         verbosity = 3, plot = true,
#         detect_codim2_bifurcation = 0,
#         start_with_eigen = false,
#         bothside = true,
#         jacobian_ma = :minaug,
#         bdlinsolver = BorderingBLS(solver = DefaultLS(), check_precision = false),
#         )

# plot(fold_po_trap1)
################################################################################
probcoll, ci = generate_ci_problem(PeriodicOrbitOCollProblem(30, 3), prob, sol, 2.; optimal_period = false)
plot(sol)

solpo = BK.newton(probcoll, ci, NewtonPar(verbose = true));

_sol = BK.get_periodic_orbit(probcoll, solpo.u,1)
plot(_sol.t, _sol[1:2,:]')

opts_po_cont = ContinuationPar(opts_br, max_steps = 50, tol_stability = 1e-8, n_inversion = 6)
@reset opts_po_cont.newton_options.verbose = true
brpo_fold = continuation(probcoll, ci, PALC(), opts_po_cont;
    verbosity = 3, plot = true,
    argspo...
    )

prob2 = @set probcoll.prob_vf.lens = @optic _.ϵ
brpo_pd = continuation(prob2, ci, PALC(), ContinuationPar(opts_po_cont, dsmax = 5e-3, max_steps=250);
    verbosity = 3, plot = true,
    argspo...
    )
pd = get_normal_form(brpo_pd, 1)
#########
# pd branch switching
brpo_pd_sw = continuation(deepcopy(brpo_pd), 1,
                ContinuationPar(brpo_pd.contparams, max_steps = 150);
                verbosity = 0, plot = true,
                # δp = 0.001, ampfactor = 0.1,
                detailed = true, prm = false,
                linear_algo = MatrixBLS(),
                argspo...,)
plot(brpo_pd, brpo_pd_sw)
########
# codim 2 Fold
opts_pocoll_fold = ContinuationPar(brpo_fold.contparams,
                                  detect_bifurcation = 3,
                                  max_steps = 120,
                                  p_min = 0.,
                                  p_max = 1.2,
                                  n_inversion = 4,
                                  plot_every_step = 10)
@reset opts_pocoll_fold.newton_options.tol = 1e-12
fold_po_coll1 = continuation(brpo_fold, 1, (@optic _.ϵ), opts_pocoll_fold;
        verbosity = 3, plot = true,
        detect_codim2_bifurcation = 0,
        start_with_eigen = false,
        bothside = true,
        jacobian_ma = :minaug,
        normC = norminf,
        callback_newton = BK.cbMaxNorm(10),
        bdlinsolver = BorderingBLS(solver = DefaultLS(), check_precision = false),
        )

fold_po_coll2 = @time continuation(brpo_fold, 2, (@optic _.ϵ), opts_pocoll_fold;
        verbosity = 3, plot = true,
        detect_codim2_bifurcation = 0,
        update_minaug_every_step = 1,
        start_with_eigen = false,
        bothside = true,
        jacobian_ma = :minaug,
        normC = norminf,
        callback_newton = BK.cbMaxNorm(10),
        bdlinsolver = BorderingBLS(solver = DefaultLS(), check_precision = false),
        )

# codim 2 PD
opts_pocoll_pd = ContinuationPar(brpo_pd.contparams, detect_bifurcation = 3, max_steps = 60, p_min = -1., plot_every_step = 10, dsmax = 3e-3, ds = 1e-3)
@reset opts_pocoll_pd.newton_options.tol = 1e-9
pd_po_coll = @time continuation(brpo_pd, 1, (@optic _.b0), opts_pocoll_pd;
        verbosity = 3, plot = true,
        detect_codim2_bifurcation = 0,
        update_minaug_every_step = 1,
        start_with_eigen = false,
        usehessian = false,
        jacobian_ma = :minaug,
        # jacobian_ma = :autodiff,
        # jacobian_ma = :finiteDifferences,
        normC = norminf,
        callback_newton = BK.cbMaxNorm(10),
        bothside = true,
        # bdlinsolver = BorderingBLS(solver = DefaultLS(), check_precision = false),
        )

plot(fold_po_coll1, pd_po_coll)

#####
fold_po_coll2 = continuation(brpo_fold, 2, (@optic _.ϵ), opts_pocoll_fold;
        verbosity = 3, plot = true,
        detect_codim2_bifurcation = 0,
        update_minaug_every_step = 1,
        start_with_eigen = false,
        bothside = true,
        jacobian_ma = :minaug,
        callback_newton = BK.cbMaxNorm(10),
        bdlinsolver = BorderingBLS(solver = DefaultLS(), check_precision = false),
        )
plot(fold_po_coll1, fold_po_coll2, ylims = (0, 0.49))
plot!(pd_po_coll, vars = (:ϵ, :b0))


#####
# find the NS case
par_pop2 = @set par_pop.b0 = 0.4
sol2 = solve(remake(prob_de, p = par_pop2, u0 = [0.1,0.1,1,0], tspan=(0,1000)), Rodas5())
sol2 = solve(remake(sol2.prob, tspan = (0,10), u0 = sol2[end]), Rodas5())
plot(sol2, xlims= (8,10))

probcoll, ci = generate_ci_problem(PeriodicOrbitOCollProblem(26, 3; update_section_every_step = 0), re_make(prob, params = sol2.prob.p), sol2, 1.2)

brpo_ns = continuation(probcoll, ci, PALC(), ContinuationPar(opts_po_cont; max_steps = 50, ds = -0.001);
    verbosity = 3, plot = true,
    argspo...,
    # bothside = true,
    )

get_normal_form(brpo_ns, 1)

prob2 = @set probcoll.prob_vf.lens = @optic _.ϵ
brpo_pd = continuation(prob2, ci, PALC(), ContinuationPar(opts_po_cont, dsmax = 5e-3);
    verbosity = 3, plot = true,
    argspo...,
    bothside = true,
    )
get_normal_form(brpo_pd, 2)

# codim 2 PD
opts_pocoll_pd = ContinuationPar(brpo_ns.contparams, detect_bifurcation = 3, max_steps = 40, p_min = 1.e-2, plot_every_step = 1, dsmax = 1e-2, ds = 1e-3)
@reset opts_pocoll_pd.newton_options.tol = 1e-10
pd_po_coll2 = continuation(brpo_pd, 2, (@optic _.b0), opts_pocoll_pd;
        verbosity = 3, plot = true,
        detect_codim2_bifurcation = 2,
        start_with_eigen = false,
        usehessian = false,
        jacobian_ma = :minaug,
        update_minaug_every_step = 1,
        # jacobian_ma = :autodiff,
        # jacobian_ma = :finiteDifferences,
        normC = norminf,
        callback_newton = BK.cbMaxNorm(1),
        bothside = true,
        # bdlinsolver = BorderingBLS(solver = DefaultLS(), check_precision = false),
        )

plot(fold_po_coll1, ylims = (0, 0.49))
plot!(fold_po_coll2)
# plot!(pd_po_coll, vars = (:ϵ, :b0))
plot!(pd_po_coll2, vars = (:ϵ, :b0))

ns = get_normal_form(brpo_ns, 1)

opts_pocoll_ns = ContinuationPar(brpo_pd.contparams, detect_bifurcation = 1, max_steps = 20, p_min = 0., plot_every_step = 1, dsmax = 1e-2, ds = 1e-3)
ns_po_coll = continuation(brpo_ns, 1, (@optic _.ϵ), opts_pocoll_ns;
        verbosity = 3, plot = true,
        detect_codim2_bifurcation = 1,
        start_with_eigen = false,
        usehessian = false,
        jacobian_ma = :minaug,
        # jacobian_ma = :autodiff,
        # jacobian_ma = :finiteDifferences,
        normC = norminf,
        callback_newton = BK.cbMaxNorm(10),
        bothside = true,
        # bdlinsolver = BorderingBLS(solver = DefaultLS(), check_precision = false),
        )

plot!(ns_po_coll, vars = (:ϵ, :b0))
plot!(pd_po_coll2, vars = (:ϵ, :b0))

#####
# find the PD case
par_pop2 = @set par_pop.b0 = 0.45
sol2 = solve(remake(prob_de, p = par_pop2, u0 = [0.1,0.1,1,0], tspan=(0,1000)), Rodas5())
sol2 = solve(remake(sol2.prob, tspan = (0,10), u0 = sol2[end]), Rodas5())
plot(sol2, xlims= (8,10))

probcoll, ci = generate_ci_problem(PeriodicOrbitOCollProblem(26, 3; update_section_every_step = 0), re_make(prob, params = sol2.prob.p), sol2, 1.2)

prob2 = @set probcoll.prob_vf.lens = @optic _.ϵ
brpo_pd = continuation(prob2, ci, PALC(), ContinuationPar(opts_po_cont, dsmax = 5e-3);
    verbosity = 3, plot = true,
    argspo...,
    bothside = true,
    )

# codim 2 PD
opts_pocoll_pd = ContinuationPar(brpo_pd.contparams, detect_bifurcation = 3, max_steps = 40, p_min = 1.e-2, plot_every_step = 1, dsmax = 1e-2, ds = -1e-3)
@reset opts_pocoll_pd.newton_options.tol = 1e-10
pd_po_coll2 = continuation(brpo_pd, 2, (@optic _.b0), opts_pocoll_pd;
        verbosity = 3, plot = false,
        detect_codim2_bifurcation = 2,
        start_with_eigen = false,
        usehessian = false,
        jacobian_ma = :minaug,
        # jacobian_ma = :autodiff,
        # jacobian_ma = :finiteDifferences,
        normC = norminf,
        callback_newton = BK.cbMaxNorm(10),
        bothside = true,
        # bdlinsolver = BorderingBLS(solver = DefaultLS(), check_precision = false),
        )

plot(fold_po_coll1, ylims = (0, 0.49))
plot!(fold_po_coll2)
# plot!(pd_po_coll, vars = (:ϵ, :b0))
plot!(pd_po_coll2, vars = (:ϵ, :b0))
################################################################################
######    Shooting ########
probsh, cish = generate_ci_problem( ShootingProblem(M=3), prob, prob_de, sol, 2.; alg = Rodas5(), parallel = true)

solpo = newton(probsh, cish, NewtonPar(verbose = true))

_sol = BK.get_periodic_orbit(probsh, solpo.u, sol.prob.p)
plot(_sol.t, _sol[1:2,:]')

opts_po_cont = setproperties(opts_br, max_steps = 50, save_eigenvectors = true, detect_loop = true, tol_stability = 1e-3)
@reset opts_po_cont.newton_options.verbose = false
br_fold_sh = continuation(probsh, cish, PALC(tangent = Bordered()), opts_po_cont;
    verbosity = 3, plot = true,
    argspo...)
pt = get_normal_form(br_fold_sh, 1)

probsh2 = @set probsh.lens = @optic _.ϵ
brpo_pd_sh = continuation(probsh2, cish, PALC(), opts_po_cont;
    verbosity = 3, plot = true,
    argspo...
    )
pt = get_normal_form(brpo_pd_sh, 1)

# codim 2 Fold
opts_posh_fold = ContinuationPar(br_fold_sh.contparams, detect_bifurcation = 3, max_steps = 200, p_min = 0.01, p_max = 1.2)
@error "it fails if the tolerance tol is too high"
@reset opts_posh_fold.newton_options.tol = 1e-12
fold_po_sh1 = continuation(br_fold_sh, 2, (@optic _.ϵ), opts_posh_fold;
        verbosity = 2, plot = true,
        detect_codim2_bifurcation = 2,
        jacobian_ma = :minaug,
        start_with_eigen = false,
        bothside = true,
        callback_newton = BK.cbMaxNorm(1),
        )

fold_po_sh2 = continuation(br_fold_sh, 1, (@optic _.ϵ), opts_posh_fold;
        verbosity = 2, plot = true,
        detect_codim2_bifurcation = 2,
        jacobian_ma = :minaug,
        start_with_eigen = false,
        bothside = true,
        callback_newton = BK.cbMaxNorm(1),
        )

# codim 2 PD
opts_posh_pd = ContinuationPar(brpo_pd_sh.contparams, detect_bifurcation = 3, max_steps = 40, p_min = -1.)
@reset opts_posh_pd.newton_options.tol = 1e-12
@error "it fails if the tolerance tol is too high"
@reset opts_posh_pd.newton_options.verbose = true
pd_po_sh = continuation(brpo_pd_sh, 1, (@optic _.b0), opts_posh_pd;
        verbosity = 0, plot = true,
        detect_codim2_bifurcation = 0,
        jacobian_ma = :minaug,
        # jacobian_ma = :autodiff,
        # jacobian_ma = :finiteDifferences,
        usehessian = false,
        start_with_eigen = false,
        bothside = true,
        callback_newton = BK.cbMaxNorm(1),
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

probshns, ci = generate_ci_problem(ShootingProblem(M=3), re_make(prob, params = sol2.prob.p), remake(prob_de, p = par_pop2), sol2, 1.; alg = Rodas5())

brpo_ns = continuation(probshns, ci, PALC(), ContinuationPar(opts_po_cont; max_steps = 50, ds = -0.001);
    verbosity = 0, plot = true,
    argspo...,
    # bothside = true,
    callback_newton = BK.cbMaxNorm(1),
    )

ns = get_normal_form(brpo_ns, 1)

# codim 2 NS
opts_posh_ns = ContinuationPar(brpo_ns.contparams, detect_bifurcation = 0, max_steps = 100, p_min = -0., p_max = 1.2, ds = 0.001)
@reset opts_posh_ns.newton_options.tol = 1e-12
@reset opts_posh_ns.newton_options.verbose = true
ns_po_sh = continuation(brpo_ns, 1, (@optic _.ϵ), opts_posh_ns;
        verbosity = 2, plot = true,
        detect_codim2_bifurcation = 0,
        start_with_eigen = false,
        usehessian = false,
        jacobian_ma = :minaug,
        # jacobian_ma = :autodiff,
        # jacobian_ma = :finiteDifferences,
        normC = norminf,
        bothside = false,
        callback_newton  = BK.cbMaxNorm(1),
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

probshpd, ci = generate_ci_problem(ShootingProblem(M=3), re_make(prob, params = sol2.prob.p), remake(prob_de, p = par_pop2), sol2, 1.; alg = Rodas5())

prob2 = @set probshpd.lens = @optic _.ϵ
brpo_pd = continuation(prob2, ci, PALC(), ContinuationPar(opts_po_cont, dsmax = 5e-3);
    verbosity = 3, plot = true,
    argspo...,
    bothside = true,
    )

get_normal_form(brpo_pd, 2)
# codim 2 PD
opts_pocoll_pd = ContinuationPar(brpo_pd.contparams, detect_bifurcation = 3, max_steps = 40, p_min = 1.e-2, plot_every_step = 10, dsmax = 1e-2, ds = -1e-3)
@reset opts_pocoll_pd.newton_options.tol = 1e-12
pd_po_sh2 = continuation(brpo_pd, 2, (@optic _.b0), opts_pocoll_pd;
        verbosity = 3, plot = true,
        detect_codim2_bifurcation = 2,
        start_with_eigen = false,
        usehessian = false,
        jacobian_ma = :minaug,
        # jacobian_ma = :autodiff,
        # jacobian_ma = :finiteDifferences,
        normC = norminf,
        callback_newton = BK.cbMaxNorm(10),
        bothside = true,
        # bdlinsolver = BorderingBLS(solver = DefaultLS(), check_precision = false),
        )

plot(fold_po_sh1, fold_po_sh2, branchlabel = ["FOLD", "FOLD"])
# plot!(ns_po_sh, vars = (:ϵ, :b0), branchlabel = "NS")
# plot!(pd_po_sh, vars = (:ϵ, :b0), branchlabel = "PD")
plot!(pd_po_sh2, vars = (:ϵ, :b0), branchlabel = "PD2")