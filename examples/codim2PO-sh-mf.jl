# CAREFUL: VERY ADVANCED EXAMPLE
using Revise
using Test, ForwardDiff, Plots, LinearAlgebra
using BifurcationKit, Test
const BK = BifurcationKit

using ComponentArrays # this is for SciMLSensitivity and adjoint of flow
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

par_pop = ComponentArray( K = 1., r = 2π, a = 4π, b0 = 0.25, e = 1., d = 2π, ϵ = 0.2, )

z0 = [0.1,0.1,1,0]

prob = BifurcationProblem(Pop!, z0, par_pop, (@optic _.b0); record_from_solution = (x, p; k...) -> (x = x[1], y = x[2], u = x[3]))

opts_br = ContinuationPar(p_min = 0., p_max = 20.0, ds = 0.002, dsmax = 0.01, n_inversion = 6, detect_bifurcation = 3, max_bisection_steps = 25, nev = 4, max_steps = 20000)
@reset opts_br.newton_options.verbose = true

################################################################################
using DifferentialEquations
prob_de = ODEProblem(Pop!, z0, (0, 600), par_pop)
alg = Rodas5()
# alg = Vern9()
sol = DifferentialEquations.solve(prob_de, alg)
prob_de = ODEProblem(Pop!, sol.u[end], (0,5.), par_pop, reltol = 1e-10, abstol = 1e-12)
sol = DifferentialEquations.solve(prob_de, Rodas5())
plot(sol)
################################################################################
argspo = (record_from_solution = (x, p; k...) -> begin
        xtt = BK.get_periodic_orbit(p.prob, x, set(getparams(p.prob), BK.getlens(p.prob), p.p))
        return (max = maximum(xtt[1,:]),
                min = minimum(xtt[1,:]),
                period = getperiod(p.prob, x, set(getparams(p.prob), BK.getlens(p.prob), p.p)))
    end,
    plot_solution = (X, p; k...) -> begin
        x = X isa BorderedArray ? X.u : X
        xtt = BK.get_periodic_orbit(p.prob, x, set(getparams(p.prob), BK.getlens(p.prob), p.p))
        plot!(xtt.t, xtt[1,:]; label = "x", k...)
        plot!(xtt.t, xtt[2,:]; label = "y", k...)
        # plot!(br; subplot = 1, putspecialptlegend = false)
    end)
################################################################################
using Test, ForwardDiff
import DifferentiationInterface as DI

probsh0 = ShootingProblem(M=1)

probshMatrix, = generate_ci_problem( ShootingProblem(M=3), prob, prob_de, sol, 2.; alg = Rodas5())
probsh, cish = generate_ci_problem( ShootingProblem(M=3), prob, prob_de, sol, 2.; alg = Rodas5(),
            jacobian = BK.AutoDiffMF()
            # jacobian = BK.FiniteDifferencesMF()
            )
######
# il faut un jvp precis pour la monodromy sinon ca ne marche pas bien
@error "c'est pas bon ca car flowDE n'as pas de structure pour les fonctions"
######
function flow(x0, prob0, tm, p = prob0.p)
    prob = remake(prob0, u0 = x0, tspan = (0, tm), p = p)
    sol = DifferentialEquations.solve(prob, Rodas5())
    return sol[end]
end

sol0_f = rand(4)
flow(rand(4), prob_de, 1)

dϕ = ForwardDiff.jacobian(x->flow(x, prob_de, 1), sol0_f)
# jvp
res1 = ForwardDiff.derivative(t->flow(sol0_f .+ t .* sol0_f, prob_de, 1), zero(eltype(sol0_f)))
@test norm(res1 - dϕ * sol0_f, Inf) < 1e-8

res1, = DI.pushforward(x->flow(x, prob_de, 1), DI.AutoForwardDiff(), sol0_f, (sol0_f,))
@test norm(res1 - dϕ * sol0_f, Inf) < 1e-7

# vjp
res1, = DI.pullback(x->flow(x, prob_de, 1), DI.AutoForwardDiff(), sol0_f, (sol0_f,))
@test norm(res1 - dϕ' * sol0_f, Inf) < 1e-8
######


@reset probsh.flow.vjp = (x,p,dx,tm) -> DI.pullback(z->flow(z, prob_de,tm,p), DI.AutoForwardDiff(), x, (dx,))[1]

lspo = GMRESIterativeSolvers(verbose = false, N = length(cish), abstol = 1e-12, reltol = 1e-10)
eigpo = EigKrylovKit(x₀ = rand(4))
optnpo = NewtonPar(verbose = true, linsolver = lspo, eigsolver = eigpo)
solpo = BK.newton(probsh, cish, optnpo)

_sol = BK.get_periodic_orbit(probsh, solpo.u, sol.prob.p)
plot(_sol.t, _sol[1:2,:]')

opts_po_cont = setproperties(opts_br, max_steps = 50, save_eigenvectors = true, detect_loop = true, tol_stability = 1e-3, newton_options = optnpo)
@reset opts_po_cont.newton_options.verbose = true
br_fold_sh = continuation(probsh, cish, PALC(tangent = Bordered()), opts_po_cont;
    verbosity = 3, plot = true,
    linear_algo = MatrixFreeBLS(@set lspo.N = lspo.N+1),
    argspo...)
pt = get_normal_form(br_fold_sh, 1)

probsh2 = @set probsh.lens = @optic _.ϵ
brpo_pd_sh = continuation(probsh2, cish, PALC(), opts_po_cont;
    verbosity = 3, plot = true,
    linear_algo = MatrixFreeBLS(@set lspo.N = lspo.N+1),
    argspo...
    )

# pt = get_normal_form(brpo_pd_sh, 1)

# codim 2 Fold
opts_posh_fold = ContinuationPar(br_fold_sh.contparams, detect_bifurcation = 0, max_steps = 20, p_min = 0.01, p_max = 1.2)
@reset opts_posh_fold.newton_options.tol = 1e-9

# use this option for jacobian_ma = BK.FiniteDifferencesMF(), otherwise do not
@reset opts_posh_fold.newton_options.linsolver.solver.N = opts_posh_fold.newton_options.linsolver.solver.N+1
# fold_po_sh1 = continuation(br_fold_sh, 2, (@optic _.ϵ), opts_posh_fold;
#     verbosity = 2, plot = true,
#     detect_codim2_bifurcation = 0,
#     # jacobian_ma = :finiteDifferencesMF,

#     jacobian_ma = :minaug,
#     bdlinsolver = MatrixFreeBLS(@set lspo.N = lspo.N+1),
#     # linear_algo = MatrixFreeBLS(@set lspo.N = lspo.N+2),
#     # # usehessian = false,

#     start_with_eigen = true,
#     bothside = true,
#     callback_newton = BK.cbMaxNorm(1),
#     )

# fold_po_sh2 = continuation(br_fold_sh, 1, (@optic _.ϵ), opts_posh_fold;
#         verbosity = 2, plot = true,
#         detect_codim2_bifurcation = 0,
#         # jacobian_ma = :minaug,
#         jacobian_ma = :finiteDifferencesMF,
#         bdlinsolver = MatrixFreeBLS(@set lspo.N = lspo.N+1),
#         start_with_eigen = false,
#         bothside = true,
#         callback_newton = BK.cbMaxNorm(1),
#         )

# plot(fold_po_sh1, fold_po_sh2, branchlabel = ["FOLD", "FOLD"])

# codim 2 PD
opts_posh_pd = ContinuationPar(brpo_pd_sh.contparams, detect_bifurcation = 3, max_steps = 40, p_min = -1.)
@reset opts_posh_pd.newton_options.tol = 1e-8
# use this option for jacobian_ma = BK.FiniteDifferencesMF(), otherwise do not
# @reset opts_posh_pd.newton_options.linsolver.solver.N = opts_posh_pd.newton_options.linsolver.solver.N+1
pd_po_sh = continuation(brpo_pd_sh, 1, (@optic _.b0), opts_posh_pd;
    verbosity = 2, plot = false,
    detect_codim2_bifurcation = 0,
    usehessian = false,
    jacobian_ma = BK.MinAug(),
    # jacobian_ma = BK.FiniteDifferencesMF(),
    bdlinsolver = MatrixFreeBLS(@set lspo.N = lspo.N+1),
    start_with_eigen = false,
    bothside = true,
    callback_newton = BK.cbMaxNorm(1),
    )

plot(pd_po_sh)

# plot(fold_po_sh1, fold_po_sh2, branchlabel = ["FOLD", "FOLD"])
plot!(pd_po_sh, vars = (:ϵ, :b0), branchlabel = "PD")


#####
# find the PD NS case
par_pop2 = @set par_pop.b0 = 0.45
sol2 = DifferentialEquations.solve(remake(prob_de, p = par_pop2, u0 = [0.1,0.1,1,0], tspan=(0,1000)), Rodas5())
sol2 = DifferentialEquations.solve(remake(sol2.prob, tspan = (0,10), u0 = sol2[end]), Rodas5())
plot(sol2, xlims= (8,10))

probshns, ci = generate_ci_problem( ShootingProblem(M=3), re_make(prob, params = sol2.prob.p), remake(prob_de, p = par_pop2), sol2, 1.; alg = Rodas5(),
            jacobian = BK.AutoDiffMF()
            # jacobian = BK.FiniteDifferencesMF()
            )

@reset probshns.flow.vjp = (x,p,dx,tm) -> DI.pullback(z->flow(z, prob_de,tm,p), DI.AutoForwardDiff(), x, (dx,))[1]

brpo_ns = continuation(probshns, ci, PALC(), ContinuationPar(opts_po_cont; max_steps = 50, ds = -0.001);
    verbosity = 3, plot = true,
    argspo...,
    # bothside = true,
    callback_newton = BK.cbMaxNorm(1),
    linear_algo = MatrixFreeBLS(@set lspo.N = lspo.N+1),
    )

ns = get_normal_form(brpo_ns, 1)

# codim 2 NS
opts_posh_ns = ContinuationPar(brpo_ns.contparams, detect_bifurcation = 0, max_steps = 100, p_min = -0., p_max = 1.2)
@reset opts_posh_ns.newton_options.tol = 1e-9

ns_po_sh = continuation(brpo_ns, 1, (@optic _.ϵ), opts_posh_ns;
        verbosity = 2, plot = false,
        detect_codim2_bifurcation = 0,
        start_with_eigen = false,
        usehessian = false,
        jacobian_ma = BK.MinAug(),
        # jacobian_ma = :autodiff,
        # jacobian_ma = :finiteDifferences,
        normC = norminf,
        bothside = false,
        callback_newton = BK.cbMaxNorm(1),
        bdlinsolver = MatrixFreeBLS(@set lspo.N = lspo.N+2),
        # bdlinsolver = BorderingBLS(@set lspo.N = lspo.N+2),
        )

plot(ns_po_sh, vars = (:ϵ, :b0), branchlabel = "NS")
plot!(pd_po_sh, vars = (:ϵ, :b0), branchlabel = "PD")
plot!(fold_po_sh1, vars = (:ϵ, :b0), branchlabel = "FOLD")
plot!(fold_po_sh2, vars = (:ϵ, :b0), branchlabel = "FOLD")
