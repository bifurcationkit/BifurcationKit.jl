# using Revise, Plots
using Test, ForwardDiff, LinearAlgebra
using BifurcationKit, Test
const BK = BifurcationKit

using ComponentArrays # this is for SciMLSensitivity and adjoint of flow
###################################################################################################
function Pop!(du, X, p, t = 0)
    (; r,K,a,ϵ,b0,e,d) = p
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
z0 = [0.1, 0.1, 1, 0]
prob = BifurcationProblem(Pop!, z0, par_pop, (@optic _.b0); record_from_solution = (x, p) -> (x = x[1], y = x[2], u = x[3]))
opts_br = ContinuationPar(p_min = 0., p_max = 20.0, ds = 0.002, dsmax = 0.01, n_inversion = 6, detect_bifurcation = 3, max_bisection_steps = 25, nev = 4)
@reset opts_br.newton_options.verbose = true
################################################################################
import OrdinaryDiffEq as ODE
prob_de = ODE.ODEProblem(Pop!, z0, (0, 600.), par_pop)
alg = ODE.Rodas5()
sol = ODE.solve(prob_de, alg)
prob_de = ODE.ODEProblem(Pop!, sol.u[end], (0, 5), par_pop, reltol = 1e-10, abstol = 1e-12)
sol = ODE.solve(prob_de, ODE.Rodas5())
################################################################################
argspo = (record_from_solution = (x, p; k...) -> begin
        xtt = BK.get_periodic_orbit(p.prob, x, p.p)
        return (max = maximum(xtt[1,:]),
                min = minimum(xtt[1,:]),
                period = getperiod(p.prob, x, p.p))
    end,)
################################################################################
# @info "import AD"
# import AbstractDifferentiation as AD
# @info "import Zygote"
# using Zygote, SciMLSensitivity

probsh, cish = generate_ci_problem( ShootingProblem(M=3), deepcopy(prob), deepcopy(prob_de), deepcopy(sol), 2.; 
    alg = ODE.Rodas5(),
    jacobian = BK.AutoDiffMF(),
    # jacobian = BK.FiniteDifferencesMF(),
    # parallel = true,
    )

function flow(x0, prob0, tm, p = prob0.p)
    prob = remake(prob0, u0 = x0, tspan = (0, tm), p = p)
    sol = ODE.solve(prob, Rodas5())
    return sol[end]
end

lspo = GMRESIterativeSolvers(verbose = false, N = length(cish), abstol = 1e-12, reltol = 1e-10)
eigpo = EigKrylovKit(x₀ = rand(4))
optnpo = NewtonPar(verbose = true, linsolver = lspo, eigsolver = eigpo)
solpo = newton(probsh, cish, optnpo)

_sol = BK.get_periodic_orbit(probsh, solpo.u, sol.prob.p)
# plot(_sol.t, _sol[1:2,:]')

opts_po_cont = setproperties(opts_br, max_steps = 50, save_eigenvectors = true, detect_loop = true, tol_stability = 1e-3, newton_options = optnpo)
@reset opts_po_cont.newton_options.verbose = false
br_fold_sh = continuation(probsh, cish, PALC(tangent = Bordered()), opts_po_cont;
    # verbosity = 3, plot = true,
    linear_algo = MatrixFreeBLS(lspo),
    argspo...)
pt = get_normal_form(br_fold_sh, 1)

probsh2 = @set probsh.lens = @optic _.ϵ
brpo_pd_sh = continuation(probsh2, cish, PALC(), opts_po_cont;
    # verbosity = 3, plot = true,
    # linear_algo = MatrixFreeBLS(@set lspo.N = lspo.N+1),
    linear_algo = MatrixFreeBLS(lspo),
    argspo...
    )
pt = get_normal_form(brpo_pd_sh, 1)

# codim 2 Fold
opts_posh_fold = ContinuationPar(br_fold_sh.contparams, detect_bifurcation = 0, max_steps = 0, p_min = 0.01, p_max = 1.2)
@reset opts_posh_fold.newton_options.tol = 1e-8
# @reset opts_posh_fold.newton_options.linsolver.solver.N = opts_posh_fold.newton_options.linsolver.solver.N+1
@reset opts_posh_fold.newton_options.verbose = false
@reset opts_posh_fold.newton_options.linsolver.solver.verbose=0
# fold_po_sh1 = continuation(br_fold_sh, 2, (@optic _.ϵ), opts_posh_fold;
#     # verbosity = 3, #plot = true,
#     detect_codim2_bifurcation = 0,
#     jacobian_ma = :finiteDifferencesMF,
#     bdlinsolver = MatrixFreeBLS(lspo),
#     start_with_eigen = true,
#     callback_newton = BK.cbMaxNorm(1),
#     )

# @test fold_po_sh1.kind isa BK.FoldPeriodicOrbitCont

# codim 2 PD
@info "--> PD curve"
opts_posh_pd = ContinuationPar(brpo_pd_sh.contparams, detect_bifurcation = 0, max_steps = 4, p_min = -1.)
@reset opts_posh_pd.newton_options.tol = 1e-8
pd_po_sh = continuation(brpo_pd_sh, 1, (@optic _.b0), opts_posh_pd;
    # verbosity = 3, #plot = true,
    detect_codim2_bifurcation = 0,
    usehessian = false,
    # jacobian_ma = BK.MinAug(),
    jacobian_ma = BK.FiniteDifferencesMF(),
    # bdlinsolver = BorderingBLS(@set lspo.N=12),
    bdlinsolver = MatrixFreeBLS(lspo),
    start_with_eigen = false,
    callback_newton = BK.cbMaxNorm(1),
    update_minaug_every_step = 0,
    )

# plot(pd_po_sh)
# plot(fold_po_sh1, fold_po_sh2, branchlabel = ["FOLD", "FOLD"])
#     plot!(pd_po_sh, vars = (:ϵ, :b0), branchlabel = "PD")


#####
# find the PD NS case
par_pop2 = @set par_pop.b0 = 0.45
sol2 = ODE.solve(ODE.remake(prob_de, p = par_pop2, u0 = [0.1,0.1,1,0], tspan=(0,1000)), ODE.Rodas5())
sol2 = ODE.solve(ODE.remake(sol2.prob, tspan = (0,10), u0 = sol2[end]), ODE.Rodas5())
# plot(sol2, xlims= (8,10))

probshns, ci = generate_ci_problem( ShootingProblem(M=3), re_make(prob, params = sol2.prob.p), ODE.remake(prob_de, p = par_pop2), sol2, 1.; alg = ODE.Rodas5(),
            jacobian = BK.AutoDiffMF()
            )

# @reset probshns.flow.vjp = (x,p,dx,tm) -> AD.pullback_function(AD.ZygoteBackend(), z->flow(z, prob_de,tm,p), x)(dx)[1]

brpo_ns = continuation(probshns, ci, PALC(), ContinuationPar(opts_po_cont; max_steps = 50, ds = -0.001);
    # verbosity = 3, plot = true,
    argspo...,
    callback_newton = BK.cbMaxNorm(1),
    linear_algo = MatrixFreeBLS(lspo),
    )

# ns = get_normal_form(brpo_ns, 1)

# codim 2 NS
@info "--> NS curve"
opts_posh_ns = ContinuationPar(brpo_ns.contparams, detect_bifurcation = 0, max_steps = 0, p_min = -0., p_max = 1.2)
@reset opts_posh_ns.newton_options.tol = 1e-8
@reset opts_posh_ns.newton_options.linsolver.solver.verbose = 0
@reset opts_posh_ns.newton_options.verbose = false
ns_po_sh = continuation(brpo_ns, 1, (@optic _.ϵ), opts_posh_ns;
        # verbosity = 2, plot = true,
        detect_codim2_bifurcation = 0,
        start_with_eigen = false,
        usehessian = false,
        # jacobian_ma = BK.MinAug(),
        jacobian_ma = BK.FiniteDifferencesMF(),
        normC = norminf,
        bothside = false,
        callback_newton = BK.cbMaxNorm(1),
        bdlinsolver = MatrixFreeBLS(lspo),
        )
@test ns_po_sh.kind isa BK.NSPeriodicOrbitCont

# plot(ns_po_sh, vars = (:ϵ, :b0), branchlabel = "NS")
#     plot!(pd_po_sh, vars = (:ϵ, :b0), branchlabel = "PD")
#     plot!(fold_po_sh1, vars = (:ϵ, :b0), branchlabel = "FOLD")
#     plot!(fold_po_sh2, vars = (:ϵ, :b0), branchlabel = "FOLD")
