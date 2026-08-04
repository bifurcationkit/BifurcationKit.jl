# using Revise#, Plots
using LinearAlgebra, Test
using BifurcationKit
const BK = BifurcationKit
import OrdinaryDiffEq as ODE
####################################################################################################
record_from_solution(x, p; k...) = (u1 = x[1], u2 = x[2])

function lur(u, p, t = 0.) # use t = Float64 for type constant function
    (; α, β) = p
    x, y, z = u
    return [y, z, -α * z - β * y - x + x^2]
end

# jacobian-vector product of lur
function dlur(u, du, p)
    (; α, β) = p
    x = u[1]
    dx, dy, dz = du
    return [dy, dz, -α * dz - β * dy - dx + 2 * x * dx]
end

# stacked system [lur(u); jvp of lur along du] used to compute the monodromy
function lurStack(u, p, t = 0.)
    x = u[1:3]
    dx = u[4:6]
    f = lur(x, p)
    jvp = dlur(x, dx, p)
    [f[1], f[2], f[3], jvp[1], jvp[2], jvp[3]]
end

function plotPO(x, p; k...)
    xtt = get_periodic_orbit(p.prob, x, p.p)
    plot!(xtt.t, xtt[1,:]; markersize = 2, marker = :d, k...)
    plot!(xtt.t, xtt[2,:]; k...)
    plot!(xtt.t, xtt[3,:]; legend = false, k...)
end

# record function
function recordPO(x, p; k...)
    xtt = get_periodic_orbit(p.prob, x, p.p)
    period = getperiod(p.prob, x, p.p)
    return (max = maximum(xtt[1,:]), min = minimum(xtt[1,:]), period = period)
end
####################################################################################################
prob = BK.ODEBifProblem(lur, zeros(3), (α = -1.0, β = 1.), (@optic _.α); record_from_solution)
opts_br = ContinuationPar(p_min = -1.4, p_max = 1.8, ds = -0.01, dsmax = 0.01, n_inversion = 8, detect_bifurcation = 3, max_bisection_steps = 25, nev = 3, plot_every_step = 20, max_steps = 1000)
opts_br = @set opts_br.newton_options.verbose = false
br = continuation(prob, PALC(tangent = Bordered()), opts_br; bothside = true, normC = norminf)
# plot(br)

let
opts_br = ContinuationPar(p_min = -1.4, p_max = 1.8, ds = -0.01, dsmax = 0.01, n_inversion = 8, detect_bifurcation = 3, max_bisection_steps = 25, nev = 3, plot_every_step = 20, max_steps = 1000)
opts_br = @set opts_br.newton_options.verbose = false
br = continuation(prob, PALC(tangent = Bordered()), opts_br;
bothside = true, normC = norminf)

# plot(br)
####################################################################################################
# continuation parameters
opts_po_cont = ContinuationPar(dsmax = 0.02, dsmin = 1e-4, p_max = 1.1, max_steps = 80, tol_stability = 1e-4, ds = -0.01)

Mt = 120 # number of time sections
br_po = continuation(
        br, 2, opts_po_cont,
        Trapeze(M = Mt; update_section_every_step = 1, jacobian = BK.Dense());
        ampfactor = 1., δp = 0.01,
        verbosity = 0, plot = false,
        record_from_solution = recordPO,
        plot_solution = plotPO,
        finalise_solution = (z, tau, step, contResult; prob = nothing, kwargs...) -> begin
                return z.u[end] < 40
                true
            end,
        normC = norminf)

# plot(br, br_po)
# plot(br_po, vars=(:param, :period))

show(br_po)

# test normal forms
for _ind in (1,3,16)
    if _ind <= length(br_po.specialpoint) &&
        length(br_po.specialpoint) >= 3 &&
        br_po.specialpoint[_ind].type ∈ (:bp, :pd, :ns)
            println("")
            local pt = get_normal_form(br_po, _ind; verbose = true)
            predictor(pt, 0.1, 1.)
            show(pt)
    end
end

# aBS from PD
br_po_pd = continuation(br_po, 1, setproperties(br_po.contparams, detect_bifurcation = 3, max_steps = 5, ds = 0.01, dsmax = 0.01, plot_every_step = 10);
    verbosity = 0, plot = false,
    ampfactor = .2, δp = -0.005,
    usedeflation = true,
    normC = norminf
    )

# plot(br, br_po, br_po_pd, xlims=(0.0,1.5))
####################################################################################################
# case of collocation
opts_po_cont = ContinuationPar(dsmax = 0.03, ds= -0.0001, dsmin = 1e-4, p_max = 1.8, p_min=-0.9, max_steps = 120, newton_options = NewtonPar(tol = 1e-11,  max_iterations = 25), nev = 3, tol_stability = 1e-4, detect_bifurcation = 3, plot_every_step = 20, save_sol_every_step = 1, n_inversion = 8)

for meshadapt in (false, true)
    local br_po = continuation(
                br, 2, opts_po_cont,
                Collocation(40, 4; meshadapt, K = 200);
                alg = PALC(),
                record_from_solution = recordPO,
                plot_solution = plotPO,
                normC = norminf)
    @test br_po.specialpoint[1].param ≈  0.63031334 rtol = 1e-4
    @test br_po.specialpoint[2].param ≈ -0.63031334 rtol = 1e-4

    # test normal forms
    for _ind in (1,)
        if length(br_po.specialpoint) >=1 && br_po.specialpoint[_ind].type ∈ (:bp, :pd, :ns)
            println("")
            for prm in (true, false)
                pt = get_normal_form(br_po, _ind; verbose = true, prm = Val(prm))
                show(pt)
            end
        end
    end

    pd = get_normal_form(br_po, 1; verbose = false, prm = Val(true))
    @test_skip pt.nf.a * pt.nf.c3 > 0
    predictor(pd, 0.1, 1)
    pd = get_normal_form(br_po, 1; verbose = false, prm = Val(false))
    predictor(pd, 0.1, 1)
    @test pd.nf.nf.b3 ≈ -0.30509421737255177 rtol=1e-2 # reference value computed with ApproxFun
    # @test pd.nf.nf.a  ≈ 0.020989802220981707 rtol=1e-3 # reference value computed with ApproxFun

    # aBS from PD
    continuation(br_po, 1, setproperties(br_po.contparams, detect_bifurcation = 3, max_steps = 5, ds = 0.01, dsmax = 0.01, plot_every_step = 10);
        ampfactor = .2, δp = -0.005,
        usedeflation = true,
    )
end

# ----- test saved_solution interface -----
let
    x0 = rand(10)
    mesh0 = collect(range(0, 1, length=11))
    saved = BK.POSavedSolutionAndState(mesh0, x0, mesh0, zeros(10))
    @test BK.saved_solution(x0) === x0
    @test BK.saved_solution(saved) === x0
end
####################################################################################################
probsh = ODE.ODEProblem(lur, zeros(3), (0., 1), BK.getparams(prob); abstol = 1e-12, reltol = 1e-10)
probsh_monodromy = ODE.ODEProblem(lurStack, zeros(6), (0., 1), BK.getparams(prob); abstol = 1e-12, reltol = 1e-10)

# continuation parameters
opts_po_cont = ContinuationPar(dsmax = 0.02, ds= -0.001, dsmin = 1e-4, max_steps = 122, tol_stability = 1e-5, detect_bifurcation = 3, plot_every_step = 10, n_inversion = 8, nev = 3)

br_po = continuation(
    br, 2, opts_po_cont,
    Shooting(10, probsh, ODE.Vern9(), probsh_monodromy, ODE.Vern9(); parallel = false, abstol = 1e-12, reltol = 1e-10);
    # verbosity = 2,    plot = true,
    record_from_solution = recordPO,
    plot_solution = plotPO,
    callback_newton = BK.cbMaxNorm(10),
    normC = norminf)

pt = BK.get_normal_form(br_po, 1)
# Period-Doubling bifurcation point of periodic orbit
# ├─ Period = 6.364071672903722 -> 12.728143345807444
# ├─ Problem: Shooting
# SuperCritical - Period-Doubling bifurcation point at α ≈ 0.6303003064801065
# ┌─ Normal form:
# ├        x ─▶ x⋅(a⋅δα - 1 + c⋅x²)
# ├─ a = 8.833724502729645
# └─ c = 25.631010143700337
# BK.predictor(pt, 0.1,1).δp #0.1
# a and b must have the same sign
@test_skip pt.nf.a * pt.nf.c3 > 0

show(br_po)

# plot(br, br_po)
# plot(br_po, vars=(:param, :period))

# test the saved solutions
for ind in eachindex(br_po.sol)
    pt = br_po.sol[ind]
    pars = BK.setparam(br_po, pt.p)
    wrap = BK.getprob(br_po)
    BK.restore_problem!(wrap, pt.x, pars)
    # @error "" BK.residual(wrap, BK.saved_solution(pt.x), pars)
    @test norminf(BK.residual(wrap, BK.saved_solution(pt.x), pars)) < br_po.contparams.newton_options.tol
end

for ind in eachindex(br_po.specialpoint)
    pt = br_po.specialpoint[ind]
    pars = BK.setparam(br_po, pt.param)
    wrap = BK.getprob(br_po)
    BK.restore_problem!(wrap, pt.x, pars)
    # @error "" BK.residual(wrap, BK.saved_solution(pt.x), pars)
    @test norminf(BK.residual(wrap, BK.saved_solution(pt.x), pars)) < br_po.contparams.newton_options.tol
end

@test br_po.specialpoint[1].param ≈ 0.63031334 rtol = 1e-4
@test br_po.specialpoint[2].param ≈ -0.63031334 atol = 1e-2

# test showing normal form
for _ind in (1,3)
    if length(br_po.specialpoint) >=3 && br_po.specialpoint[_ind].type ∈ (:pd, :ns)
        println("")
        local pt = get_normal_form(br_po, _ind; verbose = true)
        show(pt)
        predictor(pt, 0.1, 1.)
        show(pt)
    end
end

# aBS from PD
br_po_pd = continuation(br_po, 1, setproperties(br_po.contparams, max_steps = 5, ds = -0.02, plot_every_step = 1, save_sol_every_step = 1);
    # verbosity = 0, plot = false,
    # usedeflation = true,
    ampfactor = .1, δp = -0.005,
    record_from_solution = recordPO,
    normC = norminf,
    callback_newton = BK.cbMaxNorm(10),
    )

# plot(br_po, br_po_pd)
#######################################
opts_po_cont_ps = @set opts_po_cont.newton_options.tol = 1e-11
# @set opts_po_cont_ps.dsmax = 0.0025
br_po = continuation(br, 2, opts_po_cont_ps,
    PoincareShooting(3, probsh, ODE.Vern9(); parallel = false, update_section_every_step = 1, jacobian = BK.AutoDiffDenseAnalytical());
    # verbosity = 3, plot=true,
    callback_newton = BK.cbMaxNorm(10),
    record_from_solution = recordPO,
    plot_solution = plotPO,
    normC = norminf)

# plot(br_po, br)

# test the saved solutions
for ind in eachindex(br_po.sol)
    pt = br_po.sol[ind]
    pars = BK.setparam(br_po, pt.p)
    wrap = BK.getprob(br_po)
    BK.restore_problem!(wrap, pt.x, pars)
    # @error "" BK.residual(wrap, BK.saved_solution(pt.x), pars)
    @test norminf(BK.residual(wrap, BK.saved_solution(pt.x), pars)) < br_po.contparams.newton_options.tol
end

for ind in eachindex(br_po.specialpoint)
    pt = br_po.specialpoint[ind]
    pars = BK.setparam(br_po, pt.param)
    wrap = BK.getprob(br_po)
    BK.restore_problem!(wrap, pt.x, pars)
    # @error "" BK.residual(wrap, BK.saved_solution(pt.x), pars)
    @test norminf(BK.residual(wrap, BK.saved_solution(pt.x), pars)) < br_po.contparams.newton_options.tol
end

show(br_po)
# test showing normal form
for _ind in (1,)
    if length(br_po.specialpoint) >=1 && br_po.specialpoint[_ind].type ∈ (:bp, :pd, :ns)
        println("")
        pt = get_normal_form(br_po, _ind; verbose = true)
        predictor(pt, 0.1, 1.)
        show(pt)
    end
end

# aBS from PD
# br_po_pd = BK.continuation(br_po, 1, setproperties(br_po.contparams, detect_bifurcation = 0, max_steps = 3, ds = -0.01, plot_every_step = 1);
#     # verbosity = 3, plot = true,
#     ampfactor = .1, δp = -0.005,
#     normC = norminf,
#     callback_newton = BK.cbMaxNorm(10),
#     record_from_solution = recordPO,
#     plot_solution = plotPO,
#     )

# plot(br_po_pd, br_po)
end
