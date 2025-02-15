# using Revise, Plots, AbbreviatedStackTraces
using LinearAlgebra, Test
using BifurcationKit, Test
const BK = BifurcationKit

recordFromSolution(x, p; k...) = (u1 = x[1], u2 = x[2])
####################################################################################################
function lur!(dz, u, p, t)
    (; α, β) = p
    x, y, z = u
    dz[1] = y
    dz[2] = z
    dz[3] = -α * z - β * y - x + x^2
    dz
end

lur(z, p) = lur!(similar(z), z, p, 0)
par_lur = (α = -1.0, β = 1.)
z0 = zeros(3)
prob = BifurcationProblem(lur, z0, par_lur, (@optic _.α); record_from_solution = recordFromSolution)

opts_br = ContinuationPar(p_min = -1.4, p_max = 1.8, ds = -0.01, dsmax = 0.01, n_inversion = 8, detect_bifurcation = 3, max_bisection_steps = 25, nev = 3, plot_every_step = 20, max_steps = 1000)
opts_br = @set opts_br.newton_options.verbose = false
br = continuation(prob, PALC(tangent = Bordered()), opts_br;
bothside = true, normC = norminf)

# plot(br)
####################################################################################################
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
# newton parameters
optn_po = NewtonPar(tol = 1e-8,  max_iterations = 25)

# continuation parameters
opts_po_cont = ContinuationPar(dsmax = 0.03, ds= 0.0001, dsmin = 1e-4, p_max = 1.8, p_min=-5., max_steps = 122, newton_options = (@set optn_po.tol = 1e-8), nev = 3, tol_stability = 1e-4, detect_bifurcation = 3, plot_every_step = 20, save_sol_every_step=1, n_inversion = 6)

Mt = 90 # number of time sections
br_po = continuation(
        br, 2, opts_po_cont,
        PeriodicOrbitTrapProblem(M = Mt; update_section_every_step = 1,
        jacobian = :Dense);
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
        length(br_po.specialpoint) >=3 &&
        br_po.specialpoint[_ind].type ∈ (:bp, :pd, :ns)
            println("")
            pt = get_normal_form(br_po, _ind; verbose = true)
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
opts_po_cont = ContinuationPar(dsmax = 0.03, ds= -0.0001, dsmin = 1e-4, p_max = 1.8, p_min=-0.9, max_steps = 120, newton_options = (@set optn_po.tol = 1e-11), nev = 3, tol_stability = 1e-4, detect_bifurcation = 3, plot_every_step = 20, save_sol_every_step = 1, n_inversion = 8)

for meshadapt in (false, true)
    br_po = continuation(
        br, 2, opts_po_cont,
        PeriodicOrbitOCollProblem(40, 4; meshadapt, K = 200);
        alg = PALC(),
        ampfactor = 1., δp = 0.01,
        record_from_solution = recordPO,
        plot_solution = plotPO,
        normC = norminf)

    # test normal forms
    for _ind in (1,)
        if length(br_po.specialpoint) >=1 && br_po.specialpoint[_ind].type ∈ (:bp, :pd, :ns)
            println("")
            for prm in (true, false)
                pt = get_normal_form(br_po, _ind; verbose = true, prm)
                show(pt)
            end
        end
    end

    pd = get_normal_form(br_po, 1; verbose = false, prm = true)
    predictor(pd, 0.1, 1)
    pd = get_normal_form(br_po, 1; verbose = false, prm = false)
    predictor(pd, 0.1, 1)
    @test pd.nf.nf.b3 ≈ -0.30509421737255177 rtol=1e-3 # reference value computed with ApproxFun
    # @test pd.nf.nf.a  ≈ 0.020989802220981707 rtol=1e-3 # reference value computed with ApproxFun

    # aBS from PD
    continuation(br_po, 1, setproperties(br_po.contparams, detect_bifurcation = 3, max_steps = 5, ds = 0.01, dsmax = 0.01, plot_every_step = 10);
    ampfactor = .2, δp = -0.005,
    usedeflation = true,
    )

end
####################################################################################################
using OrdinaryDiffEq

probsh = ODEProblem(lur!, copy(z0), (0., 1000.), par_lur; abstol = 1e-12, reltol = 1e-10)

optn_po = NewtonPar(tol = 1e-12, max_iterations = 25)

# continuation parameters
opts_po_cont = ContinuationPar(dsmax = 0.02, ds= -0.001, dsmin = 1e-4, max_steps = 122, newton_options = (@set optn_po.tol = 1e-12), tol_stability = 1e-5, detect_bifurcation = 3, plot_every_step = 10, n_inversion = 6, nev = 3)

br_po = continuation(
    br, 2, opts_po_cont,
    ShootingProblem(15, probsh, Rodas5P(); parallel = false, update_section_every_step = 1);
    # ampfactor = 1., δp = 0.0051,
    # verbosity = 3,    plot = true,
    record_from_solution = recordPO,
    plot_solution = plotPO,
    # finalise_solution = (z, tau, step, contResult; prob = nothing, kwargs...) -> begin
    #         BK.haseigenvalues(contResult) && Base.display(contResult.eig[end].eigenvals)
    #         return z.u[end] < 30 && length(contResult.specialpoint) < 3
    #         true
    #     end,
    callback_newton = BK.cbMaxNorm(10),
    normC = norminf)

show(br_po)

# plot(br, br_po)
# plot(br_po, vars=(:param, :period))

@test br_po.specialpoint[1].param ≈ 0.63030057 rtol = 1e-4
@test br_po.specialpoint[2].param ≈ -0.63030476 rtol = 1e-4

# test showing normal form
for _ind in (1,3)
    if length(br_po.specialpoint) >=3 && br_po.specialpoint[_ind].type ∈ (:bp, :pd, :ns)
        println("")
        pt = get_normal_form(br_po, _ind; verbose = true, δ = 1e-5)
        show(pt)
        predictor(pt, 0.1, 1.)
        show(pt)
    end
end

# aBS from PD
br_po_pd = continuation(br_po, 1, setproperties(br_po.contparams, detect_bifurcation = 3, max_steps = 5, ds = 0.01, plot_every_step = 1, save_sol_every_step = 1);
    # verbosity = 0, plot = false,
    usedeflation = true,
    ampfactor = .1, δp = -0.005,
    record_from_solution = recordPO,
    normC = norminf,
    callback_newton = BK.cbMaxNorm(10),
    )

# plot(br_po, br_po_pd)
#######################################
@info "testLure Poincare"
opts_po_cont_ps = @set opts_po_cont.newton_options.tol = 1e-9
@set opts_po_cont_ps.dsmax = 0.0025
br_po = continuation(br, 2, opts_po_cont_ps,
    PoincareShootingProblem(2, probsh, Rodas4P(); parallel = false, reltol = 1e-6, update_section_every_step = 1, jacobian = BK.AutoDiffDenseAnalytical());
    ampfactor = 1., δp = 0.0051, 
    # verbosity = 3, plot=true,
    callback_newton = BK.cbMaxNorm(10),
    record_from_solution = recordPO,
    plot_solution = plotPO,
    normC = norminf)

# plot(br_po, br)

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
