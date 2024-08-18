using Revise
using Test
using Plots
# using GLMakie; Makie.inline!(true)
using BifurcationKit, Test
const BK = BifurcationKit
####################################################################################################
function COm!(du, u, p, t = 0)
    (;q1,q2,q3,q4,q5,q6,k) = p
    x, y, s = u
    z = 1-x-y-s
    du[1] = 2q1 * z^2 - 2q5 * x^2 - q3 * x * y
    du[2] = q2 * z - q6 * y - q3 * x * y
    du[3] = q4 * (z - k * s)
    du
end

par_com = (q1 = 2.5, q2 = 2.0, q3 = 10., q4 = 0.0675, q5 = 1., q6 = 0.1, k = 0.4)

z0 = [0.001137, 0.891483, 0.062345]

prob = BifurcationProblem(COm!, z0, par_com, (@optic _.q2); record_from_solution = (x, p) -> (x = x[1], y = x[2], s = x[3]))

opts_br = ContinuationPar(dsmax = 0.015, dsmin=1e-4, ds=1e-4, p_min = 0.5, p_max = 2.0, n_inversion = 6, detect_bifurcation = 3, nev = 3)
br = @time continuation(prob, PALC(), opts_br;
    plot = true, verbosity = 0,
    normC = norminf,
    bothside = true)

BK.plot(br)#markersize=4, legend=:topright, ylims=(0,0.16))
####################################################################################################
# periodic orbits
function plotSolution(x, p; k...)
    xtt = BK.get_periodic_orbit(p.prob, x, p.p)
    plot!(xtt.t, xtt[1,:]; label = "", k...)
    plot!(xtt.t, xtt[2,:]; label = "", k...)
    plot!(xtt.t, xtt[3,:]; label = "", k...)
    plot!(xtt.t, xtt[1,:]; label = "", marker =:d, markersize = 1.5, k...)
    plot!(br; subplot = 1, putspecialptlegend = false, xlims = (1.02, 1.07))
end

function plotSolution(ax, x, p; ax1 = nothing, k...)
    @info "plotsol Makie"
    xtt = BK.get_periodic_orbit(p.prob, x, p.p)
    lines!(ax1, br)
    lines!(ax, xtt.t, xtt[1,:]; k...)
    lines!(ax, xtt.t, xtt[2,:]; k...)
    lines!(ax, xtt.t, xtt[3,:]; k...)
    scatter!(ax, xtt.t, xtt[1,:]; markersize = 1.5, k...)
end

args_po = (    record_from_solution = (x, p) -> begin
        xtt = BK.get_periodic_orbit(p.prob, x, @set par_com.q2 = p.p)
        return (max = maximum(xtt[1,:]),
                min = minimum(xtt[1,:]),
                period = getperiod(p.prob, x, @set par_com.q2 = p.p))
    end,
    plot_solution = plotSolution,
    normC = norminf)

opts_po_cont = ContinuationPar(opts_br, dsmax = 2., ds= 2e-2, dsmin = 1e-6, p_max = 5., p_min=-5.,
max_steps = 300, detect_bifurcation = 0, plot_every_step = 10)

# @set! opts_po_cont.newton_options.verbose = false
# @set! opts_po_cont.newton_options.tol = 1e-11
# @set! opts_po_cont.newton_options.max_iterations = 10
# @set! opts_po_cont.newton_options.linesearch = true

# using DifferentialEquations
# prob_ode = ODEProblem(COm!, copy(z0), (0., 1000.), par_com; abstol = 1e-11, reltol = 1e-9)

function callbackCO(state; fromNewton = false, kwargs...)
    # check that the solution is not too far
    δ0 = 1e-3
    z0 = get(state, :z0, nothing)
    p  = get(state, :p, nothing)
    if state.residual > 1
        @error "Reject Newton step, res too big!!"
        return false
    end
    # abort of the δp is too large
    if ~fromNewton && ~isnothing(z0)
        # @show abs(p - z0.p)
        return abs(p - z0.p) <= 2e-3
    end
    return true
end

brpo = @time continuation(br, 5, opts_po_cont,
    PeriodicOrbitOCollProblem(60, 5 ; meshadapt = false, K = 1000, verbose_mesh_adapt = true);
    # ShootingProblem(25, prob_ode, TaylorMethod(25); parallel = true; update_section_every_step = 1, jacobian = BK.AutoDiffDense());
    verbosity = 3, plot = true,
    normC = norminf,
    # alg = PALC(tangent = Bordered()),
    # alg = PALC(),
    # alg = MoorePenrose(tangent=PALC(tangent = Bordered()), method = BK.direct),
    δp = 0.0005,
    callback_newton = callbackCO,
    args_po...
    )

BK.plot(br, brpo, branchlabel = ["eq","max"])
xlims!((1.037, 1.055))
scatter!(br)
plot!(brpo.param, brpo.min, label = "min", xlims = (1.037, 1.055))
####################################################################################################
sn_codim2 = continuation(br, 3, (@optic _.k), ContinuationPar(opts_br, p_max = 3.2, p_min = 0., detect_bifurcation = 0, dsmin=1e-5, ds = -0.001, dsmax = 0.05, n_inversion = 6, detect_event = 2, detect_fold = false) ; plot = true,
    verbosity = 3,
    normC = norminf,
    update_minaug_every_step = 1,
    start_with_eigen = true,
    # record_from_solution = (u,p; kw...) -> (x = u.u[1] ),
    bothside = true,
    bdlinsolver = MatrixBLS()
    )

BK.plot(sn_codim2)#, real.(sn_codim2.BT), ylims = (-1,1), xlims=(0,2))
BK.plot(sn_codim2, vars=(:q2, :x), branchlabel = "Fold", plotstability = false);plot!(br,xlims=(0.8,1.8))

hp_codim2 = continuation((@set br.alg.tangent = Bordered()), 2, (@optic _.k), ContinuationPar(opts_br, p_min = 0., p_max = 2.8, detect_bifurcation = 0, ds = -0.0001, dsmax = 0.08, dsmin = 1e-4, n_inversion = 6, detect_event = 2, detect_loop = true, max_steps = 50, detect_fold=false) ; plot = true,
    verbosity = 0,
    normC = norminf,
    detect_codim2_bifurcation = 2,
    update_minaug_every_step = 1,
    start_with_eigen = true,
    # record_from_solution = (u,p; kw...) -> (x = u.u[1] ),
    bothside = true,
    bdlinsolver = MatrixBLS())

BK.plot(sn_codim2, vars=(:q2, :x), branchlabel = "Fold", plotcirclesbif = true)
plot!(hp_codim2, vars=(:q2, :x), branchlabel = "Hopf",plotcirclesbif = true)
plot!(br,xlims=(0.6,1.5))