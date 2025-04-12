using Revise
using Plots
using BifurcationKit
const BK = BifurcationKit
####################################################################################################
function COm!(du, u, p, t = 0)
    q1,q2,q3,q4,q5,q6,k = p
    x, y, s = u
    z = 1-x-y-s
    du[1] = 2q1 * z^2 - 2q5 * x^2 - q3 * x * y
    du[2] = q2 * z - q6 * y - q3 * x * y
    du[3] = q4 * (z - k * s)
    du
end

par_com = (q1 = 2.5, q2 = 2.0, q3 = 10., q4 = 0.0675, q5 = 1., q6 = 0.1, k = 0.4)

z0 = [0.001137, 0.891483, 0.062345]

prob = BifurcationProblem(COm!, z0, par_com, (@optic _.q2); record_from_solution = (x, p; k...) -> (x = x[1], y = x[2], s = x[3]))

opts_br = ContinuationPar(dsmax = 0.015, dsmin=1e-4, ds=1e-4, p_min = 0.5, p_max = 2.0, n_inversion = 6, detect_bifurcation = 3, nev = 3)
br = @time continuation(prob, PALC(), opts_br;
    plot = true, verbosity = 0,
    normC = norminf,
    # bothside = true
    )

plot(br)
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

args_po = (    record_from_solution = (x, p; k...) -> begin
        xtt = BK.get_periodic_orbit(p.prob, x, @set par_com.q2 = p.p)
        return (max = maximum(xtt[1,:]),
                min = minimum(xtt[1,:]),
                period = getperiod(p.prob, x, @set par_com.q2 = p.p))
    end,
    plot_solution = plotSolution,
    normC = norminf)

opts_po_cont = ContinuationPar(opts_br, dsmax = 1., ds= 2e-2, dsmin = 1e-6, p_max = 5., p_min=-5.,
max_steps = 300, detect_bifurcation = 0, plot_every_step = 10)

brpo = @time continuation(br, 2, opts_po_cont,
    PeriodicOrbitOCollProblem(50, 3 ; jacobian = BK.DenseAnalyticalInplace(), meshadapt = true, K = 1000, verbose_mesh_adapt = true, update_section_every_step = 0);
    # verbosity = 0, plot = true,
    normC = norminf,
    alg = PALC(tangent = Bordered()),
    # alg = PALC(),
    # alg = MoorePenrose(tangent=PALC(tangent = Bordered()), method = BK.direct),
    δp = 0.00025,
    linear_algo = COPBLS(),
    callback_newton = BK.cbMaxNormAndΔp(1., 1.5e-3),
    bothside = true,
    args_po...
    )

plot(br, brpo, branchlabel = ["eq","max"])
xlims!((1.037, 1.055))
scatter!(br)
plot!(brpo.param, brpo.min, label = "min", xlims = (1.037, 1.055))
####################################################################################################
sn_codim2 = continuation(br, 3, (@optic _.k), ContinuationPar(opts_br, p_max = 3.2, p_min = 0., detect_bifurcation = 0, ds = -0.001, n_inversion = 6) ; plot = true,
    normC = norminf,
    update_minaug_every_step = 1,
    bothside = true,
    )

plot(sn_codim2)
plot(sn_codim2, vars=(:q2, :x), branchlabel = "Fold", plotstability = false);plot!(br,xlims=(0.8,1.8))

hp_codim2 = continuation((@set br.alg.tangent = Bordered()), 2, (@optic _.k), ContinuationPar(opts_br, p_min = 0., p_max = 2.8, detect_bifurcation = 0, dsmax = 0.02, dsmin = 1e-4, n_inversion = 6, max_steps = 50) ; plot = true,
    verbosity = 3,
    normC = norminf,
    detect_codim2_bifurcation = 2,
    update_minaug_every_step = 1,
    bothside = true,
    bdlinsolver = MatrixBLS())

plot(sn_codim2, vars=(:q2, :x), branchlabel = "Fold")
plot!(hp_codim2, vars=(:q2, :x), branchlabel = "Hopf")
plot!(br,xlims=(0.6,1.5))