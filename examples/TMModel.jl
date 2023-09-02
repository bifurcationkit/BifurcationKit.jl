using Revise, Test, Parameters
using Plots
# using GLMakie; Makie.inline!(true)
using BifurcationKit
const BK = BifurcationKit
####################################################################################################
function TMvf!(dz, z, p, t = 0)
    @unpack J, α, E0, τ, τD, τF, U0 = p
    E, x, u = z
    SS0 = J * u * x * E + E0
    SS1 = α * log(1 + exp(SS0 / α))
    dz[1] = (-E + SS1) / τ
    dz[2] = (1.0 - x) / τD - u * x * E
    dz[3] = (U0 - u) / τF +  U0 * (1.0 - u) * E
    dz
end

par_tm = (α = 1.5, τ = 0.013, J = 3.07, E0 = -2.0, τD = 0.200, U0 = 0.3, τF = 1.5, τS = 0.007) #2.87
z0 = [0.238616, 0.982747, 0.367876 ]
prob = BifurcationProblem(TMvf!, z0, par_tm, (@lens _.E0); record_from_solution = (x, p) -> (E = x[1], x = x[2], u = x[3]),)

opts_br = ContinuationPar(p_min = -10.0, p_max = -0.9, ds = 0.04, dsmax = 0.125, n_inversion = 8, detect_bifurcation = 3, max_bisection_steps = 25, nev = 3)
br = continuation(prob, PALC(tangent = Bordered()), opts_br; plot = true, normC = norminf)

BK.plot(br, plotfold=false)
####################################################################################################
hopfpt = get_normal_form(br, 4)

# newton parameters
optn_po = NewtonPar(verbose = false, tol = 1e-8,  max_iterations = 9)

# continuation parameters
opts_po_cont = ContinuationPar(dsmax = 0.1, ds= 0.0005, dsmin = 1e-4, p_max = 0., p_min=-2., max_steps = 120, newton_options = (@set optn_po.tol = 1e-8), nev = 3, tol_stability = 1e-6, detect_bifurcation = 2, plot_every_step = 20, save_sol_every_step=1)

# arguments for periodic orbits
function plotSolution(x, p; k...)
    xtt = BK.get_periodic_orbit(p.prob, x, p.p)
    @show size(xtt[:,:]) maximum(xtt[1,:])
    plot!(xtt.t, xtt[1,:]; label = "E", k...)
    plot!(xtt.t, xtt[2,:]; label = "x", k...)
    plot!(xtt.t, xtt[3,:]; label = "u", k...)
    plot!(br; subplot = 1, putspecialptlegend = false)
end

args_po = (	record_from_solution = (x, p) -> begin
		xtt = BK.get_periodic_orbit(p.prob, x, p.p)
		return (max = maximum(xtt[1,:]),
				min = minimum(xtt[1,:]),
				period = getPeriod(p.prob, x, p.p))
	end,
	plot_solution = plotSolution,
	normC = norminf)

br_potrap = continuation(br, 4, opts_po_cont,
    PeriodicOrbitTrapProblem(M = 250, jacobian = :Dense, update_section_every_step = 0);
    verbosity = 2, plot = false,
    args_po...,
    callbackN = BK.cbMaxNorm(10.),
    )

plot(br, br_potrap, markersize = 3)
plot!(br_potrap.param, br_potrap.min, label = "")
####################################################################################################
# based on collocation
hopfpt = get_normal_form(br, 4)

# newton parameters
optn_po = NewtonPar(verbose = false, tol = 1e-8,  max_iterations = 10)

# continuation parameters
opts_po_cont = ContinuationPar(dsmax = 0.1, ds= 0.0001, dsmin = 1e-4, p_max = 0., p_min=-5., max_steps = 110, newton_options = (@set optn_po.tol = 1e-7), nev = 3, tol_stability = 1e-5, detect_bifurcation = 2, plot_every_step = 40, save_sol_every_step=1)

br_pocoll = @time continuation(
    br, 4, opts_po_cont,
    PeriodicOrbitOCollProblem(30, 5, update_section_every_step = 0, meshadapt = true);
    verbosity = 2,
    plot = true,
    args_po...,
    plotSolution = (x, p; k...) -> begin
        xtt = BK.get_periodic_orbit(p.prob, x, p.p)
        plot!(xtt.t, xtt[1,:]; label = "", marker =:d, markersize = 1.5, k...)
        plot!(br; subplot = 1, putspecialptlegend = false)

    end,
    callbackN = BK.cbMaxNorm(1000.),
    )

plot(br, br_pocoll, markersize = 3)
    # plot!(br_pocoll.param, br_pocoll.min, label = "")
    # plot!(br, br_potrap, markersize = 3)
    # plot!(br_potrap.param, br_potrap.min, label = "", marker = :d)
####################################################################################################
# idem with Standard shooting
using DifferentialEquations#, TaylorIntegration

# this is the ODEProblem used with `DiffEqBase.solve`
probsh = ODEProblem(TMvf!, copy(z0), (0., 1000.), par_tm; abstol = 1e-10, reltol = 1e-9)

opts_po_cont = ContinuationPar(dsmax = 0.09, ds= -0.0001, dsmin = 1e-4, p_max = 0., p_min=-5., max_steps = 120, newton_options = NewtonPar(optn_po; tol = 1e-6, max_iterations = 7), nev = 3, tol_stability = 1e-8, detect_bifurcation = 2, plot_every_step = 10, save_sol_every_step=1)

br_posh = @time continuation(
    br, 4,
    # arguments for continuation
    opts_po_cont,
    # this is where we tell that we want Standard Shooting
    ShootingProblem(15, probsh, Rodas4P(), parallel = true, update_section_every_step = 1, jacobian = BK.AutoDiffDense(),);
    # ShootingProblem(15, probsh, TaylorMethod(15), parallel = false);
    ampfactor = 1.0, δp = 0.0005,
    usedeflation = true,
    linearAlgo = MatrixBLS(),
    verbosity = 2,    plot = true,
    args_po...,
    )

plot(br_posh, br, markersize=3)
    # plot(br, br_potrap, br_posh, markersize=3)
####################################################################################################
# idem with Poincaré shooting

opts_po_cont = ContinuationPar(dsmax = 0.02, ds= 0.001, dsmin = 1e-6, p_max = 0., p_min=-5., max_steps = 50, newton_options = NewtonPar(optn_po;tol = 1e-9, max_iterations=15), nev = 3, tol_stability = 1e-8, detect_bifurcation = 2, plot_every_step = 5, save_sol_every_step = 1)

br_popsh = @time continuation(
    br, 4,
    # arguments for continuation
    opts_po_cont,
    # this is where we tell that we want Poincaré Shooting
    PoincareShootingProblem(5, probsh, Rodas4P(); parallel = true, update_section_every_step = 1, jacobian = BK.AutoDiffDenseAnalytical(), abstol = 1e-10, reltol = 1e-9);
    alg = PALC(tangent = Bordered()),
    ampfactor = 1.0, δp = 0.005,
    # usedeflation = true,
    linearAlgo = MatrixBLS(),
    verbosity = 2, plot = true,
    args_po...,
    callbackN = BK.cbMaxNorm(1e1),
    record_from_solution = (x, p) -> (return (max = getmaximum(p.prob, x, @set par_tm.E0 = p.p), period = getperiod(p.prob, x, @set par_tm.E0 = p.p))),
    normC = norminf)

plot(br, br_popsh, markersize=3)
