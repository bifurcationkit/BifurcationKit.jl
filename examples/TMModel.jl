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
    dz[3] = (U0 - u) / τF +  U0 * (1 - u) * E
    dz
end

par_tm = (α = 1.5, τ = 0.013, J = 3.07, E0 = -2.0, τD = 0.200, U0 = 0.3, τF = 1.5, τS = 0.007)
z0 = [0.238616, 0.982747, 0.367876 ]
prob = BifurcationProblem(TMvf!, z0, par_tm, (@lens _.E0); record_from_solution = (x, p) -> (E = x[1], x = x[2], u = x[3]),)

opts_br = ContinuationPar(p_min = -10.0, p_max = -0.9, dsmax = 0.1, n_inversion = 8, nev = 3)
br = continuation(prob, PALC(tangent = Bordered()), opts_br; plot = true, normC = norminf)

BK.plot(br, plotfold=false)
####################################################################################################
# continuation parameters
opts_po_cont = ContinuationPar(opts_br, dsmin = 1e-4, ds = 1e-4, max_steps = 80, tol_stability = 1e-6, detect_bifurcation = 2, plot_every_step = 20)

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
                period = getperiod(p.prob, x, p.p))
    end,
    plot_solution = plotSolution,
    normC = norminf
    )

br_potrap = continuation(br, 4, opts_po_cont,
    PeriodicOrbitTrapProblem(M = 150);
    verbosity = 2, plot = true,
    args_po...,
    callback_newton = BK.cbMaxNorm(1.),
    )

plot(br, br_potrap, markersize = 3)
plot!(br_potrap.param, br_potrap.min, label = "")
####################################################################################################
# continuation parameters
opts_po_cont = ContinuationPar(opts_br, ds= 0.0001, dsmin = 1e-4, max_steps = 10, tol_stability = 1e-5, detect_bifurcation = 2, plot_every_step = 10)

br_pocoll = @time continuation(
    br, 4, opts_po_cont,
    PeriodicOrbitOCollProblem(30, 5; meshadapt = true);
    verbosity = 2,
    plot = true,
    args_po...,
    plot_solution = (x, p; k...) -> begin
        xtt = BK.get_periodic_orbit(p.prob, x, p.p)
        plot!(xtt.t, xtt[1,:]; label = "", marker =:d, markersize = 1.5, k...)
        plot!(br; subplot = 1, putspecialptlegend = false)
    end,
    )

plot(br, br_pocoll, markersize = 3)
####################################################################################################
# idem with Standard shooting
using DifferentialEquations

# this is the ODEProblem used with `DiffEqBase.solve`
prob_ode = ODEProblem(TMvf!, copy(z0), (0., 1000.), par_tm; abstol = 1e-11, reltol = 1e-9)

opts_po_cont = ContinuationPar(opts_br, ds= -0.0001, dsmin = 1e-4, max_steps = 120, newton_options = NewtonPar(tol = 1e-6, max_iterations = 7), tol_stability = 1e-8, detect_bifurcation = 2, plot_every_step = 10, save_sol_every_step=1)

br_posh = @time continuation(
    br, 4,
    # arguments for continuation
    opts_po_cont,
    # this is where we tell that we want Standard Shooting
    ShootingProblem(15, prob_ode, Rodas4P(), parallel = true,);
    linear_algo = MatrixBLS(),
    verbosity = 2,
    plot = true,
    args_po...,
    )

plot(br_posh, br, markersize=3)
####################################################################################################
# idem with Poincaré shooting

opts_po_cont = ContinuationPar(opts_br, dsmax = 0.02, ds= 0.0001, max_steps = 50, newton_options = NewtonPar(tol = 1e-9, max_iterations=15), tol_stability = 1e-6, detect_bifurcation = 2, plot_every_step = 5)

br_popsh = @time continuation(
    br, 4,
    # arguments for continuation
    opts_po_cont,
    # this is where we tell that we want Poincaré Shooting
    PoincareShootingProblem(3, prob_ode, Rodas4P(); parallel = true);
    # usedeflation = true,
    linear_algo = MatrixBLS(),
    verbosity = 2, plot = true,
    args_po...,
    callback_newton = BK.cbMaxNorm(1e0),
    normC = norminf)

plot(br, br_popsh, markersize=3)
