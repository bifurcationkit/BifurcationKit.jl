using BifurcationKit, CairoMakie
const BK = BifurcationKit
let 
    F_simple(x, p; k = 2) = @. p[1] * x + x^(k+1)/(k+1) + 0.01
    opts = ContinuationPar(p_min = -3., detect_bifurcation = 3)
    prob = ODEBifProblem(F_simple, zeros(10), -1.5, (@optic _))
    br = continuation(prob, PALC(tangent=Bordered()), opts, plot = true)
    BK.plot(br)[1]
    BK.plot(br, br)
    BK.plot_eigenvals(br, true)
    BK.plot_eigenvals(br, false)
end

# test plotting deflated continuation
let 
    F_simple(x, p; k = 2) = @. p[1] * x + x^(k+1)/(k+1) + 0.01
    prob = BK.ODEBifProblem(F_simple, [0.], 0.5, (@optic _))
    opts = ContinuationPar(p_min = -3., detect_bifurcation = 3)
    alg = BK.DefCont(deflation_operator = DeflationOperator(2, .001, [[0.]]),
        perturb_solution = (x,p,id) -> (x .+ 0.1 .* rand(length(x)))
    )
    brdc = continuation(prob, alg,
        ContinuationPar(opts, ds = -0.001, max_steps = 800, newton_options = NewtonPar(verbose = false, max_iterations = 6), plot_every_step = 40, detect_bifurcation = 3);
        plot = true, verbosity = 0,
        callback_newton = BK.cbMaxNorm(1e3))
    BK.plot(brdc)
end

# test plotting bifurcation diagram
let
    Fbp(x, p) = [x[1] * (3.23 .* p.μ - p.x2 * x[1] + p.x3 * x[1]^2) + x[2], 
            -x[2] + p.γ * x[1]^2]
    opts_br = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds = 0.01, p_max = 0.4, p_min = -0.5, detect_bifurcation = 3, newton_options = NewtonPar(tol = 1e-14), max_steps = 100, n_inversion = 8)
    par_pf = setproperties((μ = -0.2, ν = 0, x2 = 1.12, x3 = 0.234, γ = 0.) ; x2 = 0.0, x3 = -1.0, γ = 1.422)
    prob_pf = ODEBifProblem(Fbp, [0., 0], par_pf, (@optic _.μ);record_from_solution = (x,p;k...)->(x[1],))
    bdiag = bifurcationdiagram(prob_pf, PALC(#=tangent=Bordered()=#), 2,
        setproperties(opts_br; p_min = -1.0, p_max = .5, ds = 0.01, dsmax = 0.05, n_inversion = 6, detect_bifurcation = 3, max_bisection_steps = 30, newton_options = NewtonPar(tol = 1e-12), max_steps = 15);
        verbosediagram = true, normC = norminf, plot = true)
    BK.plot(bdiag)
end

let
    BK.plot_periodic_potrap(rand(10*100+1)|>vec, 10, 100; ratio = 1)
    BK.plot_periodic_potrap(rand(2*10*100+1)|>vec, 10, 100; ratio = 2)
end