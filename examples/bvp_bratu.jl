using Revise
using BifurcationKit, LinearAlgebra, Plots
using Test

# ==============================================================================
# Bratu Problem BVP Example
# ==============================================================================

# 1. Define the vector field (first-order form)
# u'' + 10(a * exp(u₁) - 1 - b u₁²/2) = 0  =>  u₁' = u₂, u₂' = -10(a * exp(u₁) - 1 - b u₁²/2)
function Fbratu(x, p)
    return [x[2], -10*(p.a * (exp(x[1]) - 1 - p.b * x[1]^2/2))]
end

# 2. Define boundary conditions: x₁(0) = 0, x₁(1) = 0
function gbratu(u0, uT, p)
    return [u0[1], uT[1]]
end

# 3. Create BVP Model
# State dimension is 2 (u, u')
# Fixed interval [0, 1] => phase condition fixes T=1.0
model = BifurcationKit.BVP.BVPModel(Fbratu, gbratu; n=2)

# 4. Discretize using Collocation method
# Using 201 points for better accuracy
disc = BifurcationKit.BVP.Collocation(Ntst=40, m=5)
bvp = BifurcationKit.BVP.discretize(model, disc)

# 5. Set up parameters and initial guess
# At p₁ = 0, the solution is u(t) = 0, u'(t) = 0
params = (a = 0.5, b = 0.)
t_vals = LinRange(0, 1, 101)
x0 = zeros(2 * (1 + disc.m * disc.Ntst))
# x0[end] = 1.0 # Interval length T = 1.0

# 6. Create BVPBifProblem
# We record max(u) to plot the bifurcation diagram
prob = BifurcationKit.BVP.BVPBifProblem(bvp, x0, params, (@optic _.a);
    record_from_solution = (x, p; k...) -> begin
        u = BifurcationKit.get_time_slices(x[1:end], 2, disc.m, disc.Ntst)
        return (max_u = norm(x, 2), s = sum(x))
    end,
    plot_solution = (x, p; kwargs...) -> begin
        u = BifurcationKit.get_time_slices(x[1:end], 2, disc.m, disc.Ntst)
        plot!(u[1, :]; ylabel="u(t)", title="Bratu Solution (p₁=)", kwargs...)
    end
)

# 7. Setup Continuation Parameters
optn = NewtonPar(tol = 1e-10, verbose=false)
optc = ContinuationPar(
    p_min = 0.1,
    p_max = 10.05,
    dsmax = 0.1,
    ds = 0.01,
    detect_bifurcation = 3,
    # detect_fold = false,
    newton_options = optn,
    max_steps = 200,
    nev = 20,
    n_inversion = 6
)
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Perform initial continuation
println("\nComputing primary branch for Bratu BVP (Collocation)...")
br = continuation(prob, PALC(), optc;
    plot = true,
    verbosity = 0,
    normC = norminf,
)

BifurcationKit.BVP.get_solution_bvp(br, 1)

plot(br)
plot(br, vars = (:param, :s))
@test br.specialpoint[1].param ≈ pi^2/10     atol = 1e-4
@test br.specialpoint[2].param ≈ 2^2*pi^2/10 atol = 1e-4
@test br.specialpoint[4].param ≈ 3^2*pi^2/10 atol = 1e-4
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NORMAL FORM COMPUTATION
get_normal_form(br, 1; autodiff=false)
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BRANCH SWITCHING
br2 = continuation(br, 1, ContinuationPar(optc, max_steps=30); autodiff = false, bothside = true)
plot(br, br2, vars = (:param, :s))
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AUTOMATIC BIFURCATION DIAGRAM
diagram = bifurcationdiagram(prob, br, 2, BifurcationKit.getcontparams(br); autodiff = false, plot = true)
diagram = bifurcationdiagram(prob, br, 2, BifurcationKit.getcontparams(br); autodiff = false, plot = true)
plot(diagram, vars = (:param, :s), legend = false)
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CODIMENSION 2
bp_codim = continuation(br, 1, (@optic _.b), ContinuationPar(optc, p_min = -1.);
            verbosity = 0,
            jacobian_ma = BifurcationKit.MinAug(), # autodiff is too slow
            usehessian = false,        # not yet defined for BVPBifProblem
            )
plot(bp_codim)
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# deflated continuation
deflationOp = DeflationOperator(2, dot, 1.0, [zero(BifurcationKit.getu0(prob))])
perturb_solution(sol, p, id) = sol .+ 0.1 .* rand(length(sol))
alg = DefCont(;deflation_operator = deflationOp, perturb_solution, max_branches = 10)
# br = @time continuation(
#     prob, alg,
#     setproperties(optc; ds = 0.001, dsmin=1e-5, max_steps = 20000,
#         p_max = 10., p_min = 0.005, detect_bifurcation = 0, plot_every_step = 100,
#         newton_options = setproperties(optn; tol = 1e-9, max_iterations = 100, verbose = false));
#     normC = norminf,
#     verbosity = 1,
#     callback_newton = BifurcationKit.cbMaxNorm(1e3)
#     )


println("\nExample complete!")
