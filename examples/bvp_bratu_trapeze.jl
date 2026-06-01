using Revise
using BifurcationKit, LinearAlgebra, Plots
using Test
const BK = BifurcationKit

function record_from_solution(x, p; k...)
    return (max_u = norm(x, 2), s = sum(x))
end

function plot_solution(x, p; iter, kwargs...)
    prob = BK.getprob(iter)
    sol = BK.BVP.get_solution_bvp(prob, x, p)
    plot!(sol.t, sol.u[1, :]; label="",  kwargs...)
end

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
# Fixed interval [0, 1]
model = BK.BVP.BVPModel(Fbratu, gbratu; n=2)

# 4. Discretize using Trapeze method
# Using 201 points for better accuracy
disc = BK.BVP.Trapeze(M=201)
bvp = BK.BVP.discretize(model, disc)

# 5. Set up parameters and initial guess
# At p₁ = 0, the solution is u(t) = 0, u'(t) = 0
params = (a = 0.5, b = 0.)
x0 = rand(2 * disc.M) .* 0.1

# 6. Create BVPBifProblem
prob = BK.BVP.BVPBifProblem(bvp, x0, params, (@optic _.a);
    record_from_solution,
    plot_solution
)

# 7. Setup Continuation Parameters
optn = NewtonPar(tol = 1e-10, verbose=false)
optc = ContinuationPar(
    p_min = 0.1,
    p_max = 10.05,
    dsmax = 0.1,
    ds = 0.01,
    detect_bifurcation = 3,
    newton_options = optn,
    max_steps = 200,
    nev = 20,
    n_inversion = 6
)
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Perform initial continuation
println("\nComputing primary branch for Bratu BVP (Trapeze)...")
br = continuation(prob, PALC(), optc;
    plot = true,
    verbosity = 0,
    normC = norminf,
)

plot(br)
plot(br, vars = (:param, :s))
bps = filter(sp -> sp.type == :bp, br.specialpoint)
@test bps[1].param ≈ pi^2/10     atol = 1e-2
@test bps[2].param ≈ 2^2*pi^2/10 atol = 1e-2
@test bps[3].param ≈ 3^2*pi^2/10 atol = 1e-2
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NORMAL FORM COMPUTATION
get_normal_form(br, 1; autodiff=false)
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BRANCH SWITCHING
br2 = continuation(br, 1, ContinuationPar(optc, max_steps=30); autodiff = false, bothside = true)
plot(br, br2, vars = (:param, :s))
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AUTOMATIC BIFURCATION DIAGRAM
diagram = bifurcationdiagram(prob, br, 2, BK.getcontparams(br); autodiff = false, plot = true)
plot(diagram, vars = (:param, :s), legend = false)
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CODIMENSION 2
bp_codim = continuation(br, bp_index, (@optic _.b), ContinuationPar(optc, p_min = -1.);
            verbosity = 0,
            jacobian_ma = BK.MinAug(), # autodiff is too slow
            usehessian = false,        # not yet defined for BVPBifProblem
            )
plot(bp_codim)
