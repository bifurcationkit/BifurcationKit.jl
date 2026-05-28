using BifurcationKit, LinearAlgebra, ForwardDiff
using Test

# ==============================================================================
# Bratu Problem BVP Example (Trapeze)
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

let
# 3. Create BVP Model
# State dimension is 2 (u, u')
# Fixed interval [0, 1]
model = BifurcationKit.BVP.BVPModel(Fbratu, gbratu; n=2)

# Nonuniform mesh smoke test (M points -> M-1 interval weights)
mesh_steps = [1.0 + 0.2 * sin(2pi * (i - 1) / 29) for i in 1:29]
disc_nonuniform = BifurcationKit.BVP.Trap(M=30, mesh=mesh_steps)
bvp_nonuniform = BifurcationKit.BVP.discretize(model, disc_nonuniform)
x_nonuniform = zeros(2 * disc_nonuniform.M)
p_nonuniform = (a = 0.5, b = 0.0)
r_nonuniform = BifurcationKit.BVP.bvp_residual(bvp_nonuniform, x_nonuniform, p_nonuniform)
@test all(isfinite, r_nonuniform)

J_nonuniform = ForwardDiff.jacobian(z -> BifurcationKit.BVP.bvp_residual(bvp_nonuniform, z, p_nonuniform), x_nonuniform)
@test all(isfinite, J_nonuniform)

# 4. Discretize using Trapeze method
disc = BifurcationKit.BVP.Trap(M=150)
bvp = BifurcationKit.BVP.discretize(model, disc)

# 5. Set up parameters and initial guess
# At p₁ = 0, the solution is u(t) = 0, u'(t) = 0
params = (a = 0.5, b = 0.)
x0 = zeros(2 * disc.M)

# 6. Create BVPBifProblem
prob = BifurcationKit.BVP.BVPBifProblem(bvp, x0, params, (@optic _.a))

# 7. Setup Continuation Parameters
optn = NewtonPar(tol = 1e-10, verbose=false)
optc = ContinuationPar(
    p_min = 0.1,
    p_max = 10.0,
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
br = continuation(prob, PALC(), optc;
    plot = false,
    verbosity = 0,
    normC = norminf,
)

# Residual sanity check on the trivial solution at a = 0
params_trivial = (a = 0.0, b = 0.0)
r0 = BifurcationKit.BVP.bvp_residual(bvp, zero(x0), params_trivial)
@test length(r0) == length(x0)
@test norm(r0, Inf) ≤ 1e-14

bps = filter(sp -> sp.type == :bp, br.specialpoint)
@test bps[1].param ≈ pi^2/10 atol = 1e-2
@test bps[2].param ≈ 2^2*pi^2/10 atol = 1e-2
@test bps[3].param ≈ 3^2*pi^2/10 atol = 1e-2
@test isnothing( BifurcationKit.BVP.get_solution_bvp(br, 1) ) == false

bp_index = findfirst(sp -> sp.type == :bp, br.specialpoint)
@test !isnothing(bp_index)
bp_index = bp_index::Int
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NORMAL FORM COMPUTATION
get_normal_form(br, bp_index; autodiff=false)
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BRANCH SWITCHING
br2 = continuation(br, bp_index, ContinuationPar(optc, max_steps=30); autodiff = false, bothside = true)
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AUTOMATIC BIFURCATION DIAGRAM
diagram = bifurcationdiagram(prob, br, 2, BifurcationKit.getcontparams(br); autodiff = false, plot = false)
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CODIMENSION 2
bp_codim = continuation(br, bp_index, (@optic _.b), ContinuationPar(optc, p_min = -1.);
            verbosity = 0,
            jacobian_ma = BifurcationKit.MinAug(), # autodiff is too slow
            usehessian = false,        # not yet defined for BVPBifProblem
            )
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# deflated continuation
deflationOp = DeflationOperator(2, dot, 1.0, [zero(BifurcationKit.getu0(prob))])
perturb_solution(sol, p, id) = sol .+ 0.1 .* rand(length(sol))
alg = DefCont(;deflation_operator = deflationOp, perturb_solution, max_branches = 10)

br = @time continuation(
    prob, alg,
    setproperties(optc; ds = 0.001, dsmin=1e-5, max_steps = 10,
        p_max = 10., p_min = 0.005, detect_bifurcation = 0,
        newton_options = setproperties(optn; tol = 1e-9, max_iterations = 100, verbose = false));
    normC = norminf,
    callback_newton = BifurcationKit.cbMaxNorm(1e3),
    plot = false
    )

# only 2 branches
@test length(br) == 2

end
