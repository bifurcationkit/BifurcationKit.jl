using BifurcationKit, LinearAlgebra, ForwardDiff
using Test
import OrdinaryDiffEq as ODE
const BK = BifurcationKit

# ==============================================================================
# Bratu Problem BVP Example (Trapeze)
# ==============================================================================

# 1. Define the vector field (first-order form)
# u'' + 10(a * exp(u₁) - 1 - b u₁²/2) = 0  =>  u₁' = u₂, u₂' = -10(a * exp(u₁) - 1 - b u₁²/2)
Fbratu(x, p, t = 0) = [x[2], -10*(p.a * (exp(x[1]) -0*x[1] - p.c - p.b * x[1]^2/2))]

# 2. Define boundary conditions: x₁(0) = 0, x₁(1) = 0
gbratu(u0, uT, p) = [u0[1], uT[1]]

let
# 3. Create BVP Model
# State dimension is 2 (u, u')
# Fixed interval [0, 1]
params = (a = 0.05, b = 0., c = 1.0)
odeprob = ODE.ODEProblem(Fbratu, zeros(2), (0,1), params)
model = BK.BVP.BVPModel(odeprob, gbratu; n=2)


disc = BK.BVP.Shooting(10, ODE.Vern9(), true)
d_bvp = BK.BVP.discretize(model, disc; abstol = 1e-12, reltol = 1e-10)

# 5. Set up parameters and initial guess
# At p₁ = 0, the solution is u(t) = 0, u'(t) = 0
t_vals = LinRange(0, 1, disc.M+1)[1:end-1]
x0 = mapreduce(t -> t*(1-t).*[1,1], vcat, t_vals)

# 6. Create BVPBifProblem
prob = BK.BVP.BVPBifProblem(d_bvp, x0, params, (@optic _.a);
    # record_from_solution,
    # plot_solution,
)


# 7. Setup Continuation Parameters
optn = NewtonPar(tol = 1e-10, verbose=false)
sol = BK.solve(prob, Newton(), optn)


optc = ContinuationPar(
    p_min = 0.01,
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
params_trivial = (a = 0.0, b = 0.0, c=1.)
r0 = BK.BVP.bvp_residual(d_bvp, zero(x0), params_trivial)
@test length(r0) == length(x0)
@test norm(r0, Inf) ≤ 1e-14

bps = filter(sp -> sp.type == :bp, br.specialpoint)
@test bps[1].param ≈ pi^2/10 atol = 1e-2
@test bps[2].param ≈ 2^2*pi^2/10 atol = 1e-2
@test bps[3].param ≈ 3^2*pi^2/10 atol = 1e-2
@test isnothing( BK.BVP.get_solution_bvp(br, 1) ) == false

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
diagram = bifurcationdiagram(prob, br, 2, BK.getcontparams(br); autodiff = false, plot = false)
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CODIMENSION 2
bp_codim = continuation(br, bp_index, (@optic _.b), ContinuationPar(optc, p_min = -1.);
            verbosity = 0,
            jacobian_ma = BK.MinAug(), # autodiff is too slow
            usehessian = false,        # not yet defined for BVPBifProblem
            )
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# deflated continuation
deflationOp = DeflationOperator(2, dot, 1.0, [zero(BK.getu0(prob))])
perturb_solution(sol, p, id) = sol .+ 0.1 .* rand(length(sol))
alg = DefCont(;deflation_operator = deflationOp, perturb_solution, max_branches = 10)

br = @time continuation(
    prob, alg,
    setproperties(optc; ds = 0.001, dsmin=1e-5, max_steps = 10,
        p_max = 10., p_min = 0.005, detect_bifurcation = 0,
        newton_options = setproperties(optn; tol = 1e-9, max_iterations = 100, verbose = false));
    normC = norminf,
    callback_newton = BK.cbMaxNorm(1e3),
    plot = false
    )

# only 2 branches
@test length(br) <= 2

end
