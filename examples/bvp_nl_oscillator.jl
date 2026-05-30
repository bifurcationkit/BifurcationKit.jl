using Revise
using BifurcationKit, LinearAlgebra, Plots
using Test
const BK = BifurcationKit
import OrdinaryDiffEq as ODE

function record_from_solution(x, p; iter, k...)
    return (z0 = x[1], max_u = norm(x, 2), s = sum(x))
end

function plot_solution(x, p; iter, kwargs...)
    prob = BK.getprob(iter)
    sol = BK.BVP.get_solution_bvp(prob, x, p)
    plot!(sol.t, sol.u[1, :]; label="",  kwargs...)
end

# ==============================================================================

# 1. Define the vector field (first-order form)
function Fosc(x, p, _=0)
    (;λ) = p
    r = λ/(2pi)
    z, z1, t = x
    [z1,
    -(r/25 * z1 -z / 5 + 8/16 * z^3 ) / r^2 + 2/5*cos(2*pi*t)/r^2,
    1]
end

# 2. Define boundary conditions: x₁(0) = 0, x₁(1) = 0
gosc(u0, uT, p) = [u0[1]-uT[1], u0[2]-uT[2], u0[3]]

# 3. Create BVP Model
# State dimension is 2 (u, u')
params = (λ = 3.0, )
model = BK.BVP.BVPModel(Fosc, gosc; n=3)

# odeprob = ODE.ODEProblem(Fosc, zeros(3), (0,1), params)
# model = BK.BVP.BVPModel(odeprob, gosc; n=3)

# 4. Discretize using Collocation method
disc = BK.BVP.Collocation(Ntst=40, m=5)
# disc = BifurcationKit.BVP.Shooting(10, ODE.Vern9(), true)
bvp = BK.BVP.discretize(model, disc)

# 5. Set up parameters and initial guess
t_vals = LinRange(0, 1, 101)
x0 = BK.BVP.generate_solution(bvp, t-> 0.0t*(1-t)*[1,1,0] + [0,0,t])

# 6. Create BVPBifProblem
prob = BK.BVP.BVPBifProblem(bvp, x0, params, (@optic _.λ);
    record_from_solution,
    # plot_solution,
    # jacobian = BK.DenseAnalytical(),
)

BK.residual(prob, prob.u0, prob.params)
BK.jacobian(prob, prob.u0, prob.params)

# 7. Setup Continuation Parameters
optn = NewtonPar(tol = 1e-10, verbose=false)

BK.solve(prob, Newton(), @set optn.verbose = true)

optc = ContinuationPar(
    p_min = 1/20.,
    p_max = 10.05,
    dsmax = 1.1,
    ds = -0.01,
    detect_bifurcation = 0,
    # detect_fold = false,
    newton_options = optn,
    max_steps = 400,
    nev = 20,
    n_inversion = 6
)
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Perform initial continuation
br = @time continuation(prob, PALC(), optc;
    plot = true,
    verbosity = 2,
    normC = norminf,
)

plot(br, applytoX = inv, xlabel = "1/λ")
