using Revise

using BifurcationKit, LinearAlgebra, Plots
using Test
import OrdinaryDiffEq as ODE
const BK = BifurcationKit

function record_from_solution(x, p; k...)
    u = BK.get_time_slices(x[1:end], 2, disc.m, disc.Ntst)
    return (max_u = norm(x, 2), s = sum(x))
end

function plot_solution(x, p; kwargs...)
    u = BK.get_time_slices(x[1:end], 2, disc.m, disc.Ntst)
    plot!(u[1, :]; ylabel="u(t)", title="Bratu Solution (p₁=)", kwargs...)
end

# ==============================================================================
# Bratu Problem BVP Example
# ==============================================================================

# 1. Define the vector field (first-order form)
# u'' + p₁ * exp(u) = 0  =>  u₁' = u₂, u₂' = -10(a * exp(u₁) - 1 - b u₁²/2)
function Fbratu(x, p, t = 0)
    return [x[2], -10*(p.a * (exp(x[1]) - p.c - p.b * x[1]^2/2))]
end

# 2. Define boundary conditions: x₁(0) = 0, x₁(1) = 0
function gbratu(u0, uT, p)
    return [u0[1], uT[1]]
end

# 3. Create BVP Model
# State dimension is 2 (u, u')
# Fixed interval [0, 1] => phase condition fixes T=1.0
params = (a = 0.5, b = 0., c = 0.1)
odeprob = ODE.ODEProblem(Fbratu, zeros(2), (0,1), params)
model = BifurcationKit.BVP.BVPModel(odeprob, gbratu; n=2)

# 4. Discretize using Collocation method
# Using 201 points for better accuracy
disc = BifurcationKit.BVP.Shooting(2, ODE.Vern9(), false)
bvp = BifurcationKit.BVP.discretize(model, disc; abstol = 1e-12, reltol = 1e-10)

# 5. Set up parameters and initial guess
# At p₁ = 0, the solution is u(t) = 0, u'(t) = 0
t_vals = LinRange(0, 1, disc.M+1)[1:end-1]
x0 = mapreduce(t -> 0.1.*[t,t], vcat, t_vals)

# 6. Create BVPBifProblem
# We record max(u) to plot the bifurcation diagram
prob = BifurcationKit.BVP.BVPBifProblem(bvp, x0, params, (@optic _.a);
    record_from_solution,
    plot_solution,
)

# 7. Setup Continuation Parameters
optn = NewtonPar(tol = 1e-10, verbose=false)
optc = ContinuationPar(
    p_min = 0.1,
    p_max = 10.05,
    dsmax = 0.1,
    ds = 0.01,
    detect_bifurcation = 0,
    # detect_fold = false,
    newton_options = optn,
    max_steps = 200,
    nev = 20,
    n_inversion = 6
)
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Perform initial continuation
println("\nComputing primary branch for Bratu BVP (Shooting)...")
br = continuation(prob, PALC(), optc;
    plot = true,
    verbosity = 0,
    normC = norminf,
)

plot(br)
plot(br, vars = (:param, :s))
@test br.specialpoint[1].param ≈ pi^2/10 atol = 1e-4
@test br.specialpoint[2].param ≈ 2^2*pi^2/10 atol = 1e-4
@test br.specialpoint[4].param ≈ 3^2*pi^2/10 atol = 1e-4
