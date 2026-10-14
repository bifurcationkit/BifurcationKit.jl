using BifurcationKit, Test
import OrdinaryDiffEq as ODE
const BK = BifurcationKit

# ==============================================================================
# Regression test for the uninitialized residual tail of the BVP Shooting
# discretization.
#
# bvp_residual used to allocate `out = similar(X)` of length n*M+1 (trailing
# "period" slot advertised by generate_solution/total_dim) but wrote only
# out[1:n*M], so `residual(prob, x, p)` — read at Newton.jl:74 — returned
# uninitialized memory (NaN / subnormals, differing between identical calls)
# and the AutoDiffDense Jacobian carried a garbage last row. Newton was
# nondeterministic.
#
# Fix (fixed-interval design): the time span is fixed by the BVPModel, there is
# no trailing slot; residuals fully write an output of length solution_dim and
# reject wrong-size input. Unknowns like tf must be constant state components.
# ==============================================================================

let
    # harmonic oscillator on [0, 2π], periodic boundary conditions
    F(u, p, t=0) = [u[2], -p.μ * u[1]]
    g(u0, uT, p) = u0 .- uT
    n, M = 2, 4

    odeprob = ODE.ODEProblem(F, [1.0, 0.0], (0.0, 2π), (μ = 1.0,))
    model = BK.BVP.BVPModel(odeprob, g; n)
    bvp = BK.BVP.discretize(model, BK.BVP.Shooting(M, ODE.Tsit5(), false))

    x0 = BK.BVP.generate_solution(bvp, t -> [cos(t), sin(t)])
    @test length(x0) == n * M                       # no trailing period slot

    prob = BK.BVP.BVPBifProblem(bvp, x0, (μ = 1.0,), (@optic _.μ))

    # residual must be fully written: identical inputs, identical outputs
    r1 = BK.residual(prob, prob.u0, prob.params)
    r2 = BK.residual(prob, prob.u0, prob.params)
    @test length(r1) == n * M
    @test r1 == r2

    # wrong-size input must fail loudly instead of reading uninitialized memory
    @test_throws ArgumentError BK.BVP.bvp_residual(bvp, vcat(x0, 0.0), (μ = 1.0,))
    @test_throws ArgumentError BK.BVP.bvp_residual(bvp, x0[1:end-1], (μ = 1.0,))

    # Newton on the Shooting path is deterministic and converges
    sol = BK.solve(prob, BK.Newton(), NewtonPar(tol = 1e-10, verbose = false))
    @test BK.converged(sol)
end
