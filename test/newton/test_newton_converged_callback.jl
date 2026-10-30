# using Revise
using Test
using BifurcationKit, LinearAlgebra
const BK = BifurcationKit

# Newton callback whose tolerance grows with the size of the iterate
struct ScaledTolerance{T}
    tol::T
end
function (check::ScaledTolerance)(state; k...)
    if state.residual < check.tol * maximum(abs, state.x)
        return NewtonConverged()
    end
    return true
end

let
    F(x, p) = @. x^3 / 3 + x - p.r
    J(x, p) = diagm(0 => 1 .+ x .^ 2)
    par = (r = 0.5,)
    prob = BifurcationProblem(F, fill(2.0, 3), par, (@optic _.r); J)
    opts = NewtonPar(tol = 1e-12)

    strict = BK.solve(prob, Newton(), opts)
    @test BK.converged(strict)
    @test last(strict.residuals) < 1e-12

    loosened = BK.solve(prob, Newton(), opts; callback = ScaledTolerance(1e-3))
    @test BK.converged(loosened)
    @test loosened.itnewton < strict.itnewton
    @test 1e-12 < last(loosened.residuals) < 1e-3 * maximum(abs, loosened.u)

    # `false` at the last call refuses a point under `tol`; `true` leaves the verdict to `tol`
    @test ~BK.converged(BK.solve(prob, Newton(), opts; callback = (state; k...) -> state.step == 0))
    @test BK._converged(true, true) && ~BK._converged(false, true) && BK._converged(false, NewtonConverged())
    @test BK._verdict_and(NewtonConverged(), true) === NewtonConverged()
    @test BK._verdict_and(NewtonConverged(), false) === false

    # a zero `tol` is unreachable: the correctors converge at the callback's verdict only
    unreachable = ContinuationPar(p_min = 0.0, p_max = 1.0, ds = 0.05, max_steps = 200, detect_bifurcation = 0, newton_options = NewtonPar(tol = 0.0, max_iterations = 10))
    for alg in (PALC(), MoorePenrose(), Natural())
        floored = continuation(prob, alg, unreachable; callback_newton = ScaledTolerance(1e-10))
        @test maximum(floored.param) >= 1.0
        # Natural records its predictor's parameter at the bound, so only points inside the box are checked
        @test all(point -> norm(F(point.x, (r = point.p,)), Inf) < 1e-10 * maximum(abs, point.x), filter(point -> point.p <= 1.0, floored.sol))
    end
end

# low precision and `Multiple` pass the verdict through
let
    F(x, p) = @. x^3 / 3 + x - p.r
    J(x, p) = diagm(0 => 1 .+ x .^ 2)
    prob = BifurcationProblem(F, fill(2f0, 3), (r = 0.5f0,), (@optic _.r); J)
    opts = NewtonPar(tol = 0f0, max_iterations = 10)
    sol = BK.solve(prob, Newton(), opts; callback = ScaledTolerance(1f-3))
    @test BK.converged(sol)
    @test eltype(sol.u) == Float32

    unreachable = ContinuationPar(p_min = 0f0, p_max = 1f0, ds = 0.05f0, max_steps = 200, detect_bifurcation = 0, newton_options = opts)
    floored = continuation(prob, Multiple(PALC(θ = 0.5f0), copy(prob.u0), 0.01f0, 3; dsfact = 1.5f0), unreachable; callback_newton = ScaledTolerance(1f-3))
    @test maximum(floored.param) >= 1
end
