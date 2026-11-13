using Test
using BifurcationKit
const BK = BifurcationKit

# a step under the state's floating-point resolution moves nothing: a secant continuation stops at
# the last point it moved to, with a finite tangent and predictor
@testset "PALC secant: a step that moves nothing stops the continuation" begin
    F(x, p) = x .- p.λ
    prob = BifurcationProblem(F, [0f0], (λ = 0f0,), (@optic _.λ))
    # every corrector iterate past λ = 1/2 is refused, so the steps halve toward λ = 1/2 until one moves nothing
    wall(iterate; fromNewton = true, kwargs...) = fromNewton || iterate.p <= 0.5f0
    options = ContinuationPar(ds = 0.1f0, dsmin = floatmin(Float32), dsmax = 0.1f0, p_min = 0f0, p_max = 1f0, max_steps = 10_000, detect_fold = false, detect_bifurcation = 0, newton_options = NewtonPar(tol = 1f-5))
    iter = ContIterable(prob, PALC(), options; callback_newton = wall, verbosity = 0)
    next = iterate(iter)
    state = last(next)
    while ~isnothing(next)
        state = last(next)
        next = iterate(iter, state)
    end
    @test state.stopcontinuation
    @test BK.getp(state) <= 0.5f0
    @test all(isfinite, state.τ.u) && isfinite(state.τ.p)
    @test all(isfinite, state.z_pred.u) && isfinite(state.z_pred.p)
end

# the stall check sees through a nested `BorderedArray` state
@testset "stalled_step compares nested states by coordinates" begin
    state(u, p, q) = BorderedArray(BorderedArray(u, p), q)
    @test BK.stalled_step(state([1f0, 2f0], 3f0, 4f0), state([1f0, 2f0], 3f0, 4f0))
    @test ~BK.stalled_step(state([1f0, 2f0], 3f0, 4f0), state([1f0, 2f0], 3f0, 5f0))
    @test ~BK.stalled_step(state([1f0, 2f0], 3f0, 4f0), state([1f0, 2f0], 0f0, 4f0))
    @test ~BK.stalled_step(state([1f0, 2f0], 3f0, 4f0), state([1f0, 9f0], 3f0, 4f0))
end
