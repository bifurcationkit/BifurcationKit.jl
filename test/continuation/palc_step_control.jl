using Test, BifurcationKit, LinearAlgebra
const BK = BifurcationKit
####################################################################################################
# the pure rules of CorrectorQuality
let
    tol = 1e-8
    control = CorrectorQuality(max_distance = 0.4)   # Gambit's values: 0.4, 0.6, 1.1, η = 0.1
    @test (control.max_distance, control.max_contraction, control.max_growth, control.η) == (0.4, 0.6, 1.1, 0.1)
    # the distance test is off unless a threshold is given, the contraction test is on
    @test CorrectorQuality().max_distance === nothing
    @test CorrectorQuality().max_contraction == 0.6

    quality = BK.StepQuality(control)
    # no Newton step: the largest growth
    @test BK.growth_factor(control, quality) ≈ 1.1

    # first step: rejected from max_distance on
    @test ~BK.rejects_step(control, quality, 0.39, tol)
    @test BK.rejects_step(control, quality, 0.4, tol)
    # a short first step still allows the largest growth, a long one slows it down by sqrt(d / max_distance) * max_growth
    @test BK.growth_factor(control, BK.observe(control, quality, 0.1, tol)) ≈ 1.1
    @test BK.growth_factor(control, BK.observe(control, quality, 0.3, tol)) ≈ inv(1.1 * sqrt(0.3 / 0.4))

    # later steps: rejected when d_k / (d_{k-1} + tol η) exceeds max_contraction
    after_first = BK.observe(control, quality, 0.3, tol)
    @test ~BK.rejects_step(control, after_first, 0.18, tol)
    @test BK.rejects_step(control, after_first, 0.19, tol)
    contraction = 0.1 / (0.3 + tol * 0.1)
    @test BK.contraction_rate(control, after_first, 0.1, tol) ≈ contraction
    after_second = BK.observe(control, after_first, 0.1, tol)
    # the deceleration is the largest one asked for by any step
    @test after_second.deceleration ≈ max(1.1 * sqrt(0.3 / 0.4), 1.1 * sqrt(contraction / 0.6))
    @test after_second.steps == 2

    # a test that is off neither rejects a step nor slows the next one
    off = CorrectorQuality(max_distance = nothing, max_contraction = nothing)
    @test ~BK.rejects_step(off, BK.StepQuality(off), 1e6, tol)
    @test ~BK.rejects_step(off, BK.observe(off, BK.StepQuality(off), 1e6, tol), 1e12, tol)
    @test BK.growth_factor(off, BK.observe(off, BK.observe(off, BK.StepQuality(off), 1e6, tol), 1e12, tol)) ≈ 1.1
    # the growth is capped by max_growth when no test asked for more deceleration
    @test BK.growth_capped(control, quality)
    @test BK.growth_capped(control, BK.observe(control, quality, 0.1, tol))
    @test ~BK.growth_capped(control, BK.observe(control, quality, 0.3, tol))
    # max_growth = nothing leaves ds to the Newton iteration count: no factor, tests still reject
    unfactored = CorrectorQuality(max_distance = 0.4, max_growth = nothing)
    @test BK.growth_factor(unfactored, BK.observe(unfactored, BK.StepQuality(unfactored), 0.3, tol)) === nothing
    @test BK.rejects_step(unfactored, BK.StepQuality(unfactored), 0.4, tol)
    # the distance test is off alone
    no_distance = CorrectorQuality()
    @test ~BK.rejects_step(no_distance, BK.StepQuality(no_distance), 1e6, tol)
    @test BK.rejects_step(no_distance, BK.observe(no_distance, BK.StepQuality(no_distance), 1.0, tol), 0.7, tol)

    # thresholds follow the float type of the continuation
    @test BK.CorrectorQuality{Float32}(control) isa CorrectorQuality{Float32}
    @test BK.CorrectorQuality{Float32}(no_distance).max_distance === nothing
    @test CorrectorQuality(max_distance = 0.4f0, max_contraction = 0.6f0, max_growth = 1.1f0, η = 0.1f0) isa CorrectorQuality{Float32}
    @test CorrectorQuality(max_contraction = 0.6f0, max_growth = 1.1f0, η = 0.1f0) isa CorrectorQuality{Float32}
    @test_throws AssertionError CorrectorQuality(max_growth = 1.0)
    @test_throws AssertionError CorrectorQuality(max_distance = 0.0)
    @test_throws AssertionError CorrectorQuality(max_contraction = -1.0)
end
####################################################################################################
# the cosine of the angle between the tangent and the chord of a step, in the norm of the arc length constraint
let
    dotθ = BK.DotTheta()
    τ = BorderedArray([1.0], 0.0)
    z = BorderedArray([0.0], 0.0)
    @test BK.chord_cosine(dotθ, τ, z, BorderedArray([2.0], 0.0), 0.5) ≈ 1
    @test BK.chord_cosine(dotθ, τ, z, BorderedArray([-2.0], 0.0), 0.5) ≈ -1
    @test BK.chord_cosine(dotθ, τ, z, BorderedArray([0.0], 3.0), 0.5) ≈ 0 atol = 1e-15
    @test BK.chord_cosine(dotθ, τ, z, BorderedArray([1.0], 1.0), 0.5) ≈ cos(π / 4)
    # an empty chord does not turn
    @test BK.chord_cosine(dotθ, τ, z, z, 0.5) == 1
end
####################################################################################################
# continuation with the step control: u^3 - u - (p - 1) = 0 has two folds, p ≈ 0.615 and p ≈ 1.385
let
    F(u, p) = @. u^3 - u - (p.λ - 1)
    make_problem(T) = BifurcationProblem(F, T[-1.3247179], (λ = zero(T),), (@optic _.λ))
    options(T; ds = 0.05, dsmax = 0.5) = ContinuationPar(ds = T(ds), dsmin = T(1e-6), dsmax = T(dsmax), a = T(0.5), p_min = zero(T), p_max = T(2), η = T(150), tol_stability = T(1e-10), dsmin_bisection = T(1e-16), tol_bisection_eigenvalue = T(1e-16), tol_param_bisection_event = T(1e-16), max_steps = 200, detect_fold = true, detect_bifurcation = 0, newton_options = NewtonPar(tol = sqrt(eps(T))))

    # both are off by default
    @test PALC().step_control === nothing
    @test ~PALC().orientation_check

    for tangent in (Secant(), Bordered())
        br = continuation(make_problem(Float64), PALC(; tangent, step_control = CorrectorQuality(), orientation_check = true), options(Float64))
        # it follows the branch through both folds
        @test count(sp -> sp.type == :fold, br.specialpoint) == 2
        @test all(point -> norm(F(point.x, (λ = point.p,)), Inf) < 1e-7, br.sol)
        # ds grows by at most max_growth per step (the recorded ds is the one of the next step), whatever the rejections
        ds = abs.(br.branch.ds)
        # (the last step may end on the boundary p_max, where the Natural corrector is used)
        @test all(ds[2:(end - 1)] .<= 1.1 .* ds[begin:(end - 2)] .* (1 + 1e-12))
        # and it does grow when the corrector is good
        @test maximum(ds) > 2 * first(ds)
    end

    # without them ds follows the Newton iteration count, which grows it faster than max_growth
    legacy = continuation(make_problem(Float64), PALC(step_control = nothing, orientation_check = false), options(Float64))
    @test count(sp -> sp.type == :fold, legacy.specialpoint) == 2
    legacy_ds = abs.(legacy.branch.ds)
    @test any(legacy_ds[2:(end - 1)] .> 1.1 .* legacy_ds[begin:(end - 2)])

    # with max_growth = nothing ds follows the Newton iteration count again, the tests still refuse steps
    unfactored = continuation(make_problem(Float64), PALC(step_control = CorrectorQuality(; max_distance = 0.4, max_growth = nothing)), options(Float64))
    @test count(sp -> sp.type == :fold, unfactored.specialpoint) == 2
    unfactored_ds = abs.(unfactored.branch.ds)
    @test any(unfactored_ds[2:(end - 1)] .> 1.1 .* unfactored_ds[begin:(end - 2)])

    # a short distance to the curve limits the step, the other test off
    only_distance(max_distance) = CorrectorQuality(; max_distance, max_contraction = nothing)
    loose = continuation(make_problem(Float64), PALC(step_control = only_distance(nothing)), options(Float64))
    tight = continuation(make_problem(Float64), PALC(step_control = only_distance(1e-3)), options(Float64))
    @test maximum(abs, tight.branch.ds) < maximum(abs, loose.branch.ds)
    @test length(tight.branch) > length(loose.branch)
    @test all(point -> norm(F(point.x, (λ = point.p,)), Inf) < 1e-7, tight.sol)

    # the continuation keeps the float type of the problem
    for T in (Float32, Float64)
        control = CorrectorQuality(; max_distance = T(0.4), max_contraction = T(0.6), max_growth = T(1.1), η = T(0.1))
        br = continuation(make_problem(T), PALC(θ = T(0.5), step_control = control), options(T))
        @test eltype(br.branch.ds) == T
        @test all(point -> point.p isa T, br.sol)
        @test count(sp -> sp.type == :fold, br.specialpoint) == 2
    end
end
####################################################################################################
# orientation_check: determinant orientation of a Bordered tangent, rejection on a negative dot product
let
    F(u, p) = @. u^2 + p.λ^2 - 1
    prob = BifurcationProblem(F, [sqrt(0.75)], (λ = -0.5,), (@optic _.λ))
    options = ContinuationPar(ds = 0.3, dsmin = 1e-6, dsmax = 0.3, p_min = -2., p_max = 2., max_steps = 30, detect_fold = false, detect_bifurcation = 0, newton_options = NewtonPar(tol = 1e-10))
    dotθ = BK.DotTheta()

    point(φ::Float64) = BorderedArray([cos(φ)], sin(φ))
    unit_tangent(φ::Float64) = BorderedArray([-sin(φ) / sqrt(0.5)], cos(φ) / sqrt(0.5))
    # in the norm of the arc length constraint (θ = 1/2) the angle between tangents is the angle between points
    φ0 = -π / 6
    iter = ContIterable(prob, PALC(; tangent = Bordered(), orientation_check = true), options)
    state = iterate(iter)[1]
    state.z = point(φ0)
    state.τ = unit_tangent(φ0)
    state.ds = 0.3
    state.orientation = 1
    # the determinant orientation is that of the curve, the same on both sides of a quarter turn, where alignment with the previous tangent
    # would reverse it
    for φ in (φ0 + 0.4, φ0 + 2.0, φ0 - 2.0)
        aligned = BK.bordered_tangent(state, iter, point(φ), dotθ)
        oriented = BK.bordered_tangent(state, iter, point(φ), dotθ; by_determinant = true)
        @test oriented.u ≈ unit_tangent(φ).u rtol = 1e-6
        @test oriented.p ≈ unit_tangent(φ).p rtol = 1e-6
        @test dotθ(state.τ, aligned, 0.5) ≥ 0
        cosine, next = BK.turn_cosine(Bordered(), state, iter, point(φ), dotθ)
        @test cosine ≈ cos(φ - φ0) atol = 1e-6
        @test next.u ≈ oriented.u
    end
    # the orientation sign flips the whole field
    state.orientation = -1
    @test BK.bordered_tangent(state, iter, point(φ0 + 0.4), dotθ; by_determinant = true).p ≈ -unit_tangent(φ0 + 0.4).p rtol = 1e-6
    # a Secant tangent is the chord: its cosine is that of consecutive chords, and the sign of ds is the direction
    state.orientation = 1
    for ds in (0.3, -0.3)
        state.ds = ds
        cosine, next = BK.turn_cosine(Secant(), state, iter, point(φ0 + sign(ds) * 0.4), dotθ)
        @test cosine ≈ cos(0.2) rtol = 1e-6
        @test isnothing(next)
        @test first(BK.turn_cosine(Secant(), state, iter, point(φ0 - sign(ds) * 0.4), dotθ)) ≈ -cos(0.2) rtol = 1e-6
    end

    # the start sets the sign of the orientation to the direction of travel, whichever way ds points
    for ds in (0.3, -0.3), tangent in (Secant(), Bordered())
        plain = continuation(prob, PALC(; tangent), ContinuationPar(options; ds))
        checked = continuation(prob, PALC(; tangent, orientation_check = true), ContinuationPar(options; ds))
        @test checked.branch.param ≈ plain.branch.param rtol = 1e-10
        @test length(checked.branch) == 31
    end

    # a corrector landing past a quarter turn is refused with the check and accepted without it
    # (a Secant chord that makes progress cannot turn back, so only Bordered is tested)
    far = point(φ0 + 2.0)
    for (check, turns_back) in ((true, true), (false, false))
        let alg = PALC(; tangent = Bordered(), orientation_check = check)
            iter = ContIterable(prob, alg, options)
            state = iterate(iter)[1]
            state.z = point(φ0)
            state.τ = unit_tangent(φ0)
            state.orientation = 1
            state.z_pred = far
            state.ds = dotθ(far.u .- state.z.u, state.τ.u, far.p - state.z.p, state.τ.p, 0.5)
            BK.corrector!(state, iter, alg)
            @test BK.converged(state) == ~turns_back
            @test (state.z.p ≈ far.p) == ~turns_back
        end
    end
end
