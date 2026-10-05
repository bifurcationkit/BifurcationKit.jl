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
    @test BK.growth_factor(quality) ≈ 1.1

    # first step: rejected from max_distance on
    @test ~BK.rejects_step(control, quality, 0.39, tol)
    @test BK.rejects_step(control, quality, 0.4, tol)
    # a short first step still allows the largest growth, a long one slows it down by sqrt(d / max_distance) * max_growth
    @test BK.growth_factor(BK.observe(control, quality, 0.1, tol)) ≈ 1.1
    @test BK.growth_factor(BK.observe(control, quality, 0.3, tol)) ≈ inv(1.1 * sqrt(0.3 / 0.4))

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
    @test BK.growth_factor(BK.observe(off, BK.observe(off, BK.StepQuality(off), 1e6, tol), 1e12, tol)) ≈ 1.1
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
    @test PALC().max_angle === nothing

    for tangent in (Secant(), Bordered())
        br = continuation(make_problem(Float64), PALC(; tangent, step_control = CorrectorQuality(), max_angle = π / 2), options(Float64))
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
    legacy = continuation(make_problem(Float64), PALC(step_control = nothing, max_angle = nothing), options(Float64))
    @test count(sp -> sp.type == :fold, legacy.specialpoint) == 2
    legacy_ds = abs.(legacy.branch.ds)
    @test any(legacy_ds[2:(end - 1)] .> 1.1 .* legacy_ds[begin:(end - 2)])

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
# the angle test: the angle between the tangents at both ends of a step
let
    F(u, p) = @. u^2 + p.λ^2 - 1
    prob = BifurcationProblem(F, [sqrt(0.75)], (λ = -0.5,), (@optic _.λ))
    options = ContinuationPar(ds = 0.3, dsmin = 1e-6, dsmax = 0.3, p_min = -2., p_max = 2., max_steps = 30, detect_fold = false, detect_bifurcation = 0, newton_options = NewtonPar(tol = 1e-10))
    dotθ = BK.DotTheta()

    point(φ) = BorderedArray([cos(φ)], sin(φ))
    unit_tangent(φ) = BorderedArray([-sin(φ) / sqrt(0.5)], cos(φ) / sqrt(0.5))
    # on the circle u^2 + λ^2 = 1, in the norm of the arc length constraint (θ = 1/2, one component), the angle between the tangents at two points is the angle between the points
    φ0 = -π / 6
    for tangent in (Secant(), Bordered())
        iter = ContIterable(prob, PALC(; tangent, max_angle = π / 2), options)
        state = iterate(iter)[1]
        state.z = point(φ0)
        state.τ = unit_tangent(φ0)
        state.ds = 0.3
        # a Bordered tangent is the tangent at the new point, a Secant one stands for it by the chord
        turn = if tangent isa Bordered
            0.4
        else
            0.2
        end
        cosine, next = BK.turn_cosine(tangent, state, iter, point(φ0 + 0.4), dotθ)
        @test cosine ≈ cos(turn) rtol = 1e-6
        @test isnothing(next) == (tangent isa Secant)
        if tangent isa Bordered
            @test next.u ≈ unit_tangent(φ0 + 0.4).u rtol = 1e-6
            @test next.p ≈ unit_tangent(φ0 + 0.4).p rtol = 1e-6
        end
        # a radial chord is at right angles to the tangent
        if tangent isa Secant
            @test first(BK.turn_cosine(tangent, state, iter, BorderedArray([1.3cos(φ0)], 1.3sin(φ0)), dotθ)) ≈ 0 atol = 1e-12
            # a negative ds walks along `-τ`
            state.ds = -0.3
            @test first(BK.turn_cosine(tangent, state, iter, point(φ0 - 0.4), dotθ)) ≈ cos(0.2) rtol = 1e-6
            @test first(BK.turn_cosine(tangent, state, iter, point(φ0 + 0.4), dotθ)) ≈ -cos(0.2) rtol = 1e-6
        end
    end

    # on this circle consecutive tangents turn by 0.44 rad (Bordered) and consecutive chords by 0.46 rad (Secant) with ds = 0.3:
    # π / 4 refuses no step, either way round, and the Bordered tangent computed for the test is the one the next predictor takes
    for tangent in (Secant(), Bordered()), ds in (0.3, -0.3)
        plain = continuation(prob, PALC(; tangent), ContinuationPar(options; ds))
        checked = continuation(prob, PALC(; tangent, max_angle = π / 4), ContinuationPar(options; ds))
        @test checked.branch.param == plain.branch.param
        @test length(checked.branch) == 31
    end

    # a threshold under the angle of a step refuses it and halves ds until the tangents are close enough:
    # the same 30 steps cover less of the circle
    for tangent in (Secant(), Bordered())
        plain = continuation(prob, PALC(; tangent), options)
        tight = continuation(prob, PALC(; tangent, max_angle = 1e-2), options)
        @test sum(abs, diff(tight.branch.param)) < sum(abs, diff(plain.branch.param)) / 2
    end

    # the derived limit: the turn a predictor of length |ds| follows within max_distance, at most the cap
    control = CorrectorQuality(; max_distance = 0.05)
    @test BK.turn_limit(:derived, π / 2, control, 0.1) ≈ 1
    @test BK.turn_limit(:derived, π / 2, control, -0.1) ≈ 1
    @test BK.turn_limit(:derived, π / 2, control, 0.01) == π / 2
    @test BK.turn_limit(:derived, 0.3, control, 0.1) == 0.3
    @test BK.turn_limit(0.7, π / 2, control, 0.1) == 0.7
    @test isnothing(BK.turn_limit(nothing, π / 2, control, 0.1))
    @test_throws AssertionError PALC(max_angle = :derived)
    @test_throws AssertionError PALC(max_angle = :derived, step_control = CorrectorQuality())
    @test_throws AssertionError PALC(max_angle = :other)
    @test_throws AssertionError PALC(max_angle = 4.0)
    @test_throws AssertionError PALC(max_angle_cap = 0.0)
    # a distance that allows every step here (limit above the circle's turn) refuses none, a small cap refuses them
    for tangent in (Secant(), Bordered())
        plain = continuation(prob, PALC(; tangent, step_control = CorrectorQuality(; max_distance = 1.0, max_contraction = nothing)), options)
        loose = continuation(prob, PALC(; tangent, max_angle = :derived, step_control = CorrectorQuality(; max_distance = 1.0, max_contraction = nothing)), options)
        @test loose.branch.param == plain.branch.param
        capped = continuation(prob, PALC(; tangent, max_angle = :derived, max_angle_cap = 1e-2, step_control = CorrectorQuality(; max_distance = 1.0, max_contraction = nothing)), options)
        @test sum(abs, diff(capped.branch.param)) < sum(abs, diff(plain.branch.param)) / 2
    end
end
