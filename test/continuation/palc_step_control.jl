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
# the sign of the dot product between the tangent and the chord of a step, in the norm of the arc length constraint
let
    dotθ = BK.DotTheta()
    τ = BorderedArray([1.0], 0.0)
    z = BorderedArray([0.0], 0.0)
    @test BK.chord_dot(dotθ, τ, z, BorderedArray([2.0], 0.0), 0.5) > 0
    @test BK.chord_dot(dotθ, τ, z, BorderedArray([-2.0], 0.0), 0.5) < 0
    @test BK.chord_dot(dotθ, τ, z, BorderedArray([0.0], 3.0), 0.5) ≈ 0 atol = 1e-15
    # an empty chord does not turn
    @test BK.chord_dot(dotθ, τ, z, z, 0.5) == 0
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
# the orientation test: the sign of the dot product between the direction of travel and the chord of a step
let
    F(u, p) = @. u^2 + p.λ^2 - 1
    prob = BifurcationProblem(F, [sqrt(0.75)], (λ = -0.5,), (@optic _.λ))
    options = ContinuationPar(ds = 0.3, dsmin = 1e-6, dsmax = 0.3, p_min = -2., p_max = 2., max_steps = 30, detect_fold = false, detect_bifurcation = 0, newton_options = NewtonPar(tol = 1e-10))
    dotθ = BK.DotTheta()

    iter = ContIterable(prob, PALC(orientation_check = true), options)
    state = iterate(iter)[1]
    state.τ = BorderedArray([0.0], 1.0)
    state.z = BorderedArray([0.5], 0.0)
    ahead = BorderedArray([0.5], 0.3)
    behind = BorderedArray([0.5], -0.3)
    state.ds = 0.3
    @test BK.orientation_dot(state, iter, ahead, dotθ) > 0
    @test BK.orientation_dot(state, iter, behind, dotθ) < 0
    # a negative ds walks along `-τ`
    state.ds = -0.3
    @test BK.orientation_dot(state, iter, ahead, dotθ) < 0
    @test BK.orientation_dot(state, iter, behind, dotθ) > 0

    # with a good corrector the test never refuses a step, either way round and with either predictor
    for tangent in (Secant(), Bordered()), ds in (0.3, -0.3)
        plain = continuation(prob, PALC(; tangent), ContinuationPar(options; ds))
        checked = continuation(prob, PALC(; tangent, orientation_check = true), ContinuationPar(options; ds))
        @test checked.branch.param == plain.branch.param
        @test length(checked.branch) == 31
    end
end
