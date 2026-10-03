using Test, BifurcationKit, LinearAlgebra
const BK = BifurcationKit
####################################################################################################
# the pure rules of the Allgower-Georg acceptance
let
    tol = 1e-8
    acceptance = AllgowerGeorg()   # Gambit's: 0.4, 0.6, 1.1, alignment 0, η = 0.1
    @test (acceptance.max_distance, acceptance.max_contraction, acceptance.max_growth, acceptance.min_alignment, acceptance.η) == (0.4, 0.6, 1.1, 0.0, 0.1)

    quality = BK.StepQuality(acceptance)
    # no Newton step: the largest growth
    @test BK.growth_factor(quality) ≈ 1.1

    # first step: rejected from max_distance on
    @test ~BK.rejects_step(acceptance, quality, 0.39, tol)
    @test BK.rejects_step(acceptance, quality, 0.4, tol)
    # a short first step still allows the largest growth, a long one slows it down by sqrt(d / max_distance) * max_growth
    @test BK.growth_factor(BK.observe(acceptance, quality, 0.1, tol)) ≈ 1.1
    @test BK.growth_factor(BK.observe(acceptance, quality, 0.3, tol)) ≈ inv(1.1 * sqrt(0.3 / 0.4))

    # later steps: rejected when d_k / (d_{k-1} + tol η) exceeds max_contraction
    after_first = BK.observe(acceptance, quality, 0.3, tol)
    @test ~BK.rejects_step(acceptance, after_first, 0.18, tol)
    @test BK.rejects_step(acceptance, after_first, 0.19, tol)
    contraction = 0.1 / (0.3 + tol * 0.1)
    @test BK.contraction_rate(acceptance, after_first, 0.1, tol) ≈ contraction
    after_second = BK.observe(acceptance, after_first, 0.1, tol)
    # the deceleration is the largest one asked for by any step
    @test after_second.deceleration ≈ max(1.1 * sqrt(0.3 / 0.4), 1.1 * sqrt(contraction / 0.6))
    @test after_second.steps == 2

    # each test is switched off alone
    off = AllgowerGeorg(max_distance = Inf, max_contraction = Inf)
    @test ~BK.rejects_step(off, BK.StepQuality(off), 1e6, tol)
    @test ~BK.rejects_step(off, BK.observe(off, BK.StepQuality(off), 1e6, tol), 1e12, tol)
    @test BK.growth_factor(BK.observe(off, BK.observe(off, BK.StepQuality(off), 1e6, tol), 1e12, tol)) ≈ 1.1

    # thresholds follow the float type of the continuation
    @test BK.AllgowerGeorg{Float32}(acceptance) isa AllgowerGeorg{Float32}
    @test AllgowerGeorg(max_distance = 0.4f0, max_contraction = 0.6f0, max_growth = 1.1f0, min_alignment = 0f0, η = 0.1f0) isa AllgowerGeorg{Float32}
    @test_throws AssertionError AllgowerGeorg(max_growth = 1.0)
    @test_throws AssertionError AllgowerGeorg(min_alignment = 2.0)
end
####################################################################################################
# cosine between the tangent and the chord of a step, in the norm of the arc length constraint
let
    dotθ = BK.DotTheta()
    τ = BorderedArray([1.0], 0.0)
    z = BorderedArray([0.0], 0.0)
    @test BK.chord_alignment(dotθ, τ, z, BorderedArray([2.0], 0.0), 0.5) ≈ 1
    @test BK.chord_alignment(dotθ, τ, z, BorderedArray([-2.0], 0.0), 0.5) ≈ -1
    @test BK.chord_alignment(dotθ, τ, z, BorderedArray([0.0], 3.0), 0.5) ≈ 0 atol = 1e-15
    # 45 degrees in the weighted norm (θ = 1/2: both components weigh the same)
    @test BK.chord_alignment(dotθ, τ, z, BorderedArray([1.0], 1.0), 0.5) ≈ sqrt(2) / 2
    # an empty chord does not turn
    @test BK.chord_alignment(dotθ, τ, z, z, 0.5) == 1
end
####################################################################################################
# continuation with the acceptance: u^3 - u - (p - 1) = 0 has two folds, p ≈ 0.615 and p ≈ 1.385
let
    F(u, p) = @. u^3 - u - (p.λ - 1)
    make_problem(T) = BifurcationProblem(F, T[-1.3247179], (λ = zero(T),), (@optic _.λ))
    options(T; ds = 0.05, dsmax = 0.5) = ContinuationPar(ds = T(ds), dsmin = T(1e-6), dsmax = T(dsmax), a = T(0.5), p_min = zero(T), p_max = T(2), η = T(150), tol_stability = T(1e-10), dsmin_bisection = T(1e-16), tol_bisection_eigenvalue = T(1e-16), tol_param_bisection_event = T(1e-16), max_steps = 200, detect_fold = true, detect_bifurcation = 0, newton_options = NewtonPar(tol = sqrt(eps(T))))

    # without it, nothing changes
    @test PALC().step_acceptance === nothing
    plain = continuation(make_problem(Float64), PALC(), options(Float64))
    explicit = continuation(make_problem(Float64), PALC(step_acceptance = nothing), options(Float64))
    @test plain.branch.param == explicit.branch.param
    @test plain.branch.ds == explicit.branch.ds

    for tangent in (Secant(), Bordered())
        br = continuation(make_problem(Float64), PALC(; tangent, step_acceptance = AllgowerGeorg()), options(Float64))
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

    # a short distance to the curve limits the step, the other tests off
    only_distance(max_distance) = AllgowerGeorg(; max_distance, max_contraction = Inf, min_alignment = -1.0)
    loose = continuation(make_problem(Float64), PALC(step_acceptance = only_distance(Inf)), options(Float64))
    tight = continuation(make_problem(Float64), PALC(step_acceptance = only_distance(1e-3)), options(Float64))
    @test maximum(abs, tight.branch.ds) < maximum(abs, loose.branch.ds)
    @test length(tight.branch) > length(loose.branch)
    @test all(point -> norm(F(point.x, (λ = point.p,)), Inf) < 1e-7, tight.sol)

    # the continuation keeps the float type of the problem
    for T in (Float32, Float64)
        acceptance = AllgowerGeorg(; max_distance = T(0.4), max_contraction = T(0.6), max_growth = T(1.1), min_alignment = T(0), η = T(0.1))
        br = continuation(make_problem(T), PALC(θ = T(0.5), step_acceptance = acceptance), options(T))
        @test eltype(br.branch.ds) == T
        @test all(point -> point.p isa T, br.sol)
        @test count(sp -> sp.type == :fold, br.specialpoint) == 2
    end
end
####################################################################################################
# a turn that is sharper than min_alignment is refused: continuation around the unit circle u^2 + p^2 = 1
let
    F(u, p) = @. u^2 + p.λ^2 - 1
    prob = BifurcationProblem(F, [sqrt(0.75)], (λ = -0.5,), (@optic _.λ))
    options = ContinuationPar(ds = 0.05, dsmin = 1e-6, dsmax = 0.3, p_min = -2., p_max = 2., max_steps = 40, detect_fold = false, detect_bifurcation = 0, newton_options = NewtonPar(tol = 1e-10))

    # cosine between successive chords, which are the tangents of the Secant predictor
    function turns(br)
        points = [[only(point.x), point.p] for point in br.sol]
        chords = diff(points)
        weighted(a, b) = 0.5 * a[1] * b[1] + 0.5 * a[2] * b[2]
        return [weighted(a, b) / sqrt(weighted(a, a) * weighted(b, b)) for (a, b) in zip(chords, Iterators.drop(chords, 1))]
    end

    unconstrained = AllgowerGeorg(max_distance = Inf, max_contraction = Inf, min_alignment = -1.0)
    constrained = AllgowerGeorg(max_distance = Inf, max_contraction = Inf, min_alignment = 0.99)
    br_free = continuation(prob, PALC(step_acceptance = unconstrained), options)
    br_turn = continuation(prob, PALC(step_acceptance = constrained), options)
    # the steps of the free run turn more than the limit, so the test discriminates
    @test minimum(turns(br_free)) < 0.99
    @test minimum(turns(br_turn)) >= 0.99 - 1e-8
    @test maximum(abs, br_turn.branch.ds) < maximum(abs, br_free.branch.ds)

    # a negative ds walks the circle the other way: the chord is tested against `sign(ds) τ`, so the continuation is not refused
    backwards = ContinuationPar(options; ds = -0.05)
    br_back = continuation(prob, PALC(step_acceptance = constrained), backwards)
    @test length(br_back.branch) > 20
    @test br_back.branch.param[2] < br_back.branch.param[1]
    @test minimum(turns(br_back)) >= 0.99 - 1e-8
    @test length(continuation(prob, PALC(step_acceptance = AllgowerGeorg()), backwards).branch) > 5
end
