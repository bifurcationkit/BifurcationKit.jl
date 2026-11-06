using Test
using BifurcationKit, LinearAlgebra
const BK = BifurcationKit

@testset "Multiple continuation" begin
    ################################################################################################
    # (2) unit test: arithmetic of the predictors
    # the ii-th predictor must be z + ii * ds * τ, built from the base predictor z + ds * τ
    @testset "predictor arithmetic" begin
        z = BorderedArray([1.0, 2.0], 0.3)
        τ = BorderedArray([0.25, -0.5], 1.2)
        ds = 0.4
        # base predictor z + ds * τ (this is what `addtangent!` stores in state.z_pred)
        z_pred0 = BK._copy(z)
        BK.VI.add!(z_pred0, τ, ds)
        @test z_pred0.u ≈ z.u .+ ds .* τ.u
        @test z_pred0.p ≈ z.p + ds * τ.p
        # the base inputs are not mutated
        @test z.u == [1.0, 2.0] && z.p == 0.3
        for ii in 1:6
            zpred = BK._multiple_predictor(z_pred0, τ, ii, ds)
            @test zpred.u ≈ z.u .+ ii * ds .* τ.u
            @test zpred.p ≈ z.p + ii * ds * τ.p
        end
        # z_pred0 is not mutated by the calls above
        @test z_pred0.u ≈ z.u .+ ds .* τ.u
        @test z_pred0.p ≈ z.p + ds * τ.p
    end
    ################################################################################################
    # (1) affine branch: F(x, p) = x - (a p + b), solution x = a p + b
    # every predictor z + ii * ds * τ lies exactly on the branch and satisfies the
    # arclength constraint ii * ds, hence:
    # - every step converges in at most 1 newton iteration with the largest predictor (ii = nb)
    # - the branch advances by nb * ds along the tangent at each step (like pmcont)
    # - ds grows by dsfact at every step until ds * dsfact > dsmax
    # This test fails if the predictors are built from a stale tangent (B1) or compound (B2),
    # or if all predictors target the same arclength ds (B7).
    @testset "affine branch" begin
        a = [1.0, -0.5]; b = [0.3, 0.7]
        x0 = [1.5, -1.2] # not parallel to a and ≠ 0: exposes a stale tangent
        F_aff(x, p) = x .- (a .* p.p .+ b) # p is the full params container
        prob = BifurcationProblem(F_aff, x0, (p = 0.0,), (@optic _.p);
                    J = (x, p) -> [1.0 0.0; 0.0 1.0],
                    record_from_solution = (x, p; k...) -> (x = copy(x), p = p))
        opts = ContinuationPar(ds = 0.05, dsmin = 1e-6, dsmax = 0.4,
                    p_min = -10.0, p_max = 10.0, max_steps = 100,
                    newton_options = NewtonPar(tol = 1e-12, max_iterations = 8),
                    detect_bifurcation = 0, detect_fold = false)
        alg = Multiple(copy(x0), 0.01, 3)
        br = continuation(prob, alg, opts; verbosity = 0)
        @test length(br) > 10
        # all points on the branch, one newton iteration per step
        for pt in br.branch
            @test pt.x ≈ a .* pt.p .+ b atol = 1e-8
        end
        @test all(pt.itnewton <= 1 for pt in br.branch)
        # the tangent is constant along the branch: τ_p = 1/‖(a, 1)‖_θ with θ = 0.5
        τ_p = 1 / sqrt(0.5 * (a[1]^2 + a[2]^2) / 2 + 0.5)
        # each step advances by nb * ds * τ_p (largest predictor accepted), not by ds * τ_p
        @test br.branch[2].p ≈ 3 * 0.05 * τ_p atol = 1e-12
        for k in 2:5
            @test br.branch[k+1].p - br.branch[k].p ≈ 3 * (0.05 * 1.5^(k-1)) * τ_p atol = 1e-10
        end
        # ds growth: dsfact = 1.5 each step until ds * dsfact > dsmax
        @test br.branch[1].ds == 0.05
        for k in 1:5
            @test br.branch[k+1].ds ≈ 0.05 * 1.5^k
        end
        @test br.branch[end].ds ≈ 0.05 * 1.5^5 # stuck at ds * dsfact > dsmax
    end
    ################################################################################################
    # (3) unit test: step size control
    @testset "step size control" begin
        v = BorderedArray([1.0, 2.0], 0.3)
        cp = ContinuationPar(ds = 0.1, dsmin = 1e-3, dsmax = 1.0)
        # not converged, |ds| >= (nb+1)*dsmin → ds/(nb+1), continuation carries on
        alg = Multiple(copy([1.0, 2.0]), 0.1, 3)
        state = BK.ContState(; z_pred = v, τ = v, z = v, z_old = v, ds = 0.05, converged = false)
        @test_logs (:error, r"Decrease ds") match_mode=:any BK._step_size_control_multiple!(state, cp, 0, alg)
        @test state.ds ≈ 0.05 / (3 + 1)
        @test state.stopcontinuation == false
        # not converged, |ds| < (nb+1)*dsmin → increase pmimax, keep ds
        alg = Multiple(copy([1.0, 2.0]), 0.1, 3)
        state = BK.ContState(; z_pred = v, τ = v, z = v, z_old = v, ds = 0.003, converged = false)
        BK._step_size_control_multiple!(state, cp, 0, alg)
        @test alg.pmimax == 2
        @test state.ds ≈ 0.003
        @test state.stopcontinuation == false
        # not converged, |ds| < (nb+1)*dsmin and pmimax == imax → stop continuation
        alg = Multiple(copy([1.0, 2.0]), 0.1, 3); alg.pmimax = alg.imax
        state = BK.ContState(; z_pred = v, τ = v, z = v, z_old = v, ds = 0.003, converged = false)
        BK._step_size_control_multiple!(state, cp, 0, alg)
        @test state.stopcontinuation == true
        # converged with the largest predictor and ds * dsfact <= dsmax → ds *= dsfact, reset pmimax
        alg = Multiple(copy([1.0, 2.0]), 0.1, 3); alg.pmimax = 3; alg.currentind = 3
        state = BK.ContState(; z_pred = v, τ = v, z = v, z_old = v, ds = 0.1, converged = true)
        BK._step_size_control_multiple!(state, cp, 0, alg)
        @test state.ds ≈ 0.15
        @test alg.pmimax == 1
        # converged but not with the largest predictor → ds unchanged
        alg = Multiple(copy([1.0, 2.0]), 0.1, 3); alg.currentind = 2
        state = BK.ContState(; z_pred = v, τ = v, z = v, z_old = v, ds = 0.1, converged = true)
        BK._step_size_control_multiple!(state, cp, 0, alg)
        @test state.ds ≈ 0.1
    end
    ################################################################################################
    # tangent update: the tangent must be recomputed at each accepted point (secant of the two
    # last points), stay normalized, and the accepted step must be a multiple ii * ds (1 ≤ ii ≤ nb)
    # along the CURRENT tangent. This catches a stale tangent (B1).
    @testset "tangent update" begin
        F_circle(x, p) = x .- [cos(p.p), sin(p.p)] # p is the full params container
        prob = BifurcationProblem(F_circle, [1.0, 0.0], (p = 0.0,), (@optic _.p);
                    J = (x, p) -> [1.0 0.0; 0.0 1.0])
        θ = 0.5
        dotθ(u1, u2, p1, p2) = θ * (u1 ⋅ u2) / 2 + (1 - θ) * p1 * p2
        taus = []
        function finalise(z, tau, step, contRes; k...)
            push!(taus, (z = copy(z.u), p = z.p, tau_u = copy(tau.u), tau_p = tau.p))
            true
        end
        opts = ContinuationPar(ds = 0.05, dsmin = 1e-6, dsmax = 0.1,
                    p_min = 0.0, p_max = 100.0, max_steps = 30, # no boundary clamp
                    newton_options = NewtonPar(tol = 1e-12, max_iterations = 15),
                    detect_bifurcation = 0, detect_fold = false)
        br = continuation(prob, Multiple(copy([1.0, 0.0]), 0.5, 3), opts; verbosity = 0, finalise_solution = finalise)
        @test length(taus) > 10
        # τ is normalized in the θ-norm and satisfies the secant formula τ_k = (z_k - z_{k-1})/‖.‖_θ
        for k in 1:length(taus)
            @test sqrt(dotθ(taus[k].tau_u, taus[k].tau_u, taus[k].tau_p, taus[k].tau_p)) ≈ 1 atol = 1e-12
            if k >= 2
                dz = taus[k].z .- taus[k-1].z; dp = taus[k].p - taus[k-1].p
                n = sqrt(dotθ(dz, dz, dp, dp))
                @test taus[k].tau_u ≈ dz ./ n atol = 1e-12
                @test taus[k].tau_p ≈ dp / n atol = 1e-12
            end
        end
        # the accepted step advances by ii * ds along the CURRENT tangent (ii ∈ 1..nb)
        for k in 1:length(taus)-1
            dz = taus[k+1].z .- taus[k].z; dp = taus[k+1].p - taus[k].p
            ds_k = br.branch[k+1].ds # ds used for step k+1
            ii = dotθ(dz, taus[k].tau_u, dp, taus[k].tau_p) / ds_k
            @test isapprox(ii, round(ii); atol = 1e-6)
            @test 1 <= round(ii) <= 3
        end
        # τ rotates with p: the tangent is updated, not the stale initial one
        angles = [atan(t.tau_u[2], t.tau_u[1]) for t in taus]
        @test angles[end] - angles[1] > 1.0
    end
    ################################################################################################
    # (4) curved branch: circle (analytic solution) + fold crossing with u^2 = p
    @testset "curved branches" begin
        # circle: F(x, p) = x - [cos p, sin p]
        F_circle(x, p) = x .- [cos(p.p), sin(p.p)] # p is the full params container
        prob = BifurcationProblem(F_circle, [1.0, 0.0], (p = 0.0,), (@optic _.p);
                    J = (x, p) -> [1.0 0.0; 0.0 1.0],
                    record_from_solution = (x, p; k...) -> (x = copy(x), p = p))
        opts = ContinuationPar(ds = 0.1, dsmin = 1e-6, dsmax = 0.3,
                    p_min = 0.0, p_max = 6.5, max_steps = 200,
                    newton_options = NewtonPar(tol = 1e-12, max_iterations = 15),
                    detect_bifurcation = 0, detect_fold = false)
        br = continuation(prob, Multiple(copy([1.0, 0.0]), 0.05, 4), opts; verbosity = 0)
        @test br.param[end] > 2π - 0.5 # went around the circle
        err = maximum(norm(pt.x .- [cos(pt.p), sin(pt.p)], Inf) for pt in br.branch)
        @test err < 1e-8
        # fold crossing: F(u, p) = p - u^2, continue backward from (1, 1) through the fold at p = 0
        F_fold(u, p) = [p.p - u[1]^2] # p is the full params container
        prob_fold = BifurcationProblem(F_fold, [1.0], (p = 1.0,), (@optic _.p);
                    J = (u, p) -> reshape([-2u[1]], 1, 1),
                    record_from_solution = (u, p; k...) -> (u = copy(u), p = p))
        opts_fold = ContinuationPar(opts; ds = -0.05, p_min = -0.01, p_max = 1.5, max_steps = 150)
        br_fold = continuation(prob_fold, Multiple(copy([1.0]), 0.05, 4), opts_fold; verbosity = 0)
        # passed the fold: p decreased to 0 and came back, u changed sign
        @test minimum(pt.p for pt in br_fold.branch) < 0.05
        @test any(pt.u[1] < -0.9 for pt in br_fold.branch)
        # all points on the branch p = u^2
        @test maximum(abs(pt.p - pt.u[1]^2) for pt in br_fold.branch) < 1e-8
    end
end
