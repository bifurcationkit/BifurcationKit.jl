using Test
using BifurcationKit, LinearAlgebra
using NonlinearSolve:
    NewtonRaphson,
    TrustRegion,
    BoundedTrustRegion,
    JacobianReuse,
    LUFactorization,
    KrylovJL_GMRES
import NonlinearSolve
# loads the extension; its `BorderingBLS` is not BifurcationKit's
import BorderedLinearSolve
const BK = BifurcationKit
####################################################################################################
# Bratu problem u'' + λ exp(u) = 0, which has a fold at λ ≈ 3.51
function F_bratu(x, p)
    n = length(x)
    h = 1 / (n + 1)
    # in the type of `p.λ` too, for the automatic `∂ₚF` of the jet
    out = similar(x, promote_type(eltype(x), typeof(p.λ)))
    for i in eachindex(x)
        left = i == 1 ? zero(eltype(x)) : x[i - 1]
        right = i == n ? zero(eltype(x)) : x[i + 1]
        out[i] = (left - 2x[i] + right) / h^2 + p.λ * exp(x[i])
    end
    out
end

# the tridiagonal Jacobian of `F_bratu`, dense, and as the function `dx -> J dx` of a matrix-free problem
function J_bratu(x, p)
    n = length(x)
    h = 1 / (n + 1)
    return Tridiagonal(
        fill(1 / h^2, n - 1),
        -2 / h^2 .+ p.λ .* exp.(x),
        fill(1 / h^2, n - 1),
    )
end

J_bratu_free(x, p) = dx -> J_bratu(x, p) * dx

# `precs` of `KrylovJL_GMRES`: the LU of the dense Jacobian (bordered by PALC), counting its builds
struct DenseLUPreconditioner
    "the parameter of a solve at fixed parameter, whose unknown is the state alone"
    λ::Float64
    builds::Base.RefValue{Int}
end

# the state and the parameter of the unknown `u` of the corrector: `ArrayPartition(x, [p])` for PALC, `x` at the fixed parameter `λ`
dense_point(u::BK.RecursiveArrayTools.ArrayPartition, λ) = (u.x[1], only(u.x[2]))
dense_point(u::AbstractVector, λ) = (u, λ)

function (preconditioner::DenseLUPreconditioner)(A, ::Any)
    x, λ = dense_point(A.u, preconditioner.λ)
    preconditioner.builds[] += 1
    return (lu(BK.corrector_matrix(A.p, A.u, Matrix(J_bratu(x, (; λ))))), I)
end

prob_bratu = BifurcationProblem(F_bratu, zeros(20), (λ = 0.01,), (@optic _.λ))
opts_bratu = ContinuationPar(
    p_min = 0.0,
    p_max = 4.0,
    ds = 0.05,
    dsmax = 0.2,
    max_steps = 60,
    detect_bifurcation = 0,
    newton_options = NewtonPar(tol = 1.0e-10),
)

@testset "NonlinearSolveCorrector" begin
    br_default = continuation(prob_bratu, PALC(), opts_bratu)

    exact = NewtonRaphson(; linsolve = LUFactorization(), jacobian_reuse = false)

    @testset "PALC passes the fold and follows the same branch" begin
        # exact Newton steps: the same corrector iterates as the default one
        br = continuation(
            prob_bratu,
            PALC(; corrector = NonlinearSolveCorrector(exact)),
            opts_bratu,
        )
        @test length(br.sol) == length(br_default.sol)
        @test maximum(abs, br.branch.param .- br_default.branch.param) < 1.0e-8
        @test maximum(
            point -> norm(point[1].x - point[2].x, Inf),
            zip(br.sol, br_default.sol),
        ) <
            1.0e-8
        # the branch goes beyond the fold, p decreases after it
        @test any(diff(br.branch.param) .< 0)
    end

    @testset "other algorithms stay on the branch" begin
        # a modified Newton method (Jacobian reuse) or a trust region takes other steps: the points still solve F = 0 on the same branch
        for alg in (
                NewtonRaphson(;
                    linsolve = LUFactorization(),
                    jacobian_reuse = JacobianReuse(),
                ),
                # the dogleg trust region: the default Moré subproblem solves the least-squares system `[J; √λD]`, which is not bordered
                TrustRegion(;
                    linsolve = LUFactorization(),
                    jacobian_reuse = false,
                    subproblem = NonlinearSolve.NonlinearSolveBase.TrustRegionSubproblem.Dogleg,
                ),
            )
            br = continuation(
                prob_bratu,
                PALC(; corrector = NonlinearSolveCorrector(alg)),
                opts_bratu,
            )
            @test all(point -> norm(F_bratu(point.x, (λ = point.p,)), Inf) < 1.0e-8, br.sol)
            @test any(diff(br.branch.param) .< 0)
            @test maximum(point -> point.p, br.sol) ≈
                maximum(point -> point.p, br_default.sol) atol = 0.02
        end
    end

    @testset "a Jacobian carried across steps stays on the branch" begin
        # `reuse_jacobian = true`: the policy of `alg`, not each new step, decides when the Jacobian is rebuilt
        alg = NewtonRaphson(;
            linsolve = LUFactorization(),
            jacobian_reuse = JacobianReuse(max_age = 50),
        )
        br = continuation(
            prob_bratu,
            PALC(; corrector = NonlinearSolveCorrector(alg; reuse_jacobian = true)),
            opts_bratu,
        )
        @test all(point -> norm(F_bratu(point.x, (λ = point.p,)), Inf) < 1.0e-8, br.sol)
        @test any(diff(br.branch.param) .< 0)
        @test maximum(point -> point.p, br.sol) ≈
            maximum(point -> point.p, br_default.sol) atol = 0.02
    end

    @testset "Newton solve" begin
        prob = re_make(prob_bratu; params = (λ = 1.0,), u0 = fill(0.1, 20))
        sol0 = BK.solve(prob, Newton(), NewtonPar(tol = 1.0e-12))
        sol1 = BK.solve(prob, NonlinearSolveCorrector(exact), NewtonPar(tol = 1.0e-12))
        @test BK.converged(sol1)
        @test norm(sol0.u - sol1.u, Inf) < 1.0e-10
    end

    @testset "Natural continuation" begin
        opts = ContinuationPar(opts_bratu; p_max = 3.0, max_steps = 20)
        br0 = continuation(prob_bratu, Natural(), opts)
        br1 = continuation(
            prob_bratu,
            Natural(; corrector = NonlinearSolveCorrector(exact)),
            opts,
        )
        @test length(br1.sol) == length(br0.sol)
        @test maximum(point -> norm(point[1].x - point[2].x, Inf), zip(br1.sol, br0.sol)) <
            1.0e-8
    end

    @testset "a step past p_max ends the branch on p_max" begin
        # the first step goes to p = 0.01 + 0.5 > p_max
        opts = ContinuationPar(
            opts_bratu;
            p_max = 0.2,
            ds = 0.5,
            dsmax = 0.5,
            max_steps = 5,
        )
        br = continuation(
            prob_bratu,
            PALC(; corrector = NonlinearSolveCorrector(exact)),
            opts,
        )
        @test all(br.branch.param .<= 0.2)
        @test br.branch.param[end] == 0.2
        @test norm(F_bratu(br.sol[end].x, (λ = br.sol[end].p,)), Inf) < 1.0e-8
    end

    @testset "a corrector converging past p_max from an interior predictor is corrected at p_max" begin
        # below the centre of the circle x² + a² = 1, the corrector raises `a` beyond its predictor: from a = -1/2, a predictor short of
        # p_max = -0.148 converges past it (the tangent has norm 1 for the dot product with θ = 1/2, so the predictor moves `ds √2`)
        F_circle(x, p) = [x[1]^2 + p.a^2 - 1]
        prob_circle = BifurcationProblem(F_circle, [sqrt(0.75)], (a = -0.5,), (@optic _.a))
        for along in (0.38, 0.4)
            opts = ContinuationPar(
                p_min = -1.0,
                p_max = -0.148,
                ds = along / sqrt(2),
                dsmax = 0.5,
                max_steps = 50,
                detect_bifurcation = 0,
                newton_options = NewtonPar(tol = 1.0e-10),
            )
            br = continuation(prob_circle, PALC(; corrector = NonlinearSolveCorrector(exact)), opts)
            @test all(br.branch.param .<= opts.p_max)
            @test br.branch.param[end] == opts.p_max
            @test abs(F_circle(br.sol[end].x, (a = br.sol[end].p,))[1]) < 1.0e-8
        end
    end

    @testset "an algorithm with box constraints ends the branch at p_max" begin
        opts = ContinuationPar(opts_bratu; p_max = 3.0, max_steps = 80)
        br = continuation(
            prob_bratu,
            PALC(; corrector = NonlinearSolveCorrector(BoundedTrustRegion(;
                linsolve = LUFactorization(),
            ))),
            opts,
        )
        @test all(point -> norm(F_bratu(point.x, (λ = point.p,)), Inf) < 1.0e-8, br.sol)
        @test all(br.branch.param .<= 3.0)
        @test maximum(br.branch.param) > 2.9
    end

    # the algorithms that clamp a trial point into the box (`bounds_handling = BoundsProjection()`) exist only in newer NonlinearSolve
    projecting = if isdefined(NonlinearSolve, :BoundsProjection)
        NewtonRaphson(;
            linsolve = LUFactorization(),
            bounds_handling = NonlinearSolve.BoundsProjection(),
        )
    else
        nothing
    end
    if !isnothing(projecting)
        @testset "an algorithm that projects onto the box ends the branch at p_max" begin
            opts = ContinuationPar(opts_bratu; p_max = 3.0, max_steps = 80)
            br = continuation(
                prob_bratu,
                PALC(; corrector = NonlinearSolveCorrector(projecting)),
                opts,
            )
            @test all(point -> norm(F_bratu(point.x, (λ = point.p,)), Inf) < 1.0e-8, br.sol)
            @test all(br.branch.param .<= 3.0)
            @test maximum(br.branch.param) > 2.9
        end
    end

    @testset "a dense linear solver rejects matrix-free Jacobians" begin
        prob = BifurcationProblem(
            F_bratu,
            zeros(5),
            (λ = 0.1,),
            (@optic _.λ);
            J = J_bratu_free,
        )
        @test_throws ArgumentError BK.solve(
            prob,
            NonlinearSolveCorrector(NewtonRaphson()),
            NewtonPar(),
        )
    end

    prob_free = BifurcationProblem(
        F_bratu,
        zeros(20),
        (λ = 0.01,),
        (@optic _.λ);
        J = J_bratu_free,
    )
    free_policy = JacobianReuse(max_age = 50)

    @testset "a matrix-free Newton solve reaches the root of the dense one" begin
        prob = re_make(prob_free; params = (λ = 1.0,), u0 = fill(0.1, 20))
        sol0 = BK.solve(
            re_make(prob_bratu; params = (λ = 1.0,), u0 = fill(0.1, 20)),
            Newton(),
            NewtonPar(tol = 1.0e-12),
        )
        builds = Ref(0)
        alg = NewtonRaphson(;
            linsolve = KrylovJL_GMRES(; precs = DenseLUPreconditioner(1.0, builds)),
            jacobian_reuse = free_policy,
        )
        sol1 = BK.solve(prob, NonlinearSolveCorrector(alg), NewtonPar(tol = 1.0e-9))
        @test BK.converged(sol1)
        @test norm(sol0.u - sol1.u, Inf) < 1.0e-7
        # the preconditioner is built when the policy refreshes it, not at every iterate
        @test builds[] < sol1.itnewton
    end

    @testset "matrix-free PALC follows the dense branch, rebuilding the preconditioner only when reuse refreshes it" begin
        # BifurcationKit's own parts of the continuation (start point, tangent) need an iterative solver too
        linsolver = GMRESKrylovKit(dim = 20, rtol = 1.0e-12, atol = 1.0e-12)
        opts = ContinuationPar(
            p_min = 0.0,
            p_max = 4.0,
            ds = 0.05,
            dsmax = 0.2,
            max_steps = 60,
            detect_bifurcation = 0,
            newton_options = NewtonPar(; tol = 1.0e-9, linsolver),
        )
        builds_fresh, builds_reused = Ref(0), Ref(0)
        for (builds, reuse_jacobian) in ((builds_fresh, false), (builds_reused, true))
            # GMRES at the default relative tolerance leaves residuals above `tol`, which BifurcationKit rejects as a failed step
            alg = NewtonRaphson(;
                linsolve = KrylovJL_GMRES(;
                    precs = DenseLUPreconditioner(NaN, builds),
                    atol = 1.0e-12,
                    rtol = 1.0e-12,
                ),
                jacobian_reuse = free_policy,
            )
            corrector = NonlinearSolveCorrector(alg; reuse_jacobian)
            br = continuation(
                prob_free,
                PALC(; corrector, bls = MatrixFreeBLS(linsolver)),
                opts,
            )
            @test all(point -> norm(F_bratu(point.x, (λ = point.p,)), Inf) < 1.0e-8, br.sol)
            @test any(diff(br.branch.param) .< 0)
            @test maximum(point -> point.p, br.sol) ≈
                maximum(point -> point.p, br_default.sol) atol = 0.02
        end
        @test builds_reused[] < builds_fresh[]
    end

    @testset "the state can be any AbstractArray" begin
        # the Bratu problem with its 20 unknowns as a 4×5 matrix
        F_mat(x, p) = reshape(F_bratu(vec(x), p), size(x))
        J_mat(x, p) = dx -> reshape(J_bratu(vec(x), p) * vec(dx), size(dx))
        prob_mat = BifurcationProblem(F_mat, zeros(4, 5), (λ = 0.01,), (@optic _.λ); J = J_mat)
        linsolver = GMRESKrylovKit(dim = 20, rtol = 1.0e-12, atol = 1.0e-12)
        opts = ContinuationPar(
            p_min = 0.0,
            p_max = 4.0,
            ds = 0.05,
            dsmax = 0.2,
            max_steps = 60,
            detect_bifurcation = 0,
            newton_options = NewtonPar(; tol = 1.0e-9, linsolver),
        )
        alg = NewtonRaphson(; linsolve = KrylovJL_GMRES(; atol = 1.0e-12, rtol = 1.0e-12))
        br = continuation(
            prob_mat,
            PALC(; corrector = NonlinearSolveCorrector(alg), bls = MatrixFreeBLS(linsolver)),
            opts,
        )
        @test all(point -> point.x isa Matrix && size(point.x) == (4, 5), br.sol)
        @test all(point -> norm(F_bratu(vec(point.x), (λ = point.p,)), Inf) < 1.0e-8, br.sol)
        @test any(diff(br.branch.param) .< 0)
        @test maximum(point -> point.p, br.sol) ≈
            maximum(point -> point.p, br_default.sol) atol = 0.02
    end

    @testset "∂F/∂p of the bordered Jacobian is the R01 of the problem's jet" begin
        ext = Base.get_extension(BK, :NonlinearSolveExt)
        # a residual whose forward finite difference in `a` (step sqrt(eps)) is off by 1e-3 relative
        ω = 1.0e5
        F_fast(x, par) = x .- sin(ω * par.a)
        par = (a = 0.3,)
        x = zeros(3)
        τ = BK.BorderedArray(ones(3), 1.0)
        bordered_dFdp(prob) = ext.palc_jacobian(
            ext.ArrayPartition(x, [par.a]),
            ext.PALCCache(prob, par, (; τ), BK.DotTheta(), 0.5),
        ).dFdp

        # the default jet differentiates automatically
        prob_auto = BifurcationProblem(F_fast, x, par, (@optic _.a))
        @test bordered_dFdp(prob_auto) ≈ fill(-ω * cos(ω * par.a), 3) rtol = 1.0e-10
        prob_fd = BifurcationProblem(F_fast, x, par, (@optic _.a); R01 = BK.FiniteDifferences())
        @test !isapprox(bordered_dFdp(prob_fd), fill(-ω * cos(ω * par.a), 3); rtol = 1.0e-10)
        # a derivative passed by the user is the one used
        prob_user = BifurcationProblem(F_fast, x, par, (@optic _.a); R01 = (x, par) -> fill(7.0, length(x)))
        @test bordered_dFdp(prob_user) == fill(7.0, 3)
    end

    @testset "Fold and Hopf continuation" begin
        # the CO oxidation model of `test/fold_codim_2/codim2.jl`, with its Hopf point 2 and point 3, from which a fold curve is continued
        function COm(u, p)
            (; q1, q2, q3, q4, q5, q6, k) = p
            x, y, s = u
            z = 1 - x - y - s
            out = similar(u, promote_type(eltype(u), eltype(q2), eltype(k)))
            out[1] = 2 * q1 * z^2 - 2 * q5 * x^2 - q3 * x * y
            out[2] = q2 * z - q6 * y - q3 * x * y
            out[3] = q4 * z - k * q4 * s
            out
        end
        par_com = (q1 = 2.5, q2 = 2.0, q3 = 10.0, q4 = 0.0675, q5 = 1.0, q6 = 0.1, k = 0.4)
        prob_com = BifurcationProblem(COm, [0.07, 0.2, 5.0], par_com, (@optic _.q2))
        opts_com = ContinuationPar(p_min = 0.6, p_max = 2.5, ds = 0.002, dsmax = 0.01, n_inversion = 4, detect_bifurcation = 3, max_bisection_steps = 25, nev = 2, max_steps = 20000)
        br_com = continuation(prob_com, PALC(), opts_com; normC = norminf, bothside = true)
        @test br_com.specialpoint[2].type == :hopf

        alg = PALC(; corrector = NonlinearSolveCorrector(exact))
        codim2 = (;
            bdlinsolver = MatrixBLS(),
            start_with_eigen = true,
            update_minaug_every_step = 1,
            detect_codim2_bifurcation = 2,
            jacobian_ma = BK.AutoDiff(),
        )
        opts_codim2 = ContinuationPar(opts_com, p_max = 1.0, p_min = 0.0, detect_bifurcation = 1, max_steps = 50, save_sol_every_step = 1, detect_event = 2)

        fold0 = continuation(br_com, 3, (@optic _.k), opts_codim2; codim2...)
        fold1 = continuation(br_com, 3, (@optic _.k), opts_codim2; alg, codim2...)
        # the same branch and the same Bogdanov-Takens point as the default corrector
        @test fold1.kind isa BK.FoldCont
        @test length(fold1.branch) == length(fold0.branch)
        @test maximum(abs, fold1.branch.param .- fold0.branch.param) < 1.0e-8
        @test [point.type for point in fold1.specialpoint] == [point.type for point in fold0.specialpoint]
        @test fold1.specialpoint[1].type == :bt
        @test fold1.specialpoint[1].param ≈ fold0.specialpoint[1].param rtol = 1.0e-5

        opts_hopf = ContinuationPar(opts_codim2, ds = -0.001)
        hopf0 = continuation(br_com, 2, (@optic _.k), opts_hopf; codim2...)
        hopf1 = continuation(br_com, 2, (@optic _.k), opts_hopf; alg, codim2...)
        # the same branch and the same generalized Hopf points as the default corrector
        @test hopf1.kind isa BK.HopfCont
        @test length(hopf1.branch) == length(hopf0.branch)
        @test maximum(abs, hopf1.branch.param .- hopf0.branch.param) < 1.0e-8
        @test [point.type for point in hopf1.specialpoint] == [point.type for point in hopf0.specialpoint]
        @test count(point -> point.type == :gh, hopf1.specialpoint) == 2
        for (point1, point0) in zip(hopf1.specialpoint, hopf0.specialpoint)
            @test point1.param ≈ point0.param rtol = 1.0e-5
        end
    end
end
