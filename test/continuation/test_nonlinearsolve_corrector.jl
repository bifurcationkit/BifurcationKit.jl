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
const BK = BifurcationKit
####################################################################################################
# Bratu problem u'' + λ exp(u) = 0, which has a fold at λ ≈ 3.51
function F_bratu(x, p)
    n = length(x)
    h = 1 / (n + 1)
    out = similar(x)
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
    bordered::Bool
    λ::Float64
    builds::Base.RefValue{Int}
end

function (preconditioner::DenseLUPreconditioner)(A, ::Any)
    w = A.u
    x = w
    λ = preconditioner.λ
    if preconditioner.bordered
        x = w[begin:end - 1]
        λ = w[end]
    end
    preconditioner.builds[] += 1
    return (lu(BK.corrector_matrix(A.p, w, Matrix(J_bratu(x, (; λ))))), I)
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
                TrustRegion(; linsolve = LUFactorization(), jacobian_reuse = false),
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

    @testset "a corrector ending outside [p_min, p_max] has not converged" begin
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
            linsolve = KrylovJL_GMRES(; precs = DenseLUPreconditioner(false, 1.0, builds)),
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
                    precs = DenseLUPreconditioner(true, 0.0, builds),
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
end
