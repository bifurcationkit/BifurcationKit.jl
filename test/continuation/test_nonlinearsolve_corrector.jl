using Test
using BifurcationKit, LinearAlgebra
using NonlinearSolve: NewtonRaphson, TrustRegion, JacobianReuse, LUFactorization
const BK = BifurcationKit
####################################################################################################
# Bratu problem u'' + λ exp(u) = 0, which has a fold at λ ≈ 3.51
function F_bratu(x, p)
    n = length(x)
    h = 1 / (n + 1)
    out = similar(x)
    for i in eachindex(x)
        left = i == 1 ? zero(eltype(x)) : x[i-1]
        right = i == n ? zero(eltype(x)) : x[i+1]
        out[i] = (left - 2x[i] + right) / h^2 + p.λ * exp(x[i])
    end
    out
end

prob_bratu = BifurcationProblem(F_bratu, zeros(20), (λ = 0.01,), (@optic _.λ))
opts_bratu = ContinuationPar(p_min = 0., p_max = 4., ds = 0.05, dsmax = 0.2, max_steps = 60, detect_bifurcation = 0, newton_options = NewtonPar(tol = 1e-10))

@testset "NonlinearSolveCorrector" begin
    br_default = continuation(prob_bratu, PALC(), opts_bratu)

    exact = NewtonRaphson(; linsolve = LUFactorization(), jacobian_reuse = false)

    @testset "PALC passes the fold and follows the same branch" begin
        # exact Newton steps: the same corrector iterates as the default one
        br = continuation(prob_bratu, PALC(; corrector = NonlinearSolveCorrector(exact)), opts_bratu)
        @test length(br.sol) == length(br_default.sol)
        @test maximum(abs, br.branch.param .- br_default.branch.param) < 1e-8
        @test maximum(point -> norm(point[1].x - point[2].x, Inf), zip(br.sol, br_default.sol)) < 1e-8
        # the branch goes beyond the fold, p decreases after it
        @test any(diff(br.branch.param) .< 0)
    end

    @testset "other algorithms stay on the branch" begin
        # a modified Newton method (Jacobian reuse) or a trust region takes other steps: the points still solve F = 0 on the same branch
        for alg in (NewtonRaphson(; linsolve = LUFactorization(), jacobian_reuse = JacobianReuse()),
                    TrustRegion(; linsolve = LUFactorization(), jacobian_reuse = false))
            br = continuation(prob_bratu, PALC(; corrector = NonlinearSolveCorrector(alg)), opts_bratu)
            @test all(point -> norm(F_bratu(point.x, (λ = point.p,)), Inf) < 1e-8, br.sol)
            @test any(diff(br.branch.param) .< 0)
            @test maximum(point -> point.p, br.sol) ≈ maximum(point -> point.p, br_default.sol) atol = 0.02
        end
    end

    @testset "Newton solve" begin
        prob = re_make(prob_bratu; params = (λ = 1.,), u0 = fill(0.1, 20))
        sol0 = BK.solve(prob, Newton(), NewtonPar(tol = 1e-12))
        sol1 = BK.solve(prob, NonlinearSolveCorrector(exact), NewtonPar(tol = 1e-12))
        @test BK.converged(sol1)
        @test norm(sol0.u - sol1.u, Inf) < 1e-10
    end

    @testset "Natural continuation" begin
        opts = ContinuationPar(opts_bratu; p_max = 3., max_steps = 20)
        br0 = continuation(prob_bratu, Natural(), opts)
        br1 = continuation(prob_bratu, Natural(; corrector = NonlinearSolveCorrector(exact)), opts)
        @test length(br1.sol) == length(br0.sol)
        @test maximum(point -> norm(point[1].x - point[2].x, Inf), zip(br1.sol, br0.sol)) < 1e-8
    end

    @testset "a corrector ending outside [p_min, p_max] has not converged" begin
        # the first step goes to p = 0.01 + 0.5 > p_max
        opts = ContinuationPar(opts_bratu; p_max = 0.2, ds = 0.5, dsmax = 0.5, max_steps = 5)
        br = continuation(prob_bratu, PALC(; corrector = NonlinearSolveCorrector(exact)), opts)
        @test all(br.branch.param .<= 0.2)
    end

    @testset "matrix-free Jacobians are rejected" begin
        prob = BifurcationProblem(F_bratu, zeros(5), (λ = 0.1,), (@optic _.λ); J = (x, p) -> (dx -> dx))
        @test_throws ArgumentError BK.solve(prob, NonlinearSolveCorrector(NewtonRaphson()), NewtonPar())
    end
end
