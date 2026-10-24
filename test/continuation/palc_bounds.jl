using Test, BifurcationKit, LinearAlgebra
const BK = BifurcationKit
####################################################################################################
# u^3 - u - (p - 1) = 0 has two folds, p ≈ 0.615 and p ≈ 1.385, inside p ∈ [0, 2]. F throws outside
# the box, so a corrector that evaluates it there fails. The sheet between the folds is |u| < 1/√3:
# a corrector that clamps p back into the box jumps from u ≈ -0.92 to u ≈ 0.92 and never visits it.
let
    function F(u, p)
        if p.λ < 0 || p.λ > 2
            error("F evaluated outside [p_min, p_max] at λ = $(p.λ)")
        end
        return @. u^3 - u - (p.λ - 1)
    end
    prob = BifurcationProblem(F, [-1.3247179], (λ = 0.0,), (@optic _.λ))
    for linesearch in (false, true)
        opts = ContinuationPar(ds = 0.05, dsmin = 1e-6, dsmax = 0.5, p_min = 0.0, p_max = 2.0, max_steps = 200,
                               detect_fold = true, detect_bifurcation = 0,
                               newton_options = NewtonPar(tol = 1e-8, linesearch = linesearch))
        br = continuation(prob, PALC(), opts)
        @test any(point -> abs(point.x[1]) < 1 / sqrt(3), br.sol)
        @test all(point -> norm(F(point.x, (λ = point.p,)), Inf) < 1e-7, br.sol)
        @test count(sp -> sp.type == :fold, br.specialpoint) == 2
    end
end
