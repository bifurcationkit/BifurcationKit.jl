# using Revise
using Test
using BifurcationKit, LinearAlgebra
const BK = BifurcationKit

let
    F(x, p) = @. p.r * x + x^3 / 3 + 0.01
    J(x, p) = diagm(0 => p.r .+ x .^ 2)
    # a wrong `J`: a corrector that read it would need more iterations, and the step control would change `ds`
    wrong_J(x, p) = 1.5 .* J(x, p)
    FJ(x, p) = (F(x, p), J(x, p))

    par = (r = -1.0,)
    u0 = zeros(3)
    opts = ContinuationPar(p_min = -1.0, p_max = 1.0, ds = 0.05, max_steps = 30, detect_bifurcation = 0, newton_options = NewtonPar(tol = 1e-10))
    separate = BifurcationProblem(F, u0, par, (@optic _.r); J)
    fused = BifurcationProblem(F, u0, par, (@optic _.r); J = wrong_J, FJ)

    @test ~BK.has_FJ(separate)
    @test BK.has_FJ(fused)
    x = rand(3)
    @test all(BK.residual_and_jacobian(separate, x, par) .== (F(x, par), J(x, par)))
    @test all(BK.residual_and_jacobian(fused, x, par) .== (F(x, par), J(x, par)))

    # the corrector's jacobians of the fused run come from FJ, the true one
    br_separate = continuation(separate, PALC(tangent = Secant()), opts)
    br_fused = continuation(fused, PALC(tangent = Secant()), opts)
    @test length(br_fused) == length(br_separate)
    @test br_fused.param ≈ br_separate.param
    @test all(norm(a.x - b.x) < 1e-10 for (a, b) in zip(br_fused.sol, br_separate.sol))
end
