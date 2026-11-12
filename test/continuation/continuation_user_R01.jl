# using Revise
using Test
using BifurcationKit, LinearAlgebra
const BK = BifurcationKit

let
    F(x, p) = @. x^3 / 3 + x - p.r^2
    dFdr(x, p) = fill(-2p.r, length(x))
    par = (r = 0.1,)
    u0 = fill(0.01, 3)

    exact = BifurcationProblem(F, u0, par, (@optic _.r); R01 = dFdr)
    differenced = BifurcationProblem(F, u0, par, (@optic _.r); R01 = BK.FiniteDifferences(), delta = 1e-2)
    x = rand(3)
    @test BK._dFdp(exact, x, par) == dFdr(x, par)
    @test BK._dFdp(differenced, x, par) ≈ (F(x, (r = par.r + 1e-2,)) - F(x, par)) / 1e-2
    @test BK._dFdp(differenced, x, par) != dFdr(x, par)

    # a finite difference of step `delta = 10` is far off `∂F/∂r`; the passed `R01` is used instead
    reference = BifurcationProblem(F, u0, par, (@optic _.r); R01 = BK.FiniteDifferences(), delta = 1e-8)
    coarse = BifurcationProblem(F, u0, par, (@optic _.r); R01 = dFdr, delta = 10.0)
    opts = ContinuationPar(p_min = 0.0, p_max = 1.0, ds = 0.05, max_steps = 40, detect_bifurcation = 0, newton_options = NewtonPar(tol = 1e-10))
    for tangent in (Secant(), Bordered())
        br_reference = continuation(reference, PALC(; tangent), opts)
        br_coarse = continuation(coarse, PALC(; tangent), opts)
        @test length(br_coarse) == length(br_reference)
        @test br_coarse.param ≈ br_reference.param rtol = 1e-6
    end
    for method in (BK.direct, BK.iterative)
        br_reference = continuation(reference, MoorePenrose(; method), opts)
        br_coarse = continuation(coarse, MoorePenrose(; method), opts)
        @test length(br_coarse) == length(br_reference)
        @test br_coarse.param ≈ br_reference.param rtol = 1e-6
    end
end
