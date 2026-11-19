using Test, BifurcationKit
const BK = BifurcationKit
####################################################################################################
# a predictor past p_max is shortened along its line to the bound. Corrected at fixed λ from a full step out, it
# would converge onto the parallel branch x = λ + 3
let
    F(x, p) = (x .- p.λ) .* (x .- p.λ .- 3)
    prob = BifurcationProblem(F, [0.0], (λ = 0.0,), (@optic _.λ))
    options = ContinuationPar(ds = 4.0, dsmax = 4.0, p_min = -1.0, p_max = 1.0, detect_fold = false, detect_bifurcation = 0)
    for tangent in (Secant(), Bordered())
        br = continuation(prob, PALC(; tangent), options)
        @test br.branch.param[end] == 1.0
        @test br.branch.x[end] ≈ 1.0
    end
end

# a run that starts on a bound shortens its first predictor too
let
    F(x, p) = (x .- p.λ) .* (x .- p.λ .- 3)
    prob = BifurcationProblem(F, [0.0], (λ = 0.0,), (@optic _.λ))
    options = ContinuationPar(ds = 4.0, dsmax = 4.0, p_min = 0.0, p_max = 1.0, detect_fold = false, detect_bifurcation = 0)
    for tangent in (Secant(), Bordered())
        br = continuation(prob, PALC(; tangent), options)
        @test br.branch.param[end] == 1.0
        @test br.branch.x[end] ≈ 1.0
    end
end
