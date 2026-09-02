# Tests of the Hopf normal form for DAEMassBifProblem (constant mass matrix)
# and of the DAEMassBifProblem wrapper API.
using BifurcationKit, Test
import LinearAlgebra as LA
const BK = BifurcationKit

function Fsl2(x, p)
    (; r, μ, ν, c3, c5) = p
    u1, u2 = x[1], x[2]
    ua = u1^2 + u2^2
    return [r * u1 - ν * u2 - ua * (c3 * u1 - μ * u2) - c5 * ua^2 * u1,
            r * u2 + ν * u1 - ua * (c3 * u2 + μ * u1) - c5 * ua^2 * u2]
end

par_sl = (r = -0.1, μ = 0.132, ν = 1.0, c3 = 1.123, c5 = 0.2)

# locate a Hopf point of the (wrapped) problem and compute its normal form
function hopf_nf(prob; start_with_eigen = Val(true))
    opts = ContinuationPar(dsmin = 0.001, dsmax = 0.02, ds = 0.01,
        p_max = 0.1, p_min = -0.3, detect_bifurcation = 3, n_inversion = 10)
    br = BK.continuation(prob, PALC(), opts; normC = norminf)
    hp = BK.hopf_normal_form(prob, br, 1; start_with_eigen)
    return hp.ω, hp.nf.a, hp.nf.b
end

let
    # identity mass matrix must reproduce the ODE normal form exactly:
    # for Fsl2 the (unscaled) coefficients are ω = 1, a = 1 and b/2 = -c3 + i⋅μ
    prob_ode = BK.ODEBifProblem(Fsl2, zeros(2), par_sl, (@optic _.r))
    daeprob = BK.DAEMassBifProblem(prob_ode, LA.I)
    ω, a, b = hopf_nf(daeprob; start_with_eigen = Val(false))
    @test ω ≈ 1 atol = 1e-9
    @test a ≈ 1 atol = 1e-8
    @test b / 2 ≈ (-par_sl.c3 + im * par_sl.μ) atol = 1e-6
end

let
    # M = α⋅I: the (generalized) Hopf frequency is divided by α and the
    # bifurcation remains supercritical. Coefficients a/b are NOT asserted
    # here (mass normalization is work in progress, see the warning below).
    prob_ode = BK.ODEBifProblem(Fsl2, zeros(2), par_sl, (@optic _.r))
    α = 2.0
    daeprob = BK.DAEMassBifProblem(prob_ode, α * LA.I(2))
    ω, a, b = hopf_nf(daeprob; start_with_eigen = Val(false))
    @test 2ω ≈ 1 atol = 1e-9
    @test 2a ≈ 1 atol = 1e-8
    @test 2(b / 2) ≈ (-par_sl.c3 + im * par_sl.μ) atol = 1e-6
end

let
    # `start_with_eigen = Val(true)` is not supported for a DAE problem
    prob_ode = BK.ODEBifProblem(Fsl2, zeros(2), par_sl, (@optic _.r))
    daeprob = BK.DAEMassBifProblem(prob_ode, LA.I(2))
    @test_throws ErrorException hopf_nf(daeprob; start_with_eigen = Val(true))
end

let
    # DAEMassBifProblem wrapper: mass matrix accessors & re_make
    B = [1. 0.5; 0. 0.]
    prob = BK.ODEBifProblem(Fsl2, zeros(2), par_sl, (@optic _.r))
    daeprob = BK.DAEMassBifProblem(prob, B)
    @test BK.getmassmatrix(daeprob, zeros(2), par_sl) === B
    # UniformScaling constant mass is supported
    daeI = BK.DAEMassBifProblem(prob, LA.I)
    @test BK.getmassmatrix(daeI, zeros(2), par_sl) ≈ LA.I(2)
    # function (parameter dependent) mass matrix
    Bfun = (x, p) -> (p.r + 1) .* LA.I(2)
    daefun = BK.DAEMassBifProblem(prob, Bfun)
    @test BK.getmassmatrix(daefun, zeros(2), par_sl) ≈ (par_sl.r + 1) .* LA.I(2)
    # re_make replaces the mass matrix and forwards the usual keywords
    daeprob2 = BK.re_make(daeprob; M = 2 .* LA.I(2), u0 = [1.0, -1.0])
    @test BK.getu0(daeprob2) == [1.0, -1.0]
    @test BK.getmassmatrix(daeprob2, zeros(2), par_sl) ≈ 2 .* LA.I(2)
    @test BK.getlens(daeprob2) == BK.getlens(prob)
end
