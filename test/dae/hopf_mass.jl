# Tests of the Hopf normal form for DAEMassBifProblem (constant mass matrix)
# and of the DAEMassBifProblem wrapper API.
using BifurcationKit, Test
import LinearAlgebra as LA
import ForwardDiff
const BK = BifurcationKit

# marker for a state dependent mass matrix (the default `ConstantMass` kind
# asserts that M does not depend on the state)
struct StateDependentMass <: BK.AbstractDAEMassType end

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
    # `dF` is the directional derivative (JVP) of the residual: it is forwarded
    # to the wrapped problem and matches the ForwardDiff derivative
    u = [0.2, -0.3]; du = [0.1, 0.5]
    @test BK.dF(daeprob, u, par_sl, du) ≈ BK.dF(prob, u, par_sl, du)
    @test BK.dF(daeprob, u, par_sl, du) ≈ ForwardDiff.derivative(t -> BK.residual(daeprob, u .+ t .* du, par_sl), 0)
end

let
    # Nonconstant mass matrix M(x, p) depending on both the state and the
    # parameters. The Hopf MA linearization must include the ∂ₓM (→ σxx) and
    # ∂ₚM (→ σₚ) contributions of the bordered scalar σ1. We check the analytic
    # linear solve `_hopf_MA_linear_solver` against the Jacobian of the MA
    # residual, for both the finite-differences and the Hessian branches.
    Mfun(x, p) = [(1 + p.r) + 0.1 * x[1]^2      0.05 * x[1] * x[2];
                  0.05 * x[1] * x[2]            (1 - p.r) + 0.2 * x[2]^2]

    # complex-capable second differential of Fsl2 (symmetric bilinear form)
    _jv(x, p, dx) = ForwardDiff.derivative(t -> Fsl2(x .+ t .* dx, p), 0.0)
    _d2real(x, p, dx1, dx2) = ForwardDiff.derivative(t -> _jv(x .+ t .* dx2, p, dx1), 0.0)
    d2Fcc(x, p, dx1, dx2) = _d2real(x, p, real.(dx1), real.(dx2)) .-
                            _d2real(x, p, imag.(dx1), imag.(dx2)) .+
                            im .* (_d2real(x, p, real.(dx1), imag.(dx2)) .+
                                   _d2real(x, p, imag.(dx1), real.(dx2)))

    # a generic point, not necessarily a Hopf point
    x0 = [0.07, -0.11]; p0 = -0.15; ω = 0.83
    a = ComplexF64[0.3 + 0.7im, -0.6 + 0.2im]
    b = ComplexF64[0.9 - 0.1im, 0.4 + 0.5im]
    duu = [0.4, -0.2]; dup = 0.13; duω = -0.07

    for usehess in (false, true)
        prob_ode = BK.ODEBifProblem(Fsl2, zeros(2), par_sl, (@optic _.r); R01 = BK.AutoDiff(), R11 = BK.AutoDiff(), d2F = d2Fcc)
        daeprob = BK.DAEMassBifProblem{StateDependentMass}(prob_ode, Mfun; R01 = BK.AutoDiff(), ∇xM = BK.AutoDiff())

        @test BK.is_mass_matrix_constant(daeprob) == false

        𝐇 = BK.HopfMinimallyAugmentedFormulation(daeprob, copy(a), copy(b),
                                                 BK.DefaultLS(), BK.MatrixBLS();
                                                 usehessian = usehess)
        @test 𝐇.usehessian == usehess

        par0 = par_sl
        X0 = vcat(x0, p0, ω)
        prob_h = BK.HopfMAProblem(𝐇)
        Jfwd = ForwardDiff.jacobian(Z -> BK.residual(prob_h, Z, par0), X0)
        _res_fd = Jfwd \ vcat(duu, dup, duω)
    
        dX, dp, dω = BK._hopf_MA_linear_solver(x0, p0, ω, 𝐇, par0, duu, dup, duω)
        # the analytic linear solve inverts the Jacobian of the MA residual
        @test Jfwd * vcat(dX, dp, dω) ≈ vcat(duu, dup, duω) rtol = 1e-6 atol = 1e-8
        @test norminf(_res_fd-vcat(dX, dp, dω)) < (usehess ? 1e-14 : 1e-7)

        # the analytic matrix based Jacobian `jacobian(..., MinAugMatrixBased)`
        # matches the auto-differentiated Jacobian of the MA residual
        prob_mb = BK.HopfMAProblem(𝐇, BK.MinAugMatrixBased(), nothing, nothing, nothing, nothing)
        Jan = BK.jacobian(prob_mb, X0, par0)
        @test Jan ≈ Jfwd rtol = 1e-6 atol = 1e-8
    end
end

let
    # Zero vector field: the bordered scalar σ1 then depends on x, p and ω only
    # through the mass matrix M(x, p). This isolates the mass contributions
    # ∂ₓσ, ∂ₚσ and ∂_ωσ of the bordered scalar.
    N = 2
    Mfun(x, p) = [(1 + p.r) + 0.1 * x[1]      0.05 * x[2];
                  0.05 * x[1]            (1 - 5p.r) + 0.2 * x[2]]
    F0(x, p) = zero(x)

    # Mfun(x, p) = LA.I(2)
    # F0(x, p) = [-x[1]+x[1]*x[2]*p.r, x[1]]

    prob0 = BK.ODEBifProblem(F0, zeros(N), par_sl, (@optic _.r), R01 = BK.AutoDiff(), R11 = BK.AutoDiff())
    dae0 = BK.DAEMassBifProblem{StateDependentMass}(prob0, Mfun; R01 = BK.FiniteDifferences(), ∇xM = BK.AutoDiff())
    @test BK.is_mass_matrix_constant(dae0) == false

    x0 = [0.07, -0.11]; p0 = -0.15; ω = 0.83
    a = ComplexF64[0.3 + 0.7im, -0.6 + 0.2im]
    b = ComplexF64[0.9 - 0.1im, 0.4 + 0.5im]
    par = par_sl
    X0 = vcat(x0, p0, ω)

    𝐇 = BK.HopfMinimallyAugmentedFormulation(dae0, copy(a), copy(b),
                                             BK.DefaultLS(), BK.MatrixBLS();
                                             usehessian = false)
    prob_h  = BK.HopfMAProblem(𝐇)
    prob_mb = BK.HopfMAProblem(𝐇, BK.MinAugMatrixBased(), nothing, nothing, nothing, nothing)

    Jfd = ForwardDiff.jacobian(Z -> BK.residual(prob_h, Z, par), X0)
    Jan = BK.jacobian(prob_mb, X0, par)
    # the last two rows are the (real, imag) parts of ∂ₓσ, ∂ₚσ, ∂_ωσ:
    #   columns 1:N        ←  ∂ₓσ   (state)
    #   column  N+1        ←  ∂ₚσ   (parameter)
    #   column  N+2        ←  ∂_ωσ  (frequency)
    # @info "" norminf(Jan[end-1:end, 1:N] - Jfd[end-1:end, 1:N])
    # @info "" norminf(Jan - Jfd) Jan - Jfd Jan
    @test Jan[end-1:end, N+1] ≈ Jfd[end-1:end, N+1] atol = 1e-8#rtol = 1e-5 atol = 1e-7
    @test Jan[end-1:end, N+2] ≈ Jfd[end-1:end, N+2] #rtol = 1e-5 atol = 1e-7
    @test Jan[end-1:end, 1:N] ≈ Jfd[end-1:end, 1:N] atol = 1e-7 #rtol = 1e-10

    # the mass derivative helpers, against central differences of the mass part
    (;M, σ1, σ2, v, w, ϵ2, par0) = BK._get_bordered_terms(𝐇, x0, p0, ω, par)

    lens = BK.getlens(𝐇)
    h = 1e-8

    # reference: iω⟨w,Mv⟩ - σ1⟨w,Ma⟩ - conj(σ2)conj⟨v,Mb⟩ evaluated with dM
    dσ_mass(dM) = Complex(0, ω) * BK.dot_with_mass(w, dM, v) -
                  σ1 * BK.dot_with_mass(w, dM, 𝐇.a) -
                  conj(σ2) * conj(BK.dot_with_mass(v, dM, 𝐇.b))

    # ∂ₓσ_mass is a vector
    σx_mass = BK._dₓσ_mass(dae0, x0, par0, 𝐇.a, 𝐇.b, v, w, σ1, σ2, ω)
    @test length(σx_mass) == N
    for i in 1:N
        eᵢ = zeros(N); eᵢ[i] = 1.0
        dM = (BK.getmassmatrix(dae0, x0 .+ h .* eᵢ, par0) -
              BK.getmassmatrix(dae0, x0 .- h .* eᵢ, par0)) / (2h)
        @test σx_mass[i] ≈ dσ_mass(dM) rtol = 1e-4
    end

    # ∂ₚσ_mass
    dMp = (BK.getmassmatrix(dae0, x0, set(par0, lens, p0 + h)) -
           BK.getmassmatrix(dae0, x0, set(par0, lens, p0 - h))) / (2h)
    @test BK._dₚσ_mass(dae0, x0, par0, 𝐇.a, 𝐇.b, v, w, σ1, σ2, ω) ≈ dσ_mass(dMp) rtol = 1e-4

    # the mass derivative interfaces (defaults: finite differences / ForwardDiff)
    for i in 1:N
        eᵢ = zeros(N); eᵢ[i] = 1.0
        ref = (BK.dot_with_mass(w, BK.getmassmatrix(dae0, x0 .+ h .* eᵢ, par0), v) -
               BK.dot_with_mass(w, BK.getmassmatrix(dae0, x0 .- h .* eᵢ, par0), v)) / (2h)
        @test BK.∇_x_mass_matrix(dae0, x0, par0, v, w)[i] ≈ ref rtol = 1e-4
    end
    refp = (BK.dot_with_mass(w, BK.getmassmatrix(dae0, x0, set(par0, lens, p0 + h)), v) -
            BK.dot_with_mass(w, BK.getmassmatrix(dae0, x0, set(par0, lens, p0 - h)), v)) / (2h)
    @test BK.R01_mass_matrix(dae0, x0, par0, v, w) ≈ refp rtol = 1e-4

    # ∂_ω of the mass part of σ1 is i⋅⟨w, M v⟩
    σ_mass_ω(z) = Complex(0, z) * BK.dot_with_mass(w, M, v) -
                  σ1 * BK.dot_with_mass(w, M, 𝐇.a) -
                  conj(σ2) * conj(BK.dot_with_mass(v, M, 𝐇.b))
    @test Complex{eltype(ω)}(0, 1) * BK.dot_with_mass(w, M, v) ≈
          (σ_mass_ω(ω + h) - σ_mass_ω(ω - h)) / (2h) rtol = 1e-4
end


