# Test of the matrix-free / iterative (linear solver + eigensolver) path for the
# Hopf normal form of a DAEMassBifProblem with a constant mass matrix.
# The results are compared against the dense reference (DefaultLS + DefaultEig).
using BifurcationKit, Test
import LinearAlgebra as LA
import ForwardDiff as FwD
const BK = BifurcationKit

# vector field Fsl2 (Hopf in (u1, u2)) padded with stable real modes. The state
# dimension is large enough for the iterative eigensolver: the detection routine
# requests max(n_unstable + 5, nev) eigenvalues, so a shift-invert Arnoldi solve
# on the pencil (J, M) needs dimension ≳ 7.
function Fsl2_3(x, p)
    (; r, μ, ν, c3) = p
    u1, u2 = x[1], x[2]
    ua = u1^2 + u2^2
    out = [r * u1 - ν * u2 - ua * (c3 * u1 - μ * u2),
           r * u2 + ν * u1 - ua * (c3 * u2 + μ * u1)]
    for i in 3:length(x)
        push!(out, -i * x[i]) # stable modes -3, -4, ...
    end
    return out
end

function run_hopf_dae_mf(ls, eigls; verbosity = 0, N = 9, mf = false)
    par_sl = (r = -0.1, μ = 0.132, ν = 1.0, c3 = 1.123)
    # constant (diagonal) mass matrix, invertible
    if ~mf
        Mass = LA.Diagonal([2.0; ones(N - 1)])
        prob_ode = BK.ODEBifProblem(Fsl2_3, zeros(N), par_sl, (@optic _.r))
        daeprob = BK.DAEMassBifProblem(prob_ode, Mass)
    else
        prob_ode = BK.ODEBifProblem(Fsl2_3, zeros(N), par_sl, (@optic _.r); 
            J = (x,p) -> (dx -> BK._jvp_fwd(Fsl2_3, x,p,dx) ),
            Jᵗ = (x,p) -> (dx -> FwD.jacobian(z->Fsl2_3(z,p),x)' * dx),
            )
        daeprob = BK.DAEMassBifProblem(prob_ode,  LA.Diagonal([2.0; ones(N - 1)]))
    end
    opts = ContinuationPar(
        dsmin = 0.001, dsmax = 0.02, ds = 0.01,
        p_max = 0.1, p_min = -0.3,
        detect_bifurcation = 3, n_inversion = 10, nev = 2,
        newton_options = NewtonPar(tol = 1e-9, max_iterations = 40,
                                   linsolver = ls, eigsolver = eigls))
    br = BK.continuation(daeprob, PALC(), opts; verbosity, normC = BK.norminf)

    ind_h = findfirst(pt -> pt.type == :hopf, br.specialpoint)
    @test ind_h !== nothing

    hp = if mf == false
            BK.hopf_normal_form(daeprob, br, ind_h; start_with_eigen = Val(false))
        else
            BK.hopf_normal_form(daeprob, br, ind_h; start_with_eigen = Val(false), bls = BorderingBLS(ls) )
        end
    return (;
            p_hopf = br.specialpoint[ind_h].param,
            hp,
            ω = hp.ω,
            a = hp.nf.a,
            b = hp.nf.b)
end

let
    # dense reference
    ref = run_hopf_dae_mf(DefaultLS(), DefaultEig(); mf = false)
    @test ref.p_hopf ≈ ref.p_hopf atol = 1e-4
    @test ref.ω ≈ ref.ω rtol = 1e-6
    @test ref.a ≈ ref.a rtol = 1e-5
    @test ref.b ≈ ref.b rtol = 1e-5

    # sanity check against the mass-scaled normal form: the frequency of the pair
    # for a diagonal mass with entries (m1, m2) = (2, 1) is ω = ν / √(m1·m2) = 1/√2
    @test ref.ω ≈ 1 / √2 atol = 1e-8

    # matrix-free / iterative path: iterative linear solver (GMRES) and iterative
    # eigensolver on the generalized problem (J, M)
    mf = run_hopf_dae_mf(GMRESIterativeSolvers(reltol = 1e-12), EigArpack(;v0 = rand(9)); mf = true)

    @test mf.p_hopf ≈ ref.p_hopf atol = 1e-4
    @test mf.ω ≈ ref.ω rtol = 1e-6
    @test mf.a ≈ ref.a rtol = 1e-5
    @test mf.b ≈ ref.b rtol = 1e-5

    # sanity check against the mass-scaled normal form: the frequency of the pair
    # for a diagonal mass with entries (m1, m2) = (2, 1) is ω = ν / √(m1·m2) = 1/√2
    @test ref.ω ≈ 1 / √2 atol = 1e-8
end

