using BifurcationKit, Test
import LinearAlgebra as LA
import ForwardDiff
const BK = BifurcationKit

let
    F_dae_fold(x, pars) = [x[2], pars.p1 + pars.p2 * x[1] - x[1]^2 + x[2]]
    pars = (p1 = 0.01, p2 = -0.01)
    B = [1. 0; 0. 0]
    bifprob = ODEBifProblem(F_dae_fold, zeros(2), pars, (@optic _.p1); record_from_solution = (x,p;k...) -> x[1])
    opts_br = BK.ContinuationPar(p_min = -0.5, p_max = 0.5, ds = 0.01, dsmax = 0.05, nev = 2)
    daeproblem = BK.DAEMassBifProblem(bifprob, B)
    br = BK.continuation(daeproblem, BK.PALC(), opts_br; normC = BK.norminf, verbosity = 0, bothside = true)
    # get_normal_form(br, 2; start_with_eigen = Val(false))
    # plot(br)
    brfold = continuation(br, 2, (@optic _.p2); bothside = true)
    # plot(brfold)
end

let
    F_dae_bt(x, pars) = [x[2], pars.p1 + pars.p2 * x[2] - x[1]^2 + x[1]*x[2], x[3]-x[2]]
    pars = (p1 = 0.01, p2 = -0.1)
    B = zeros(3,3);B[1,1]=1;B[2,2]=1
    bifprob = ODEBifProblem(F_dae_bt, rand(3), pars, (@optic _.p1); record_from_solution = (x,p;k...) -> x[1])
    opts_br = BK.ContinuationPar(p_min = -0.5, p_max = 0.5, ds = 0.01, dsmax = 0.05, nev = 2)
    daeproblem = BK.DAEMassBifProblem(bifprob, B)
    br = BK.continuation(daeproblem, BK.PALC(), opts_br; normC = BK.norminf, verbosity = 0, bothside = true)
    # get_normal_form(br, 3)
    # plot(br)
    # locate the Fold and Hopf points of the equilibrium branch by their type:
    # their order in `br.specialpoint` is not deterministic (random initial guess)
    ind_fold = findfirst(pt -> pt.type in (:fold, :bp, :nd), br.specialpoint)
    ind_hopf = findfirst(pt -> pt.type == :hopf, br.specialpoint)
    @test ind_fold !== nothing
    @test ind_hopf !== nothing

    brfold = continuation(br, ind_fold, (@optic _.p2), ContinuationPar(br.contparams, detect_bifurcation = 0 ); bothside = true)
    @test :bt in [pt.type for pt in brfold.specialpoint]
    brhopf = continuation(br, ind_hopf, (@optic _.p2); bothside = true)
    @test :bt in [pt.type for pt in brhopf.specialpoint]

    # the Bogdanov-Takens normal form (from a branch) is not implemented for DAE.
    # `nev` is passed explicitly so that the default `eigenvalsfrombif` is not
    # evaluated (the fold curve does not store eigen-elements) and the guard in
    # `bogdanov_takens_normal_form` is reached.
    ind_bt = findfirst(pt -> pt.type == :bt, brfold.specialpoint)
    @test ind_bt !== nothing
    @test_throws "Constant DAE not taken into account!" get_normal_form(brfold, ind_bt; nev = 2)

    ind_bt_h = findfirst(pt -> pt.type == :bt, brhopf.specialpoint)
    @test ind_bt_h !== nothing
    @test_throws "Constant DAE not taken into account!" get_normal_form(brhopf, ind_bt_h; nev = 2)
    # plot(brfold, brhopf)

    #############################################
    # Hopf MA problem: analytic jacobian of `MinAugMatrixBased`
    # (a) the matrix based Hopf continuation detects the same BT point as the
    #     default (AutoDiff) one
    brhopf_mb = continuation(br, ind_hopf, (@optic _.p2);
                bothside = true,
                jacobian_ma = BK.MinAugMatrixBased())
    @test :bt in [pt.type for pt in brhopf_mb.specialpoint]
    bt_ad  = filter(pt -> pt.type == :bt, brhopf.specialpoint)
    bt_mb  = filter(pt -> pt.type == :bt, brhopf_mb.specialpoint)
    @test !isempty(bt_ad) && !isempty(bt_mb)
    @test bt_mb[1].param ≈ bt_ad[1].param atol = 1e-3

    # (b) the analytic jacobian `jacobian(prob, X, par)` of the Hopf MA problem
    #     with `MinAugMatrixBased` matches the auto-differentiated jacobian of the
    #     MA residual (this is what `jacobian_ma = AutoDiff()` computes)
    ind_h = findfirst(pt -> pt.type == :hopf, br.specialpoint)
    @test ind_h !== nothing
    bifpt = br.specialpoint[ind_h]
    ω = imag(br.eig[bifpt.idx].eigenvals[bifpt.ind_ev])
    X0 = vcat(bifpt.x, bifpt.param, ω) # MA state (x, p1, ω)
    par0 = set(BK.getparams(br), (@optic _.p1), bifpt.param)

    prob_mb = brhopf_mb.prob
    J_mb = BK.jacobian(prob_mb, X0, par0)
    J_ad = ForwardDiff.jacobian(X -> BK.residual(prob_mb, X, par0), X0)
    # the analytic jacobian uses finite differences internally (step δ) for the
    # derivatives of the bordered quantities; close to the BT point the σ-rows
    # are ill-conditioned, so we use a loose tolerance
    @test J_mb ≈ J_ad rtol = 5e-2 atol = 1e-6
end

# function to record information from the solution
recordFromSolution(x, p; k...) = (u1 = BK.norminf(x), x1 = x[1], x2 = x[2], x3 = x[3], x4 = x[4])
# vector field
f(x, p) = p.Is * (exp(p.q * x) - 1)
IE(x1, x2, p) = -f(x2, p) + f(x1, p) / p.αF
IC(x1, x2, p) = f(x2, p)/ p.αR - f(x1, p)
function Colpitts!(dz, z, p, t = 0)
    (;C1, C2, L, R, Is, q, αF, αR, V, μ) = p
    x1, x2, x3, x4 = z
    dz[1] = (x1 - V) / R + IE(x1, x2, p)
    dz[2] = x3 + IC(x1, x2, p)
    dz[3] = -x3-x4
    dz[4] = -μ+x2
    dz
end

let
# parameter values
par_Colpitts = (C1 = 1.0, C2 = 1.0, L = 1.0, R = 1/4., Is = 1e-16, q = 40., αF = 0.99, αR = 0.5, μ = 0.5, V = 6.)
# initial condition
z0 = [0.9957,0.7650,19.81,-19.81]
# mass matrix
Be(x, pars) = [-(pars.C1+pars.C2) pars.C2 0 0;pars.C2 -pars.C2 0 0;pars.C1 0 0 0; 0 0 pars.L 0]
# we group the differentials together
bifprob = ODEBifProblem(Colpitts!, z0, par_Colpitts, (@optic _.μ); record_from_solution = recordFromSolution)
daeproblem = BK.DAEMassBifProblem(bifprob, Be)
opts_br = BK.ContinuationPar(p_min = -0.4, p_max = 6.8, ds = 0.01, dsmax = 0.05, nev = 4, plot_every_step = 3, max_steps = 1000)
# opts_br = @set opts_br.newton_options.verbose = true
br = BK.continuation(daeproblem, BK.PALC(), opts_br; normC = BK.norminf, verbosity = 0, bothside = true)
# plot(br, vars = (:param, :x1)) |> display
hp = BK.hopf_normal_form(daeproblem, br, 2; start_with_eigen = Val(false))
@test hp isa BK.Hopf
@test hp.ω > 0
@test hp.type in (:SuperCritical, :SubCritical, :Singular)

brhopf = continuation(br, 2, (@optic _.C1), ContinuationPar(BK.getcontparams(br), p_max = 10., max_steps = 50, dsmax = 0.01);
            start_with_eigen = false,
            # verbosity = 2,
            detect_codim2_bifurcation = 2,
            bothside = true,
            # jacobian_ma = BK.AutoDiff(),
            linear_algo = MatrixBLS(),
            # bdlinsolver = MatrixBLS(),
            )
# plot(brhopf)
end