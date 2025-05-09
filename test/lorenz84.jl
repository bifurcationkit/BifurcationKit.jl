# using Revise
using Test, ForwardDiff, LinearAlgebra
# using Plots
using BifurcationKit, Test
const BK = BifurcationKit

####################################################################################################
function Lor(u, p, t = 0)
    (;α,β,γ,δ,G,F,T) = p
    X,Y,Z,U = u
    [
        -Y^2 - Z^2 - α*X + α*F - γ*U^2,
        X*Y - β*X*Z - Y + G,
        β*X*Y + X*Z - Z,
        -δ*U + γ*U*X + T
    ]
end

parlor = (α = 1//4, β = 1, G = .25, δ = 1.04, γ = 0.987, F = 1.7620532879639, T = .0001265)

opts_br = ContinuationPar(p_min = -1.5, p_max = 3.0, ds = 0.001, dsmax = 0.025,
    # options to detect codim 1 bifurcations using bisection
    detect_bifurcation = 3,
    # Optional: bisection options for locating bifurcations
    n_inversion = 6, max_bisection_steps = 25,
    # number of eigenvalues
    nev = 4, max_steps = 252)

@reset opts_br.newton_options.max_iterations = 25

z0 =  [2.9787004394953343, -0.03868302503393752,  0.058232737694740085, -0.02105288273117459]

recordFromSolutionLor(u::AbstractVector, p; k...) = (X = u[1], Y = u[2], Z = u[3], U = u[4])
recordFromSolutionLor(u::BorderedArray, p; k...) = recordFromSolutionLor(u.u, p)

prob = BK.BifurcationProblem(Lor, z0, parlor, (@optic _.F);
    record_from_solution = recordFromSolutionLor,)

br = @time continuation(re_make(prob, params = setproperties(parlor;T=0.04,F=3.)),
     PALC(tangent = Bordered()),
    opts_br;
    normC = norminf,
    bothside = true)

@test br.alg.tangent isa Bordered
@test br.alg.bls isa MatrixBLS

@test prod(br.specialpoint[2].interval .≈ (2.8598634135619982, 2.859897757930758))
@test prod(br.specialpoint[3].interval .≈ (2.467211879219629, 2.467246154619121))
@test prod(br.specialpoint[4].interval .≈ (1.619657484413436, 1.6196654620692468))
@test prod(br.specialpoint[5].interval .≈ (1.5466483726208073, 1.5466483727182652))
####################################################################################################
# this part is for testing the spectrum
@reset opts_br.newton_options.verbose = false

# be careful here, Bordered predictor not good for Fold continuation
sn_codim2_test = continuation((@set br.alg.tangent = Secant()), 5, (@optic _.T), ContinuationPar(opts_br, p_max = 3.2, p_min = -0.1, detect_bifurcation = 1, dsmin=1e-5, ds = -0.001, dsmax = 0.005, n_inversion = 8, save_sol_every_step = 1, max_steps = 60) ;
    normC = norminf,
    detect_codim2_bifurcation = 1,
    update_minaug_every_step = 1,
    start_with_eigen = true,
    record_from_solution = recordFromSolutionLor,
    bdlinsolver = MatrixBLS(),
    )

# plot(sn_codim2_test)

@test sn_codim2_test.specialpoint[1].param ≈ +0.02058724 rtol = 1e-5
@test sn_codim2_test.specialpoint[2].param ≈ +0.00004983 atol = 1e-8
@test sn_codim2_test.specialpoint[3].param ≈ -0.00045281 rtol = 1e-5
@test sn_codim2_test.specialpoint[4].param ≈ -0.02135893 rtol = 1e-5

@test sn_codim2_test.eig[1].eigenvecs != nothing

hp_codim2_test = continuation(br, 2, (@optic _.T), ContinuationPar(opts_br, ds = -0.001, dsmax = 0.02, dsmin = 1e-4, n_inversion = 6, save_sol_every_step = 1, max_steps = 100) ;
    normC = norminf,
    detect_codim2_bifurcation = 2,
    update_minaug_every_step = 1,
    start_with_eigen = true,
    record_from_solution = recordFromSolutionLor,
    bothside = true,
    bdlinsolver = MatrixBLS())

@test hp_codim2_test.specialpoint[2].param ≈ +0.02627393 rtol = 1e-5
@test hp_codim2_test.specialpoint[3].param ≈ -0.02627430 atol = 1e-8

@test hp_codim2_test.eig[1].eigenvecs != nothing

####################################################################################################
"""
This function test if the eigenvalues are well computed during a branch of Hopf/Fold.
This test works if the problem is not updated because we dont record prob.a and prob.b
"""
function testEV(br, verbose = false)
    verbose && println("\n\n\n"*"="^50)
    prob_ma = br.prob.prob
    prob_vf = prob_ma.prob_vf
    eig = DefaultEig()
    lens1 = BK.getlens(br)
    lens2 = prob.lens
    par0 = BK.getparams(br)
    ϵ = br.contparams.newton_options.tol

    for (ii, pt) in enumerate(br.branch)
        # we make sure the parameters are set right
        step = pt.step
        verbose && (println("="^50); @info step ii)
        x0 = BK.getvec(br.sol[ii].x, prob_ma)
        p0 = BK.getp(br.sol[ii].x, prob_ma)[1]
        if prob_ma isa HopfProblemMinimallyAugmented
            ω0 = BK.getp(br.sol[ii].x, prob_ma)[2]
        end
        p1 = br.sol[ii].p
        @test p1 == lens1(pt)
        @test p0 == lens2(pt)

        # we test the functional
        par1 = BK.set(par0, lens1, p1)
        par1 = BK.set(par1, lens2, p0)
        @test par1.T == pt.T && par1.F == pt.F
        resf = prob_vf.VF.F(x0, par1)
        @test norminf(resf) < ϵ
        if prob_ma isa FoldProblemMinimallyAugmented
            res = prob_ma(x0, p0, BK.set(par0, lens1, p1))
        else
            res = prob_ma(x0, p0, ω0, BK.set(par0, lens1, p1))
        end
        @test resf == res[1]
        verbose && @show res
        @test norminf(res[1]) < 100ϵ
        @test norminf(res[2]) < 100ϵ
        par0 = merge(BK.getparams(br), pt)

        # we test the eigenvalues
        vp = eigenvals(br, step)
        J = BK.jacobian(prob_vf.VF, x0, par1)
        vp2 = eig(J, 4)[1]
        verbose && display(hcat(vp, vp2))
        @test vp == vp2
    end
end

testEV(sn_codim2_test)
testEV(hp_codim2_test)
####################################################################################################
@reset opts_br.newton_options.verbose = false
sn_codim2 = nothing
for _jac in (:autodiff, :minaug, :finiteDifferences, :MinAugMatrixBased)
    # be careful here, Bordered predictor not good for Fold continuation
    sn_codim2 = @time continuation((@set br.alg.tangent = Secant()), 5, (@optic _.T), ContinuationPar(opts_br, p_max = 3.2, p_min = -0.1, detect_bifurcation = 1, dsmin=1e-5, ds = -0.001, dsmax = 0.015, n_inversion = 10, save_sol_every_step = 1, max_steps = 30, max_bisection_steps = 55) ; verbosity = 0,
        normC = norminf,
        jacobian_ma = _jac,
        # jacobian_ma = :minaug,
        detect_codim2_bifurcation = 1,
        update_minaug_every_step = 1,
        start_with_eigen = true,
        record_from_solution = recordFromSolutionLor,
        bdlinsolver = MatrixBLS())

    @test sn_codim2.specialpoint[1].type == :bt
    @test sn_codim2.specialpoint[2].type == :zh
    @test sn_codim2.specialpoint[3].type == :zh
    @test sn_codim2.specialpoint[4].type == :bt

    @test sn_codim2.eig[1].eigenvecs != nothing

    btpt = get_normal_form(sn_codim2, 1; nev = 4, verbose = true)
    @test norm(eigvals(BK.jacobian(br.prob, btpt.x0, btpt.params))[1:2], Inf) < 0.02
    HC = BK.predictor(btpt, Val(:HopfCurve), 0.)
        HC.hopf(0.)

    @test btpt.nf.a ≈ 0.20776621366525655
    @test btpt.nf.b ≈ 0.5773685192880018
    # plot(sn_codim2, vars=(:F, :T), branchlabel = "SN")
    #     _S = LinRange(0., 0.001, 100)
    #     plot!([HC.hopf(s)[1] for s in _S], [HC.hopf(s)[2] for s in _S], label = "Hpred")
    #     # plot!(hp_codim2_1, vars=(:F, :T), branchlabel = "Hopf1")

    # test for Zero-Hopf
    zh = BK.get_normal_form(sn_codim2, 2, verbose = true)
    show(zh)
    BK.predictor(zh, Val(:HopfCurve), 0.1).hopf(0.)
    BK.predictor(zh, Val(:HopfCurve), 0.1).x0(0.)
    BK.predictor(zh, Val(:HopfCurve), 0.1).ω(0.)

    # locate BT point with newton algorithm and compute the normal form
    _bt = BK.bt_point(sn_codim2, 1) # does nothing

    solbt = newton(sn_codim2, 1; options = NewtonPar(br.contparams.newton_options; verbose = false, tol = 1e-15), start_with_eigen = true, jacobian_ma = :finitedifferences)
    @test BK.converged(solbt)
    solbt = newton(sn_codim2, 1; options = NewtonPar(br.contparams.newton_options; verbose = false, tol = 1e-15), start_with_eigen = true, jacobian_ma = :minaug)
    @test BK.converged(solbt)
    solbt = newton(sn_codim2, 1; options = NewtonPar(br.contparams.newton_options; verbose = false, tol = 1e-15), start_with_eigen = false, jacobian_ma = :autodiff)
    @test BK.converged(solbt)
    solbt = newton(sn_codim2, 1; options = NewtonPar(br.contparams.newton_options; verbose = false, tol = 1e-15), start_with_eigen = true, jacobian_ma = :autodiff)
    @test BK.converged(solbt)
    @test norm(eigvals(BK.jacobian(br.prob, solbt.u.x0, solbt.u.params))[1:2], Inf) < 1e-8

    if sn_codim2.specialpoint[1].x isa BorderedArray
        sn_codim2_forbt = @set sn_codim2.specialpoint[1].x.u = Array(solbt.u.x0)
        @reset sn_codim2_forbt.specialpoint[1].x.p = solbt.u.params.F
    else
        sn_codim2_forbt = @set sn_codim2.specialpoint[1].x = vcat(Array(solbt.u.x0), solbt.u.params.F)
    end
    @reset sn_codim2_forbt.specialpoint[1].param = solbt.u.params.T

    bpbt_2 = get_normal_form(sn_codim2_forbt, 1; nev = 4, verbose = true)
    @test bpbt_2.nf.a ≈ 0.2144233509273467
    @test bpbt_2.nf.b ≈ 0.6065145518280868

    @test bpbt_2.nfsupp.γ ≈ -1.2655376039398163 rtol = 1e-3
    @test bpbt_2.nfsupp.c ≈ 12.35040633066114 rtol = 1e-3
    @test bpbt_2.nfsupp.K10 ≈ [10.994145052508442, 5.261454635012957] rtol = 1e-3
    @test bpbt_2.nfsupp.K11 ≈ [1.4057052343089358, 0.24485405521091583] rtol = 1e-3
    @test bpbt_2.nfsupp.K2 ≈ [2.445742562066124, 1.1704560432349538] rtol = 1e-3
    @test bpbt_2.nfsupp.d ≈ -0.23814643486558454 rtol = 1e-3
    @test bpbt_2.nfsupp.e ≈ -2.8152510696740043 rtol = 1e-3
    @test bpbt_2.nfsupp.a1 ≈ 0.588485870443459 rtol = 1e-3
    @test bpbt_2.nfsupp.b1 ≈ 1.2381458099504048 rtol = 1e-3
    @test bpbt_2.nfsupp.H0001 ≈ [1.2666466468447481, -0.11791034083511988, -0.26313225842609955, -0.5338271838915466] rtol = 1e-3
    @test bpbt_2.nfsupp.H0010 ≈ [15.651509120793042, -1.1750214928055762, -3.2016608356146423, -6.424103770005164] rtol = 1e-3
    @test bpbt_2.nfsupp.H0002 ≈ [-0.34426541029040103, 0.7403628764888541, 0.5020796040084594, 0.7211107457956355] rtol = 1e-3 rtol = 1e-3
    @test bpbt_2.nfsupp.H1001 ≈ [0.8609019479520158, 0.3666091456682787, 0.09272126477464948, -1.1252591151814477] rtol = 1e-3
    @test bpbt_2.nfsupp.H2000 ≈ [-1.1430891994241816, 0.5090981254844374, 0.4300904962638521, -0.4240003230561569] rtol = 1e-3
    # test branch switching from BT points
    hp_codim2_2 = continuation(sn_codim2, 1, ContinuationPar(opts_br, ds = -0.001, dsmax = 0.02, dsmin = 1e-4, n_inversion = 6, detect_bifurcation = 1, p_max = 15.) ;
        normC = norminf,
        detect_codim2_bifurcation = 2,
        update_minaug_every_step = 1,
        jacobian_ma = _jac,
        record_from_solution = recordFromSolutionLor,
        bdlinsolver = MatrixBLS())
end

####################################################################################################
# test events
sn_codim2 = @time continuation((@set br.alg.tangent = Secant()), 5, (@optic _.T), ContinuationPar(opts_br, p_max = 3.2, p_min = -0.1, detect_bifurcation = 1, dsmin=1e-5, ds = -0.001, dsmax = 0.015, n_inversion = 10, save_sol_every_step = 1, max_steps = 30, max_bisection_steps = 55) ; verbosity = 0,
    normC = norminf,
    # jacobian_ma = _jac,
    # jacobian_ma = :minaug,
    detect_codim2_bifurcation = 2,
    update_minaug_every_step = 1,
    start_with_eigen = true,
    # bothside = true,
    record_from_solution = recordFromSolutionLor,
    event = SaveAtEvent((0.001,.01)),
    bdlinsolver = MatrixBLS())

@test sn_codim2.specialpoint |> length == 7
@test sn_codim2.specialpoint[2].type == Symbol("save-2")
@test sn_codim2.specialpoint[3].type == Symbol("save-1")

hp_codim2_1 = continuation(br, 3, (@optic _.T), ContinuationPar(opts_br, ds = -0.001, dsmax = 0.02, dsmin = 1e-4, n_inversion = 6, save_sol_every_step = 1, detect_bifurcation = 1, max_steps = 100)  ;
    # verbosity = 3, plot = true,
    normC = norminf,
    detect_codim2_bifurcation = 2,
    update_minaug_every_step = 1,
    start_with_eigen = true,
    bothside = false,
    jacobian_ma = :autodiff,
    # jacobian_ma = :minaug,
    event = SaveAtEvent((-0.05,.0)),
    record_from_solution = recordFromSolutionLor,
    bdlinsolver = MatrixBLS())

@test hp_codim2_1.specialpoint |> length == 4
@test hp_codim2_1.specialpoint[2].type == Symbol("save-2")
@test hp_codim2_1.specialpoint[3].type == Symbol("save-1")
####################################################################################################
sn_codim2 = @time continuation((@set br.alg.tangent = Secant()), 5, (@optic _.T), ContinuationPar(opts_br, p_max = 3.2, p_min = -0.1, detect_bifurcation = 1, dsmin=1e-5, ds = -0.001, dsmax = 0.015, n_inversion = 10, save_sol_every_step = 1, max_steps = 30, max_bisection_steps = 55) ; verbosity = 0,
    normC = norminf,
    # jacobian_ma = _jac,
    # jacobian_ma = :minaug,
    detect_codim2_bifurcation = 2,
    update_minaug_every_step = 1,
    start_with_eigen = true,
    record_from_solution = recordFromSolutionLor,
    bdlinsolver = MatrixBLS())

hp_codim2_1 = continuation(br, 3, (@optic _.T), ContinuationPar(opts_br, ds = -0.001, dsmax = 0.02, dsmin = 1e-4, n_inversion = 6, save_sol_every_step = 1, detect_bifurcation = 1, max_steps = 100)  ;
    verbosity = 0,
    normC = norminf,
    detect_codim2_bifurcation = 2,
    update_minaug_every_step = 1,
    start_with_eigen = true,
    bothside = true,
    jacobian_ma = :autodiff,
    # jacobian_ma = :minaug,
    record_from_solution = recordFromSolutionLor,
    bdlinsolver = MatrixBLS())

@test hp_codim2_1.alg.tangent isa Bordered
@test ~(hp_codim2_1.alg.bls isa MatrixBLS)
@test hp_codim2_1.prob.prob.linbdsolver isa MatrixBLS

@test hp_codim2_1.specialpoint |> length == 5
@test hp_codim2_1.specialpoint[2].type == :bt
@test hp_codim2_1.specialpoint[3].type == :gh
@test hp_codim2_1.specialpoint[4].type == :hh

get_normal_form(hp_codim2_1, 2)

# plot(sn_codim2, vars=(:X,:U))
# plot!(hp_codim2_1, vars=(:X,:U))

get_normal_form(hp_codim2_1, 2; nev = 4, verbose = true)

nf = get_normal_form(hp_codim2_1, 3; nev = 4, verbose = true, detailed = true)
@test nf.nf.ω ≈ 0.6903636672622595 atol = 1e-5
@test nf.nf.l2 ≈ 0.15555332623343107 atol = 1e-3
@test nf.nf.G32 ≈ 1.8694569030805148 - 49.456355483784634im atol = 1e-3
@test nf.nf.γ₁₀₁ ≈ 0.41675854806948004 - 0.3691568377673768im atol = 1e-3
@test nf.nf.γ₁₁₀ ≈ 0.03210697158629905 + 0.34913987438180344im atol = 1e-3
@test nf.nf.γ₂₀₁ ≈ 6.5060917177185535 - 1.276445931785017im atol = 1e-3
@test nf.nf.γ₂₁₀ ≈ -2.005158175714135 - 1.8446801200912402im atol = 1e-3
_pred = BK.predictor(nf, Val(:FoldPeriodicOrbitCont), 0.1)
_pred.orbit(0.1)

nf = get_normal_form(hp_codim2_1, 4; nev = 4, verbose = true, detailed = true)
_pred = predictor(nf, Val(:NS), .01)
_pred.ns1(0.1)
_pred.ns2(0.1)

# locate BT point with newton algorithm
_bt = BK.bt_point(hp_codim2_1, 2)
solbt = newton(hp_codim2_1, 2; options = NewtonPar(br.contparams.newton_options;verbose = true))

eigvals(BK.jacobian(prob, solbt.u.x0, solbt.u.params))

# curv of Hopf points from BT
sn_from_bt = continuation(hp_codim2_1, 2, ContinuationPar(opts_br, ds = -0.001, dsmax = 0.02, dsmin = 1e-4, n_inversion = 6, detect_bifurcation = 1, p_max = 15.) ;
    normC = norminf,
    detect_codim2_bifurcation = 2,
    update_minaug_every_step = 1,
    record_from_solution = recordFromSolutionLor,
    bdlinsolver = MatrixBLS())
@test sn_from_bt.kind isa BK.FoldCont

# curve of Hopf points from ZH
zh = get_normal_form(sn_codim2, 2, detailed = true)
_pred = BK.predictor(zh, Val(:NS), 0.1)
_pred.orbit(0.1)

hp_from_zh = continuation(sn_codim2, 2, ContinuationPar(opts_br, ds = -0.001, dsmax = 0.02, dsmin = 1e-4, n_inversion = 6, detect_bifurcation = 1, max_steps = 100) ;
    plot = false, verbosity = 0,
    normC = norminf,
    # δp = 0.001,
    detect_codim2_bifurcation = 2,
    update_minaug_every_step = 1,
    bothside = false,
    )
@test hp_from_zh.kind isa BK.HopfCont
@test length(hp_from_zh.γ.specialpoint) == 5

# curve of Hopf points from HH
hp_from_hh = continuation(hp_from_zh, 4, ContinuationPar(opts_br, ds = 0.001, dsmax = 0.02, dsmin = 1e-4, n_inversion = 6, detect_bifurcation = 1, max_steps = 120) ;
    plot = false, verbosity = 0,
    normC = norminf,
    # jacobian_ma = :min_aug,
    # δp = 0.001,
    detect_codim2_bifurcation = 2,
    update_minaug_every_step = 1,
    bothside = false,
    )
@test hp_from_zh.kind isa BK.HopfCont

# plot(sn_codim2,vars=(:X, :U),)
#     plot!(sn_from_bt, vars=(:X, :U),)
#     plot!(hp_codim2_1, vars=(:X, :U), branchlabel = "Hopf")
#     plot!(hp_from_zh, vars=(:X, :U), branchlabel = "Hopf-ZH")
#     plot!(hp_from_hh, vars=(:X, :U), branchlabel = "Hopf-HH")

# test getters for branches
BK.getlens(hp_from_hh)
BK.getparams(hp_from_hh)
####################################################################################################
# branching from Bautin to Fold of periodic orbits
using OrdinaryDiffEq
prob_ode = ODEProblem(Lor, z0, (0, 1), BK.getparams(hp_codim2_1), reltol = 1e-10, abstol = 1e-12)

opts_fold_po = ContinuationPar(hp_codim2_1.contparams, dsmax = 0.01, detect_bifurcation = 0, max_steps = 3, detect_event = 0, ds = 0.001)
@reset opts_fold_po.newton_options.verbose = false
@reset opts_fold_po.newton_options.tol = 1e-8

for probPO in (
                PeriodicOrbitOCollProblem(20, 3), 
                ShootingProblem(9, prob_ode, Rodas5(), parallel = true)
              )
    @info probPO
    fold_po = continuation(hp_codim2_1, 3, opts_fold_po, probPO;
            normC = norminf,
            δp = 0.02,
            update_minaug_every_step = 0,
            jacobian_ma = :minaug,
            # callback_newton =  BK.cbMaxNormAndΔp(1e1, 0.025),
            # verbosity = 2, plot = true,
            )
    
    @test fold_po.kind == BifurcationKit.FoldPeriodicOrbitCont()
end 
####################################################################################################
# branching HH to NS of periodic orbits
prob_ode = ODEProblem(Lor, z0, (0, 1), hp_codim2_1.contparams, reltol = 1e-8, abstol = 1e-10)
opts_ns_po = ContinuationPar(hp_codim2_1.contparams, dsmax = 0.02, detect_bifurcation = 1, max_steps = 10, ds = -0.01, detect_event = 0)
# @reset opts_ns_po.newton_options.verbose = true
@reset opts_ns_po.newton_options.tol = 1e-12
@reset opts_ns_po.newton_options.max_iterations = 10
for probPO in (PeriodicOrbitOCollProblem(20, 3, update_section_every_step = 1), ShootingProblem(5, prob_ode, Rodas5(), parallel = true, update_section_every_step = 1))
    ns_po = continuation(hp_codim2_1, 4, opts_ns_po, 
        probPO;
        usehessian = false,
        detect_codim2_bifurcation = 0,
        δp = 0.02,
        update_minaug_every_step = 1,
        whichns = 2,
        jacobian_ma = :minaug,
        # verbosity = 3, plot = true,
        )
    # test that the Floquet coefficients equal     ns_po.ωₙₛ
    @test abs(imag(ns_po.eig[end].eigenvals[2])) ≈ ns_po[end].ωₙₛ
end