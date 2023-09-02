# using Revise
# using Plots
using Test, ForwardDiff, Parameters, LinearAlgebra
using BifurcationKit, Test
const BK = BifurcationKit
####################################################################################################
function COm(u, p)
    @unpack q1, q2, q3, q4, q5, q6, k = p
    x, y, s = u
    z = 1-x-y-s
    [
        2q1 * z^2 - 2q5 * x^2 - q3 * x * y,
        q2 * z - q6 * y - q3 * x * y,
        q4 * (z - k * s)
    ]
end

par_com = (q1 = 2.5, q2 = 1., q3 = 10., q4 = 0.0675, q5 = 1., q6 = 0.1, k = 0.4)

z0 = [0.001137, 0.891483, 0.062345]

prob = BifurcationProblem(COm, z0, par_com, (@lens _.q2);
        record_from_solution = (x, p) -> (x = x[1], y = x[2], s = x[3]))

opts_br = ContinuationPar(p_min = 0.5, p_max = 2.3, ds = 0.002, dsmax = 0.01, n_inversion = 6, detect_bifurcation = 3, max_bisection_steps = 25, nev = 3, max_steps = 100)

br = continuation(prob, PALC(), opts_br; normC = norminf, bothside = true)

@test br.kind == BK.EquilibriumCont()
@test br.specialpoint[2].param ≈ 1.04099606
@test br.specialpoint[3].param ≈ 1.05220029
@test br.specialpoint[4].param ≈ 1.04204851
@test br.specialpoint[5].param ≈ 1.05158367
####################################################################################################
@set! opts_br.newton_options = NewtonPar(max_iterations = 10, tol = 1e-12)

sn_codim2 = continuation(br, 3, (@lens _.k),
    ContinuationPar(opts_br, p_max = 2.2, p_min = 0., ds = -0.001, dsmax = 0.05, n_inversion = 8, max_steps = 50) ;
    normC = norminf,
    detect_codim2_bifurcation = 2,
    update_minaug_every_step = 1,
    # record_from_solution = (u,p; kw...) -> (x = BK.getVec(u)[1] ),
    bothside = true,
    )

# start_with_eigen = true, # ce truc fout la merde
# le probleme est que prob.a change avec bothside. detect_bifurcation = 1 fout la merde aussi


@test sn_codim2.kind == BK.FoldCont()
@test sn_codim2.specialpoint[2].type == :bt
@test sn_codim2.specialpoint[3].type == :cusp
@test sn_codim2.specialpoint[4].type == :bt

@test sn_codim2.specialpoint[2].param ≈ 0.97139757 atol = 1e-5
@test sn_codim2.specialpoint[3].param ≈ 0.35665351 rtol = 1e-4
@test sn_codim2.specialpoint[4].param ≈ 0.7223392465523879

@test sn_codim2.specialpoint[2].printsol.k     ≈ 0.971397 rtol = 1e-4
@test sn_codim2.specialpoint[2].printsol.q2 ≈ 1.417628 rtol = 1e-4
@test sn_codim2.specialpoint[4].printsol.k     ≈ 0.722339 rtol = 1e-4
@test sn_codim2.specialpoint[4].printsol.q2 ≈ 1.161199 rtol = 1e-4

# cusp normal form
cp = get_normal_form(sn_codim2, 3)

@test isapprox(cp.nf.c, 0.362; rtol = 1e-2)

# case of BT point
bt = get_normal_form(sn_codim2, 2; autodiff = false)
bt = get_normal_form(sn_codim2, 2; autodiff = true)

@test isapprox(bt.nf.a |> abs, 0.083784; rtol = 1e-4)
@test isapprox(bt.nf.b |> abs, 2.1363; rtol = 1e-4)
@test isapprox(bt.nfsupp.K2, [-13.1155, 51.17]; atol = 1e-2)
@test isapprox(bt.nfsupp.d, -0.1778; rtol = 1e-3)
@test isapprox(bt.nfsupp.e, -7.1422; rtol = 1e-4)
@test isapprox(abs(bt.nfsupp.a1), abs( -0.8618 ); rtol = 1e-3)
@test isapprox(abs(bt.nfsupp.b1), abs( -7.1176 ); rtol = 1e-3)

# very close to BT, stop event location at 1e-12
brh = (@set br.alg.tangent = Bordered())
hp_codim2 = continuation(brh, 2, (@lens _.k), ContinuationPar(opts_br, p_min = 0., p_max = 2.8, detect_bifurcation = 1, ds = -0.0001, dsmax = 0.02, dsmin = 1e-4, n_inversion = 12, max_steps = 150, max_bisection_steps = 35 ) ;
    normC = norminf,
    # verbosity = 3, plot = true,
    detect_codim2_bifurcation = 2,
    update_minaug_every_step = 1,
    start_with_eigen = true,
    bothside = true,
    bdlinsolver = MatrixBLS())

show(hp_codim2)

@test hp_codim2.kind == BK.HopfCont()
@test hp_codim2.specialpoint[2].type == :bt
@test hp_codim2.specialpoint[3].type == :gh
@test hp_codim2.specialpoint[4].type == :gh
if length(hp_codim2.specialpoint) > 4
    @test hp_codim2.specialpoint[5].type == :bt
end

@test hp_codim2.specialpoint[3].printsol.k  ≈ 0.305879 rtol = 1e-3
@test hp_codim2.specialpoint[3].printsol.q2 ≈ 0.924255 rtol = 1e-3
@test hp_codim2.specialpoint[4].printsol.k  ≈ 0.232    rtol = 1e-2
@test hp_codim2.specialpoint[4].printsol.q2 ≈ 0.896099 rtol = 1e-2

gh = get_normal_form(hp_codim2, 4)
@test isapprox(gh.nf.G21, -24.9125im; rtol = 1e-4)
@test isapprox(gh.nf.G32, -9322.74711 + 51539.23090im; rtol = 1e-4)
@test isapprox(gh.nf.l2, -776.89; rtol = 1e-4)
####################################################################################################
# branch switching at BT
hp_from_bt = continuation(sn_codim2, 4 , setproperties(sn_codim2.contparams, ds = 0.001, max_steps = 200);
        # verbosity = 3, plot  = true,
        δp = 1e-5,
        normC = norminf,
        update_minaug_every_step = 1,
        bothside = true,
        # recordFromSolution = (u,p; kw...) -> (x = BK.getVec(u)[1] ),
        )

@test hp_from_bt.kind == BK.HopfCont()
@test hp_from_bt.specialpoint[2].type == :bt
@test hp_from_bt.specialpoint[3].type == :gh
@test hp_from_bt.specialpoint[4].type == :gh
@test hp_from_bt.specialpoint[5].type == :bt
# bt = computeNormalForm(sn_codim2, 3; verbose = true)
#     display(bt.x0)
#     show(bt)
# HC = predictor(bt, Val(:HopfCurve), 0.)
#
# plot(sn_codim2, vars=(:q2, :k))
#     _S = LinRange(0, 1e-7, 1000)
#     plot!([HC.hopf(s)[1] for s in _S], [HC.hopf(s)[2] for s in _S], linewidth=3, label = "Hpred")
#     plot!(hp_codim2, vars=(:q2, :k), color = :black)
#
# HC.ω(0.01)
#
# HC = predictor(bt, Val(:HopfCurve), 0.)
#     HC.EigenVec(0.01)
#
#
# plot(sn_codim2, vars=(:q2, :x))
#     plot!(hp_codim2, vars=(:q2, :x))
#     plot!(hp_from_bt, vars=(:q2, :x), marker = :d)
