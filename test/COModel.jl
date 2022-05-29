# using Revise,
using Test, ForwardDiff, Parameters, Setfield, LinearAlgebra
# using Plots
using BifurcationKit, Test
const BK = BifurcationKit

norminf(x) = norm(x, Inf)
####################################################################################################
function COm(u, p)
	@unpack q1,q2,q3,q4,q5,q6,k = p
	x, y, s = u
	z = 1-x-y-s
	[
		2q1 * z^2 - 2q5 * x^2 - q3 * x * y,
		q2 * z - q6 * y - q3 * x * y,
		q4 * (z - k * s)
	]
end
jet = BK.getJet(COm; matrixfree=false)

par_com = (q1 = 2.5, q2 = 1., q3 = 10., q4 = 0.0675, q5 = 1., q6 = 0.1, k = 0.4)

z0 = [0.001137, 0.891483, 0.062345]

opts_br = ContinuationPar(pMin = 0.5, pMax = 2.3, ds = 0.002, dsmax = 0.01, nInversion = 6, detectBifurcation = 3, maxBisectionSteps = 25, nev = 3, maxSteps = 20000)

br, = continuation(jet[1], jet[2], z0, par_com, (@lens _.q2), opts_br;
	recordFromSolution = (x, p) -> (x = x[1], y = x[2], s = x[3]),
	normC = norminf, bothside = true)
####################################################################################################
@set! opts_br.newtonOptions = NewtonPar(maxIter = 10, tol = 1e-12)
sn_codim2, = continuation(jet[1:2]..., br, 2, (@lens _.k), ContinuationPar(opts_br, pMax = 3.2, pMin = 0., detectBifurcation = 1, dsmin=1e-5, ds = -0.001, dsmax = 0.05, nInversion = 6) ;
	normC = norminf,
	detectCodim2Bifurcation = 2,
	updateMinAugEveryStep = 1,
	startWithEigen = true,
	d2F = jet[3], d3F = jet[4],
	recordFromSolution = (u,p; kw...) -> (x = BK.getVec(u)[1] ),
	bothside=true,
	bdlinsolver = MatrixBLS()
	)

@test sn_codim2.specialpoint[1].type == :bt
@test sn_codim2.specialpoint[2].type == :cusp
@test sn_codim2.specialpoint[3].type == :bt
@test sn_codim2.specialpoint[1].printsol.k 	≈ 0.971397 rtol = 1e-4
@test sn_codim2.specialpoint[1].printsol.q2 ≈ 1.417628 rtol = 1e-4
@test sn_codim2.specialpoint[3].printsol.k 	≈ 0.722339 rtol = 1e-4
@test sn_codim2.specialpoint[3].printsol.q2 ≈ 1.161199 rtol = 1e-4

# cusp normal form
cp = computeNormalForm(jet..., sn_codim2, 2; verbose = true)

@test isapprox(cp.nf.c, 0.362; rtol = 1e-2)

# case of BT point
bt = computeNormalForm(jet..., sn_codim2, 1; autodiff = false)
bt = computeNormalForm(jet..., sn_codim2, 1; autodiff = true)

@test isapprox(bt.nf.a |> abs, 0.083784; rtol = 1e-4)
@test isapprox(bt.nf.b |> abs, 2.1363; rtol = 1e-4)
@test isapprox(bt.nfsupp.K2, [-13.113, 51.15]; rtol = 1e-4)
@test isapprox(bt.nfsupp.d, -0.1778; rtol = 1e-3)
@test isapprox(bt.nfsupp.e, -7.1422; rtol = 1e-4)
@test isapprox(bt.nfsupp.a1, abs( -0.8618 ); rtol = 1e-4)
@test isapprox(bt.nfsupp.b1, abs( -7.1176 ); rtol = 1e-4)

hp_codim2, = continuation(jet[1:2]..., br, 1, (@lens _.k), ContinuationPar(opts_br, pMin = 0., pMax = 2.8, detectBifurcation = 1, ds = -0.0001, dsmax = 0.02, dsmin = 1e-4, nInversion = 12, maxSteps = 150, maxBisectionSteps = 35 ) ;
	normC = norminf,
	tangentAlgo = BorderedPred(),
	# linearAlgo = MatrixBLS(),
	detectCodim2Bifurcation = 2,
	updateMinAugEveryStep = 1,
	startWithEigen = true,
	d2F = jet[3], d3F = jet[4],
	bothside = true,
	bdlinsolver = MatrixBLS())

show(hp_codim2)

# @test hp_codim2.specialpoint[1].type == :bt
@test hp_codim2.specialpoint[2].type == :gh
@test hp_codim2.specialpoint[3].type == :gh
if length(hp_codim2.specialpoint) == 4
	@test hp_codim2.specialpoint[4].type == :bt
end

@test hp_codim2.specialpoint[2].printsol.k 	≈ 0.305879 rtol = 1e-3
@test hp_codim2.specialpoint[2].printsol.q2 ≈ 0.924255 rtol = 1e-3
@test hp_codim2.specialpoint[3].printsol.k 	≈ 0.232 rtol = 1e-2
@test hp_codim2.specialpoint[3].printsol.q2 ≈ 0.896099 rtol = 1e-2

gh = computeNormalForm(jet..., hp_codim2, 3)
@test isapprox(gh.nf.G21, -24.9125im; rtol = 1e-4)
@test isapprox(gh.nf.G32, -9322.74711 + 51539.23090im; rtol = 1e-4)
@test isapprox(gh.nf.l2, -776.89; rtol = 1e-4)
####################################################################################################
# branch switching at BT
# hp_from_bt, = continuation(jet..., sn_codim2, 3 , setproperties(sn_codim2.contparams, ds = 0.001, maxSteps = 200);
# 		verbosity = 3, plot  = true,
# 		δp = 1e-5,
# 		normC = norminf,
# 		updateMinAugEveryStep = 1,
# 		bothside = true,
# 		recordFromSolution = (u,p; kw...) -> (x = BK.getVec(u)[1] ),
# 		)
# bt = computeNormalForm(jet..., sn_codim2, 3; verbose = true)
# 	display(bt.x0)
# 	show(bt)
# HC = predictor(bt, Val(:HopfCurve), 0.)
#
# plot(sn_codim2, vars=(:q2, :k))
# 	_S = LinRange(0, 1e-7, 1000)
# 	plot!([HC.hopf(s)[1] for s in _S], [HC.hopf(s)[2] for s in _S], linewidth=3, label = "Hpred")
# 	plot!(hp_codim2, vars=(:q2, :k), color = :black)
#
# HC.ω(0.01)
#
# HC = predictor(bt, Val(:HopfCurve), 0.)
# 	HC.EigenVec(0.01)
#
#
# plot(sn_codim2, vars=(:q2, :x))
# 	plot!(hp_codim2, vars=(:q2, :x))
# 	plot!(hp_from_bt, vars=(:q2, :x), marker = :d)
