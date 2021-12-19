using Revise, Test, ForwardDiff, Parameters, Setfield, Plots, LinearAlgebra
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

par_com = (q1 = 2.5, q2 = 2.0, q3 = 10., q4 = 0.0675, q5 = 1., q6 = 0.1, k = 0.4)

z0 = [0.001137, 0.891483, 0.062345]

opts_br = ContinuationPar(pMin = 0.5, pMax = 2.0, ds = 0.002, dsmax = 0.01, nInversion = 6, detectBifurcation = 3, maxBisectionSteps = 25, nev = 3, maxSteps = 20000)
	@set! opts_br.newtonOptions.verbose = true
	br, = @time continuation(jet[1], jet[2], z0, par_com, (@lens _.q2), opts_br;
		recordFromSolution = (x, p) -> (x = x[1], y = x[2], s = x[3]),
		plot = false, verbosity = 3, normC = norminf,
	bothside = true)
	show(br)

	
plot(br, plotfold=false, markersize=4, legend=:topright, ylims=(0,0.16))
####################################################################################################
@set! opts_br.newtonOptions.verbose = true
@set! opts_br.newtonOptions.maxIter = 10
opts_br = @set opts_br.newtonOptions.tol = 1e-12

sn, = newton(jet[1:2]..., br, 2; options = opts_br.newtonOptions, bdlinsolver = MatrixBLS())

hp, = newton(jet[1:2]..., br, 1; options = NewtonPar( opts_br.newtonOptions; maxIter = 10),startWithEigen=true, d2F = jet[3])

BK.hopfNormalForm(jet..., hp, opts_br.newtonOptions.linsolver)
hpnf = computeNormalForm(jet..., br, 1)

sn_codim2, = continuationFold(jet[1:2]..., br, 2, (@lens _.k), ContinuationPar(opts_br, pMax = 3.2, pMin = 0., detectBifurcation = 0, dsmin=1e-5, ds = -0.001, dsmax = 0.05, nInversion = 6, detectEvent = 2, detectFold = false) ; plot = true,
	verbosity = 3,
	normC = norminf,
	updateMinAugEveryStep = 1,
	startWithEigen = true,
	recordFromSolution = (u,p; kw...) -> (x = u.u[1] ),
	bothside=true,
	bdlinsolver = MatrixBLS()
	)

using Test
@test sn_codim2.specialpoint[1].printsol.k 	≈ 0.971397 rtol = 1e-4
@test sn_codim2.specialpoint[1].printsol.q2 ≈ 1.417628 rtol = 1e-4
@test sn_codim2.specialpoint[3].printsol.k 	≈ 0.722339 rtol = 1e-4
@test sn_codim2.specialpoint[3].printsol.q2 ≈ 1.161199 rtol = 1e-4

plot(sn_codim2)#, real.(sn_codim2.BT), ylims = (-1,1), xlims=(0,2))

plot(sn_codim2, vars=(:q2, :x), branchlabel = "Fold", plotstability = false);plot!(br,xlims=(0.8,1.8))

hp_codim2, = continuation(jet[1:2]..., br, 1, (@lens _.k), ContinuationPar(opts_br, pMin = 0., pMax = 2.8, detectBifurcation = 0, ds = -0.0001, dsmax = 0.08, dsmin = 1e-4, nInversion = 6, detectEvent = 2, detectLoop = true, maxSteps = 50, detectFold=false) ; plot = true,
	verbosity = 3,
	normC = norminf,
	tangentAlgo = BorderedPred(),
	detectCodim2Bifurcation = 2,
	updateMinAugEveryStep = 1,
	startWithEigen = true,
	recordFromSolution = (u,p; kw...) -> (x = u.u[1] ),
	d2F = jet[3], d3F = jet[4],
	bothside = true,
	bdlinsolver = MatrixBLS())

@test hp_codim2.branch[5].l1 |> real 		≈ 33.15920 rtol = 1e-1
@test hp_codim2.specialpoint[1].printsol.k 	≈ 0.305879 rtol = 1e-3
@test hp_codim2.specialpoint[1].printsol.q2 ≈ 0.924255 rtol = 1e-3
@test hp_codim2.specialpoint[2].printsol.k 	≈ 0.235550 rtol = 1e-4
@test hp_codim2.specialpoint[2].printsol.q2 ≈ 0.896099 rtol = 1e-4

plot(sn_codim2, vars=(:q2, :x), branchlabel = "Fold", plotcirclesbif = true)
	plot!(hp_codim2, vars=(:q2, :x), branchlabel = "Hopf",plotcirclesbif = true)
	plot!(br,xlims=(0.6,1.5))

plot(sn_codim2, vars=(:k, :q2), branchlabel = "Fold")
	plot!(hp_codim2, vars=(:k, :q2), branchlabel = "Hopf",)

plot(hp_codim2, vars=(:q2, :x), branchlabel = "Hopf")
####################################################################################################
