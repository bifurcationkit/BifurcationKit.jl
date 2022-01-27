# using Revise,
using Test, ForwardDiff, Parameters, Setfield, LinearAlgebra
# using Plots
using BifurcationKit, Test
const BK = BifurcationKit

norminf = x -> norm(x, Inf)
####################################################################################################
function Lor(u, p)
	@unpack α,β,γ,δ,G,F,T = p
	X,Y,Z,U = u
	[
		-Y^2 - Z^2 - α*X + α*F - γ*U^2,
		X*Y - β*X*Z - Y + G,
		β*X*Y + X*Z - Z,
		-δ*U + γ*U*X + T
	]
end
jet = BK.getJet(Lor;matrixfree=false)
parlor = (α = 1//4, β = 1, G = .25, δ = 1.04, γ = 0.987, F = 1.7620532879639, T = .0001265)

opts_br = ContinuationPar(pMin = -1.5, pMax = 3.0, ds = 0.002, dsmax = 0.15, nInversion = 6, detectBifurcation = 3, maxBisectionSteps = 25, nev = 4, maxSteps = 200, plotEveryStep = 30)

z0 =  [2.9787004394953343, -0.03868302503393752,  0.058232737694740085, -0.02105288273117459]

br, = @time continuation(jet[1], jet[2], z0, setproperties(parlor;T=0.04,F=3.), (@lens _.F), opts_br;
	recordFromSolution = (x, p) -> (X = x[1], Y = x[2], Z= x[3], U = x[4]),
	normC = norminf,
	tangentAlgo = BorderedPred(),
	bothside = true)
####################################################################################################
# this part is for testing the spectrum
recordFromSolutionLor(x, p) = (u = BK.getVec(x);(X = u[1], Y = u[2], Z = u[3], U = u[4]))
@set! opts_br.newtonOptions.verbose = false

sn_codim2_test, = continuation(jet[1:2]..., br, 4, (@lens _.T), ContinuationPar(opts_br, pMax = 3.2, pMin = -0.1, detectBifurcation = 1, dsmin=1e-5, ds = -0.001, dsmax = 0.005, nInversion = 8, saveSolEveryStep = 1, maxSteps = 60) ;
	normC = norminf,
	detectCodim2Bifurcation = 1,
	updateMinAugEveryStep = 1,
	startWithEigen = true,
	recordFromSolution = recordFromSolutionLor,
	d2F = jet[3], d3F = jet[4],
	bdlinsolver = MatrixBLS(),
	)

hp_codim2_test, = continuation(jet[1:2]..., br, 1, (@lens _.T), ContinuationPar(opts_br, ds = -0.001, dsmax = 0.02, dsmin = 1e-4, nInversion = 6, saveSolEveryStep = 1) ;
	normC = norminf,
	tangentAlgo = BorderedPred(),
	detectCodim2Bifurcation = 1,
	updateMinAugEveryStep = 1,
	startWithEigen = true,
	recordFromSolution = recordFromSolutionLor,
	d2F = jet[3], d3F = jet[4],
	bothside = true,
	bdlinsolver = MatrixBLS())
####################################################################################################
"""
This function test if the eigenvalues are well computed during a branch of Hopf/Fold.
This test works if the problem is not updated because we dont record prob.a and prob.b
"""
function testEV(br, verbose = false)
	verbose && println("\n\n\n"*"="^50)
	prob = br.functional
	eig = DefaultEig()
	lens1 = br.lens
	lens2 = prob.lens
	par0 = br.params
	ϵ = br.contparams.newtonOptions.tol

	for (ii, pt) in enumerate(br.branch)
		# we make sure the parameters are set right
		step = pt.step
		verbose && (println("="^50); @info step ii)
		x0 = BK.getVec(br.sol[ii].x, prob)
		p0 = BK.getP(br.sol[ii].x, prob)[1]
		if prob isa HopfProblemMinimallyAugmented
			ω0 = BK.getP(br.sol[ii].x, prob)[2]
		end
		p1 = br.sol[ii].p
		@test p1 == get(pt, lens1)
		@test p0 == get(pt, lens2)

		# we test the functional
		par1 = set(par0, lens1, p1)
		par1 = set(par1, lens2, p0)
		@test par1.T == pt.T && par1.F == pt.F
		resf = prob.F(x0, par1)
		@test norminf(resf) < ϵ
		if prob isa FoldProblemMinimallyAugmented
			res = prob(x0, p0, set(par0, lens1, p1))
		else
			res = prob(x0, p0, ω0, set(par0, lens1, p1))
		end
		@test resf == res[1]
		verbose && @show res
		@test norminf(res[1]) < 100ϵ
		@test norminf(res[2]) < 100ϵ
		par0 = merge(br.params, pt)

		# we test the eigenvalues
		vp = eigenvals(br, step)
		J = jet[2](x0, par1)
		vp2 = eig(J, 4)[1]
		verbose && display(hcat(vp, vp2))
		@test vp == vp2
	end
end

testEV(sn_codim2_test)
testEV(hp_codim2_test)
####################################################################################################
@set! opts_br.newtonOptions.verbose = false

sn_codim2, = continuation(jet[1:2]..., br, 4, (@lens _.T), ContinuationPar(opts_br, pMax = 3.2, pMin = -0.1, detectBifurcation = 1, dsmin=1e-5, ds = -0.001, dsmax = 0.015, nInversion = 10, saveSolEveryStep = 1, maxSteps = 30, maxBisectionSteps = 55) ;
	normC = norminf,
	detectCodim2Bifurcation = 2,
	updateMinAugEveryStep = 1,
	startWithEigen = true,
	recordFromSolution = recordFromSolutionLor,
	d2F = jet[3], d3F = jet[4],
	bdlinsolver = MatrixBLS())

@test sn_codim2.specialpoint[1].type == :bt
@test sn_codim2.specialpoint[2].type == :zh
@test sn_codim2.specialpoint[3].type == :zh
@test sn_codim2.specialpoint[4].type == :bt

btpt = computeNormalForm(jet..., sn_codim2, 1; nev = 4)
HC = BK.predictor(btpt, Val(:HopfCurve), 0.)
	HC.hopf(0.)

# plot(sn_codim2, vars=(:F, :T), branchlabel = "SN")
# 	_S = LinRange(0., 0.001, 100)
# 	plot!([HC.hopf(s)[1] for s in _S], [HC.hopf(s)[2] for s in _S], label = "Hpred")
# 	# plot!(hp_codim2_1, vars=(:F, :T), branchlabel = "Hopf1")

# test for Zero-Hopf
zh = BK.zeroHopfNormalForm(jet..., sn_codim2, 2)
show(zh)
BK.predictor(zh, Val(:HopfCurve), 0.1).hopf(0.)
BK.predictor(zh, Val(:HopfCurve), 0.1).x0(0.)
BK.predictor(zh, Val(:HopfCurve), 0.1).ω(0.)
####################################################################################################
hp_codim2_1, = continuation(jet[1:2]..., br, 2, (@lens _.T), ContinuationPar(opts_br, ds = -0.001, dsmax = 0.015, dsmin = 1e-4, nInversion = 6, saveSolEveryStep = 1, detectBifurcation = 1) ;
	normC = norminf,
	tangentAlgo = BorderedPred(),
	detectCodim2Bifurcation = 2,
	updateMinAugEveryStep = 1,
	startWithEigen = true,
	recordFromSolution = recordFromSolutionLor,
	d2F = jet[3], d3F = jet[4],
	bothside = true,
	linearAlgo = MatrixBLS(),
	bdlinsolver = MatrixBLS())

@test hp_codim2_1.specialpoint[1].type == :bt
@test hp_codim2_1.specialpoint[2].type == :gh
@test hp_codim2_1.specialpoint[3].type == :hh

# plot(hp_codim2_1, vars=(:F,:T))

computeNormalForm(jet..., hp_codim2_1, 1; nev = 4, verbose=true)
computeNormalForm(jet..., hp_codim2_1, 2; nev = 4, verbose=true)

# hp_codim2_2, = continuation(jet..., sn_codim2, 4, ContinuationPar(opts_br, ds = -0.001, dsmax = 0.02, dsmin = 1e-4, nInversion = 6, detectBifurcation = 0) ;
# 	normC = norminf,
# 	tangentAlgo = BorderedPred(),
# 	detectCodim2Bifurcation = 2,
# 	updateMinAugEveryStep = 1,
# 	startWithEigen = true,
# 	recordFromSolution = recordFromSolutionLor,
# 	# bothside = true,
# 	bdlinsolver = MatrixBLS())

# computeNormalForm(jet..., hp_codim2_2, 1; nev = 4, verbose=true)

# plot(hp_codim2_1, vars=(:F,:T),ylims=(0,0.06), xlims=(1,3))
# plot(hp_codim2_2, vars=(:F,:T),ylims=(-0.06,0.06), xlims=(1,3))
#
#
# plot(sn_codim2, vars=(:F, :T), branchlabel = "SN")
# 	plot!(hp_codim2_1, vars=(:F, :T), branchlabel = "Hopf1")
	# plot!(hp_codim2_2, vars=(:F, :T), branchlabel = "Hopf2")
# 	ylims!(-0.06,0.09);xlims!(1,3.5)
#
# plot(sn_codim2, vars=(:X, :U), branchlabel = "SN")
# 	plot!(hp_codim2_1, vars=(:X, :U), branchlabel = "Hopf1")
# 	plot!(hp_codim2_2, vars=(:X, :U), branchlabel = "Hopf2")
# 	ylims!(-0.7,0.75);xlims!(0.95,1.5)
