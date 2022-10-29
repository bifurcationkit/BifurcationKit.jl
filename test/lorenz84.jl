# using Revise
using Test, ForwardDiff, Parameters, Setfield, LinearAlgebra
# using Plots
using BifurcationKit, Test
const BK = BifurcationKit

norminf(x) = norm(x, Inf)
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

parlor = (α = 1//4, β = 1, G = .25, δ = 1.04, γ = 0.987, F = 1.7620532879639, T = .0001265)

opts_br = ContinuationPar(pMin = -1.5, pMax = 3.0, ds = 0.001, dsmax = 0.025,
	# options to detect codim 1 bifurcations using bisection
	detectBifurcation = 3,
	# Optional: bisection options for locating bifurcations
	nInversion = 6, maxBisectionSteps = 25,
	# number of eigenvalues
	nev = 4, maxSteps = 252)

@set! opts_br.newtonOptions.maxIter = 25

z0 =  [2.9787004394953343, -0.03868302503393752,  0.058232737694740085, -0.02105288273117459]

recordFromSolutionLor(u::AbstractVector, p) = (X = u[1], Y = u[2], Z = u[3], U = u[4])
recordFromSolutionLor(u::BorderedArray, p) = recordFromSolutionLor(u.u, p)

prob = BK.BifurcationProblem(Lor, z0, parlor, (@lens _.F);
	recordFromSolution = recordFromSolutionLor,)

br = @time continuation(reMake(prob, params = setproperties(parlor;T=0.04,F=3.)),
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
@set! opts_br.newtonOptions.verbose = false

# be careful here, Bordered predictor not good for Fold continuation
sn_codim2_test = continuation((@set br.alg.tangent = Secant()), 5, (@lens _.T), ContinuationPar(opts_br, pMax = 3.2, pMin = -0.1, detectBifurcation = 1, dsmin=1e-5, ds = -0.001, dsmax = 0.005, nInversion = 8, saveSolEveryStep = 1, maxSteps = 60) ;
	normC = norminf,
	detectCodim2Bifurcation = 1,
	updateMinAugEveryStep = 1,
	startWithEigen = true,
    recordFromSolution = recordFromSolutionLor,
	bdlinsolver = MatrixBLS(),
	)

# plot(sn_codim2_test)

@test sn_codim2_test.specialpoint[1].param ≈ +0.02058724 rtol = 1e-5
@test sn_codim2_test.specialpoint[2].param ≈ +0.00004983 atol = 1e-8
@test sn_codim2_test.specialpoint[3].param ≈ -0.00045281 rtol = 1e-5
@test sn_codim2_test.specialpoint[4].param ≈ -0.02135893 rtol = 1e-5

@test sn_codim2_test.eig[1].eigenvecs != nothing

hp_codim2_test = continuation(br, 2, (@lens _.T), ContinuationPar(opts_br, ds = -0.001, dsmax = 0.02, dsmin = 1e-4, nInversion = 6, saveSolEveryStep = 1) ;
	normC = norminf,
	detectCodim2Bifurcation = 2,
	updateMinAugEveryStep = 1,
	startWithEigen = true,
    recordFromSolution = recordFromSolutionLor,
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
	lens1 = BK.getLens(br)
	lens2 = prob.lens
	par0 = BK.getParams(br)
	ϵ = br.contparams.newtonOptions.tol

	for (ii, pt) in enumerate(br.branch)
		# we make sure the parameters are set right
		step = pt.step
		verbose && (println("="^50); @info step ii)
		x0 = BK.getVec(br.sol[ii].x, prob_ma)
		p0 = BK.getP(br.sol[ii].x, prob_ma)[1]
		if prob_ma isa HopfProblemMinimallyAugmented
			ω0 = BK.getP(br.sol[ii].x, prob_ma)[2]
		end
		p1 = br.sol[ii].p
		@test p1 == get(pt, lens1)
		@test p0 == get(pt, lens2)

		# we test the functional
		par1 = set(par0, lens1, p1)
		par1 = set(par1, lens2, p0)
		@test par1.T == pt.T && par1.F == pt.F
		resf = prob_vf.VF.F(x0, par1)
		@test norminf(resf) < ϵ
		if prob_ma isa FoldProblemMinimallyAugmented
			res = prob_ma(x0, p0, set(par0, lens1, p1))
		else
			res = prob_ma(x0, p0, ω0, set(par0, lens1, p1))
		end
		@test resf == res[1]
		verbose && @show res
		@test norminf(res[1]) < 100ϵ
		@test norminf(res[2]) < 100ϵ
		par0 = merge(BK.getParams(br), pt)

		# we test the eigenvalues
		vp = eigenvals(br, step)
		J = prob_vf.VF.J(x0, par1)
		vp2 = eig(J, 4)[1]
		verbose && display(hcat(vp, vp2))
		@test vp == vp2
	end
end

testEV(sn_codim2_test)
testEV(hp_codim2_test)
####################################################################################################
@set! opts_br.newtonOptions.verbose = false

for _jac in (:autodiff, :minaug)
	# be careful here, Bordered predictor not good for Fold continuation
	sn_codim2 = @time continuation((@set br.alg.tangent = Secant()), 5, (@lens _.T), ContinuationPar(opts_br, pMax = 3.2, pMin = -0.1, detectBifurcation = 1, dsmin=1e-5, ds = -0.001, dsmax = 0.015, nInversion = 10, saveSolEveryStep = 1, maxSteps = 30, maxBisectionSteps = 55) ; verbosity = 0,
		normC = norminf,
		jacobian_ma = _jac,
		# jacobian_ma = :minaug,
		detectCodim2Bifurcation = 1,
		updateMinAugEveryStep = 1,
		startWithEigen = true,
		recordFromSolution = recordFromSolutionLor,
		bdlinsolver = MatrixBLS())

	@test sn_codim2.specialpoint[1].type == :bt
	@test sn_codim2.specialpoint[2].type == :zh
	@test sn_codim2.specialpoint[3].type == :zh
	@test sn_codim2.specialpoint[4].type == :bt

	@test sn_codim2.eig[1].eigenvecs != nothing

	btpt = getNormalForm(sn_codim2, 1; nev = 4, verbose = true)
	@test norm(eigvals(BK.jacobian(br.prob, btpt.x0, btpt.params))[1:2], Inf) < 0.02
	HC = BK.predictor(btpt, Val(:HopfCurve), 0.)
		HC.hopf(0.)

	@test btpt.nf.a ≈ 0.20776621366525655
	@test btpt.nf.b ≈ 0.5773685192880018

	# plot(sn_codim2, vars=(:F, :T), branchlabel = "SN")
	# 	_S = LinRange(0., 0.001, 100)
	# 	plot!([HC.hopf(s)[1] for s in _S], [HC.hopf(s)[2] for s in _S], label = "Hpred")
	# 	# plot!(hp_codim2_1, vars=(:F, :T), branchlabel = "Hopf1")

	# test for Zero-Hopf
	zh = BK.getNormalForm(sn_codim2, 2)
	show(zh)
	BK.predictor(zh, Val(:HopfCurve), 0.1).hopf(0.)
	BK.predictor(zh, Val(:HopfCurve), 0.1).x0(0.)
	BK.predictor(zh, Val(:HopfCurve), 0.1).ω(0.)

	# locate BT point with newton algorithm and compute the normal form
	_bt = BK.BTPoint(sn_codim2, 1) # does nothing

	solbt = newton(sn_codim2, 1; options = NewtonPar(br.contparams.newtonOptions; verbose = false, tol = 1e-15), startWithEigen = true, jacobian = :finitedifferences)
	@test BK.converged(solbt)
	solbt = newton(sn_codim2, 1; options = NewtonPar(br.contparams.newtonOptions; verbose = false, tol = 1e-15), startWithEigen = true, jacobian = :minaug)
	@test BK.converged(solbt)
	solbt = newton(sn_codim2, 1; options = NewtonPar(br.contparams.newtonOptions; verbose = false, tol = 1e-15), startWithEigen = true, jacobian = :autodiff)
	@test BK.converged(solbt)
	@test norm(eigvals(BK.jacobian(br.prob, solbt.u.x0, solbt.u.params))[1:2], Inf) < 1e-8

	if sn_codim2.specialpoint[1].x isa BorderedArray
		sn_codim2_forbt = @set sn_codim2.specialpoint[1].x.u = Array(solbt.u.x0)
		@set! sn_codim2_forbt.specialpoint[1].x.p = solbt.u.params.F
	else
		sn_codim2_forbt = @set sn_codim2.specialpoint[1].x = vcat(Array(solbt.u.x0), solbt.u.params.F)
	end
	@set! sn_codim2_forbt.specialpoint[1].param = solbt.u.params.T

	bpbt_2 = getNormalForm(sn_codim2_forbt, 1; nev = 4, verbose = true)
	@test bpbt_2.nf.a ≈ 0.2144233509273467
	@test bpbt_2.nf.b ≈ 0.6065145518280868

	@test bpbt_2.nfsupp.γ ≈ -1.2655376039398163
		@test bpbt_2.nfsupp.c ≈ 12.35040633066114
		@test bpbt_2.nfsupp.K10 ≈ [-5.261465557678953, 5.261454635012957]
		@test bpbt_2.nfsupp.K11 ≈ [-0.24485516427537235, 0.24485405521091583]
		@test bpbt_2.nfsupp.K2 ≈ [-1.1704584730763776, 1.1704560432349538]
		@test bpbt_2.nfsupp.d ≈ -0.23814643486558454
		@test bpbt_2.nfsupp.e ≈ -2.8152510696740043
		@test bpbt_2.nfsupp.a1 ≈ 0.588485870443459
		@test bpbt_2.nfsupp.b1 ≈ 1.2381458099504048
		@test bpbt_2.nfsupp.H0001 ≈ [1.2666466468447481, -0.11791034083511988, -0.26313225842609955, -0.5338271838915466]
		@test bpbt_2.nfsupp.H0010 ≈ [15.651509120793042, -1.1750214928055762, -3.2016608356146423, -6.424103770005164]
		@test bpbt_2.nfsupp.H0002 ≈ [-0.34426541029040103, 0.7403628764888541, 0.5020796040084594, 0.7211107457956355]
		@test bpbt_2.nfsupp.H1001 ≈ [0.8609019479520158, 0.3666091456682787, 0.09272126477464948, -1.1252591151814477]
		@test bpbt_2.nfsupp.H2000 ≈ [-1.1430891994241816, 0.5090981254844374, 0.4300904962638521, -0.4240003230561569]
end

####################################################################################################
hp_codim2_1 = continuation(br, 3, (@lens _.T), ContinuationPar(opts_br, ds = -0.001, dsmax = 0.015, dsmin = 1e-4, nInversion = 6, saveSolEveryStep = 1, detectBifurcation = 1)  ;
	verbosity = 0,
	normC = norminf,
	detectCodim2Bifurcation = 2,
	updateMinAugEveryStep = 1,
	startWithEigen = true,
	bothside = true,
	jacobian_ma = :autodiff,
	# jacobian_ma = :minaug,
	recordFromSolution = recordFromSolutionLor,
	bdlinsolver = MatrixBLS())

@test hp_codim2_1.alg.tangent isa Bordered
@test ~(hp_codim2_1.alg.bls isa MatrixBLS)
@test hp_codim2_1.prob.prob.linbdsolver isa MatrixBLS

@test hp_codim2_1.specialpoint |> length == 5
@test hp_codim2_1.specialpoint[2].type == :bt
@test hp_codim2_1.specialpoint[3].type == :gh
@test hp_codim2_1.specialpoint[4].type == :hh

# plot(sn_codim2, vars=(:X,:U))
# plot!(hp_codim2_1, vars=(:X,:U))

getNormalForm(hp_codim2_1, 2; nev = 4, verbose=true)
nf = getNormalForm(hp_codim2_1, 3; nev = 4, verbose=true)

@test nf.nf.ω ≈ 0.6903636672622595 atol = 1e-5
@test nf.nf.l2 ≈ 0.15555332623343107 atol = 1e-3

# locate BT point with newton algorithm
_bt = BK.BTPoint(hp_codim2_1, 2)
solbt = newton(hp_codim2_1, 2; options = NewtonPar(br.contparams.newtonOptions;verbose = true))

eigvals(BK.jacobian(prob, solbt.u.x0, solbt.u.params))

hp_codim2_2 = continuation(sn_codim2, 1, ContinuationPar(opts_br, ds = -0.001, dsmax = 0.02, dsmin = 1e-4, nInversion = 6, detectBifurcation = 1, pMax = 15.) ;
	normC = norminf,
	detectCodim2Bifurcation = 2,
	updateMinAugEveryStep = 1,
	recordFromSolution = recordFromSolutionLor,
	bdlinsolver = MatrixBLS())

# plot!(hp_codim2_2, vars=(:X,:U))
# getNormalForm(hp_codim2_2, 1; nev = 4, verbose=true)

# plot(hp_codim2_1, vars=(:F,:T), ylims=(0,0.06), xlims=(1,3))
# plot(hp_codim2_2, vars=(:F,:T), ylims=(-0.06,0.06), xlims=(1,3))
#
#
# plot(sn_codim2, vars=(:F, :T), branchlabel = "SN")
# 	plot!(hp_codim2_1, vars=(:F, :T), branchlabel = "Hopf1")
# 	plot!(hp_codim2_2, vars=(:F, :T), branchlabel = "Hopf2")
# 	ylims!(-0.06,0.09);xlims!(1,3.5)
#
# plot(sn_codim2, vars=(:X, :U), branchlabel = "SN")
# 	plot!(hp_codim2_1, vars=(:X, :U), branchlabel = "Hopf1")
# 	plot!(hp_codim2_2, vars=(:X, :U), branchlabel = "Hopf2")
# 	ylims!(-0.7,0.75);xlims!(0.95,1.5)
