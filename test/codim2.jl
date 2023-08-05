# using Revise
using BifurcationKit
using Test, Parameters, Setfield, LinearAlgebra
# using Plots
const BK = BifurcationKit
####################################################################################################
function COm(u, p)
	@unpack q1,q2,q3,q4,q5,q6,k = p
	x, y, s = u
	z = 1-x-y-s
	out = similar(u)
	out[1] = 2 * q1 * z^2 - 2 * q5 * x^2 - q3 * x * y
	out[2] = q2 * z - q6 * y - q3 * x * y
	out[3] = q4 * z - k * q4 * s
	out
end

par_com = (q1 = 2.5, q2 = 2.0, q3 = 10., q4 = 0.0675, q5 = 1., q6 = 0.1, k = 0.4)
z0 = [0.07,0.2,05]

prob = BifurcationProblem(COm, z0, par_com, (@lens _.q2); recordFromSolution = (x, p) -> (x = x[1], y = x[2], s = x[3]))

opts_br = ContinuationPar(pMin = 0.6, pMax = 2.5, ds = 0.002, dsmax = 0.01, nInversion = 4, detectBifurcation = 3, maxBisectionSteps = 25, nev = 2, maxSteps = 20000)

	# @set! opts_br.newtonOptions.verbose = true
	alg = PALC()
	br = @time continuation(prob, alg, opts_br;
	plot = false, verbosity = 0, normC = norminf,
	bothside = true)

# plot(br, plotfold=false, markersize=4, legend=:topleft)
####################################################################################################
hp = newton(br, 2; options = NewtonPar( opts_br.newtonOptions; maxIter = 10), startWithEigen = true)

hpnf = getNormalForm(br, 5)
@test hpnf.nf.b |> real ≈ 1.070259e+01 rtol = 1e-3

hpnf = getNormalForm(br, 2)
@test hpnf.nf.b |> real ≈ 4.332247e+00 rtol = 1e-2

BK.FoldProblemMinimallyAugmented(prob)
BK.HopfProblemMinimallyAugmented(prob)
BK.PeriodDoublingProblemMinimallyAugmented(prob)
BK.NeimarkSackerProblemMinimallyAugmented(prob)
####################################################################################################
# different tests for the Fold point
snpt = getNormalForm(br, 3)

@test snpt.nf.a ≈ 0.11539539170637884 rtol = 1e-3
@test snpt.nf.b1 ≈ 0.7323167187172155 rtol = 1e-3
@test snpt.nf.b2 ≈ 0.2693795490512864 rtol = 1e-3
@test snpt.nf.b3 ≈ 12.340786210650833 rtol = 1e-3

@set! opts_br.newtonOptions.verbose = false
@set! opts_br.newtonOptions.maxIter = 10

sn = newton(br, 3; options = opts_br.newtonOptions, bdlinsolver = MatrixBLS())
# printstyled(color=:red, "--> guess for SN, p = ", br.specialpoint[2].param, ", psn = ", sn[1].p)
	# plot(br);scatter!([sn.x.p], [sn.x.u[1]])
@test BK.converged(sn) && sn.itlineartot == 8
@test sn.u.u ≈ [0.05402941507127516, 0.3022414400400177, 0.45980653206336225] rtol = 1e-4
@test sn.u.p ≈ 1.0522002878699546 rtol = 1e-4

sn = newton(br, 3; options = opts_br.newtonOptions, bdlinsolver = MatrixBLS())
@test BK.converged(sn) && sn.itlineartot == 8

sn = newton(br, 3; options = opts_br.newtonOptions, bdlinsolver = MatrixBLS(), startWithEigen = true)
@test BK.converged(sn) && sn.itlineartot == 8

sn_br = continuation(br, 3, (@lens _.k), ContinuationPar(opts_br, pMax = 1., pMin = 0., detectBifurcation = 1, maxSteps = 50, saveSolEveryStep = 1, detectEvent = 2), bdlinsolver = MatrixBLS(), startWithEigen = true, updateMinAugEveryStep = 1, jacobian_ma = :minaug, verbosity = 0, plot=false)
@test sn_br.kind isa BK.FoldCont
@test sn_br.specialpoint[1].type == :bt
@test sn_br.specialpoint[1].param ≈ 0.9716038596420551 rtol = 1e-5
@test ~isnothing(sn_br.eig)

# we test the jacobian and problem update
par_sn = BK.setParam(br, sn_br.sol[end].x.p)
par_sn = set(par_sn, BK.getLens(sn_br), sn_br.sol[end].p)
_J = BK.jacobian(prob, sn_br.sol[end].x.u, par_sn)
_eigvals, eigvec, = eigen(_J)
ind = argmin(abs.(_eigvals))
@test _eigvals[ind] ≈ 0 atol=1e-10
ζ = eigvec[:, ind]
@test sn_br.prob.prob.b ./ norm(sn_br.prob.prob.b) ≈ -ζ

_eigvals, eigvec, = eigen(_J')
ind = argmin(abs.(_eigvals))
ζstar = eigvec[:, ind]
@test sn_br.prob.prob.a ≈ ζstar
####################################################################################################
# different tests for the Hopf point
hppt = getNormalForm(br, 2)
@test hppt.nf.a ≈ 2.546719962189168 + 1.6474887797814664im
@test hppt.nf.b ≈ 4.3536804635557855 + 15.441272421860365im

@set! opts_br.newtonOptions.verbose = false

hp = newtonHopf(br, 2; options = opts_br.newtonOptions, startWithEigen = true)
# printstyled(color=:red, "--> guess for HP, p = ", br.specialpoint[1].param, ", php = ", hp[1].p)
# plot(br);scatter!([hp[1].p[1]], [hp[1].u[1]])
@test hp.converged && hp.itlineartot == 8

hp = newtonHopf(br, 2; options = opts_br.newtonOptions, startWithEigen = false, bdlinsolver = MatrixBLS())
@test hp.converged && hp.itlineartot == 12

hp = newtonHopf(br, 2; options = opts_br.newtonOptions, startWithEigen = true, bdlinsolver = MatrixBLS(), verbose = true)
@test hp.converged && hp.itlineartot == 8

# we check that we truly have a bifurcation point.
pb = hp.prob.prob
ω = hp.u.p[2]
par_hp = set(BK.getParams(br), BK.getLens(br), hp.u.p[1])
_J = pb.prob_vf.VF.J(hp.u.u, par_hp)
_eigvals, eigvec, = eigen(_J)
ind = argmin(abs.(_eigvals .- Complex(0, ω)))
@test real(_eigvals[ind]) ≈ 0 atol=1e-9
@test abs(imag(_eigvals[ind])) ≈ abs(hp.u.p[2]) atol=1e-9
ζ = eigvec[:, ind]
# reminder: pb.b should be a null vector of (J+iω)
@test pb.b ≈ ζ atol = 1e-3

hp = newton(br, 2;
	options = NewtonPar( opts_br.newtonOptions; maxIter = 10),
	startWithEigen = true,
	bdlinsolver = MatrixBLS())
# printstyled(color=:red, "--> guess for HP, p = ", br.specialpoint[1].param, ", php = ", hp.p)
# plot(br);scatter!([hp.p[1]], [hp.u[1]])

hp = newton(br, 2; options = NewtonPar( opts_br.newtonOptions; maxIter = 10),startWithEigen=true)

hp_br = continuation(br, 2, (@lens _.k), ContinuationPar(opts_br, ds = -0.001, pMax = 1., pMin = 0., detectBifurcation = 1, maxSteps = 50, saveSolEveryStep = 1, detectEvent = 2), bdlinsolver = MatrixBLS(), startWithEigen = true, updateMinAugEveryStep = 1, verbosity=0, plot=false)
@test hp_br.kind isa BK.HopfCont
@test hp_br.specialpoint[1].type == :gh
@test hp_br.specialpoint[2].type == :nd
@test hp_br.specialpoint[3].type == :gh

@test hp_br.specialpoint[1].param ≈ 0.305873681159479 rtol = 1e-5
@test hp_br.specialpoint[2].param ≈ 0.16452182436723148 rtol = 1e-5
@test hp_br.specialpoint[3].param ≈ 0.23255761094689315 atol = 1e-4

@test ~isnothing(hp_br.eig)
