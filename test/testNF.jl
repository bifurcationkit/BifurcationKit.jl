# using Revise, Plots, Test
using BifurcationKit, LinearAlgebra, Setfield, SparseArrays, ForwardDiff, Parameters
const BK = BifurcationKit
norminf = x -> norm(x, Inf)

Fbp(x, p) = [x[1] * (3.23 .* p.μ - p.x2 * x[1] + p.x3 * 0.234 * x[1]^2) + x[2], -x[2]]
par = (μ = -0.2, ν = 0, x2 = 1.12, x3 = 1.0)
####################################################################################################
opt_newton = NewtonPar(tol = 1e-9, maxIter = 20)
opts_br = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds = 0.01, pMax = 0.4, pMin = -0.5, detectBifurcation = 3, nev = 2, newtonOptions = opt_newton, maxSteps = 100, nInversion = 4, tolBisectionEigenvalue = 1e-8, dsminBisection = 1e-9)

br, = continuation(
	Fbp, [0.1, 0.1], par, (@lens _.μ),
	opts_br; plot = false, verbosity = 0, normC = norminf, recordFromSolution = (x, p) -> x[1])

####################################################################################################
# normal form computation
jet = BK.getJet(Fbp, (x, p) -> ForwardDiff.jacobian(z -> Fbp(z, p), x))

bp = BK.computeNormalForm(jet..., br, 1; verbose=false)
@test BK.isTranscritical(bp) == true

jet = BK.getJet(Fbp, (x, p) -> BK.finiteDifferences(z -> Fbp(z, p), x))
bp = BK.computeNormalForm(jet..., br, 1; verbose=false)
@test BK.isTranscritical(bp) == true
show(bp)

# normal form
nf = bp.nf

@test norm(nf.a) < 1e-10
	@test norm(nf.b1 - 3.23) < 1e-10
	@test norm(nf.b2/2 - -1.12) < 1e-10
	@test norm(nf.b3/6 - 0.234) < 1e-10

####################################################################################################
# same but when the eigenvalues are not saved in the branch but computed on the fly
br_noev, = BK.continuation(
	Fbp, [0.1, 0.1], par, (@lens _.μ),
	recordFromSolution = (x, p) -> norminf(x),
	(@set opts_br.saveEigenvectors = false); plot = false, verbosity = 0, normC = norminf)
bp = BK.computeNormalForm(jet..., br_noev, 1; verbose=false)
nf = bp.nf
@test norm(nf[1]) < 1e-10
	@test norm(nf[2] - 3.23) < 1e-10
	@test norm(nf[3]/2 - -1.12) < 1e-10
	@test norm(nf[4]/6 - 0.234) < 1e-10
####################################################################################################
# Automatic branch switching
br2, = continuation(jet..., br, 1, setproperties(opts_br; pMax = 0.2, ds = 0.01, maxSteps = 14); recordFromSolution = (x, p) -> x[1], verbosity = 0)
@test br2 isa Branch
@test BK.haseigenvalues(br2) == true
@test BK.haseigenvector(br2) == true
BK.eigenvals(br2, 1, true)
BK.getfirstusertype(br2)
@test length(br2) == 12
# plot(br,br2)

br3, = continuation(jet..., br, 1, setproperties(opts_br; ds = -0.01); recordFromSolution = (x, p) -> x[1], verbosity = 0, usedeflation = true)
# plot(br,br3)

# automatic bifurcation diagram (Transcritical)
bdiag = bifurcationdiagram(jet..., [0.1, 0.1], par,  (@lens _.μ), 2,
	(args...) -> setproperties(opts_br; pMin = -1.0, pMax = .5, ds = 0.01, dsmax = 0.05, nInversion = 6, detectBifurcation = 3, maxBisectionSteps = 30, newtonOptions = (@set opt_newton.verbose=false), maxSteps = 15);
	recordFromSolution = (x, p) -> x[1],
	tangentAlgo = BorderedPred(),
	plot = false, verbosity = 0, normC = norminf)
####################################################################################################
# Case of the pitchfork
par_pf = @set par.x2 = 0.0
par_pf = @set par_pf.x3 = -1.0
brp, = BK.continuation(
	Fbp, [0.1, 0.1], par_pf, (@lens _.μ),
	recordFromSolution = (x, p) -> x[1],
	opts_br; plot = false, verbosity = 0, normC = norminf)
bpp = BK.computeNormalForm(jet..., brp, 1; verbose=false)
show(bpp)

nf = bpp.nf
@test norm(nf[1]) < 1e-9
	@test norm(nf[2] - 3.23) < 1e-9
	@test norm(nf[3]/2 - 0) < 1e-9
	@test norm(nf[4]/6 + 0.234) < 1e-9

# test automatic branch switching
br2, = continuation(jet..., brp, 1, setproperties(opts_br; maxSteps = 19, dsmax = 0.01, ds = 0.001, detectBifurcation = 2, newtonOptions = (@set opt_newton.verbose=false)); recordFromSolution = (x, p) -> x[1], verbosity = 0, ampfactor = 1)
	# plot(brp,br2, marker=:d)

# test methods for aBS
BK.from(br2) |> BK.type
BK.from(br2) |> BK.istranscritical
BK.type(nothing)
BK.show(stdout, br2)
BK.propertynames(br2)

# automatic bifurcation diagram (Pitchfork)
bdiag = bifurcationdiagram(jet..., [0.1, 0.1], par_pf,  (@lens _.μ), 2,
	(args...) -> setproperties(opts_br; pMin = -1.0, pMax = .5, ds = 0.01, dsmax = 0.05, nInversion = 6, detectBifurcation = 3, maxBisectionSteps = 30, newtonOptions = (@set opt_newton.verbose=false), maxSteps = 15);
	recordFromSolution = (x, p) -> x[1],
	# tangentAlgo = BorderedPred(),
	plot = false, verbosity = 0, normC = norminf)
####################################################################################################
function Fbp2d(x, p)
	return [ p.α * x[1] * (3.23 .* p.μ - 0.123 * x[1]^2 - 0.234 * x[2]^2),
			 p.α * x[2] * (3.23 .* p.μ - 0.456 * x[1]^2 - 0.123 * x[2]^2),
			 -x[3]]
end

jet = BK.getJet(Fbp2d, (x, p) -> ForwardDiff.jacobian(z -> Fbp2d(z, p), x))

par = (μ = -0.2, ν = 0, α = -1)

for α in (1,1)
	global par = @set par.α = α
	br, = BK.continuation(
		Fbp2d, [0.01, 0.01, 0.01], par, (@lens _.μ),
		recordFromSolution = (x, p) -> norminf(x),
		setproperties(opts_br; nInversion = 2); plot = false, verbosity = 0, normC = norminf)
	# we have to be careful to have the same basis as for Fbp2d or the NF will not match Fbp2d
	bp2d = BK.computeNormalForm(jet..., br, 1; ζs = [[1, 0, 0.], [0, 1, 0.]]);
	show(bp2d)

	BK.nf(bp2d)
	length(bp2d)
	bp2d(rand(2), 0.2)
	bp2d(Val(:reducedForm), rand(2), 0.2)

	@test abs(bp2d.nf.b3[1,1,1,1] / 6 - -par.α * 0.123) < 1e-10
	@test abs(bp2d.nf.b3[1,1,2,2] / 2 - -par.α * 0.234) < 1e-10
	@test abs(bp2d.nf.b3[1,1,1,2] / 2 - -par.α * 0.0)   < 1e-10
	@test abs(bp2d.nf.b3[2,1,1,2] / 2 - -par.α * 0.456) < 1e-10
	@test norm(bp2d.nf.b2, Inf) < 3e-6
	@test norm(bp2d.nf.b1 - par.α * 3.23 * I, Inf) < 1e-9
	@test norm(bp2d.nf.a, Inf) < 1e-6
end

##############################
# same but when the eigenvalues are not saved in the branch but computed on the fly instead
br_noev, = BK.continuation(
	Fbp2d, [0.01, 0.01, 0.01], par, (@lens _.μ),
	recordFromSolution = (x, p) -> norminf(x),
	setproperties(opts_br; nInversion = 2, saveEigenvectors = false); plot = false, verbosity = 0, normC = norminf)
bp2d = BK.computeNormalForm(jet..., br_noev, 1; ζs = [[1, 0, 0.], [0, 1, 0.]]);
@test abs(bp2d.nf.b3[1,1,1,1] / 6 - -0.123) < 1e-15
@test abs(bp2d.nf.b3[1,1,2,2] / 2 - -0.234) < 1e-15
@test abs(bp2d.nf.b3[1,1,1,2] / 2 - -0.0)   < 1e-15
@test abs(bp2d.nf.b3[2,1,1,2] / 2 - -0.456) < 1e-15
@test norm(bp2d.nf.b2, Inf) < 3e-15
@test norm(bp2d.nf.b1 - 3.23 * I, Inf) < 1e-9
@test norm(bp2d.nf.a, Inf) < 1e-15
####################################################################################################
# vector field to test nearby secondary bifurcations
FbpSecBif(u, p) = @. -u * (p + u * (2-5u)) * (p -.15 - u * (2+20u))
dFbpSecBif(x,p)=  ForwardDiff.jacobian( z-> FbpSecBif(z,p), x)
jet = BK.getJet(FbpSecBif, dFbpSecBif)

br_snd1, = BK.continuation(
	FbpSecBif, [0.0], -0.2, (@lens _),
	recordFromSolution = (x, p) -> x[1],
	# tangentAlgo = BorderedPred(),
	setproperties(opts_br; pMin = -1.0, pMax = .3, ds = 0.001, dsmax = 0.005, nInversion = 8, detectBifurcation=3); plot = false, verbosity = 0, normC = norminf)

# plot(br_snd1)

br_snd2, = BK.continuation(
	jet..., br_snd1, 1,
	recordFromSolution = (x, p) -> x[1],
	setproperties(opts_br; pMin = -1.2, pMax = 0.2, ds = 0.001, detectBifurcation = 3, maxSteps=19, nInversion = 8, newtonOptions = NewtonPar(opts_br.newtonOptions),dsminBisection =1e-18, tolBisectionEigenvalue=1e-11, maxBisectionSteps=20); plot = false, verbosity = 0, normC = norminf,
	# tangentAlgo = BorderedPred(),
	# finaliseSolution = (z, tau, step, contResult) ->
	# 	(Base.display(contResult.eig[end].eigenvals) ;true)
	)

	# plot(plot(br_snd2.branch[1,:] |> diff, marker = :d),
	# plot(br_snd1,br_snd2, putbifptlegend=false))

bdiag = bifurcationdiagram(jet..., [0.0], -0.2, (@lens _), 2,
	(args...) -> setproperties(opts_br; pMin = -1.0, pMax = .3, ds = 0.001, dsmax = 0.005, nInversion = 8, detectBifurcation = 3,dsminBisection =1e-18, tolBisectionEigenvalue=1e-11, maxBisectionSteps=20, newtonOptions = (@set opt_newton.verbose=false));
	recordFromSolution = (x, p) -> x[1],
	# tangentAlgo = BorderedPred(),
	plot = false, verbosity = 0, normC = norminf)

# plot(bdiag; putbifptlegend=false, markersize=2, plotfold=false, title = "#branch = $(size(bdiag))")

# test calls for aBD
BK.level(bdiag)
BK.hasbranch(bdiag)
BK.from(bdiag.child[1].γ)
BK.getBranchesFromBP(bdiag, 2)
BK.getContResult(br)
BK.getContResult(getBranch(bdiag,(1,)).γ)
size(bdiag)
getBranch(bdiag, (1,))
show(stdout, bdiag)
####################################################################################################
# test of the pitchfork-D6 normal form
function FbpD6(x, p)
	return [ p.μ * x[1] + (p.a * x[2] * x[3] - p.b * x[1]^3 - p.c*(x[2]^2 + x[3]^2) * x[1]),
			 p.μ * x[2] + (p.a * x[1] * x[3] - p.b * x[2]^3 - p.c*(x[3]^2 + x[1]^2) * x[2]),
			 p.μ * x[3] + (p.a * x[1] * x[2] - p.b * x[3]^3 - p.c*(x[2]^2 + x[1]^2) * x[3])]
end
jet = BK.getJet(FbpD6, (x, p) -> ForwardDiff.jacobian(z -> FbpD6(z, p), x))

pard6 = (μ = -0.2, a = 0.3, b = 1.5, c = 2.9)

br, = BK.continuation(
	FbpD6, zeros(3), pard6, (@lens _.μ),
	recordFromSolution = (x, p) -> norminf(x),
	setproperties(opts_br; nInversion = 6, ds = 0.001); plot = false, verbosity = 0, normC = norminf)

# plot(br;  plotfold = false)
# we have to be careful to have the same basis as for Fbp2d or the NF will not match Fbp2d
bp2d = BK.computeNormalForm(jet..., br, 1; ζs = [[1, 0, 0.], [0, 1, 0.], [0, 0, 1.]])
BK.nf(bp2d)

@test abs(bp2d.nf.b3[1,1,1,1] / 6 - -pard6.b) < 1e-10
	@test abs(bp2d.nf.b3[1,1,2,2] / 2 - -pard6.c) < 1e-10
	@test abs(bp2d.nf.b2[1,2,3] - pard6.a)   < 1e-10

# test the evaluation of the normal form
x0 = rand(3); @test norm(FbpD6(x0, set(pard6, br.lens, 0.001))  - bp2d(Val(:reducedForm), x0, 0.001), Inf) < 1e-12

br1, = BK.continuation(
	jet..., br, 1,
	setproperties(opts_br; nInversion = 4, dsmax = 0.005, ds = 0.001, maxSteps = 300, pMax = 1.); plot = false, verbosity = 0, normC = norminf, recordFromSolution = (x, p) -> norminf(x))
	# plot(br1..., br, plotfold=false, putbifptlegend=false)

bdiag = bifurcationdiagram(jet..., zeros(3), pard6, (@lens _.μ), 3,
	(args...) -> setproperties(opts_br; pMin = -0.250, pMax = .4, ds = 0.001, dsmax = 0.005, nInversion = 4, detectBifurcation = 3, dsminBisection =1e-18, tolBisectionEigenvalue=1e-11, maxBisectionSteps=20, newtonOptions = (@set opt_newton.verbose=false));
	recordFromSolution = (x, p) -> norminf(x),
	# tangentAlgo = BorderedPred(),
	plot = false, verbosity = 0, normC = norminf)

# plot(bdiag; putspecialptlegend=false, markersize=2,plotfold=false);title!("#branch = $(size(bdiag))")

####################################################################################################
# test of the Hopf normal form
function Fsl2!(f, u, p, t)
	@unpack r, μ, ν, c3, c5 = p
	u1 = u[1]
	u2 = u[2]
	ua = u1^2 + u2^2

	f[1] = r * u1 - ν * u2 - ua * (c3 * u1 - μ * u2) - c5 * ua^2 * u1
	f[2] = r * u2 + ν * u1 - ua * (c3 * u2 + μ * u1) - c5 * ua^2 * u2

	return f
end

Fsl2(x, p) = Fsl2!(similar(x), x, p, 0.)
par_sl = (r = 0.5, μ = 0.132, ν = 1.0, c3 = 1.123, c5 = 0.2)
jet = BK.getJet(Fsl2, (x, p) -> ForwardDiff.jacobian(z -> Fsl2(z, p), x))

# detect hopf bifurcation
opts_br = ContinuationPar(dsmin = 0.001, dsmax = 0.02, ds = 0.01, pMax = 0.1, pMin = -0.3, detectBifurcation = 3, nev = 2, newtonOptions = (@set opt_newton.verbose = false), maxSteps = 100)

br, = BK.continuation(
	jet[1], jet[2], [0.0, 0.0], (@set par_sl.r = -0.1), (@lens _.r),
	recordFromSolution = (x, p) -> norminf(x),
	opts_br; plot = false, verbosity = 0, normC = norminf)

hp = BK.computeNormalForm(jet..., br, 1)

nf = hp.nf
BK.type(hp)

@test abs(nf.a - 1) < 1e-9
@test abs(nf.b/2 - (-par_sl.c3 + im*par_sl.μ)) < 1e-14

####################################################################################################
# same but when the eigenvalues are not saved in the branch but computed on the fly instead
br, _ = BK.continuation(
	Fsl2, [0.0, 0.0], (@set par_sl.r = -0.1), (@lens _.r),
	recordFromSolution = (x, p) -> norminf(x),
	setproperties(opts_br, saveEigenvectors = false); plot = false, verbosity = 0, normC = norminf)

hp = BK.computeNormalForm(jet..., br, 1)

nf = hp.nf

@test abs(nf.a - 1) < 1e-9
@test abs(nf.b/2 - (-par_sl.c3 + im*par_sl.μ)) < 1e-14

show(hp)
####################################################################################################
# test for the Cusp normal form
Fcusp(x, p) = [p.β1 + p.β2 * x[1] + p.c * x[1]^3]
par = (β1 = 0.0, β2 = -0.01, c = 3.)
jet  = BK.getJet(Fcusp; matrixfree=false)
br, = continuation(jet[1], jet[2], [0.01], par, (@lens _.β1), opts_br;)

sn_codim2, = continuation(jet[1:2]..., br, 1, (@lens _.β2), ContinuationPar(opts_br, detectBifurcation = 1, saveSolEveryStep = 1, maxSteps = 40) ;
	updateMinAugEveryStep = 1,
	d2F = jet[3], d3F = jet[4],
	bdlinsolver = MatrixBLS()
	)
# find the cusp point
ind = findall(map(x->x.type == :cusp, sn_codim2.specialpoint))
cuspnf = computeNormalForm(jet..., sn_codim2, ind[1])
show(cuspnf)
BK.type(btpt)
@test cuspnf.nf.c == par.c
####################################################################################################
# test for the Bogdanov-Takens normal form
Fbt(x, p) = [x[2], p.β1 + p.β2 * x[2] + p.a * x[1]^2 + p.b * x[1] * x[2]]
par = (β1 = 0.01, β2 = -0.1, a = -1., b = 1.)
jet  = BK.getJet(Fbt; matrixfree=false)
opt_newton = NewtonPar(tol = 1e-9, maxIter = 40, verbose = false)
opts_br = ContinuationPar(dsmin = 0.001, dsmax = 0.01, ds = 0.01, pMax = 0.5, pMin = -0.5, detectBifurcation = 3, nev = 2, newtonOptions = opt_newton, maxSteps = 100, nInversion = 8, tolBisectionEigenvalue = 1e-8, dsminBisection = 1e-9, saveSolEveryStep = 1)

br, = continuation(jet[1], jet[2], [0.01, 0.01], par, (@lens _.β1), opts_br; bothside = true)

sn_codim2, = continuation(jet[1:2]..., br, 1, (@lens _.β2), ContinuationPar(opts_br, detectBifurcation = 1, saveSolEveryStep = 1, maxSteps = 40) ;
	detectCodim2Bifurcation = 2,
	updateMinAugEveryStep = 1,
	d2F = jet[3], d3F = jet[4],
	bdlinsolver = MatrixBLS()
	)

hopf_codim2, = continuation(jet[1:2]..., br, 2, (@lens _.β2), ContinuationPar(opts_br, detectBifurcation = 1, saveSolEveryStep = 1, maxSteps = 40) ;
	detectCodim2Bifurcation = 2,
	updateMinAugEveryStep = 1,
	bothside = true,
	d2F = jet[3], d3F = jet[4],
	bdlinsolver = MatrixBLS(),
	)

# plot(sn_codim2, hopf_codim2, branchlabel = ["Fold", "Hopf"])

btpt = computeNormalForm(jet..., sn_codim2, 1; nev = 2)
show(btpt)
BK.type(btpt)
@test norm(btpt.nf.b * sign(sum(btpt.ζ[1])) - par.b, Inf) < 1e-5
@test norm(btpt.nf.a * sign(sum(btpt.ζ[1])) - par.a, Inf) < 1e-5
@test isapprox(abs.(btpt.ζ[1]), [1, 0])
@test isapprox(abs.(btpt.ζ[2]), [0, 1];rtol = 1e-6)
@test isapprox(abs.(btpt.ζstar[1]), [1, 0];rtol = 1e-6)

@test isapprox(btpt.nfsupp.K2, [0, 0]; atol = 1e-5)
@test isapprox(btpt.nfsupp.d, 0; atol = 1e-3)
@test isapprox(btpt.nfsupp.e, 0; atol = 1e-3)
@test isapprox(btpt.nfsupp.a1, 0; atol = 1e-3)
@test isapprox(btpt.nfsupp.b1, 0; atol = 1e-3)

HC = BK.predictor(btpt, Val(:HopfCurve), 0.)
	HC.hopf(0.)
SN = BK.predictor(btpt, Val(:FoldCurve), 0.)
Hom = BK.predictor(btpt, Val(:HomoclinicCurve), 0.)
	Hom.orbit(0,0)

# plot(sn_codim2, branchlabel = ["Fold"], vars = (:β1, :β2))
# 	_S = LinRange(-0.06, 0.06, 1000)
# 	plot!([HC.hopf(s)[1] for s in _S], [HC.hopf(s)[2] for s in _S], linewidth=5, label = "Hpred")
# 	plot!([SN.fold(s)[1] for s in _S], [SN.fold(s)[2] for s in _S], linewidth=5, label = "SNpred")
# 	_S = LinRange(-0.25, 0.25, 1000)
# 	plot!([Hom.α(s)[1] for s in _S], [Hom.α(s)[2] for s in _S], linewidth=5, label = "Hom")
#
# 	plot!(hopf_codim2, branchlabel = ["Hopf"], vars = (:β1, :β2), color = :black)
# 	xlims!(-0.001, 0.05)


# plot of the homoclinic orbit
hom1 = [Hom.orbit(t,0.1)[1] for t in LinRange(-1000, 1000, 10000)]
hom2 = [Hom.orbit(t,0.1)[2] for t in LinRange(-1000, 1000, 10000)]
# plot(hom1, hom2)
####################################################################################################
# test of the Bautin normal form
function Fsl2!(f, u, p, t)
	@unpack r, μ, ν, c3, c5 = p
	u1, u2 = u
	ua = u1^2 + u2^2
	f[1] = r * u1 - ν * u2 + ua * (c3 * u1 - μ * u2) + c5 * ua^2 * u1
	f[2] = r * u2 + ν * u1 + ua * (c3 * u2 + μ * u1) + c5 * ua^2 * u2
	return f
end

Fsl2(x, p) = Fsl2!(similar(x), x, p, 0.)
par_sl = (r = -0.5, μ = 0., ν = 1.0, c3 = 0.1, c5 = 0.3)
jet = BK.getJet(Fsl2, matrixfree=false)

@set! opts_br.newtonOptions.verbose = false
@set! opts_br.newtonOptions.tol = 1e-12
opts_br = setproperties(opts_br;nInversion = 10, maxBisectionSteps = 25)

br, = continuation( jet[1], jet[2], [0.01, 0.01], par_sl, (@lens _.r), opts_br)

hopf_codim2, = continuation(jet[1:2]..., br, 1, (@lens _.c3), ContinuationPar(opts_br, detectBifurcation = 0, saveSolEveryStep = 1, maxSteps = 40, pMin = -2., pMax = 2., ds = -0.001) ;
	detectCodim2Bifurcation = 2,
	startWithEigen = true,
	updateMinAugEveryStep = 1,
	d2F = jet[3], d3F = jet[4],
	bdlinsolver = MatrixBLS(),
	)

bautin = BifurcationKit.computeNormalForm(jet..., hopf_codim2, 1; nev = 2)
show(bautin)
BK.type(bautin)

@test bautin.nf.l2 ≈ par_sl.c5 * 4
