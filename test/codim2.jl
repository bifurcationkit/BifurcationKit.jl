# using Revise, Plots
using Test, ForwardDiff, Parameters, Setfield, LinearAlgebra
using BifurcationKit, Test
const BK = BifurcationKit

D(f, x, p, dx) = ForwardDiff.derivative(t->f(x .+ t .* dx, p), 0.)
norminf = x -> norm(x, Inf)

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
dCOm = (z, p) -> ForwardDiff.jacobian(x -> COm(x, p), z)

# we group the differentials together
d1COm(x,p,dx1) = ForwardDiff.derivative(t -> COm(x .+ t .* dx1, p), 0.)
d2COm(x,p,dx1,dx2) = ForwardDiff.derivative(t -> d1COm(x .+ t .* dx2, p, dx1), 0.)
d3COm(x,p,dx1,dx2,dx3) = ForwardDiff.derivative(t -> d2COm(x .+ t .* dx3, p, dx1, dx2), 0.)
jet  = (COm, dCOm, d2COm, d3COm)

par_com = (q1 = 2.5, q2 = 2.0, q3=10., q4 = 0.0675,q5=1.,q6=0.1,k=0.4)
z0 = [0.07,0.2,05]

opts_br = ContinuationPar(pMin = 0.6, pMax = 2.5, ds = 0.002, dsmax = 0.01, nInversion = 4, detectBifurcation = 3, maxBisectionSteps = 25, nev = 2, maxSteps = 20000)
	br, = @time continuation(jet[1], jet[2], z0, par_com, (@lens _.q2), opts_br;
	printSolution = (x, p) -> (x = x[1], y = x[2]),
	plot = false, verbosity = 0, normC = norminf,
	bothside = true)

# plot(br, plotfold=false, markersize=4, legend=:topleft)
####################################################################################################
hp, = newton(jet[1:2]..., br, 1; options = NewtonPar( opts_br.newtonOptions; maxIter = 10),startWithEigen=true, d2F = jet[3])

hpnf = computeNormalForm(jet..., br, 4)
@test hpnf.nf.b |> real ≈ 1.070259e+01 rtol = 1e-2

hpnf = computeNormalForm(jet..., br, 1)
@test hpnf.nf.b |> real ≈ 4.332247e+00 rtol = 1e-2
####################################################################################################
# different tests for the Fold point
snpt = computeNormalForm(jet..., br, 2)
@set! opts_br.newtonOptions.verbose = false
@set! opts_br.newtonOptions.maxIter = 10

sn = newton(jet[1:2]..., br, 2; options = opts_br.newtonOptions, bdlinsolver = MatrixBLS())
printstyled(color=:red, "--> guess for SN, p = ", br.specialpoint[2].param, ", psn = ", sn[1].p)
	# plot(br);scatter!([sn.x.p], [sn.x.u[1]])
@test sn[3] && sn[5] == 6

sn = newton(jet[1:2]..., br, 2; options = opts_br.newtonOptions, bdlinsolver = MatrixBLS(), d2F = d2COm)
@test sn[3] && sn[5] == 8

sn = newton(jet[1:2]..., br, 2; options = opts_br.newtonOptions, bdlinsolver = MatrixBLS(), d2F = d2COm, startWithEigen = true)
@test sn[3] && sn[5] == 8

sn_br, = continuation(jet[1:2]..., br, 2, (@lens _.k), ContinuationPar(opts_br, pMax = 1., pMin = 0., detectBifurcation = 0, maxSteps = 1, saveSolEveryStep = 1), bdlinsolver = MatrixBLS(), d2F = d2COm, startWithEigen = true, updateMinAugEveryStep = 1, verbosity = 0, plot=false)

# we test the jacobian and problem update
par_sn = set(br.params, br.lens, sn_br.sol[end].x.p)
par_sn = set(par_sn, sn_br.lens, sn_br.sol[end].p)
_J = dCOm(sn_br.sol[end].x.u, par_sn)
_eigvals, eigvec, = eigen(_J)
ind = argmin(abs.(_eigvals))
@test _eigvals[ind] ≈ 0 atol=1e-15
ζ = eigvec[:, ind]
@test sn_br.functional.b ./ norm(sn_br.functional.b) ≈ ζ

_eigvals, eigvec, = eigen(_J')
ind = argmin(abs.(_eigvals))
ζstar = eigvec[:, ind]
@test sn_br.functional.a ≈ ζstar
####################################################################################################
# different tests for the Hopf point
hppt = computeNormalForm(jet..., br, 1)

@set! opts_br.newtonOptions.verbose = false

hp = newtonHopf(jet[1:2]..., br, 1; options = opts_br.newtonOptions, startWithEigen = true)
# printstyled(color=:red, "--> guess for HP, p = ", br.specialpoint[1].param, ", php = ", hp[1].p)
# plot(br);scatter!([hp[1].p[1]], [hp[1].u[1]])
# @test hp[3] && hp[5] == 6

hp = newtonHopf(jet[1:2]..., br, 1; options = opts_br.newtonOptions, startWithEigen = false, bdlinsolver = MatrixBLS())
@test hp[3] && hp[5] == 12

hp = newtonHopf(jet[1:2]..., br, 1; options = opts_br.newtonOptions, startWithEigen = true, bdlinsolver = MatrixBLS(), verbose = true)
@test hp[3] && hp[5] == 8

# we check that we truly have a bifurcation point.
pb = hp[end]
ω = hp[1].p[2]
par_hp = set(br.params, br.lens, hp[1].p[1])
_J = dCOm(hp[1].u, par_hp)
_eigvals, eigvec, = eigen(_J)
ind = argmin(abs.(_eigvals .- Complex(0, ω)))
@test real(_eigvals[ind]) ≈ 0 atol=1e-9
@test abs(imag(_eigvals[ind])) ≈ abs(hp[1].p[2]) atol=1e-9
ζ = eigvec[:, ind]
# reminder: pb.b should be a null vector of (J+iω)
@test pb.b ≈ ζ atol = 1e-3

d2NLIFc =(x,p,dx1,dx2) -> BK.BilinearMap((_dx1, _dx2) -> d2COm(x,p,_dx1,_dx2))(dx1,dx2)
hp, = newtonHopf(jet[1:2]..., br, 1;
	options = NewtonPar( opts_br.newtonOptions; maxIter = 10),
	startWithEigen = true,
	bdlinsolver = MatrixBLS(),
	d2F = d2NLIFc)
printstyled(color=:red, "--> guess for HP, p = ", br.specialpoint[1].param, ", php = ", hp.p)
# plot(br);scatter!([hp.p[1]], [hp.u[1]])

hp, = newtonHopf(jet[1:2]..., br, 1; options = NewtonPar( opts_br.newtonOptions; maxIter = 10),startWithEigen=true, d2F = d2NLIFc)
