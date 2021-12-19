# this test is designed to test the ability of the package to use a state space that is not an AbstractArray.
# using Revise
using Test, Random, Setfield
using BifurcationKit
const BK = BifurcationKit
####################################################################################################
# test Bordered Arrays
_a = BorderedArray(zeros(2), zeros(2))
_b = BorderedArray(zeros(2), zeros(2))
BK.mul!(_a, 1., _b)
BK.axpby!(1., _a, 1., _b)
####################################################################################################
# We start with a simple Fold problem
using LinearAlgebra
function F0(x::Vector, r)
	out = r .+  x .- x.^3
end

opt_newton0 = NewtonPar(tol = 1e-11, verbose = false)
	out0, hist, flag = newton(F0, [0.8], 1., opt_newton0)

opts_br0 = ContinuationPar(dsmin = 0.001, dsmax = 0.07, ds= -0.02, pMax = 4.1, pMin = -1., newtonOptions = setproperties(opt_newton0; maxIter = 70, tol = 1e-8), detectBifurcation = 0, maxSteps = 150)

BK.isStable(opts_br0, nothing)

br0, u1 = continuation(F0, out0, 1.0, (@lens _), opts_br0, recordFromSolution = (x, p) -> x[1])

outfold, hist, flag = newton(
		F0, (x, r) -> diagm(0 => 1 .- 3 .* x.^2),
		br0, 2;
		Jᵗ = (x, r) -> diagm(0 => 1 .- 3 .* x.^2),
		d2F = (x, r, v1, v2) -> -6 .* x .* v1 .* v2,)
	# flag && printstyled(color=:red, "--> We found a Fold Point at α = ",outfold.p, ", from ", br0.specialpoint[2].param, "\n")

####################################################################################################
# Here is a more involved example

function Fb(x::BorderedArray, p)
	r, s = p
	BorderedArray(r .+  s .* x.u .- (x.u).^3, x.p - 0.0)
end

# there is no finite differences defined, so we need to provide a linearsolve
# we could also have used GMRES. We define a custom Jacobian which will be used for evaluation and jacobian inverse
struct Jacobian
	x
	r
	s
end

# We express the jacobian operator
function (J::Jacobian)(dx)
	BorderedArray( (J.s .- 3 .* ((J.x).u).^2) .* dx.u, dx.p)
end

struct linsolveBd <: BK.AbstractBorderedLinearSolver end

function (l::linsolveBd)(J, dx)
	x = J.x
	r = J.r
	out = BorderedArray(dx.u ./ (J.s .- 3 .* (x.u).^2), dx.p)
	out, true, 1
end

sol0 = BorderedArray([0.8], 0.0)

opt_newton = NewtonPar(tol = 1e-11, verbose = false, linsolver = linsolveBd())
sol, hist, flag = newton(Fb, (x, p) -> Jacobian(x, 1., 1.), sol0, (1., 1.), opt_newton)

opts_br = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= -0.01, pMax = 4.1, pMin = -1., newtonOptions = setproperties(opt_newton; maxIter = 70, tol = 1e-8), detectBifurcation = 0, maxSteps = 150, saveSolEveryStep = 1)

	br, u1 = continuation(
		Fb, (x, r) -> Jacobian(x, r[1], r[2]),
		sol,  (1., 1.), (@lens _[1]),
		opts_br; recordFromSolution = (x,p) -> x.u[1])

BK.getSolx(br,1)
BK.getSolp(br,1)

# plot(br);title!("")

outfold, hist, flag = newton(
	Fb, (x, r) -> Jacobian(x, r[1], r[2]),
	br, 1;
	Jᵗ = (x, r) -> Jacobian(x, r[1], r[2]),
	d2F = (x, r, v1, v2) -> BorderedArray(-6 .* x.u .* v1.u .* v2.u, 0.),)
		# flag && printstyled(color=:red, "--> We found a Fold Point at α = ", outfold.p, ", from ", br.specialpoint[1].param,"\n")

outfoldco, hist, flag = continuation(
	Fb, (x, r) -> Jacobian(x, r[1], r[2]),
	br, 1, (@lens _[2]), opts_br;
	Jᵗ = (x, r) -> Jacobian(x, r[1], r[2]),
	d2F = ((x, r, v1, v2) -> BorderedArray(-6 .* x.u .* v1.u .* v2.u, 0.)), plot = false)

# try with newtonDeflation
# test with Newton deflation 1
deflationOp = DeflationOperator(2, 1.0, [zero(sol)])
soldef0 = BorderedArray([0.1], 0.0)
soldef1,  = newton(
	Fb, (x, r) -> Jacobian(x, r[1], r[2]),
	soldef0, (0., 1.),
	opt_newton, deflationOp)

push!(deflationOp, soldef1)

Random.seed!(1231)
# test with Newton deflation 2
soldef2, = newton(
	Fb, (x, r) -> Jacobian(x, r[1], r[2]),
	rmul!(soldef0,rand()),  (0., 1.),
	opt_newton, deflationOp)

# test indexing
deflationOp[1]

# test length
length(deflationOp)

# test pop!
pop!(deflationOp)

# test empty
empty!(deflationOp)

####################################################################################################
using KrylovKit

function Fr(x::RecursiveVec, p)
	r, s = p
	out = similar(x)
	for ii=1:length(x)
		out[ii] .= r .+  s .* x[ii] .- x[ii].^3
	end
	out
end

# there is no finite differences defined, so we need to provide a linearsolve
# we could also have used GMRES. We define a custom Jacobian which will be used for evaluation and jacobian inverse
struct JacobianR
	x
	s
end

# We express the jacobian operator
function (J::JacobianR)(dx)
	out = similar(dx)
	for ii=1:length(out)
		out[ii] .= (J.s .- 3 .* (J.x[ii]).^2) .* dx[ii]
	end
	return out
end

struct linsolveBd_r <: BK.AbstractBorderedLinearSolver end

function (l::linsolveBd_r)(J, dx)
	x = J.x
	out = similar(dx)
	for ii=1:length(out)
		out[ii] .= dx[ii] ./ (J.s .- 3 .* (x[ii]).^2)
	end
	out, true, 1
end

opt_newton0 = NewtonPar(tol = 1e-10, maxIter = 50, verbose = false, linsolver = linsolveBd_r())
	out0, hist, flag = newton(
		Fr, (x, p) -> JacobianR(x, p[1]),
		RecursiveVec([1 .+ 0.1*rand(10) for _ = 1:2]), (0., 1.),
		opt_newton0)

Base.:copyto!(dest::RecursiveVec, in::RecursiveVec) = copyto!(dest.vecs, in.vecs)

opts_br0 = ContinuationPar(dsmin = 0.001, dsmax = 0.2, ds= -0.01, pMin = -1.1, pMax = 1.1, newtonOptions = opt_newton0)

br0, u1 = continuation(
	Fr, (x, p) -> JacobianR(x, p[1]),
	out0, (0.9, 1.), (@lens _[1]), opts_br0;
	plot = false,
	recordFromSolution = (x,p) -> x[1][1])

br0, u1 = continuation(
	Fr, (x, p) -> JacobianR(x, p[1]),
	out0, (0.9, 1.), (@lens _[1]), opts_br0;
	tangentAlgo = BorderedPred(),
	plot = false,
	recordFromSolution = (x,p) -> x[1][1])

outfold, hist, flag = newton(
	Fr, (x, p) -> JacobianR(x, p[1]),
	br0, 1; #index of the fold point
	Jᵗ = (x, r) -> JacobianR(x, r[1]),
	d2F = (x, r, v1, v2) -> RecursiveVec([-6 .* x[ii] .* v1[ii] .* v2[ii] for ii=1:length(x)]),)
		# flag && printstyled(color=:red, "--> We found a Fold Point at α = ", outfold.p, ", from ", br0.specialpoint[1].param,"\n")

outfoldco, hist, flag = continuation(
	Fr, (x, p) -> JacobianR(x, p[1]),
	br0, 1,	(@lens _[2]), opts_br0;
	Jᵗ = (x, s) -> JacobianR(x, s[1]),
	d2F = ((x, r, v1, v2) -> RecursiveVec([-6 .* x[ii] .* v1[ii] .* v2[ii] for ii=1:length(x)])),
	tangentAlgo = SecantPred(), plot = false)

outfoldco, hist, flag = continuation(
	Fr, (x, p) -> JacobianR(x, p[1]),
	br0, 1, (@lens _[2]), opts_br0;
	Jᵗ = (x, s) -> JacobianR(x, s[1]),
	d2F = ((x, r, v1, v2) -> RecursiveVec([-6 .* x[ii] .* v1[ii] .* v2[ii] for ii=1:length(x)])),
	tangentAlgo = BorderedPred(), plot = false)

# try with newtonDeflation
# test with Newton deflation 1
deflationOp = DeflationOperator(2, 1.0, [out0])
soldef1, = newton(
	Fr, (x, p) -> JacobianR(x, 0.),
	out0, (0., 1.0),
	opt_newton0, deflationOp)

push!(deflationOp, soldef1)
