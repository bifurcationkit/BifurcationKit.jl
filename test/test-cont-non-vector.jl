# this test is designed to test the ability of the package to use a state space that is not an AbstractArray.
using Test
using PseudoArcLengthContinuation
const Cont = PseudoArcLengthContinuation
####################################################################################################
# We start with a simple Fold problem
using LinearAlgebra
function F0(x::Vector, r)
	out = r .+  x .- x.^3
end

opt_newton0 = Cont.NewtonPar(tol = 1e-11, verbose = true)
	out0, hist, flag = @time Cont.newton(
		x -> F0(x, 1.),
		[0.8],
		opt_newton0)

opts_br0 = Cont.ContinuationPar(dsmin = 0.001, dsmax = 0.1, ds= -0.01, pMax = 4.1, pMin = -1, newtonOptions = opt_newton0, detect_fold = true, detect_bifurcation = false)
	opts_br0.newtonOptions.maxIter = 70
	opts_br0.newtonOptions.tol = 1e-8
	opts_br0.maxSteps = 150

	br0, u1 = @time Cont.continuation(
		(x, r) -> F0(x, r),
		out0, 1.0,
		opts_br0, printsolution = x -> x[1])

# plotBranch(br0);title!("a")

outfold, hist, flag = @time Cont.newtonFold(
	(x, r) -> F0(x, r),
	# (x, r) -> diagm(0 => 1 .- 3 .* x.^2),
	# (x, r) -> diagm(0 => 1 .- 3 .* x.^2),
	# (x, r, v1, v2) -> -6 .* x .* v1 .* v2,
	br0, 2, #index of the fold point
	opts_br0.newtonOptions)
		flag && printstyled(color=:red, "--> We found a Fold Point at α = ", outfold.p, ", from ", br0.bifpoint[1][3],"\n")

####################################################################################################
# Here is a more involved example

function Fb(x::BorderedVector, r, s = 1.0)
	out = similar(x)
	out.u = r .+  s .* x.u .- (x.u).^3
	out.p = x.p - 0.0
	out
end

# there is no finite differences defined, so we need to provide a linearsolve
# we could also have used GMRES. We define a custom Jacobian which will be used for evaluation and jacobian inverse
struct jacobian
	x
	r
	s
end

# We express the jacobian operator
function (J::jacobian)(dx)
	out = similar(dx)
	out.u = (J.s .- 3 .* ((J.x).u).^2) .* dx.u
	out.p = dx.p
	return out
end

struct linsolveBd <: Cont.LinearSolver end

function (l::linsolveBd)(J, dx)
	x = J.x
	r = J.r
	out = similar(dx)
	out.u = dx.u ./ (J.s .- 3 .* (x.u).^2)
	out.p = dx.p
	out, true, 1
end

sol = BorderedVector([0.8], 0.0)

opt_newton = Cont.NewtonPar(tol = 1e-11, verbose = true,
		linsolve = linsolveBd())
out, hist, flag = @time Cont.newton(
		x -> Fb(x, 1., 1.),
		x -> jacobian(x, 1., 1.),
		sol,
		opt_newton)

opts_br = Cont.ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= -0.01, pMax = 4.1, pMin = -1, newtonOptions = opt_newton, detect_fold = true, detect_bifurcation = false)
	opts_br.newtonOptions.maxIter = 70
	opts_br.newtonOptions.tol = 1e-8
	opts_br.maxSteps = 150

	br, u1 = @time Cont.continuation(
		(x, r) -> Fb(x, r),
		(x, r) -> jacobian(x, r, 1.),
		out, 1.,
		opts_br, printsolution = x -> x.u[1])

# plotBranch(br);title!("")

outfold, hist, flag = @time Cont.newtonFold(
	(x, r) -> Fb(x, r),
	(x, r) -> jacobian(x, r, 1.),
	(x, r) -> jacobian(x, r, 1.),
	(x, r, v1, v2) -> BorderedVector(-6 .* x.u .* v1.u .* v2.u, 0.),
	br, 1, #index of the fold point
	opts_br.newtonOptions)
		flag && printstyled(color=:red, "--> We found a Fold Point at α = ", outfold.p, ", from ", br.bifpoint[1][3],"\n")

outfoldco, hist, flag = @time Cont.continuationFold(
	(x, r, s) -> Fb(x, r, s),
	(x, r, s) -> jacobian(x, r, s),
	(x, r, s) -> jacobian(x, r, s),
	s -> ((x, r, v1, v2) -> BorderedVector(-6 .* x.u .* v1.u .* v2.u, 0.)),
	br, 1,
	1.0, plot = false,
	opts_br)
