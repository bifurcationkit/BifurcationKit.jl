using Revise, Plots
using BifurcationKit, Setfield, ForwardDiff
const BK = BifurcationKit

function FbpSecBif(u, p)
	return @. -u * (p + u * (2-5u)) * (p -.15 - u * (2+20u))
end

D(f, x, p, dx) = ForwardDiff.derivative(t -> f(x .+ t .* dx, p), 0.)
dFbpSecBif(x,p)         =  ForwardDiff.jacobian( z -> FbpSecBif(z,p), x)
d1FbpSecBif(x,p,dx1)         = D((z, p0) -> FbpSecBif(z, p0), x, p, dx1)
d2FbpSecBif(x,p,dx1,dx2)     = D((z, p0) -> d1FbpSecBif(z, p0, dx1), x, p, dx2)
d3FbpSecBif(x,p,dx1,dx2,dx3) = D((z, p0) -> d2FbpSecBif(z, p0, dx1, dx2), x, p, dx3)
jet = (FbpSecBif, dFbpSecBif, d2FbpSecBif, d3FbpSecBif)

# options for Krylov-Newton
opt_newton = NewtonPar(tol = 1e-9, verbose = false, maxIter = 20)
# options for continuation
opts_br = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds = 0.01, pMax = 0.4, pMin = -0.5, detectBifurcation = 3, nev = 2, newtonOptions = opt_newton, maxSteps = 100, nInversion = 4, tolBisectionEigenvalue = 1e-8, dsminBisection = 1e-9)

deflationOp = DeflationOperator(2.0, dot, 1.0, ([[0.]]))

br, _ = @time continuation(
	FbpSecBif, dFbpSecBif,
	-0.25, (@lens _),
	setproperties(optcont; ds = 0.001, maxSteps = 1000, pMax = 0.7, pMin = -0.5, newtonOptions = setproperties(optnew; tol = 1e-9, maxIter = 300, verbose = false)),
	deflationOp;
	verbosity = 1,
	maxBranches = 100,
	perturbSolution = (sol, p, id) -> sol .+ 0.1rand(),
	printSolution = (x, p) -> (x[1]) ,
	normN = x -> norm(x, Inf64),
	# callbackN = (x, f, J, res, iteration, itlinear, options; kwargs...) ->(true)
	)
