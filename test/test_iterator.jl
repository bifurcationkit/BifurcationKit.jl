# using Revise
using PseudoArcLengthContinuation, SparseArrays, LinearAlgebra, Plots, Setfield
const PALC = PseudoArcLengthContinuation

normInf = x -> norm(x, Inf)

k = 2
F = (x, p) -> (@. p + x - x^(k+1)/(k+1))
Jac_m = (x, p) -> diagm(0 => 1  .- x.^k)

opt_newton0 = PALC.NewtonPar(tol = 1e-11, verbose = true)
	out0, hist, flag = @time PALC.newton(
		F, [0.8], 1, opt_newton0)

opts = PALC.ContinuationPar(dsmax = 0.1, dsmin = 1e-3, ds = -0.001, maxSteps = 130, pMin = -3., pMax = 3., saveSolEveryNsteps = 0, computeEigenValues = true, detectBifurcation = true, newtonOptions = NewtonPar(verbose = false))


println("\n"*"#"^120)
	br0, u1, _ = @time PALC.continuation(F,Jac_m,[0.8],1.,(@lens _),opts;verbosity=0,printSolution = (x,p) -> x[1]) #130 => 73 points
	plot(br0);title!("")

br0.branch[:,:]'
br0.n_unstable
br0.n_imag
br0.sol
br0.eig
