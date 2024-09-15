# using Revise
using BifurcationKit, SparseArrays, LinearAlgebra, Plots
const BK = BifurcationKit

k = 2
F = (x, p) -> (@. p + x - x^(k+1)/(k+1))

prob = BK.BifurcationProblem(F, [0.8], 1., (@lens _))

opt_newton0 = BK.NewtonPar(tol = 1e-11, verbose = true)
out0 = BK.newton(prob, opt_newton0)

opts = BK.ContinuationPar(dsmax = 0.1, dsmin = 1e-3, ds = -0.001, max_steps = 130, p_min = -3., p_max = 3., save_sol_every_step = 0, detect_bifurcation = true, newton_options = NewtonPar(verbose = false))


println("\n"*"#"^120)
br0 = BK.continuation(F,Jac_m,[0.8],1.,(@lens _),opts;verbosity=0,record_from_solution = (x,p) -> x[1]) #130 => 73 points
    plot(br0);title!("")

br0.branch[:,:]'
br0.n_unstable
br0.n_imag
br0.sol
br0.eig
