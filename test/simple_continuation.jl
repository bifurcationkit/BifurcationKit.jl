using PseudoArcLengthContinuation, LinearAlgebra

println("\n\n--> Continuation method")
k = 2
N = 10
F = (x, p)-> p .* x .+ x.^(k+1)/(k+1) .+ 1.0e-2
Jac_m = (x, p) -> diagm(0 => p  .+ x.^k)
opts = PseudoArcLengthContinuation.ContinuationPar(dsmax = 0.02, dsmin=1e-2, ds=0.0001, maxSteps = 140, pMin = -3)
x0 = 0.01 * ones(N)
opts.newtonOptions.tol     = 1e-8
opts.newtonOptions.verbose       = false

br1, sol, _ = @time PseudoArcLengthContinuation.continuation(F,Jac_m,x0,-1.5,opts,verbosity=0)

br2, sol, _ = @time PseudoArcLengthContinuation.continuation(F,Jac_m,x0,-1.5,opts,verbosity=0, printsolution = x -> norm(x,2))

# test for different norm
br3, sol, _ = @time PseudoArcLengthContinuation.continuation(F,Jac_m,x0,-1.5,opts,verbosity=0, normC = x -> norm(x,Inf64))

# plotBranch(br1,marker=:d);title!("")
# plotBranch!(br3,marker=:d);title!("")
