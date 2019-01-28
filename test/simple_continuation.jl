using PseudoArcLengthContinuation, LinearAlgebra

println("\n\n--> Continuation method")
k = 2
N = 10
F = (x, p)-> p .* x .+ x.^(k+1)/(k+1) .+ 1.0e-2
Jac = (x, p, v) -> p .* v .+ (x.^k).*v
Jac_m = (x, p) -> diagm(0 => p  .+ x.^k)
opts = PseudoArcLengthContinuation.ContinuationPar(dsmax = 0.1, dsmin=1e-2, ds=0.01, maxSteps = 24, pMin = -3)
x0 = 0.01 * ones(N)
opts.newtonOptions.tol     = 1e-8
opts.newtonOptions.verbose       = false
opts.newtonOptions.damped        = true

br, sol, _ = @time PseudoArcLengthContinuation.continuation(F,Jac_m,x0,-1.5,opts,verbosity=0)
