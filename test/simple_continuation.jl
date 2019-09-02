using PseudoArcLengthContinuation, LinearAlgebra, Setfield, SparseArrays
const Cont = PseudoArcLengthContinuation

println("\n\n--> Continuation method")
k = 2
N = 10
F = (x, p) -> p .* x .+ x.^(k+1)/(k+1) .+ 1.0e-2
Jac_m = (x, p) -> diagm(0 => p  .+ x.^k)

normInf = x -> norm(x,Inf64)

opts = Cont.ContinuationPar(dsmax = 0.02, dsmin=1e-2, ds=0.0001, maxSteps = 140, pMin = -3)
x0 = 0.01 * ones(N)
opts.newtonOptions.tol     = 1e-8
opts.newtonOptions.verbose = false

br1, sol, _ = @time Cont.continuation(F,Jac_m,x0,-1.5,opts,verbosity=0) #(18.19 k allocations: 1.222 MiB)
show(br1)

br2, sol, _ = @time Cont.continuation(F,Jac_m,x0,-1.5,opts,verbosity=0, printsolution = x -> norm(x,2))

# test for different norms
br3, sol, _ = @time Cont.continuation(F,Jac_m,x0,-1.5,opts,verbosity=0, normC = normInf)

# test for linesearch in Newton method
opts.newtonOptions.linesearch = true
br4, sol, _ = @time Cont.continuation(F,Jac_m,x0,-1.5,opts,verbosity=0, normC = normInf) # (19.03 k allocations: 1.254 MiB)

# test for different ways to solve the bordered linear system arising during the continuation step
opts.newtonOptions.linesearch = false
br5, sol, _ = @time Cont.continuation(F,Jac_m,x0,-1.5,opts,verbosity=0, normC = normInf, linearalgo = :bordering)

br5, sol, _ = @time Cont.continuation(F,Jac_m,x0,-1.5,opts,verbosity=0, normC = normInf, linearalgo = :full)

# test for stopping continuation based on user defined function
finaliseSolution = (z, tau, step, contResult) -> (step < 20)
br5a, sol, _ = @time Cont.continuation(F,Jac_m,x0,-1.5,opts,verbosity=0, finaliseSolution = finaliseSolution)
@test length(br5a.branch) == 22

# test for different predictors
opts.secant = true
br6, sol, _ = @time Cont.continuation(F,Jac_m,x0,-1.5,opts,verbosity=0)

optsnat = @set opts.natural = true
optsnat.ds = 0.001
optsnat.dsmax = 0.02
optsnat.dsmin = 0.0001
br7, sol, _ = @time Cont.continuation(F,Jac_m,x0,-1.5,optsnat,verbosity=0)

# tangent prediction with Bordered predictor
opts.secant = false
br8, sol, _ = @time Cont.continuation(F,Jac_m,x0,-1.5,opts,verbosity=0)


# further testing with sparse Jacobian operator
Jac_sp_simple = (x, p) -> SparseArrays.spdiagm(0 => p  .+ x.^k)
brsp, sol, _ = @time Cont.continuation(F,Jac_sp_simple,x0,-1.5,opts,verbosity=0)
brsp, sol, _ = @time Cont.continuation(F,Jac_sp_simple,x0,-1.5,opts,verbosity=0, printsolution = x -> norm(x,2))
brsp, sol, _ = @time Cont.continuation(F,Jac_sp_simple,x0,-1.5,opts,verbosity=0,linearalgo = :bordering)
brsp, sol, _ = @time Cont.continuation(F,Jac_sp_simple,x0,-1.5,opts,verbosity=0,linearalgo = :full)
# plotBranch(br1,marker=:d);title!("")
# plotBranch!(br3,marker=:d);title!("")
