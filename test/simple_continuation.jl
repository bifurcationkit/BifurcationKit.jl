# using Revise
using PseudoArcLengthContinuation, LinearAlgebra, Setfield, SparseArrays
const PALC = PseudoArcLengthContinuation
# using Plots

k = 2
N = 10
F = (x, p) -> p .* x .+ x.^(k+1)/(k+1) .+ 1.0e-2
Jac_m = (x, p) -> diagm(0 => p  .+ x.^k)

normInf = x -> norm(x, Inf)

opts = PALC.ContinuationPar(dsmax = 0.051, dsmin = 1e-3, ds=0.001, maxSteps = 140, pMin = -3., saveSolEveryNsteps = 0, newtonOptions = NewtonPar(tol = 1e-8, verbose = false), computeEigenValues = false, saveEigenvectors = false)
x0 = 0.01 * ones(N)

opts = @set opts.doArcLengthScaling = true
br0, sol, _ = @time PALC.continuation(F,Jac_m,x0,-1.5,opts,verbosity=0) #(15.28 k allocations: 1.055 MiB)

###### Used to check type stability of the methods
# using RecursiveArrayTools
# iter = PALC.PALCIterable(F,Jac_m,x0,-1.5,opts,verbosity=0)
# state = iterate(iter)[1]
# contRes = PALC.initContRes(iter, state)
# @time continuation!(iter, state, contRes)
#
# typeof(contRes)
#
# state = iterate(iter)[1]
#     contRes = PALC.initContRes(iter, state)
#     @code_warntype continuation!(iter, state, contRes)
#####

opts = @set opts.doArcLengthScaling = false
br1, sol, _ = @time PALC.continuation(F,Jac_m,x0,-1.5,opts,verbosity=0) #(14.28 k allocations: 1001.500 KiB)
show(br1)

br2, sol, _ = @time PALC.continuation(F,Jac_m,x0,-1.5,opts,verbosity=0, printSolution = (x,p) -> norm(x,2))

# test for different norms
br3, sol, _ = @time PALC.continuation(F,Jac_m,x0,-1.5,opts,verbosity=0, normC = normInf)

# test for linesearch in Newton method
opts = @set opts.newtonOptions.linesearch = true
br4, sol, _ = @time PALC.continuation(F,Jac_m,x0,-1.5,opts,verbosity=0, normC = normInf) # (15.61 k allocations: 1.020 MiB)

# test for different ways to solve the bordered linear system arising during the continuation step
opts = @set opts.newtonOptions.linesearch = false
br5, sol, _ = @time PALC.continuation(F,Jac_m,x0,-1.5,opts,verbosity=0, normC = normInf, linearAlgo = PALC.BorderingBLS())

br5, sol, _ = @time PALC.continuation(F,Jac_m,x0,-1.5,opts,verbosity=0, normC = normInf, linearAlgo = PALC.MatrixBLS())

# test for stopping continuation based on user defined function
finaliseSolution = (z, tau, step, contResult) -> (step < 20)
br5a, sol, _ = @time PALC.continuation(F,Jac_m,x0,-1.5,opts,verbosity=2, finaliseSolution = finaliseSolution)
@test length(br5a.branch) == 21

# test for different predictors
br6, sol, _ = @time PALC.continuation(F,Jac_m,x0,-1.5,opts,verbosity=0, tangentAlgo = PALC.SecantPred())

optsnat = setproperties(opts; ds = 0.001, dsmax = 0.02, dsmin = 0.0001)
br7, sol, _ = @time PALC.continuation(F,Jac_m,x0,-1.5,optsnat,verbosity=0, tangentAlgo = PALC.NaturalPred())

# tangent prediction with Bordered predictor
br8, sol, _ = @time PALC.continuation(F,Jac_m,x0,-1.5,opts,verbosity=0, tangentAlgo = PALC.BorderedPred())


# further testing with sparse Jacobian operator
Jac_sp_simple = (x, p) -> SparseArrays.spdiagm(0 => p  .+ x.^k)
brsp, sol, _ = @time PALC.continuation(F,Jac_sp_simple,x0,-1.5,opts,verbosity=0)
brsp, sol, _ = @time PALC.continuation(F,Jac_sp_simple,x0,-1.5,opts,verbosity=0, printSolution = (x,p) -> norm(x,2))
brsp, sol, _ = @time PALC.continuation(F,Jac_sp_simple,x0,-1.5,opts,verbosity=0,linearAlgo = PALC.BorderingBLS())
brsp, sol, _ = @time PALC.continuation(F,Jac_sp_simple,x0,-1.5,opts,verbosity=0,linearAlgo = PALC.MatrixBLS())
# plotBranch(br1,marker=:d);title!("")
# plotBranch!(br8,marker=:d);title!("")
