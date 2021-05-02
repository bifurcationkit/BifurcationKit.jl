# using Revise, Plots, Test
using BifurcationKit, LinearAlgebra, Setfield, SparseArrays
const BK = BifurcationKit

k = 2
N = 1
F = (x, p) -> p[1] .* x .+ x.^(k+1)/(k+1) .+ 0.01
Jac_m = (x, p) -> diagm(0 => p[1] .+ x.^k)

####################################################################################################
# test creation of specific scalar product
_dt = BK.DotTheta()
# tests for the predictors
BK.emptypredictor!(nothing)
BK.mergefromuser(1., (a=1,))
BK.mergefromuser(rand(2), (a=1,))
BK.mergefromuser((1,2), (a=1,))
####################################################################################################

normInf = x -> norm(x, Inf)

opts = ContinuationPar(dsmax = 0.051, dsmin = 1e-3, ds=0.001, maxSteps = 140, pMin = -3., saveSolEveryStep = 0, newtonOptions = NewtonPar(tol = 1e-8, verbose = false), saveEigenvectors = false, detectBifurcation = 0)
x0 = 0.01 * ones(N)

opts = @set opts.doArcLengthScaling = true
br0, = @time continuation(F, Jac_m, x0, -1.5, (@lens _), opts) #(16.12 k allocations: 772.250 KiB)
BK.getfirstusertype(br0)
BK.propertynames(br0)
BK.computeEigenvalues(opts)
BK.computeEigenvectors(opts)

# test with callbacks
br0, = continuation(F,Jac_m,x0, -1.5, (@lens _), (@set opts.maxSteps = 3), callbackN = (x, f, J, res, iteration, itlinear, optionsN; kwargs...)->(@show nothing;true))

###### Used to check type stability of the methods
# using RecursiveArrayTools
iter = ContIterable(F, Jac_m, x0, -1.5, (@lens _), opts)
state = iterate(iter)[1]
contRes = ContResult(iter, state)
continuation!(iter, state, contRes)
eltype(iter)
length(iter)
#
typeof(contRes)

# state = iterate(iter)[1]
# 	 contRes = BK.ContResult(iter, state)
# 	 @code_warntype continuation!(iter, state, contRes)
#####

opts = ContinuationPar(opts;detectBifurcation = 1,saveEigenvectors=true)
br1, sol, _ = continuation(F,Jac_m,x0,-1.5, (@lens _),opts) #(14.28 k allocations: 1001.500 KiB)
show(br1)
length(br1)
br1[1]
BK.eigenvals(br1,20)
BK.eigenvec(br1,20,1)
BK.haseigenvector(br1)
BK.getvectortype(br1)
BK.getvectoreltype(br1)
br1.param
br1.params


br2, = continuation(F,Jac_m,x0,-1.5, (@lens _),opts, printSolution = (x,p) -> norm(x,2))

# test for different norms
br3, = continuation(F,Jac_m,x0,-1.5, (@lens _),opts, normC = normInf)

# test for linesearch in Newton method
opts = @set opts.newtonOptions.linesearch = true
br4, = continuation(F,Jac_m,x0,-1.5, (@lens _),opts, normC = normInf) # (15.61 k allocations: 1.020 MiB)

# test for different ways to solve the bordered linear system arising during the continuation step
opts = @set opts.newtonOptions.linesearch = false
br5, = continuation(F,Jac_m,x0,-1.5, (@lens _),opts, normC = normInf, linearAlgo = BorderingBLS())

br5, = continuation(F,Jac_m,x0,-1.5, (@lens _),opts, normC = normInf, linearAlgo = MatrixBLS())

# test for stopping continuation based on user defined function
finaliseSolution = (z, tau, step, contResult; k...) -> (step < 20)
br5a, = continuation(F,Jac_m,x0,-1.5, (@lens _),opts, finaliseSolution = finaliseSolution)
@test length(br5a.branch) == 21

# test for different predictors
br6, = continuation(F,Jac_m,x0,-1.5, (@lens _),opts, tangentAlgo = SecantPred())

optsnat = setproperties(opts; ds = 0.001, dsmax = 0.1, dsmin = 0.0001)
br7, = continuation(F,Jac_m,x0,-1.5, (@lens _),optsnat, tangentAlgo = NaturalPred(),printSolution = (x,p)->x[1])

# tangent prediction with Bordered predictor
br8, sol, _ = continuation(F,Jac_m,x0,-1.5, (@lens _),opts, tangentAlgo = BorderedPred(),printSolution = (x,p)->x[1])

# tangent prediction with Multiple predictor
opts9 = (@set opts.newtonOptions.verbose=true)
	opts9 = ContinuationPar(opts9; maxSteps = 48, ds = 0.015, dsmin = 1e-5, dsmax = 0.05)
	br9, = continuation(F,Jac_m,x0,-1.5, (@lens _),opts9,
	printSolution = (x,p)->x[1],
	tangentAlgo = MultiplePred(copy(x0), 0.01,13)
	)
	BK.emptypredictor!(BK.MultiplePred(copy(x0), 0.01,13))
	# plot(br9, title = "$(length(br9))",marker=:d,vars=(:p,:sol),plotfold=false)

# tangent prediction with Polynomial predictor
polpred = PolynomialPred(BorderedPred(),2,6,x0)
	opts9 = (@set opts.newtonOptions.verbose=false)
	opts9 = ContinuationPar(opts9; maxSteps = 76, ds = 0.005, dsmin = 1e-4, dsmax = 0.02, plotEveryStep = 3,)
	br10, = continuation(F, Jac_m, x0, -1.5, (@lens _), opts9,
	tangentAlgo = polpred, plot=false,
	printSolution = (x,p)->x[1],
	)
	# plot(br10) |> display
	polpred(0.1)
	BK.emptypredictor!(polpred)
	# plot(br10, title = "$(length(br10))",marker=:dplot,fold=false)
	# plot!(br9)

# polpred(0.0)
# 	for _ds in LinRange(-0.51,.1,161)
# 		_x,_p = polpred(_ds)
# 		# @show _ds,_x[1], _p
# 		scatter!([_p],[_x[1]],label="",color=:blue, markersize = 1)
# 	end
# 	scatter!(polpred.parameters,map(x->x[1],polpred.solutions), color=:green)
# 	scatter!([polpred(0.01)[2]],[polpred(0.01)[1][1]],marker=:cross)
# 	# title!("",ylims=(-0.1,0.22))
# # BK.isready(polpred)
# # polpred.coeffsPar

# test for polynomial predictor interpolation
# polpred = BK.PolynomialPred(4,9,x0)
# 	for (ii,v) in enumerate(LinRange(-5,1.,10))
# 		if length(polpred.arclengths)==0
# 			push!(polpred.arclengths, 0.1)
# 		else
# 			push!(polpred.arclengths, polpred.arclengths[end]+0.1)
# 		end
# 		push!(polpred.solutions, [v])
# 		push!(polpred.parameters, 1-v^2+0.001v^4)
# 	end
# 	BK.updatePred!(polpred)
# 	polpred(-0.5)
#
# plot()
# 	scatter(polpred.parameters, reduce(vcat,polpred.solutions))
# 	for _ds in LinRange(-1.5,.75,121)
# 		_x,_p = polpred(_ds)
# 		scatter!([_p],[_x[1]],label="", color=:blue, markersize = 1)
# 	end
# 	scatter!([polpred(0.1)[2]],[polpred(0.1)[1][1]],marker=:cross)
# 	title!("",)

# further testing with sparse Jacobian operator
Jac_sp_simple = (x, p) -> SparseArrays.spdiagm(0 => p  .+ x.^k)
brsp, = continuation(F,Jac_sp_simple,x0,-1.5, (@lens _),opts)
brsp, = continuation(F,Jac_sp_simple,x0,-1.5, (@lens _),opts, printSolution = (x,p) -> norm(x,2))
brsp, = continuation(F,Jac_sp_simple,x0,-1.5, (@lens _),opts,linearAlgo = BK.BorderingBLS())
brsp, = continuation(F,Jac_sp_simple,x0,-1.5, (@lens _),opts,linearAlgo = BK.MatrixBLS())
# plotBranch(br1,marker=:d);title!("")
# plotBranch!(br8,marker=:d);title!("")
####################################################################################################
# check bounds for all predictors / correctors
for talgo in [BorderedPred(), SecantPred()]
	brbd,  = continuation(F,Jac_m,x0,-3.0, (@lens _), opts; tangentAlgo = talgo, verbosity = 0)
	@test length(brbd) > 2
end
brbd,  = continuation(F,Jac_m,ones(N)*3,-3.0, (@lens _), ContinuationPar(opts, pMax = -2))
brbd,  = continuation(F,Jac_m,ones(N)*3,-3.2, (@lens _), ContinuationPar(opts, pMax = -2), verbosity = 3)
@test isnothing(brbd)
####################################################################################################
# testing when starting with 2 points on the branch
opts = BK.ContinuationPar(dsmax = 0.051, dsmin = 1e-3, ds=0.001, maxSteps = 140, pMin = -3., saveSolEveryStep = 0, newtonOptions = NewtonPar(verbose = false), detectBifurcation = 3)
x0 = 0.01 * ones(2)

x0, = newton(F,Jac_m,x0, -1.5, opts.newtonOptions)
x1, = newton(F,Jac_m,x0, -1.45, opts.newtonOptions)

br0, = continuation(F,Jac_m, x0, -1.5, (@lens _), opts, verbosity=0)
BK.getEigenelements(br0, br0.bifpoint[1])

br1, = continuation(F,Jac_m, x1, -1.45, x0, -1.5, (@lens _), ContinuationPar(opts; ds = -0.001))

br2, = continuation(F,Jac_m,x0, -1.5, x1, -1.45, (@lens _), opts; tangentAlgo = BorderedPred())
####################################################################################################
# test for computing both sides
br3, = continuation(F,Jac_m,x0, -1.5, (@lens _), opts; tangentAlgo = BorderedPred(), bothside = true)
####################################################################################################
# test for deflated continuation
brdc, = continuation(F,Jac_m, 0.5, (@lens _),
	ContinuationPar(opts, ds = -0.001, maxSteps = 800, newtonOptions = NewtonPar(verbose = false, maxIter = 6), plotEveryStep = 40),
	DeflationOperator(2.0, dot, .001, [[0.]]); showplot=false, verbosity = 1,
	perturbSolution = (x,p,id) -> (x  .+ 0.1 .* rand(length(x))),
	callbackN = (x, f, J, res, iteration, itlinear, options; kwargs...) -> res <1e3)
