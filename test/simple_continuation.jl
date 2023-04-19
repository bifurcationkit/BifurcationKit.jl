# using Revise
# using Plots
using Test
using BifurcationKit, LinearAlgebra, Setfield, SparseArrays
const BK = BifurcationKit

k = 2
N = 1
F = (x, p) -> p[1] .* x .+ x.^(k+1)/(k+1) .+ 0.01
Jac_m = (x, p) -> diagm(0 => p[1] .+ x.^k)
####################################################################################################
# test creation of specific scalar product
_dt = BK.DotTheta(dot)
_dt = BK.DotTheta()
# tests for the predictors
BK.mergefromuser(1., (a = 1,))
BK.mergefromuser(rand(2), (a = 1,))
BK.mergefromuser((1, 2), (a = 1,))

BK.Fold(rand(2), 0.1, 0.1, (@lens _.p), rand(2), rand(2),1., :fold) |> BK.type
BK._displayLine(1, 1, (1,1))
BK._displayLine(1, nothing, (1,1))
####################################################################################################
# test branch kinds
BK.FoldCont()
BK.HopfCont()
BK.PDCont()

# Codim2 periodic orbit
BK.FoldPeriodicOrbitCont()
BK.PDPeriodicOrbitCont()
BK.NSPeriodicOrbitCont()
####################################################################################################
# test continuation algorithm
BK.empty(Natural())
BK.empty(PALC())
BK.empty(PALC(tangent = Bordered()))
BK.empty(BK.MoorePenrose(tangent = PALC(tangent = Bordered())))
BK.empty(PALC(tangent = Polynomial(Bordered(), 2, 6, rand(1))))
####################################################################################################
normInf(x) = norm(x, Inf)

opts = ContinuationPar(dsmax = 0.051, dsmin = 1e-3, ds=0.001, maxSteps = 140, pMin = -3., saveSolEveryStep = 0, newtonOptions = NewtonPar(tol = 1e-8, verbose = false), saveEigenvectors = false, detectBifurcation = 3)
x0 = 0.01 * ones(N)

prob = BK.BifurcationProblem(F, x0, -1.5, (@lens _); J = Jac_m)
BK.isInplace(prob)
BK.getVectorType(prob)
show(prob)

br0 = @time continuation(prob, PALC(doArcLengthScaling = true), opts) #(17.98 k allocations: 1.155 MiB)
BK.getfirstusertype(br0)
BK.propertynames(br0)
BK.computeEigenvalues(opts)
BK.saveEigenvectors(opts)
BK.from(br0)
BK.getProb(br0)
br0[1]
br0[end]
BK.bifurcation_points(br0)

branch = Branch(br0, rand(2));
branch[end]
###### start at pMin and see if it continues the solution
for alg in (Natural(), PALC(), PALC(tangent = Bordered()), Multiple(copy(x0), 0.01,13), PALC(tangent=Polynomial(Bordered(), 2, 6, copy(x0))), MoorePenrose())
	br0 = continuation(reMake(prob, params = opts.pMin), alg, opts)
	@test length(br0) > 10
	br0 = continuation(reMake(prob, params = opts.pMax), Natural(), ContinuationPar(opts, ds = -0.01))
	@test length(br0) > 10
end

# test with callbacks
br0 = continuation(prob, PALC(), (@set opts.maxSteps = 3), callbackN = (state; kwargs...)->(true))

###### Used to check type stability of the methods
# using RecursiveArrayTools
iter = ContIterable(prob, PALC(), opts)
state = iterate(iter)[1]
# test copy, copyto!
state1 = copy(state);copyto!(state1, state)
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

opts = ContinuationPar(opts; detectBifurcation = 3, saveEigenvectors=true)
br1 = continuation(prob, PALC(), opts) #(14.28 k allocations: 1001.500 KiB)
show(br1)
length(br1)
br1[1]
BK.eigenvals(br1,20)
BK.eigenvec(br1,20,1)
BK.haseigenvector(br1)
BK.getvectortype(br1)
BK.getvectoreltype(br1)
BK.hassolution(Branch(br1, nothing)) # test the method
br1.param
BK.getParams(br1)

@set! prob.recordFromSolution = (x,p) -> norm(x,2)
br2 = continuation(prob, PALC(), opts)
BK.arcLengthScaling(0.5, PALC(), BorderedArray(rand(2), 0.1), true)

# test for different norms
br3 = continuation(prob, PALC(), opts, normC = normInf)

# test for linesearch in Newton method
opts = @set opts.newtonOptions.linesearch = true
br4 = continuation(prob, PALC(), opts, normC = normInf) # (15.61 k allocations: 1.020 MiB)

# test for different ways to solve the bordered linear system arising during the continuation step
opts = @set opts.newtonOptions.linesearch = false
br5 = continuation(prob, PALC(bls = BorderingBLS()), opts, normC = normInf)

br5 = continuation(prob, PALC(bls = MatrixBLS()), opts, normC = normInf)

# test for stopping continuation based on user defined function
finaliseSolution = (z, tau, step, contResult; k...) -> (step < 20)
br5a = continuation(prob, PALC(), opts, finaliseSolution = finaliseSolution)
@test length(br5a.branch) == 21

# test for different predictors
br6 = continuation(prob, PALC(tangent = Secant()), opts)

optsnat = setproperties(opts; ds = 0.001, dsmax = 0.1, dsmin = 0.0001)
br7 = continuation((@set prob.recordFromSolution = (x,p)->x[1]), Natural(), optsnat)

# tangent prediction with Bordered predictor
br8 = continuation(prob, PALC(tangent = Bordered()), opts)

# tangent prediction with Multiple predictor
opts9 = (@set opts.newtonOptions.verbose=false)
	opts9 = ContinuationPar(opts9; maxSteps = 48, ds = 0.015, dsmin = 1e-5, dsmax = 0.05)
	br9 = continuation(prob,  Multiple(copy(x0), 0.01,13), opts9)
	BK.empty!(Multiple(copy(x0), 0.01, 13))
	# plot(br9, title = "$(length(br9))",marker=:d, vars=(:param, :x),plotfold=false)
## same but with failed prediction
opts9_1 = ContinuationPar(opts9, dsmax = 0.2, maxSteps = 125, ds = 0.1)
	@set! opts9_1.newtonOptions.tol = 1e-14
	@set! opts9_1.newtonOptions.verbose = false
	@set! opts9_1.newtonOptions.maxIter = 3
	br9_1 = continuation(prob,  Multiple(copy(x0), 1e-4,7), opts9_1, verbosity = 0)
	@test length(br9_1) == 126
	BK.empty!(Multiple(copy(x0), 0.01, 13))


# tangent prediction with Polynomial predictor
polpred = Polynomial(Bordered(), 2, 6, x0)
	opts9 = (@set opts.newtonOptions.verbose=false)
	opts9 = ContinuationPar(opts9; maxSteps = 76, ds = 0.005, dsmin = 1e-4, dsmax = 0.02, plotEveryStep = 3,)
	br10 = continuation(prob, PALC(tangent = polpred), opts9,
	plot=false,
	)
	# plot(br10) |> display
	polpred(0.1)
	BK.empty!(polpred)
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

# tangent prediction with Moore Penrose
opts11 = (@set opts.newtonOptions.verbose=false)
opts11 = ContinuationPar(opts11; maxSteps = 50, ds = 0.015, dsmin = 1e-5, dsmax = 0.15)
br11 = continuation(prob, MoorePenrose(), opts11; verbosity = 0)
br11 = continuation(prob, MoorePenrose(method = BK.pInv), opts11; verbosity = 0)
br11 = continuation(prob, MoorePenrose(method = BK.iterative), opts11; verbosity = 0)
# plot(br11)

MoorePenrose(method = BK.iterative)


# further testing with sparse Jacobian operator
prob_sp = @set prob.VF.J = (x, p) -> SparseArrays.spdiagm(0 => p .+ x.^k)
brsp = continuation(prob_sp, PALC(), opts)
brsp = continuation(prob_sp, PALC(bls = BK.BorderingBLS(solver = BK.DefaultLS(), checkPrecision = false)), opts)
brsp = continuation(prob_sp, PALC(bls = BK.MatrixBLS()), opts)
# plot(brsp,marker=:d)
####################################################################################################
# check bounds for all predictors / correctors
for talgo in (Bordered(), Secant())
	brbd  = continuation(prob, PALC(tangent = talgo), opts; verbosity = 0)
	@test length(brbd) > 2
end
prob.u0 .= ones(N)*3
@set! prob.params = -3.
brbd = continuation(prob, PALC(), ContinuationPar(opts, pMax = -2))
@set! prob.params = -3.2
brbd  = continuation(prob, PALC(), ContinuationPar(opts, pMax = -2), verbosity = 0)
@test isnothing(brbd)
####################################################################################################
# testing when starting with 2 points on the branch
opts = BK.ContinuationPar(dsmax = 0.051, dsmin = 1e-3, ds = 0.001, maxSteps = 140, pMin = -3., newtonOptions = NewtonPar(verbose = false), detectBifurcation = 3)
x0 = 0.01 * ones(2)

prob = BK.BifurcationProblem(F, x0, -1.5, (@lens _); J = Jac_m)
x0 = newton(prob, opts.newtonOptions)
x1 = newton((@set prob.params = -1.45), opts.newtonOptions)

br0 = continuation(prob, PALC(), opts, verbosity=3)
BK.getEigenelements(br0, br0.specialpoint[1])
BK.detectLoop(br0, x0.u, -1.45)

@set! prob.params = -1.5
br1 = continuation(prob, PALC(), ContinuationPar(opts; ds = -0.001))

br2 = continuation(prob, x0.u, -1.5, x1.u, -1.45, PALC(tangent = Bordered()), (@lens _), opts)
####################################################################################################
# test for computing both sides
br3 = continuation(prob, PALC(tangent = Bordered()), opts; bothside = true)
####################################################################################################
# test for deflated continuation
BK._perturbSolution(1, 0, 1)
BK._acceptSolution(1, 0)
BK.DCState(rand(2))

prob = BK.BifurcationProblem(F, [0.], 0.5, (@lens _); J = Jac_m)
alg = BK.DefCont(deflationOperator = DeflationOperator(2, .001, [[0.]]),
	perturbSolution = (x,p,id) -> (x .+ 0.1 .* rand(length(x)))
	)
brdc = continuation(prob, alg,
	ContinuationPar(opts, ds = -0.001, maxSteps = 800, newtonOptions = NewtonPar(verbose = false, maxIter = 6), plotEveryStep = 40, detectBifurcation = 3);
	plot=false, verbosity = 0,
	callbackN = BK.cbMaxNorm(1e3))

# test that the saved points are true solutions
norminf(x) = norm(x, Inf)
for i in 1:length(brdc)
	brs = brdc[i]
	for j=1:length(brs.sol)
		res = BifurcationKit.residual(prob, brs.sol[j].x, BifurcationKit.setParam(prob, brs.sol[j].p)) |> norminf
		@test res < brs.contparams.newtonOptions.tol
	end
end

lastindex(brdc)
brdc[1]
length(brdc)

F2(u,p) = @. -u * (p + u * (2-5u)) * (p - .15 - u * (2+20u))
prob2 = BK.BifurcationProblem(F2, [0.], 0.3, (@lens _))
brdc = continuation(prob2,
	BK.DefCont(deflationOperator = DeflationOperator(2, .001, [[0.], [0.05]]); maxBranches = 6),
	ContinuationPar(opts, dsmin = 1e-4, ds = -0.002, maxSteps = 800, newtonOptions = NewtonPar(verbose = false, maxIter = 15), plotEveryStep = 40, detectBifurcation = 3, pMin = -0.8);
	plot=false, verbosity = 0,
	callbackN = BK.cbMaxNorm(1e6))
