# using Revise
using Test, BifurcationKit, LinearAlgebra, Setfield, ForwardDiff
const BK = BifurcationKit

norminf = (x)->norm(x, Inf)
source_term(x; a = 0.5, b = 0.01) = 1 + (x + a*x^2)/(1 + b*x^2)
limits(x,i,N, b) = (i<1||i>N) ? b : x[i]

function F_chan(x, p)
	α, β = p
	N = length(x)
	f = similar(x)
	N = length(x)
	ind(x,i) = limits(x,i,N,β)
	for i=1:N
		f[i] = (ind(x,i-1) - 2 * x[i] + ind(x,i+1)) * (N-1)^2 + α * source_term(x[i], b = β)
	end
	return f
end

J_chan(u0, p) = ForwardDiff.jacobian(u -> F_chan(u,p), u0)
jet = BK.get3Jet(F_chan, J_chan)
par_chan = (α = 3.3, β = 0.01)

n = 101
	sol = [(i-1)*(n-i)/n^2+0.1 for i=1:n]
	opt_newton = NewtonPar(tol = 1e-9, verbose = true)
	out, hist, flag = newton(jet[1], jet[2], sol, par_chan,
							opt_newton, normN = norminf)

# _J = J_chan(sol, par_chan)
# heatmap(_J - _J', yflip=true)

# test with secant continuation
opts_br0 = ContinuationPar(dsmin = 0.01, dsmax = 0.15, ds= 0.01, pMax = 4.1, maxSteps = 150, newtonOptions = opt_newton, detectBifurcation = 3)
	br, = continuation(jet[1], jet[2],
		out, par_chan, (@lens _.α), opts_br0; )
####################################################################################################
# deflation newton solver, test of jacobian expression
deflationOp = DeflationOperator(2.0, dot, 1.0, [out])

# for testing
show(deflationOp)
length(deflationOp)
deflationOp(2out, out)
push!(deflationOp, rand(n))
deleteat!(deflationOp, 2)

chanDefPb = DeflatedProblem(jet[1], jet[2], deflationOp)

opt_def = setproperties(opt_newton; tol = 1e-10, maxIter = 1000)
outdef1, = newton(chanDefPb, out .* (1 .+0.01*rand(n)), par_chan, opt_def)

# we now compare the jacobians for the deflated problem either using finite differences or the explicit jacobian
rhs = rand(n)
J_def_fd = BK.finiteDifferences(u->chanDefPb(u, par_chan),1.5*out)
res_fd =  J_def_fd \ rhs

Jacdf = (u0, pb::DeflatedProblem, ls = opt_def.linsolve ) -> (return (u0, par_chan, pb, ls))
Jacdfsolver = DeflatedLinearSolver()

res_explicit = Jacdfsolver(Jacdf(1.5out, chanDefPb, opt_def.linsolver),rhs)[1]

# Test jacobian expression for deflated problem
@test norm(res_fd - res_explicit,Inf64) < 1e-4

opt_def = setproperties(opt_newton; tol = 1e-10, maxIter = 1000)
outdef1, = newton(
		jet[1], jet[2],
		out.*(1 .+ 0.1*rand(n)), par_chan,
		opt_def, deflationOp)

####################################################################################################
# Fold continuation, test of Jacobian expression
outfold = newtonFold(jet[1], jet[2], br, 2; startWithEigen = true)
@test  outfold[3] && outfold[4] == 2
outfold = newtonFold(jet[1], jet[2], br, 2; startWithEigen = true, issymmetric = true)
@test  outfold[3] && outfold[4] == 2
outfold = newton(jet[1], jet[2], br, 2; startWithEigen = true, issymmetric = true)
@test  outfold[3] && outfold[4] == 2

optcontfold = ContinuationPar(dsmin = 0.001, dsmax = 0.15, ds= 0.01, pMax = 4.1, pMin = 0., newtonOptions = NewtonPar(verbose=false, tol = 1e-8), maxSteps = 50, detectBifurcation = 2)
outfoldco, = continuationFold(jet[1], jet[2], br, 2, (@lens _.β), optcontfold; startWithEigen = true, updateMinAugEveryStep = 1, plot = true)
outfoldco, = continuationFold(jet[1], jet[2], br, 2, (@lens _.β), optcontfold; startWithEigen = true, updateMinAugEveryStep = 1, issymmetric = true, plot = true)

# manual handling
indfold = 1
foldpt = FoldPoint(br, indfold)
foldpb = FoldProblemMinimallyAugmented(
		jet[1], jet[2],
		nothing, nothing,
		(@lens _.α),
		br.specialpoint[indfold].x,
		br.specialpoint[indfold].x,
		opts_br0.newtonOptions.linsolver)
foldpb(foldpt, par_chan) |> norm

outfold, = newtonFold(jet[1], jet[2], foldpt, par_chan, (@lens _.α), br.specialpoint[indfold].x, br.specialpoint[indfold].x, NewtonPar(verbose=false))
	# println("--> Fold found at α = ", outfold.p, " from ", br.specialpoint[indfold].param)

# user defined Fold Problem
indfold = 1

# we define the following wrappers to be able to use ForwardDiff
Bd2Vec(x) = vcat(x.u, x.p)
Vec2Bd(x) = BorderedArray(x[1:end-1], x[end])
foldpbVec(x,p) = Bd2Vec(foldpb(Vec2Bd(x),p))

rhs = rand(n+1)
Jac_fold_fdMA(u0) = ForwardDiff.jacobian( u -> foldpbVec(u, par_chan), u0)
J_fold_fd = Jac_fold_fdMA(Bd2Vec(foldpt))
res_fd =  J_fold_fd \ rhs

Jac_fold_MA(u0, p, pb::FoldProblemMinimallyAugmented) = (return (x=u0, params=p, fldpb = pb))
jacFoldSolver = BK.FoldLinearSolverMinAug()
debugTmpForσ = zeros(n+1,n+1) # temporary array for debugging σ
res_explicit = @time jacFoldSolver(Jac_fold_MA(foldpt, par_chan, foldpb), Vec2Bd(rhs); debugArray = debugTmpForσ)

Jac_fold_MA(foldpt, 0.01, foldpb)[2]

# test whether the Jacobian Matrix for the Fold problem is correct
@test debugTmpForσ[1:end-1,1:end-1] - J_fold_fd[1:end-1,1:end-1] |> norminf == 0

# we test the expression for σp
σp_fd = J_fold_fd[end,end]
σp_fd_ana = debugTmpForσ[end,end]
@test σp_fd ≈ σp_fd_ana rtol = 1e-5

# we test the expression for σx
σx_fd = J_fold_fd[end,1:end-1]
@show norminf(σx_fd)
σx_ana = debugTmpForσ[end,1:end-1]
@test σx_fd ≈ σx_ana rtol = 1e-2

σx_fd - σx_ana |> norminf
debugTmpForσ[1:end-1,1:end-1] - J_fold_fd[1:end-1,1:end-1] |> norminf

# check our solution of the bordered problem
res_exp = debugTmpForσ \ rhs
@test norm(res_exp - Bd2Vec(res_explicit[1]), Inf64) < 1e-10
#############################################
# same with foldpb.issymmetric = true
@set! foldpb.issymmetric = true
rhs = rand(n+1)
Jac_fold_fdMA(u0) = ForwardDiff.jacobian( u -> foldpbVec(u, par_chan), u0)
J_fold_fd = Jac_fold_fdMA(Bd2Vec(foldpt))
res_fd =  J_fold_fd \ rhs

Jac_fold_MA(u0, p, pb::FoldProblemMinimallyAugmented) = (return (x=u0, params=p, fldpb = pb))
jacFoldSolver = BK.FoldLinearSolverMinAug()
debugTmpForσ = zeros(n+1,n+1) # temporary array for debugging σ
res_explicit = @time jacFoldSolver(Jac_fold_MA(foldpt, par_chan, foldpb), Vec2Bd(rhs); debugArray = debugTmpForσ)

Jac_fold_MA(foldpt, 0.01, foldpb)[2]

# test whether the Jacobian Matrix for the Fold problem is correct
@test debugTmpForσ[1:end-1,1:end-1] - J_fold_fd[1:end-1,1:end-1] |> norminf == 0

# we test the expression for σp
σp_fd = J_fold_fd[end,end]
σp_fd_ana = debugTmpForσ[end,end]
@test σp_fd ≈ σp_fd_ana rtol = 1e-5

# we test the expression for σx
σx_fd = J_fold_fd[end,1:end-1]
@show norminf(σx_fd)
σx_ana = debugTmpForσ[end,1:end-1]
@test σx_fd ≈ σx_ana rtol = 1e-2

σx_fd - σx_ana |> norminf
debugTmpForσ[1:end-1,1:end-1] - J_fold_fd[1:end-1,1:end-1] |> norminf

# check our solution of the bordered problem
res_exp = debugTmpForσ \ rhs
@test norm(res_exp - Bd2Vec(res_explicit[1]), Inf64) < 1e-10
####################################################################################################
# Use of different eigensolvers
opt_newton = NewtonPar(tol = 1e-8, verbose = false, eigsolver = EigKrylovKit())
opts_br0 = ContinuationPar(dsmin = 0.01, dsmax = 0.15, ds= 0.01, pMax = 4.1, maxSteps = 250, newtonOptions = opt_newton, detectFold = true, detectBifurcation = 1, nev = 15)

br, = continuation(jet[1], jet[2], out, (a, 0.01), (@lens _[1]), opts_br0, printSolution = (x,p)->norm(x,Inf64), plot = false, verbosity = 0)

opts_br0 = ContinuationPar(dsmin = 0.01, dsmax = 0.15, ds= 0.01, pMax = 4.1, maxSteps = 250, newtonOptions = NewtonPar(tol =1e-8), detectFold = true, detectBifurcation = 1, nev = 15)

br, = continuation(jet[1], jet[2], out, (a, 0.01), (@lens _[1]),opts_br0,plot = false, verbosity = 0)
