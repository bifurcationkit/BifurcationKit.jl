# using Revise
using Test, BifurcationKit, LinearAlgebra, Setfield
const BK = BifurcationKit

source_term(x; a = 0.5, b = 0.01) = 1 + (x + a*x^2)/(1 + b*x^2)
dsource_term(x; a = 0.5, b = 0.01) = (1-b*x^2+2*a*x)/(1+b*x^2)^2
d2source_term(x; a = 0.5, b = 0.01) = -(2*(-a+3*a*b*x^2+3*b*x-b^2*x^3))/(1+b*x^2)^3

println("\n\n\n--> Test Fold continuation")

function F_chan(x, p)
	α, β = p
	f = similar(x)
	N = length(x)
	f[1] = x[1] - β
	f[N] = x[N] - β
	for i=2:N-1
		f[i] = (x[i-1] - 2 * x[i] + x[i+1]) * (N-1)^2 + α * source_term(x[i], b = β)
	end
	return f
end

function Jac_mat(u, p)
	α, β = p
	N = length(u)
	J = zeros(N,N)
	J[1,1] = 1.0
	J[n,n] = 1.0
	for i = 2:N-1
		J[i,i-1] = (N-1)^2
		J[i,i+1] = J[i,i-1]
		J[i,i] = -2 * J[i,i-1] + α * dsource_term(u[i],b = β)
	end
	return J
end

Jac_fd(u0, p) = BK.finiteDifferences(u -> F_chan(u,p), u0)

# not really precise Finite Differences
n = 101
sol = rand(n)
sol[end] = sol[1]
J_fold_fd  = Jac_fd(sol,(3,0.01))
J_fold_exp = Jac_mat(sol,(3,0.01))
@test norm(J_fold_exp - J_fold_fd, Inf) < 1e-2

n = 101
	a = 3.3
	sol = [(i-1)*(n-i)/n^2+0.1 for i=1:n]
	opt_newton = NewtonPar(tol = 1e-8)
	out, hist, flag = @time newton(F_chan, Jac_mat, sol, (3.3, 0.01),
							opt_newton, normN = x->norm(x,Inf64))

# test with secant continuation
opts_br0 = ContinuationPar(dsmin = 0.01, dsmax = 0.15, ds= 0.01, pMax = 4.1, maxSteps = 150, newtonOptions = opt_newton, detectBifurcation = 3)
	br, _ = @time continuation(F_chan, Jac_mat,
		out, (a, 0.01), (@lens _[1]), opts_br0; printSolution= (x,p) ->norm(x, Inf))

####################################################################################################
# deflation newton solver, test of jacobian expression
deflationOp = DeflationOperator(2.0, dot, 1.0, [out])

# quick test of scalardM deflation
# BK.scalardM(deflationOp, sol, 5sol)

chanDefPb = DeflatedProblem(F_chan, Jac_mat, deflationOp)

opt_def = setproperties(opt_newton; tol = 1e-10, maxIter = 1000)
outdef1, = @time newton((u, p) -> chanDefPb(u, p),
						out .* (1 .+0.01*rand(n)), (a, 0.01),
						opt_def)

# we now compare the jacobians for the deflated problem either using finite differences or the explicit jacobian
rhs = rand(n)
J_def_fd = BK.finiteDifferences(u->chanDefPb(u, (a, 0.01)),1.5*out)
res_fd =  J_def_fd \ rhs

Jacdf = (u0, pb::DeflatedProblem, ls = opt_def.linsolve ) -> (return (u0, (a, 0.01), pb, ls))
Jacdfsolver = DeflatedLinearSolver()

res_explicit = Jacdfsolver(Jacdf(1.5out, chanDefPb, opt_def.linsolver),rhs)[1]

println("--> Test jacobian expression for deflated problem")
@test norm(res_fd - res_explicit,Inf64) < 1e-4

opt_def = setproperties(opt_newton; tol = 1e-10, maxIter = 1000)
outdef1, = @time newton(
		F_chan, Jac_mat,
		out.*(1 .+ 0.1*rand(n)), (a, 0.01),
		opt_def, deflationOp)

####################################################################################################
# Fold continuation, test of Jacobian expression
outfold, = newtonFold(F_chan, Jac_mat, br, 2; startWithEigen = true)
optcontfold = ContinuationPar(dsmin = 0.001, dsmax = 0.15, ds= 0.01, pMax = 4.1, pMin = 0., newtonOptions = NewtonPar(verbose=false, tol = 1e-8), maxSteps = 5)
outfoldco, = continuationFold(F_chan, Jac_mat, br, 2, (@lens _[2]), optcontfold; startWithEigen = true)

# manual handling
indfold = 1
foldpt = FoldPoint(br, indfold)
foldpb = FoldProblemMinimallyAugmented(
		(x, p) ->   F_chan(x, p),
		(x, p) -> (Jac_mat(x, p)),
		(x, p) -> transpose(Jac_mat(x, p)), nothing,
		(@lens _[1]),
		br.bifpoint[indfold].x,
		br.bifpoint[indfold].x,
		opts_br0.newtonOptions.linsolver)
foldpb(foldpt, (a, 0.01)) |> norm

outfold, = newtonFold(F_chan, Jac_mat, foldpt, (a, 0.01), (@lens _[1]), br.bifpoint[indfold].x, br.bifpoint[indfold].x, NewtonPar(verbose=true))
	println("--> Fold found at α = ", outfold.p, " from ", br.bifpoint[indfold].param)

# example with KrylovKit
P = Jac_mat(sol.*0,(0,0))
optils = NewtonPar(verbose=true, linsolver = GMRESKrylovKit(atol=1e-9, Pl=lu(P)), tol=1e-7)
outfold, = newtonFold(F_chan, Jac_mat, foldpt, (a, 0.01), (@lens _[1]), br.bifpoint[indfold].x,br.bifpoint[indfold].x, optils )
	println("--> Fold found at α = ", outfold.p, " from ", br.bifpoint[indfold].param)

# continuation of the fold point
outfoldco, hist, flag = @time continuation(F_chan, Jac_mat, br, indfold, (@lens _[2]), optcontfold, plot = false)

# user defined Fold Problem
indfold = 1

# we define the following wrappers to be able to use finite differences
Bd2Vec(x) = vcat(x.u, x.p)
Vec2Bd(x) = BorderedArray(x[1:end-1], x[end])
foldpbVec(x,p) = Bd2Vec(foldpb(Vec2Bd(x),p))

outfold, = newton((x, p) -> foldpbVec(x, p),
			Bd2Vec(foldpt), (a, 0.01),
			NewtonPar(verbose=true) )
	println("--> Fold found at α = ", outfold[end], " from ", br.bifpoint[indfold].param)

rhs = rand(n+1)
Jac_fold_fdMA(u0) = BK.finiteDifferences( u-> foldpbVec(u, (a, 0.01)), u0)
J_fold_fd = Jac_fold_fdMA(Bd2Vec(foldpt))
res_fd =  J_fold_fd \ rhs

Jac_fold_MA(u0, p, pb::FoldProblemMinimallyAugmented) = (return (x=u0, params=p, fldpb = pb))
jacFoldSolver = BK.FoldLinearSolverMinAug()
res_explicit = jacFoldSolver(Jac_fold_MA(foldpt, (a, 0.01), foldpb), Vec2Bd(rhs), true)

Jac_fold_MA(foldpt, 0.01, foldpb)[2]

# test whether the Jacobian Matrix for the Fold problem is correct
println("--> FD σp = ", J_fold_fd[end,end])
println("--> MA σp = ", res_explicit[end][end,end])

@test J_fold_fd[end,:] -  J_fold_fd[end,:] |> x->norm(x,Inf64) <1e-7
@test J_fold_fd[:,end] -  J_fold_fd[:,end] |> x->norm(x,Inf64) <1e-7
# this part is where we compare the FD Jacobian with Jac_chan, not really good
@test (res_explicit[end] - J_fold_fd)[1:end-1,1:end-1] |> x->norm(x,Inf64) < 1e-1


# check our solution of the bordered problem
res_exp = res_explicit[end] \ rhs
@test norm(res_exp - Bd2Vec(res_explicit[1]), Inf64) < 1e-8
####################################################################################################
# Use of different eigensolvers
opt_newton = NewtonPar(tol = 1e-8, verbose = false, eigsolver = EigKrylovKit())
opts_br0 = ContinuationPar(dsmin = 0.01, dsmax = 0.15, ds= 0.01, pMax = 4.1, maxSteps = 250, newtonOptions = opt_newton, detectFold = true, detectBifurcation = 1, nev = 15)

br, _ = @time continuation(F_chan, Jac_mat, out, (a, 0.01), (@lens _[1]), opts_br0, printSolution = (x,p)->norm(x,Inf64), plot = false, verbosity = 0)

opts_br0 = ContinuationPar(dsmin = 0.01, dsmax = 0.15, ds= 0.01, pMax = 4.1, maxSteps = 250, newtonOptions = NewtonPar(tol =1e-8), detectFold = true, detectBifurcation = 1, nev = 15)

br, _ = @time continuation(F_chan, Jac_mat, out, (a, 0.01), (@lens _[1]),opts_br0,plot = false, verbosity = 0)
