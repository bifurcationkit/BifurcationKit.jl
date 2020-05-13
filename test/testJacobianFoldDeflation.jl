# using Revise
using Test, PseudoArcLengthContinuation, LinearAlgebra, Setfield
const PALC = PseudoArcLengthContinuation

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

Jac_fd(u0, p) = PALC.finiteDifferences(u->F_chan(u,p),u0)

# not really precise Finite Differences
n = 101
sol = rand(n)
sol[end] = sol[1]
J_fold_fd  = Jac_fd(sol,(3,0.01))
J_fold_exp = Jac_mat(sol,(3,0.01))
@test (J_fold_exp - J_fold_fd) |> x->norm(x,Inf64) < 1e-2

(J_fold_exp - J_fold_fd)[10,10]

n = 101
	a = 3.3
	sol = [(i-1)*(n-i)/n^2+0.1 for i=1:n]
	opt_newton = PALC.NewtonPar(tol = 1e-8, verbose = false)
	# ca fait dans les 69.95k Allocations
	out, hist, flag = @time PALC.newton(F_chan, Jac_mat, sol, (3.3, 0.01),
							opt_newton, normN = x->norm(x,Inf64))

# test with secant continuation
opts_br0 = ContinuationPar(dsmin = 0.01, dsmax = 0.15, ds= 0.01, pMax = 4.1, maxSteps = 150, newtonOptions = opt_newton)
	br, u1 = @time PALC.continuation(F_chan, Jac_mat,
				out, (a, 0.01), (@lens _[1]), opts_br0; plot = false, verbosity = 0, printSolution= (x,p) ->norm(x,Inf64))

# test with Bordered tangent continuation
br_tg, u1 = @time PALC.continuation(F_chan, Jac_mat,
			out, (a, 0.01), (@lens _[1]), opts_br0; plot = false, verbosity = 0, printSolution= (x,p) ->norm(x,Inf64), tangentAlgo = PALC.BorderedPred())

# test with natural continuation
opts_br0 = ContinuationPar(dsmin = 0.01, dsmax = 0.05, ds= 0.01, pMax = 4.1, newtonOptions = opt_newton)
	br_nat, u1 = @time PALC.continuation(
				F_chan, Jac_mat,
				out, (a, 0.01), (@lens _[1]), opts_br0; plot = false, verbosity = 0, printSolution= (x,p) ->norm(x,Inf64), tangentAlgo = PALC.NaturalPred())

# idem with Matrix-Free solver
function dF_chan(x, dx, p)
	α, β = p
	out = similar(x)
	n = length(x)
	out[1] = dx[1]
	out[n] = dx[n]
	for i=2:n-1
		out[i] = (dx[i-1] - 2 * dx[i] + dx[i+1]) * (n-1)^2 + α * dsource_term(x[i], b = β) * dx[i]
	end
	return out
end

ls = PALC.GMRESKrylovKit(dim = 100)
	opt_newton_mf = PALC.NewtonPar(tol = 1e-11, verbose = true, linsolver = ls, eigsolver = DefaultEig())
	out_mf, hist, flag = @time newton(
		F_chan, (x, p) -> (dx -> dF_chan(x, dx, p)),
		sol, (a, 0.01),
		opt_newton_mf)

opts_cont_mf  = PALC.ContinuationPar(dsmin = 0.01, dsmax = 0.1, ds= 0.01, pMax = 4.1, nev = 5, newtonOptions = setproperties(opt_newton_mf;maxIter = 70, verbose = false, tol = 1e-8), maxSteps = 150)

brmf, u1 = @time PALC.continuation(
		F_chan, (x, p) -> (dx -> dF_chan(x, dx, p)),
		out, (a, 0.01), (@lens _[1]), opts_cont_mf, verbosity = 0)

brmf, u1 = @time PALC.continuation(
	F_chan, (x, p) -> (dx -> dF_chan(x, dx, p)),
	sol, (a, 0.01), (@lens _[1]), opts_cont_mf,
	tangentAlgo = BorderedPred(), verbosity = 0)

brmf, u1 = @time PALC.continuation(
	F_chan, (x, p) -> (dx -> dF_chan(x, dx, p)),
	sol, (a, 0.01), (@lens _[1]), opts_cont_mf,
	tangentAlgo = SecantPred(),
	linearAlgo = PALC.MatrixFreeBLS())

brmf, u1 = @time PALC.continuation(
	F_chan, (x, p) -> (dx -> dF_chan(x, dx, p)),
	sol, (a, 0.01), (@lens _[1]), opts_cont_mf,
	tangentAlgo = BorderedPred(),
	linearAlgo = PALC.MatrixFreeBLS())
####################################################################################################
# deflation newton solver, test of jacobian expression
deflationOp = DeflationOperator(2.0, (x,y) -> dot(x,y), 1.0, [out])

# quick test of scalardM deflation
# PALC.scalardM(deflationOp, sol, 5sol)

chanDefPb = DeflatedProblem(F_chan, Jac_mat, deflationOp)

opt_def = setproperties(opt_newton; tol = 1e-10, maxIter = 1000)
outdef1, _,_ = @time PALC.newton(
						(u, p) -> chanDefPb(u, p),
						out .* (1 .+0.01*rand(n)), (a, 0.01),
						opt_def)

# we now compare the jacobians for the deflated problem either using finite differences or the explicit jacobian
rhs = rand(n)
J_def_fd = PALC.finiteDifferences(u->chanDefPb(u, (a, 0.01)),1.5*out)
res_fd =  J_def_fd \ rhs

Jacdf = (u0, pb::DeflatedProblem, ls = opt_def.linsolve ) -> (return (u0, (a, 0.01), pb, ls))
Jacdfsolver = DeflatedLinearSolver()

res_explicit = Jacdfsolver(Jacdf(1.5out, chanDefPb, opt_def.linsolver),rhs)[1]

println("--> Test jacobian expression for deflated problem")
@test norm(res_fd - res_explicit,Inf64) < 1e-4

opt_def = setproperties(opt_newton; tol = 1e-10, maxIter = 1000)
outdef1, _, _ = @time newton(
		F_chan, Jac_mat,
		out.*(1 .+ 0.01*rand(n)), (a, 0.01),
		opt_def, deflationOp)
####################################################################################################
# Fold continuation, test of Jacobian expression
indfold = 1
foldpt = FoldPoint(br, indfold)
foldpb = FoldProblemMinimallyAugmented(
		(x, p) ->   F_chan(x, p),
		(x, p) -> (Jac_mat(x, p)),
		(x, p) -> transpose(Jac_mat(x, p)), nothing,
		(@lens _[1]),
		br.foldpoint[indfold][6],
		br.foldpoint[indfold][6],
		opts_br0.newtonOptions.linsolver)
foldpb(foldpt, (a, 0.01)) |> norm

outfold, _ = PALC.newtonFold(F_chan, Jac_mat, foldpt, (a, 0.01), (@lens _[1]), br.foldpoint[indfold][6],	NewtonPar(verbose=true) )
	println("--> Fold found at α = ", outfold.p, " from ", br.foldpoint[indfold].param)

# example with KrylovKit
P = Jac_mat(sol.*0,(0,0))
optils = NewtonPar(verbose=true, linsolver = GMRESKrylovKit(atol=1e-9, Pl=lu(P)), tol=1e-7)
outfold, _ = PALC.newtonFold(F_chan, Jac_mat, foldpt, (a, 0.01), (@lens _[1]), br.foldpoint[indfold][6], optils )
	println("--> Fold found at α = ", outfold.p, " from ", br.foldpoint[indfold][3])

# continuation of the fold point
optcontfold = PALC.ContinuationPar(dsmin = 0.001, dsmax = 0.15, ds= 0.01, pMax = 4.1, pMin = 0., a = 2., theta = 0.3, newtonOptions = NewtonPar(verbose=true, tol = 1e-8), maxSteps = 5)
	outfoldco, hist, flag = @time PALC.continuation(F_chan, Jac_mat, br, indfold, (a, 0.01), (@lens _[1]), (@lens _[2]), optcontfold, plot = false)

# user defined Fold Problem
indfold = 1

# we define the following wrappers to be able to use finite differences
Bd2Vec(x) = vcat(x.u, x.p)
Vec2Bd(x) = BorderedArray(x[1:end-1], x[end])
foldpbVec(x,p) = Bd2Vec(foldpb(Vec2Bd(x),p))

outfold, _ = newton((x, p) -> foldpbVec(x, p),
			Bd2Vec(foldpt), (a, 0.01),
			NewtonPar(verbose=true) )
	println("--> Fold found at α = ", outfold[end], " from ", br.foldpoint[indfold][3])

rhs = rand(n+1)
Jac_fold_fdMA(u0) = PALC.finiteDifferences( u-> foldpbVec(u, (a, 0.01)), u0)
J_fold_fd = Jac_fold_fdMA(Bd2Vec(foldpt))
res_fd =  J_fold_fd \ rhs

Jac_fold_MA(u0, p, pb::FoldProblemMinimallyAugmented) = (return (x=u0, param=p, fldpb = pb))
jacFoldSolver = PALC.FoldLinearSolverMinAug()
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
opt_newton = PALC.NewtonPar(tol = 1e-8, verbose = false, eigsolver = EigKrylovKit())
opts_br0 = ContinuationPar(dsmin = 0.01, dsmax = 0.15, ds= 0.01, pMax = 4.1, maxSteps = 250, newtonOptions = opt_newton, detectFold = true, detectBifurcation = 1, nev = 15)

br, u1 = @time PALC.continuation(F_chan, Jac_mat, out, (a, 0.01), (@lens _[1]), opts_br0, printSolution = (x,p)->norm(x,Inf64), plot = false, verbosity = 0)

opts_br0 = ContinuationPar(dsmin = 0.01, dsmax = 0.15, ds= 0.01, pMax = 4.1, maxSteps = 250, newtonOptions = NewtonPar(tol =1e-8), detectFold = true, detectBifurcation = 1, nev = 15)

br, u1 = @time PALC.continuation(F_chan, Jac_mat, out, (a, 0.01), (@lens _[1]),opts_br0,plot = false, verbosity = 0)
