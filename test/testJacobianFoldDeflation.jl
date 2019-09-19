using Test, PseudoArcLengthContinuation, LinearAlgebra
const Cont = PseudoArcLengthContinuation

source_term(x; a = 0.5, b = 0.01) = 1 + (x + a*x^2)/(1 + b*x^2)
dsource_term(x; a = 0.5, b = 0.01) = (1-b*x^2+2*a*x)/(1+b*x^2)^2
d2source_term(x; a = 0.5, b = 0.01) = -(2*(-a+3*a*b*x^2+3*b*x-b^2*x^3))/(1+b*x^2)^3

println("\n\n\n--> Test Fold continuation")

function F_chan(x, α, β = 0.)
	f = similar(x)
	N = length(x)
	f[1] = x[1] - β
	f[N] = x[N] - β
	for i=2:N-1
		f[i] = (x[i-1] - 2 * x[i] + x[i+1]) * (N-1)^2 + α * source_term(x[i], b = β)
	end
	return f
end

function Jac_mat(u, α, β = 0.)
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

Jac_fd(u0, α, β) = Cont.finiteDifferences(u->F_chan(u,α, β),u0)

# not really precise Finite Differences, I don't really undertand why
n = 101
sol = rand(n)
sol[end] = sol[1]
J_fold_fd  = Jac_fd(sol,3,0.01)
J_fold_exp = Jac_mat(sol,3,0.01)
@test (J_fold_exp - J_fold_fd) |> x->norm(x,Inf64) < 1e-2

(J_fold_exp - J_fold_fd)[10,10]

n = 101
	a = 3.3
	sol = [(i-1)*(n-i)/n^2+0.1 for i=1:n]
	opt_newton = Cont.NewtonPar(tol = 1e-8, verbose = false)
	# ca fait dans les 69.95k Allocations
	out, hist, flag = @time Cont.newton(
							x -> F_chan(x,a, 0.01),
							x -> Jac_mat(x,a, 0.01),
							sol,
							opt_newton, normN = x->norm(x,Inf64))

# test with secant continuation
opts_br0 = ContinuationPar(dsmin = 0.01, dsmax = 0.15, ds= 0.01, pMax = 4.1, maxSteps = 150, newtonOptions = opt_newton, detect_fold = true, plot_every_n_steps = 50)
	br, u1 = @time Cont.continuation(
				(x,p) -> F_chan(x,p, 0.01),
				(x,p) -> (Jac_mat(x,p, 0.01)),
				printsolution = x->norm(x,Inf64),
				out, a, opts_br0, plot = false, verbosity = 0)

# test with Bordered tangent continuation
br_tg, u1 = @time Cont.continuation(
				(x,p) -> F_chan(x,p, 0.01),
				(x,p) -> (Jac_mat(x,p, 0.01)),
				printsolution = x->norm(x,Inf64),
				out,a,opts_br0,plot = false, verbosity = 0, tangentalgo = Cont.BorderedPred())

# test with natural continuation
opts_br0 = ContinuationPar(dsmin = 0.01, dsmax = 0.05, ds= 0.01, pMax = 4.1, newtonOptions = opt_newton, detect_fold = true)
	br_nat, u1 = @time Cont.continuation(
				(x,p) -> F_chan(x,p, 0.01),
				(x,p) -> (Jac_mat(x,p, 0.01)),
				printsolution = x->norm(x,Inf64),
				out,0.,opts_br0,plot = false, verbosity = 0, tangentalgo = Cont.NaturalPred())

# idem with Matrix-Free solver
function dF_chan(x, dx, α, β = 0.)
	out = similar(x)
	n = length(x)
	out[1] = dx[1]
	out[n] = dx[n]
	for i=2:n-1
		out[i] = (dx[i-1] - 2 * dx[i] + dx[i+1]) * (n-1)^2 + α * dsource_term(x[i], b = β) * dx[i]
	end
	return out
end

ls = Cont.GMRES_KrylovKit{Float64}(dim = 100)
	opt_newton_mf = Cont.NewtonPar(tol = 1e-11, verbose = true, linsolver = ls, eigsolver = Default_eig())
	out_mf, hist, flag = @time Cont.newton(
		x -> F_chan(x, a, 0.01),
		x -> (dx -> dF_chan(x, dx, a, 0.01)),
		sol,
		opt_newton_mf)

opts_cont_mf  = Cont.ContinuationPar(dsmin = 0.01, dsmax = 0.1, ds= 0.01, pMax = 4.1, nev = 5, detect_fold = true, detect_bifurcation = false, plot_every_n_steps = 40, newtonOptions = opt_newton_mf)
	opts_cont_mf.newtonOptions.maxIter = 70
	opts_cont_mf.newtonOptions.verbose = false
	opts_cont_mf.newtonOptions.tol = 1e-8
	opts_cont_mf.maxSteps = 150

brmf, u1 = @time Cont.continuation(
		(x, p) -> F_chan(x, p, 0.01),
		(x, p) -> (dx -> dF_chan(x, dx, p, 0.01)),
		out, a, opts_cont_mf, verbosity = 0)

brmf, u1 = @time Cont.continuation(
	(x, p) -> F_chan(x, p, 0.01),
	(x, p) -> (dx -> dF_chan(x, dx, p, 0.01)),
	out, a, opts_cont_mf,
	tangentalgo = BorderedPred(), verbosity = 0)

brmf, u1 = @time Cont.continuation(
	(x, p) -> F_chan(x, p, 0.01),
	(x, p) -> (dx -> dF_chan(x, dx, p, 0.01)),
	out, a, opts_cont_mf,
	tangentalgo = SecantPred(),
	linearalgo = Cont.MatrixFreeBLS())

brmf, u1 = @time Cont.continuation(
	(x, p) -> F_chan(x, p, 0.01),
	(x, p) -> (dx -> dF_chan(x, dx, p, 0.01)),
	out, a, opts_cont_mf,
	tangentalgo = BorderedPred(),
	linearalgo = Cont.MatrixFreeBLS())
####################################################################################################
# deflation newton solver, test of jacobian expression
deflationOp = DeflationOperator(2.0,(x,y) -> dot(x,y),1.0,[out])

# quick test of scalardM deflation
# Cont.scalardM(deflationOp, sol, 5sol)

chanDefPb   = DeflatedProblem(x -> F_chan(x,a, 0.01),x -> Jac_mat(x,a, 0.01),deflationOp)

opt_def = opt_newton
opt_def.tol = 1e-10
opt_def.maxIter = 1000
outdef1, _,_ = @time Cont.newton(
						u->chanDefPb(u),
						out.*(1 .+0.01*rand(n)),
						opt_def)

# we now compare the jacobians for the deflated problem either using finite differences or the explicit jacobian
rhs = rand(n)
J_def_fd = Cont.finiteDifferences(u->chanDefPb(u),1.5*out)
res_fd =  J_def_fd \ rhs

Jacdf = (u0, pb::DeflatedProblem,ls = opt_def.linsolve ) -> (return (u0, pb, ls))
Jacdfsolver = DeflatedLinearSolver()

res_explicit = Jacdfsolver(Jacdf(1.5out, chanDefPb, opt_def.linsolver),rhs)[1]

println("--> Test jacobian expression for deflated problem")
@test norm(res_fd - res_explicit,Inf64) < 1e-4

opt_def = opt_newton
opt_def.tol = 1e-10
opt_def.maxIter = 1000

outdef1, _, _ = @time Cont.newtonDeflated(
		x -> F_chan(x, a, 0.01),
		x -> Jac_mat(x, a, 0.01),
		out.*(1 .+ 0.01*rand(n)),
		opt_def, deflationOp)
####################################################################################################
# Fold continuation, test of Jacobian expression
indfold = 1
foldpt = FoldPoint(br, indfold)
foldpb = FoldProblemMinimallyAugmented(
		(x, α) ->   F_chan(x, α, 0.01),
		(x, α) -> (Jac_mat(x, α, 0.01)),
		(x, α) -> transpose(Jac_mat(x, α, 0.01)),
		br.bifpoint[indfold][6],
		br.bifpoint[indfold][6],
		opts_br0.newtonOptions.linsolver)
foldpb(foldpt) |> norm

outfold, _ = Cont.newtonFold(
		(x, p) -> F_chan(x, p, 0.01),
		(x, p) -> Jac_mat(x, p, 0.01),
		foldpt,
		br.bifpoint[indfold][6],
		NewtonPar(verbose=true) )
	println("--> Fold found at α = ", outfold.p, " from ", br.bifpoint[indfold][3])

# continuation of the fold point
optcontfold = Cont.ContinuationPar(dsmin = 0.001, dsmax = 0.15, ds= 0.01, pMax = 4.1, pMin = 0., a = 2., theta = 0.3, newtonOptions = NewtonPar(verbose=true), maxSteps = 5)
	optcontfold.newtonOptions.tol = 1e-8
	outfoldco, hist, flag = @time Cont.continuationFold(
						(x, α, β) ->  F_chan(x, α, β),
						(x, α, β) -> Jac_mat(x, α, β),
						br, indfold,
						0.01,
						optcontfold, plot = false)

# user defined Fold Problem
indfold = 1

# we define the following wrappers to be able to use finite differences
Bd2Vec(x) = vcat(x.u, x.p)
Vec2Bd(x) = BorderedArray(x[1:end-1], x[end])
foldpbVec(x) = Bd2Vec(foldpb(Vec2Bd(x)))

outfold, _ = Cont.newton(x -> foldpbVec(x),
			# (x, α) -> Jac_mat(x, α, 0.01),
			Bd2Vec(foldpt),
			NewtonPar(verbose=true) )
	println("--> Fold found at α = ", outfold[end], " from ", br.bifpoint[indfold][3])

rhs = rand(n+1)
Jac_fold_fdMA(u0) = Cont.finiteDifferences( u-> foldpbVec(u), u0)
J_fold_fd = Jac_fold_fdMA(Bd2Vec(foldpt))
res_fd =  J_fold_fd \ rhs

Jac_fold_MA(u0, β, pb::FoldProblemMinimallyAugmented) = (return (u0, pb, x->x))
jacFoldSolver = FoldLinearSolveMinAug()
res_explicit = jacFoldSolver(Jac_fold_MA(foldpt, 0.01, foldpb), Vec2Bd(rhs), true)

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
opt_newton = Cont.NewtonPar(tol = 1e-8, verbose = false, eigsolver = eig_KrylovKit{Float64}())
opts_br0 = ContinuationPar(dsmin = 0.01, dsmax = 0.15, ds= 0.01, pMax = 4.1, maxSteps = 250, newtonOptions = opt_newton, detect_fold = true, detect_bifurcation = true, nev = 15)

br, u1 = @time Cont.continuation(
			(x,p) -> F_chan(x,p, 0.01),
			(x,p) -> (Jac_mat(x,p, 0.01)),
			printsolution = x->norm(x,Inf64),
			out,a,opts_br0,plot = false, verbosity = 0)

opts_br0 = ContinuationPar(dsmin = 0.01, dsmax = 0.15, ds= 0.01, pMax = 4.1, maxSteps = 250, newtonOptions = NewtonPar(tol =1e-8), detect_fold = true, detect_bifurcation = true, nev = 15)

br, u1 = @time Cont.continuation(
			(x,p) -> F_chan(x,p, 0.01),
			(x,p) -> (Jac_mat(x,p, 0.01)),
			printsolution = x->norm(x,Inf64),
			out,a,opts_br0,plot = false, verbosity = 0)
