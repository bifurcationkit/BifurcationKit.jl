using Revise
using Test, PseudoArcLengthContinuation, LinearAlgebra
const Cont = PseudoArcLengthContinuation

source_term(x; a = 0.5, b = 0.01) = 1 + (x + a*x^2)/(1 + b*x^2)
dsource_term(x; a = 0.5, b = 0.01) = (1-b*x^2+2*a*x)/(1+b*x^2)^2
d2source_term(x; a = 0.5, b = 0.01) = -(2*(-a+3*a*b*x^2+3*b*x-b^2*x^3))/(1+b*x^2)^3

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
sol = rand(n)
sol[end] = sol[1]
J_fold_fd = Jac_fd(sol,3,0.01)
J_fold_exp = Jac_mat(sol,3,0.01)
@test (J_fold_exp - J_fold_fd) |> x->norm(x,Inf64) < 1e-2

using Plots
Plots.heatmap(J_fold_exp - J_fold_fd)

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
							opt_newton)

# test with secant continuation
opts_br0 = ContinuationPar(dsmin = 0.01, dsmax = 0.15, ds= 0.01, pMax = 4.1, maxSteps = 250, newtonOptions = opt_newton, detect_fold = true, secant = true)
br, u1 = @time Cont.continuation(
				(x,p) -> F_chan(x,p, 0.01),
				(x,p) -> (Jac_mat(x,p, 0.01)),
				printsolution = x->norm(x,Inf64),
				out,a,opts_br0,plot = false, verbosity = 0)

# test with tangent continuation
opts_br0.secant = false
br_tg, u1 = @time Cont.continuation(
				(x,p) -> F_chan(x,p, 0.01),
				(x,p) -> (Jac_mat(x,p, 0.01)),
				printsolution = x->norm(x,Inf64),
				out,a,opts_br0,plot = false, verbosity = 0)

# test with natural continuation
opts_br0 = ContinuationPar(dsmin = 0.01, dsmax = 0.05, ds= 0.01, pMax = 4.1, newtonOptions = opt_newton, detect_fold = true, natural = true)
br_nat, u1 = @time Cont.continuation(
				(x,p) -> F_chan(x,p, 0.01),
				(x,p) -> (Jac_mat(x,p, 0.01)),
				printsolution = x->norm(x,Inf64),
				out,0.,opts_br0,plot = false, verbosity = 0)

# Cont.plotBranch(br)
# Cont.plotBranch(br_nat, marker = :d)
####################################################################################################
# deflation newton solver, test of jacobian expression
deflationOp = DeflationOperator(2.0,(x,y)->dot(x,y),1.0,[out])
chanDefPb = DeflatedProblem(x -> F_chan(x,a, 0.01),x -> Jac_mat(x,a, 0.01),deflationOp)

opt_def = opt_newton
opt_def.tol = 1e-10
opt_def.maxIter = 1000
outdef1, _,_ = @time Cont.newton(
						u->chanDefPb(u),
						out.*(1 .+0.01*rand(n)),
						opt_def)

# we now compare the jacobians for the deflated problem either using finite differences or the explicit jacobian
rhs = rand(n)
J_def_fd = Cont.finiteDifferences(u->chanDefPb(u),1.1out)
res_fd =  J_def_fd \ rhs

Jacdf = (u0, pb::DeflatedProblem,ls = opt_def.linsolve ) -> (return (u0, pb, ls))
Jacdfsolver = DeflatedLinearSolver()

res_explicit = Jacdfsolver(Jacdf(1.1out,chanDefPb,opt_def.linsolve),rhs)[1]

println("--> Test jacobian expression for deflated problem")
@test norm(res_fd - res_explicit,Inf64) < 1e-7
####################################################################################################
# Fold continuation, test of Jacobian expression
foldpt = vcat(br.bifpoint[3][5],br.bifpoint[3][3])
foldpb = FoldProblemMinimallyAugmented(
					(x, α) ->   F_chan(x, α, 0.01),
					(x, α) -> (Jac_mat(x, α, 0.01)),
					(x, α) -> transpose(Jac_mat(x, α, 0.01)),
					rand(n),
					rand(n),
					opts_br0.newtonOptions.linsolve)
foldpb(foldpt)

rhs = rand(n+1)
Jac_fold_fdMA(u0) = Cont.finiteDifferences( u-> foldpb(u), u0)
J_fold_fd = Jac_fold_fdMA(foldpt)
res_fd =  J_fold_fd \ rhs

Jac_fold_MA(u0, β, pb::FoldProblemMinimallyAugmented) = (return (u0, pb))
jacFoldSolver = FoldLinearSolveMinAug()
res_explicit = jacFoldSolver(Jac_fold_MA(foldpt,0.01,foldpb),rhs, true)

# test whether the Jacobian Matrix for the Fold problem is correct
println("--> FD σp = ", J_fold_fd[end,end])
println("--> MA σp = ", res_explicit[end][end,end])

@test J_fold_fd[end,:] -  J_fold_fd[end,:] |> x->norm(x,Inf64) <1e-7
@test J_fold_fd[:,end] -  J_fold_fd[:,end] |> x->norm(x,Inf64) <1e-7
# this part is where we compare the FD Jacobian with Jac_chan, not really good
@test (res_explicit[end] - J_fold_fd)[1:end-1,1:end-1] |> x->norm(x,Inf64) < 1e-1


# check our solution of the bordered problem
res_exp = res_explicit[end] \ rhs
@test norm(res_exp - res_explicit[1],Inf64) < 1e-8
####################################################################################################
# Use of different eigensolvers
opt_newton = Cont.NewtonPar(tol = 1e-8, verbose = false, eigsolve = eig_KrylovKit{Float64}())
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
