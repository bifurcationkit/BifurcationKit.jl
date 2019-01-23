using Revise
using PseudoArcLengthContinuation, LinearAlgebra, Plots
const Cont = PseudoArcLengthContinuation

source_term(x; a = 0.5, b = 0.01) = 1 + (x + a*x^2)/(1 + b*x^2)
dsource_term(x; a = 0.5, b = 0.01) = (1-b*x^2+2*a*x)/(1+b*x^2)^2
d2source_term(x; a = 0.5, b = 0.01) = -(2*(-a+3*a*b*x^2+3*b*x-b^2*x^3))/(1+b*x^2)^3

function F_chan(x, α, β = 0.)
	f = similar(x)
	n = length(x)
	f[1] = x[1] - β
	f[n] = x[n] - β
	for i=2:n-1
		f[i] = (x[i-1] - 2 * x[i] + x[i+1]) * (n-1)^2 + α * source_term(x[i], b = β)
	end
	return f
end

function Jac_mat(u, α, β = 0.)
	n = length(u)
	J = zeros(n, n)
	J[1, 1] = 1.0
	J[n, n] = 1.0
	for i = 2:n-1
		J[i, i-1] = (n-1)^2
		J[i, i+1] = (n-1)^2
    	J[i, i] = -2 * (n-1)^2 + α * dsource_term(u[i], b = β)
	end
	return J
end

Jac_fd(u0, α, β = 0.) = Cont.finiteDifferences(u->F_chan(u, α, β = β), u0)

n = 101
	a = 3.3
	sol = [(i-1)*(n-i)/n^2+0.1 for i=1:n]
	opt_newton = Cont.NewtonPar(tol = 1e-11, verbose = true)
	# ca fait dans les 63.59k Allocations
	out, hist, flag = @time Cont.newton(
							x -> F_chan(x, a, 0.01),
							x -> Jac_mat(x, a, 0.01),
							sol,
							opt_newton)


opts_br0 = Cont.ContinuationPar(dsmin = 0.01, dsmax = 0.15, ds= 0.01, pMax = 4.1, nev = 5, detect_fold = true, detect_bifurcation = true, plot_every_n_steps = 40)
	opts_br0.newtonOptions.maxIter = 70
	opts_br0.newtonOptions.tol = 1e-8
	opts_br0.maxSteps = 150

	br, u1 = @time Cont.continuation(
					(x, p) -> F_chan(x, p, 0.01),
					(x, p) -> (Jac_mat(x, p, 0.01)),
					out, a, opts_br0,
					printsolution = x -> norm(x, Inf64),
					plot = true,
					plotsolution = (x;kwargs...) -> (plot!(x, subplot=4, ylabel="solution", label="")))
###################################################################################################
# Example with deflation technique
deflationOp = DeflationOperator(2.0, (x, y)->dot(x, y), 1.0, [out])

opt_def = opt_newton
opt_def.tol = 1e-10
opt_def.maxIter = 1000

outdef1, _, _ = @time Cont.newtonDeflated(
						x -> F_chan(x, a, 0.01),
						x -> Jac_mat(x, a, 0.01),
						out.*(1 .+ 0.01*rand(n)),
						opt_def, deflationOp)
plot(out, label="newton")
	plot!(sol, label="init guess")
	plot!(outdef1, label="deflation-1")

#save newly found point to look for new ones
push!(deflationOp, outdef1)
outdef2, _, _ = @time Cont.newtonDeflated(
						x -> F_chan(x, a, 0.01),
						x -> Jac_mat(x, a, 0.01),
						outdef1.*(1 .+ 0.01*rand(n)),
						opt_def, deflationOp)
plot!(outdef2, label="deflation-2")
###################################################################################################
# Continuation of the Fold Point using Dense method
foldpt = FoldPoint(br.bifpoint[3])
	phi_guess = foldpt[n+1:2n]
	foldPb = (u, β)->FoldProblemMooreSpence(
					(x, α)->F_chan(x, α, β),
					(x, α)->(Jac_mat(x, α, β)),
					phi_guess,
					opts_br0.newtonOptions.linsolve)(u)

Jac_fold_fd(u0, β) = Cont.finiteDifferences( u-> foldPb(u, β), u0)

opt_fold = Cont.NewtonPar(tol = 1e-10, verbose = true, maxIter = 20)
	outfold, hist, flag = @time Cont.newton(
						x ->      foldPb(x, 0.01),
						x -> Jac_fold_fd(x, 0.01),
						foldpt,
						opt_fold)
	flag && printstyled(color=:red, "--> We found a Fold Point at α = ", outfold[end], ", β = 0.01\n")

opt_fold_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, pMax = 4.1, pMin = 0.0, a = 2., theta = 0.4)
	opt_fold_cont.maxSteps = 70

	br_fold, u1_fold = @time Cont.continuation(
					(x, β) ->      foldPb(x, β),
					(x, β) -> Jac_fold_fd(x, β),
					outfold, 0.01,
					opt_fold_cont, plot = true,
					printsolution = u -> u[end])

Cont.plotBranch(br_fold, marker=:d, xlabel="beta", ylabel = "alpha")
#################################################################################################### Continuation of the Fold Point using minimally augmented

outfold, hist, flag = @time Cont.newtonFold((x, α) -> F_chan(x, α, 0.01),
										(x, α) -> Jac_mat(x, α, 0.01),
										br, 3, #index of the fold point
										opts_br0.newtonOptions)
		flag && printstyled(color=:red, "--> We found a Fold Point at α = ", outfold[end], ", β = 0.01\n")

optcontfold = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, pMax = 4.1, pMin = 0., a = 2., theta = 0.3)
	optcontfold.newtonOptions.tol = 1e-8
	outfoldco, hist, flag = @time Cont.continuationFold(
						(x, α, β) ->  F_chan(x, α, β),
						(x, α, β) -> Jac_mat(x, α, β),
						br, 3,
						0.01,
						optcontfold)

optcontfold.newtonOptions == opts_br0.newtonOptions

Cont.plotBranch(outfoldco;xlabel="b", ylabel="a")
###################################################################################################
# GMRES example

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
	opt_newton_mf = Cont.NewtonPar(tol = 1e-11, verbose = true, linsolve = ls, eigsolve = Default_eig())
	# ca fait dans les 63.59k Allocations
	out_mf, hist, flag = @time Cont.newton(
							x -> F_chan(x, a, 0.01),
							x -> (dx -> dF_chan(x, dx, a, 0.01)),
							sol,
							opt_newton_mf)
