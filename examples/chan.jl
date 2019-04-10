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


opts_br0 = Cont.ContinuationPar(dsmin = 0.01, dsmax = 0.1, ds= 0.01, pMax = 4.1, nev = 5, detect_fold = true, detect_bifurcation = false, plot_every_n_steps = 40)
	opts_br0.newtonOptions.maxIter = 70
	opts_br0.newtonOptions.tol = 1e-8
	opts_br0.maxSteps = 150

	br, u1 = @time Cont.continuation(
		(x, p) -> F_chan(x, p, 0.01),
		(x, p) -> (Jac_mat(x, p, 0.01)),
		out, a, opts_br0,
		printsolution = x -> norm(x, Inf64),
		plot = false,
		plotsolution = (x;kwargs...) -> (plot!(x, subplot = 4, ylabel = "solution", label = "")))
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
#################################################################################################### Continuation of the Fold Point using minimally augmented
opts_br0.newtonOptions.verbose = true
opts_br0.newtonOptions.tol = 1e-10
indfold = 2

outfold, hist, flag = @time Cont.newtonFold(
			(x, α) -> F_chan(x, α, 0.01),
			(x, α) -> Jac_mat(x, α, 0.01),
			br, indfold, #index of the fold point
			opts_br0.newtonOptions)
flag && printstyled(color=:red, "--> We found a Fold Point at α = ", outfold.p, ", β = 0.01, from ", br.bifpoint[indfold][3],"\n")

optcontfold = ContinuationPar(dsmin = 0.001, dsmax = 0.15, ds= 0.01, pMax = 4.1, pMin = 0., a = 2., theta = 0.3, newtonOptions = NewtonPar(verbose=true), maxSteps = 1300)
	optcontfold.newtonOptions.tol = 1e-8
	outfoldco, hist, flag = @time Cont.continuationFold(
			(x, α, β) ->  F_chan(x, α, β),
			(x, α, β) -> Jac_mat(x, α, β),
			br, indfold,
			0.01, plot = true,
			optcontfold)
Cont.plotBranch(outfoldco, marker=:d, xlabel="beta", ylabel = "alpha", label = "");title!("")
################################################################################################### Fold Newton / Continuation when Hessian is known. Does not require state to be AbstractVector
d2F(x,p,u,v; b = 0.01) = p * d2source_term.(x; b = b) .* u .* v

outfold, hist, flag = @time Cont.newtonFold(
			(x, α) -> F_chan(x, α, 0.01),
			(x, α) -> Jac_mat(x, α, 0.01),
			(x, α) -> transpose(Jac_mat(x, α, 0.01)),
			d2F,
			br, indfold, #index of the fold point
			opts_br0.newtonOptions)
		flag && printstyled(color=:red, "--> We found a Fold Point at α = ", outfold.p, ", β = 0.01, from ", br.bifpoint[indfold][3],"\n")

optcontfold = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, pMax = 4.1, pMin = 0., a = 2., theta = 0.3, newtonOptions = NewtonPar(verbose=true), maxSteps = 1300)

outfoldco, hist, flag = @time Cont.continuationFold(
					(x, α, β) ->  F_chan(x, α, β),
					(x, α, β) -> Jac_mat(x, α, β),
					(x, α, β) -> transpose(Jac_mat(x, α, β)),
					β -> ((x, α, v1, v2) -> d2F(x,α,v1,v2; b = β)),
					br, indfold,
					0.01, plot = true,
					optcontfold)
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
	out_mf, hist, flag = @time Cont.newton(
		x -> F_chan(x, a, 0.01),
		x -> (dx -> dF_chan(x, dx, a, 0.01)),
		sol,
		opt_newton_mf)
