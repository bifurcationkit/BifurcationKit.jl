using Revise
	using PseudoArcLengthContinuation, LinearAlgebra, Plots, Setfield
	const PALC = PseudoArcLengthContinuation

N(x; a = 0.5, b = 0.01) = 1 + (x + a*x^2)/(1 + b*x^2)
dN(x; a = 0.5, b = 0.01) = (1-b*x^2+2*a*x)/(1+b*x^2)^2
d2N(x; a = 0.5, b = 0.01) = -(2*(-a+3*a*b*x^2+3*b*x-b^2*x^3))/(1+b*x^2)^3

function F_chan(x, α, β = 0.01)
	f = similar(x)
	n = length(x)
	f[1] = x[1] - β
	f[n] = x[n] - β
	for i=2:n-1
		f[i] = (x[i-1] - 2 * x[i] + x[i+1]) * (n-1)^2 + α * N(x[i], b = β)
	end
	return f
end

function Jac_mat(u, α, β = 0.01)
	n = length(u)
	J = zeros(n, n)
	J[1, 1] = 1.0
	J[n, n] = 1.0
	for i = 2:n-1
		J[i, i-1] = (n-1)^2
		J[i, i+1] = (n-1)^2
		J[i, i] = -2 * (n-1)^2 + α * dN(u[i], b = β)
	end
	return J
end

Jac_fd(u0, α, β = 0.) = finiteDifferences(u -> F_chan(u, α, β = β), u0)

n = 101
	a = 3.3
	sol = [(i-1)*(n-i)/n^2+0.1 for i=1:n]
	optnewton = NewtonPar(tol = 1e-11, verbose = true)
	# ca fait dans les 63.59k Allocations
	out, hist, flag = @time newton(
		x ->  F_chan(x, a),
		x -> Jac_mat(x, a),
		sol,
		optnewton)


optscont = ContinuationPar(dsmin = 0.01, dsmax = 0.1, ds= 0.01, pMax = 4.1, nev = 5, detectFold = true, plotEveryNsteps = 40, newtonOptions = NewtonPar(maxIter = 70, tol = 1e-8), maxSteps = 150)
	br, _ = @time continuation(
		(x, p) ->   F_chan(x, p),
		(x, p) -> (Jac_mat(x, p)),
		out, a, optscont,
		plot = true,
		plotSolution = (x;kwargs...) -> (plot!(x;ylabel="solution",label="",kwargs...))
		)
###################################################################################################
# Example with deflation technique
deflationOp = DeflationOperator(2.0, (x, y) -> dot(x, y), 1.0, [out])

optdef = setproperties(optnewton; tol = 1e-10, maxIter = 1000)

outdef1, _, _ = @time newton(
						x ->  F_chan(x, a),
						x -> Jac_mat(x, a),
						out.*(1 .+ 0.01*rand(n)),
						optdef, deflationOp)
plot(out, label="newton")
	plot!(sol, label="init guess")
	plot!(outdef1, label="deflation-1")

#save newly found point to look for new ones
push!(deflationOp, outdef1)
outdef2, _, _ = @time newton(
						x ->  F_chan(x, a),
						x -> Jac_mat(x, a),
						outdef1.*(1 .+ 0.01*rand(n)),
						optdef, deflationOp)
plot!(outdef2, label="deflation-2")
#################################################################################################### Continuation of the Fold Point using minimally augmented formulation
optscont = (@set optscont.newtonOptions = setproperties(optscont.newtonOptions; verbose = true, tol = 1e-10))

indfold = 2

outfold, _, flag = @time newtonFold(
			(x, α) ->  F_chan(x, α),
			(x, α) -> Jac_mat(x, α),
			br, indfold, #index of the fold point
			optscont.newtonOptions)
flag && printstyled(color=:red, "--> We found a Fold Point at α = ", outfold.p, ", β = 0.01, from ", br.foldpoint[indfold][3],"\n")

optcontfold = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.05, pMax = 4.1, pMin = 0., newtonOptions = NewtonPar(verbose=true, tol = 1e-8), maxSteps = 1300)
	outfoldco, _, _ = @time continuationFold(
			(x, α, β) ->  F_chan(x, α, β),
			(x, α, β) -> Jac_mat(x, α, β),
			br, indfold,
			0.01, plot = true, verbosity = 2,
			optcontfold)
PALC.plotBranch(outfoldco, xlabel="beta", ylabel = "alpha", label = "");title!("")
################################################################################################### Fold Newton / Continuation when Hessian is known. Does not require state to be AbstractVector
d2F(x, p, u, v; b = 0.01) = p * d2N.(x; b = b) .* u .* v

outfold, _, flag = @time newtonFold(
			(x, α) -> F_chan(x, α),
			(x, α) -> Jac_mat(x, α),
			br, indfold, #index of the fold point
			optscont.newtonOptions; Jt = (x, α) -> transpose(Jac_mat(x, α)),
			d2F = d2F)
		flag && printstyled(color=:red, "--> We found a Fold Point at α = ", outfold.p, ", β = 0.01, from ", br.foldpoint[indfold][3],"\n")

optcontfold = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, pMax = 4.1, pMin = 0., newtonOptions = NewtonPar(verbose=true), maxSteps = 1300)

outfoldco, _, _ = @time continuationFold(
					(x, α, β) ->  F_chan(x, α, β),
					(x, α, β) -> Jac_mat(x, α, β),
					br, indfold, 0.01, optcontfold;
					Jt = (x, α, β) -> transpose(Jac_mat(x, α, β)),
					d2F = β -> ((x, α, v1, v2) -> d2F(x,α,v1,v2; b = β)), plot = true)
###################################################################################################
# Matrix Free example
function dF_chan(x, dx, α, β = 0.01)
	out = similar(x)
	n = length(x)
	out[1] = dx[1]
	out[n] = dx[n]
	for i=2:n-1
		out[i] = (dx[i-1] - 2 * dx[i] + dx[i+1]) * (n-1)^2 + α * dN(x[i], b = β) * dx[i]
	end
	return out
end

ls = GMRESKrylovKit(dim = 100)
	optnewton_mf = NewtonPar(tol = 1e-11, verbose = true, linsolver = ls, eigsolver = DefaultEig())
	out_mf, _, flag = @time newton(
		x -> F_chan(x, a),
		x -> (dx -> dF_chan(x, dx, a)),
		sol,
		optnewton_mf)

opts_cont_mf  = ContinuationPar(dsmin = 0.01, dsmax = 0.1, ds= 0.01, pMax = 4.1, nev = 5, plotEveryNsteps = 40, newtonOptions = setproperties(optnewton_mf; maxIter = 70, tol = 1e-8), maxSteps = 150)
	brmf, _ = @time continuation(
		(x, p) -> F_chan(x, p),
		(x, p) -> (dx -> dF_chan(x, dx, p)),
		out, a, opts_cont_mf,
		# linearalgo = MatrixFreeBLS(),
		)

using SparseArrays
P = spdiagm(0 => -2 * (n-1)^2 * ones(n), -1 => (n-1)^2 * ones(n-1), 1 => (n-1)^2 * ones(n-1))
P[1,1:2] .= [1, 0.];P[end,end-1:end] .= [0, 1.]

ls = GMRESIterativeSolvers(tol = 1e-4, N = length(sol), restart = 10, maxiter=10, Pl = lu(P))
	optnewton_mf = NewtonPar(tol = 1e-11, verbose = true, linsolver = ls, eigsolver = DefaultEig())
	out_mf, _, flag = @time newton(
		x -> F_chan(x, a),
		x -> (dx -> dF_chan(x, dx, a)),
		sol,
		optnewton_mf)


plotBranch(brmf,color=:red);title!("")

# matrix free with different tangent predictor
brmf, _ = @time continuation(
	(x, p) -> F_chan(x, p),
	(x, p) -> (dx -> dF_chan(x, dx, p)),
	out, a, opts_cont_mf,
	tangentAlgo = BorderedPred(),
	# linearAlgo = PALC.MatrixFreeBLS(),
	)

plotBranch(brmf,color=:blue)

brmf, _ = @time continuation(
	(x, p) -> F_chan(x, p),
	(x, p) -> (dx -> dF_chan(x, dx, p)),
	out, a, opts_cont_mf,
	tangentAlgo = SecantPred(),
	linearAlgo = MatrixFreeBLS()
	)

plotBranch(brmf,color=:green)

brmf, _ = @time continuation(
	(x, p) -> F_chan(x, p),
	(x, p) -> (dx -> dF_chan(x, dx, p)),
	out, a, opts_cont_mf,
	tangentAlgo = BorderedPred(),
	linearAlgo = MatrixFreeBLS())

plotBranch(brmf,color=:orange)
