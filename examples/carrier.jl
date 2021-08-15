using Revise
using LinearAlgebra, Parameters, Setfield, SparseArrays, BandedMatrices

using Plots, BifurcationKit
const BK = BifurcationKit
####################################################################################################
function F_carr(x, p)
	@unpack ϵ, X, dx = p
	f = similar(x)
	n = length(x)
	f[1] = x[1]
	f[n] = x[n]
	for i=2:n-1
		f[i] = ϵ^2 * (x[i-1] - 2 * x[i] + x[i+1]) / dx^2 +
			2 * (1 - X[i]^2) * x[i] + x[i]^2-1
	end
	return f
end

function Jac_carr!(J, x, p)
	@unpack ϵ, X, dx = p
	n = length(x)
	J[band(-1)] .= ϵ^2/dx^2    									# set the diagonal band
	J[band(1)]  .= ϵ^2/dx^2										# set the super-diagonal band
	J[band(0)]  .= (-2ϵ^2 /dx^2) .+ 2 * (1 .- X.^2) .+ 2 .* x   # set the second super-diagonal band
	J[1, 1] = 1.0
	J[n, n] = 1.0
	J[1, 2] = 0.0
	J[n, n-1] = 0.0
	J
end
	# @time Jac_carr(sol, par_car)
Jac_carr(x, p) = Jac_carr!(BandedMatrix{Float64}(undef, (length(x),length(x)), (1,1)), x, p)

jet = BK.getJet(F_carr, Jac_carr)

N = 200
X = LinRange(-1,1,N)
dx = X[2] - X[1]
par_car = (ϵ = 0.7, X = X, dx = dx)
sol = -(1 .- par_car.X.^2)
norminf(x) = norm(x,Inf)
recordFromSolution(x, p) = (x[2]-x[1]) * sum(x->x^2, x)

optnew = NewtonPar(tol = 1e-8, verbose = true)
	out, = @time newton(
		F_carr, Jac_carr, sol, par_car, optnew, normN = x -> norm(x, Inf64))
	Plots.plot(out, label="Solution")

optcont = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= -0.01, pMin = 0.05, plotEveryStep = 10, newtonOptions = NewtonPar(tol = 1e-8, maxIter = 20, verbose = false), maxSteps = 300, detectBifurcation = 3, nev = 40, nInversion = 6, maxBisectionSteps = 25)
	br, = @time continuation(
		F_carr, Jac_carr, zeros(N), par_car, (@lens _.ϵ), optcont;
		plot = true, verbosity = 3,
		recordFromSolution = recordFromSolution,
		normC = norminf)

plot(br)

####################################################################################################
# Example with deflation technics
deflationOp = DeflationOperator(2, dot, 1.0, empty([out]), copy(out))
par_def = @set par_car.ϵ = 0.6

optdef = setproperties(optnew; tol = 1e-7, maxIter = 200)

function perturbsol(sol, p, id)
	sol0 = @. exp(-.01/(1-par_car.X^2)^2)
	# solp = copy(sol)
	solp = 0.02*rand(length(sol))
	# plot([sol, sol + solp * sol0], xlims=(-1,1),title = "Perturb $id") |> display
	return sol .+ solp .* sol0
end

outdef1, _, flag = @time newton(
	F_carr, Jac_carr,
	# perturbsol(deflationOp[1],0,0), par_def,
	perturbsol(-out, 0, 0), par_def,
	optdef, deflationOp;
	# callback = (x, f, J, res, iteration, itlinear, options; kwargs...) ->(res < 1e8)
	)
	flag && push!(deflationOp, outdef1)

plot(); for _s in deflationOp.roots; plot!(_s);end;title!("")
perturbsol(-deflationOp[1],0,0) |> plot
####################################################################################################
# bifurcation diagram with deflated continuation
# empty!(deflationOp)

br, _ = @time continuation(
	F_carr, Jac_carr,
	par_def, (@lens _.ϵ),
	setproperties(optcont; ds = -0.00021, dsmin=1e-5, maxSteps = 20000, pMax = 0.7, pMin = 0.05, newtonOptions = setproperties(optnew; tol = 1e-9, maxIter = 100, verbose = false), detectBifurcation = 0, plotEveryStep = 40),
	deflationOp;
	verbosity = 1,
	maxBranches = 100,
	# tangentAlgo = BorderedPred(),
	perturbSolution = perturbsol,
	recordFromSolution = recordFromSolution,
	normN = norminf,
	)

plot(br..., branchlabel = 1:length(br), legend=true)#, marker=:d)


BifurcationKit.mergeBranches(br)
####################################################################################################
# bifurcation diagram
diagram = bifurcationdiagram(jet...,
		0*out, par_car,
		(@lens _.ϵ), 2,
		(arg...) -> @set optcont.newtonOptions.verbose=false;
		recordFromSolution = (x, p) -> (x[2]-x[1]) * sum(x->x^2, x),
		plot = true)

plot(diagram, legend=false)
