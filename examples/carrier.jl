using Revise
using LinearAlgebra, Parameters, Setfield, SparseArrays, BandedMatrices

using BifurcationKit, Plots
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

function Jac_carr(x, p)
	@unpack ϵ, X, dx = p
	n = length(x)
	J = BandedMatrix{Float64}(undef, (n,n), (1,1))
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


N = 200
X = LinRange(-1,1,N)
dx = X[2] - X[1]
par_car = (ϵ = 0.7, X = X, dx = dx)
sol = -(1 .- par_car.X.^2)


optnew = NewtonPar(tol = 1e-8, verbose = true)
	out, _, flag = @time newton(
		F_carr, Jac_carr, sol, par_car, optnew, normN = x -> norm(x, Inf64))
	Plots.plot(out, label="Solution")

optcont = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= -0.01, pMin = 0.05, plotEveryStep = 10, newtonOptions = NewtonPar(tol = 1e-8, maxIter = 20, verbose = true), maxSteps = 300, detectBifurcation = 3, nev = 40)

	br, _ = @time continuation(
		F_carr, Jac_carr, 0*out, par_car, (@lens _.ϵ), optcont;
		plot = true,
		plotSolution = (x, p; kwargs...) -> plot!(x; label = "l = $(length(x))", kwargs...),
		verbosity = 2,
		printSolution = (x, p) -> (x[2]-x[1]) * sum(x.^2),
		normC = x -> norm(x, Inf64))

plot(br)

####################################################################################################
# Example with deflation technique
deflationOp = DeflationOperator(2.0, dot, 1.0, empty([out]))
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
	perturbsol(-out ,0,0), par_def,
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
	perturbSolution = perturbsol,
	printSolution = (x, p) -> (x[2]-x[1]) * sum(x.^2),
	normN = x -> norm(x, Inf64),
	# callbackN = (x, f, J, res, iteration, itlinear, options; kwargs...) ->(true)
	)

plot(br, label="")


BifurcationKit.mergeBranches(br)
####################################################################################################
# bifurcation diagram
using ForwardDiff
D(f, x, p, dx) = ForwardDiff.derivative(t -> f(x .+ t .* dx, p), 0.)
dF_carr(x,p)         =  ForwardDiff.jacobian( z-> F_carr(z,p), x)
d1F_carr(x,p,dx1)         = D((z, p0) -> F_carr(z, p0), x, p, dx1)
d2F_carr(x,p,dx1,dx2)     = D((z, p0) -> d1F_carr(z, p0, dx1), x, p, dx2)
d3F_carr(x,p,dx1,dx2,dx3) = D((z, p0) -> d2F_carr(z, p0, dx1, dx2), x, p, dx3)
jet = (F_carr, Jac_carr, d2F_carr, d3F_carr)

diagram = bifurcationdiagram(jet...,
		0*out, par_car,
		(@lens _.ϵ), 2,
		(arg...) -> @set optcont.newtonOptions.verbose=false;
		printSolution = (x, p) -> (x[2]-x[1]) * sum(x.^2),
		plot = true)

plot(diagram, legend=false)
