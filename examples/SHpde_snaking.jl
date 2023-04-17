using Revise
	using SparseArrays, LinearAlgebra, DiffEqOperators, Parameters
	using BifurcationKit
	using Plots
	const BK = BifurcationKit
################################################################################
# case of the SH equation
Nx = 200; Lx = 6.;
X = -Lx .+ 2Lx/Nx*(0:Nx-1) |> collect
hx = X[2]-X[1]

norminf(x) = norm(x, Inf64)
const _weight = rand(Nx)
normweighted(x) = norm(_weight .* x)

# Q = Neumann0BC(hx)
Q = Dirichlet0BC(hx |> typeof)
Dxx = sparse(CenteredDifference(2, 2, hx, Nx) * Q)[1]
Lsh = -(I + Dxx)^2

function R_SH(u, par)
	@unpack l, ν, L1 = par
	out = similar(u)
	out .= L1 * u .+ l .* u .+ ν .* u.^3 - u.^5
end

Jac_sp(u, par) = par.L1 + spdiagm(0 => par.l .+ 3 .* par.ν .* u.^2 .- 5 .* u.^4)
d2R(u,p,dx1,dx2) = @. p.ν * 6u*dx1*dx2 - 5*4u^3*dx1*dx2
d3R(u,p,dx1,dx2,dx3) = @. p.ν * 6dx3*dx1*dx2 - 5*4*3u^2*dx1*dx2*dx3

parSH = (l = -0.1, ν = 2., L1 = Lsh)
sol0 = 1.1cos.(X) .* exp.(-0X.^2/(2*5^2))

prob = BifurcationProblem(R_SH, sol0, parSH, (@lens _.l); J = Jac_sp,
	recordFromSolution = (x, p) -> (n2 = norm(x), nw = normweighted(x), s = sum(x), s2 = x[end ÷ 2], s4 = x[end ÷ 4], s5 = x[end ÷ 5]),
	plotSolution = (x, p;kwargs...)->(plot!(X, x; ylabel="solution", label="", kwargs...)))
####################################################################################################
optnew = NewtonPar(verbose = false, tol = 1e-12)
	# allocations 357, 0.8ms
	sol1 = @time newton(prob, optnew)
	Plots.plot(X, sol1.u)

opts = ContinuationPar(dsmin = 0.0001, dsmax = 0.01, ds = 0.01,
		newtonOptions = setproperties(optnew; maxIter = 30, tol = 1e-8), pMax = 1.,
		maxSteps = 300, plotEveryStep = 40, detectBifurcation = 3, nInversion = 4, tolBisectionEigenvalue = 1e-17, dsminBisection = 1e-7)

function cb(state; kwargs...)
	_x = get(kwargs, :z0, nothing)
	fromNewton = get(kwargs, :fromNewton, false)
	if ~fromNewton
		# if the residual is too large or if the parameter jump
		# is too big, abort continuation step
		return norm(_x.u - state.x) < 20.5 && abs(_x.p - state.p) < 0.05
	end
	true
end

kwargsC = (verbosity = 3,
	plot = true,
	# tangentAlgo = BorderedPred(),
	linearAlgo  = MatrixBLS(),
	callbackN = cb
	)

brflat = @time continuation(prob, PALC(), opts; kwargsC...)

plot(brflat, putspecialptlegend = false)
####################################################################################################
# branch switching
function optrec(x, p, l; opt = opts)
	level =  l
	if level <= 2
		return setproperties(opt; maxSteps = 300, detectBifurcation = 3, nev = Nx, detectLoop = false)
	else
		return setproperties(opt; maxSteps = 250, detectBifurcation = 3, nev = Nx, detectLoop = true)
	end
end

diagram = @time bifurcationdiagram(reMake(prob, params = @set parSH.l = -0.1), PALC(), 2, optrec; kwargsC..., halfbranch = true, verbosity = 0, usedeflation = false)

code = ()
	vars = (:param, :n2)
	plot(diagram; code = code, plotfold = false,  markersize = 2, putspecialptlegend = false, vars = vars)
	plot!(brflat, putspecialptlegend = false, vars = vars)
	title!("#branches = $(size(diagram, code))")

diagram2 = bifurcationdiagram!(diagram.γ.prob, BK.getBranch(diagram, (2,)), 3, optrec; kwargsC..., halfbranch = true)

####################################################################################################
deflationOp = DeflationOperator(2, 1.0, [sol1.u])
algdc = BK.DefCont(deflationOperator = deflationOp, maxBranches = 150, perturbSolution = (sol, p, id) -> sol .+ 0.02 .* rand(length(sol)),)

br = @time continuation(
	reMake(prob, params = @set parSH.l = -0.1), algdc,
	setproperties(opts; ds = 0.001, maxSteps = 20000, pMax = 0.25, pMin = -1., newtonOptions = setproperties(optnew; tol = 1e-9, maxIter = 15, verbose = false), saveSolEveryStep = 0, detectBifurcation = 0);
	verbosity = 1,
	normN = x -> norm(x, Inf64),
	# tangentAlgo = SecantPred(),
	# callbackN = (x, f, J, res, iteration, itlinear, options; kwargs...) ->(true)
	)

plot(br, legend=false, linewidth=1, vars = (:param, :n2))
