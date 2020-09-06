using Revise
	using SparseArrays, LinearAlgebra, DiffEqOperators, Setfield, Parameters
	using BifurcationKit
	using Plots
	const BK = BifurcationKit
################################################################################
# case of the SH equation
norminf(x) = norm(x, Inf64)
Nx = 200; Lx = 6.;
X = -Lx .+ 2Lx/Nx*(0:Nx-1) |> collect
hx = X[2]-X[1]

# Q = Neumann0BC(hx)
Q = Dirichlet0BC(hx |> typeof)
Dxx = sparse(CenteredDifference(2, 2, hx, Nx) * Q)[1]
Lsh = -(I + Dxx)^2

function R_SH(u, par)
	@unpack p, b, L1 = par
	out = similar(u)
	out .= L1 * u .- p .* u .+ b .* u.^3 - u.^5
end

Jac_sp = (u, par) -> par.L1 + spdiagm(0 => -par.p .+ 3*par.b .* u.^2 .- 5 .* u.^4)
parSH = (p = 0.7, b = 2., L1 = Lsh)
####################################################################################################
sol0 = 1.1cos.(X) .* exp.(-0X.^2/(2*5^2))
	optnew = NewtonPar(verbose = true, tol = 1e-12)
	# allocations 26.47k, 0.038s, tol = 1e-10
	sol1, hist, flag = @time BK.newton(
	R_SH, Jac_sp,
	sol0, (@set parSH.p = -1.95), optnew)
	Plots.plot(X, sol1)

opts = BK.ContinuationPar(dsmin = 0.0001, dsmax = 0.01, ds = -0.01,
		newtonOptions = setproperties(optnew; maxIter = 30, tol = 1e-8), pMin = -2.1,
		maxSteps = 300, plotEveryStep = 40, detectBifurcation = 3, nInversion = 4, tolBisectionEigenvalue = 1e-17, dsminBisection = 1e-7)

function cb(x,f,J,res,it,itl,optN; kwargs...)
	_x = get(kwargs, :z0, nothing)
	if _x isa BorderedArray
		return norm(_x.u - x) < 20.5 && abs(_x.p - kwargs[:p])<0.05
	end
	true
end

args = (verbosity = 3,
	plot = true,
	# tangentAlgo = BorderedPred(),
	linearAlgo  = MatrixBLS(),
	plotSolution = (x, p;kwargs...)->(plot!(X, x; ylabel="solution", label="", kwargs...)),
	callbackN = cb
	)

brflat, u1 = @time continuation(
	R_SH, Jac_sp, sol1, (@set parSH.p = 1.), (@lens _.p), opts;
	args...)

plot(brflat)
####################################################################################################
# branch switching
d2R(u,p,dx1,dx2) = @. p.b * 6u*dx1*dx2 - 5*4u^3*dx1*dx2
d3R(u,p,dx1,dx2,dx3) = @. p.b * 6dx3*dx1*dx2 - 5*4*3u^2*dx1*dx2*dx3
jet = (R_SH, Jac_sp, d2R, d3R)

function optrec(x, p, l; opt = opts)
	level =  l
	if level <= 2
		return setproperties(opt; maxSteps = 300, detectBifurcation = 3, nev = Nx, detectLoop = false)
	else
		return setproperties(opt; maxSteps = 250, detectBifurcation = 3, nev = Nx, detectLoop = true)
	end
end

diagram = @time bifurcationdiagram(jet..., sol1, (@set parSH.p = 1.), (@lens _.p), 4, optrec; args...)

code = ()
	plot(diagram; code = code, plotfold = false,  markersize = 2, putbifptlegend = false)
	plot!(brflat)
	title!("#branches = $(size(diagram, code))")

diagram2 = bifurcationdiagram!(jet..., BK.getBranch(diagram, (2,)),  (current = 1, maxlevel = 2), optrec; args...)

####################################################################################################
deflationOp = DeflationOperator(2.0, dot, 1.0, [sol1])

br, _ = @time continuation(
	R_SH, Jac_sp,
	(@set parSH.p = 1.), (@lens _.p),
	setproperties(opts; ds = -0.01, maxSteps = 10000, pMax = 2.7, pMin = 0.5, newtonOptions = setproperties(optnew; tol = 1e-9, maxIter = 15, verbose = false), saveSolEveryStep = 0, detectBifurcation = 0),
	deflationOp;
	verbosity = 1,
	maxBranches = 100,
	perturbSolution = (sol, p, id) -> sol .+ 0.02 .* rand(length(sol)),
	normN = x -> norm(x, Inf64),
	# tangentAlgo = SecantPred(),
	# callbackN = (x, f, J, res, iteration, itlinear, options; kwargs...) ->(true)
	)

plot(br, legend=false)
