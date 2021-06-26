using Revise
	using DiffEqOperators, Setfield, Parameters
	using BifurcationKit, LinearAlgebra, Plots, SparseArrays
	const BK = BifurcationKit

plotsol(x, Nx=Nx, Ny=Ny) = heatmap(reshape(Array(x), Nx, Ny)', color=:viridis)
plotsol!(x, Nx=Nx, Ny=Ny; kwargs...) = heatmap!(reshape(Array(x), Nx, Ny)'; color=:viridis, kwargs...)

Nx = 151
	Ny = 100
	lx = 8pi
	ly = 2*2pi/sqrt(3)

function Laplacian2D(Nx, Ny, lx, ly, bc = :Neumann)
	hx = 2lx/Nx
	hy = 2ly/Ny
	D2x = CenteredDifference(2, 2, hx, Nx)
	D2y = CenteredDifference(2, 2, hy, Ny)
	if bc == :Dirichlet
		Qx = Dirichlet0BC(typeof(hx))
		Qy = Dirichlet0BC(typeof(hy))
	elseif bc == :Neumann
		Qx = Neumann0BC(hx)
		Qy = Neumann0BC(hy)
	elseif bc == :Periodic
		Qx = PeriodicBC(hx)
		Qy = PeriodicBC(hy)
	end
	# @show norm(D2x - Circulant(D2x[1,:]))
	A = kron(sparse(I, Ny, Ny), sparse(D2x * Qx)[1]) + kron(sparse(D2y * Qy)[1], sparse(I, Nx, Nx))
	return A, D2x
end

function F_sh(u, p)
	@unpack l, ν, L1 = p
	return -L1 * u .+ (l .* u .+ ν .* u.^2 .- u.^3)
end

function dF_sh(u, p)
	@unpack l, ν, L1 = p
	return -L1 .+ spdiagm(0 => l .+ 2 .* ν .* u .- 3 .* u.^2)
end

d2F_sh(u, p, dx1, dx2) = (2 .* p.ν .* dx2 .- 6 .* dx2 .* u) .* dx1
d3F_sh(u, p, dx1, dx2, dx3) = (-6 .* dx2 .* dx3) .* dx1
jet = (F_sh, dF_sh, d2F_sh, d3F_sh)

X = -lx .+ 2lx/(Nx) * collect(0:Nx-1)
Y = -ly .+ 2ly/(Ny) * collect(0:Ny-1)

sol0 = [(cos(x) .+ cos(x/2) * cos(sqrt(3) * y/2) ) for x in X, y in Y]
	sol0 .= sol0 .- minimum(vec(sol0))
	sol0 ./= maximum(vec(sol0))
	sol0 = sol0 .- 0.25
	sol0 .*= 1.7
	heatmap(sol0', color=:viridis)

Δ, D2x = Laplacian2D(Nx, Ny, lx, ly, :Neumann)
L1 = (I + Δ)^2
par = (l = -0.1, ν = 1.3, L1 = L1)

optnew = NewtonPar(verbose = true, tol = 1e-8, maxIter = 20)
# optnew = NewtonPar(verbose = true, tol = 1e-8, maxIter = 20, eigsolver = EigArpack(0.5, :LM))
	sol_hexa, hist, flag = @time newton(F_sh, dF_sh, vec(sol0), par, optnew)
	println("--> norm(sol) = ", norm(sol_hexa, Inf64))
	plotsol(sol_hexa)

heatmapsol(0.2vec(sol_hexa) .* vec([exp(-(x+0lx)^2/25) for x in X, y in Y]))

heatmapsol(0.2vec(sol_hexa) .* vec([exp(-(x+0lx)^2/25) for x in X, y in Y]))
###################################################################################################
# recherche de solutions
deflationOp = DeflationOperator(2, 1.0, [sol_hexa])

optnew = @set optnew.maxIter = 250
outdef, _, flag, _ = @time newton(F_sh, dF_sh,
		# 0.4vec(sol_hexa) .* vec([exp(-1(x+1lx)^2/25) for x in X, y in Y]),
		0.4vec(sol_hexa) .* vec([1 .- exp(-1(x+0lx)^2/55) for x in X, y in Y]),
		par, optnew, deflationOp, normN = x -> norm(x, Inf))
	println("--> norm(sol) = ", norm(outdef))
	plotsol(outdef) |> display
	flag && push!(deflationOp, outdef)

plotsol(deflationOp[end])

plotsol(0.4vec(sol_hexa) .* vec([1 .- exp(-1(x+0lx)^2/55) for x in X, y in Y]))
###################################################################################################
optcont = ContinuationPar(dsmin = 0.0001, dsmax = 0.005, ds= -0.001, pMax = -0.0, pMin = -1.0, newtonOptions = setproperties(optnew; tol = 1e-9, maxIter = 15, verbose = false), maxSteps = 146, detectBifurcation = 3, nev = 40, dsminBisection = 1e-9, nInversion = 6, tolBisectionEigenvalue= 1e-19)
	optcont = @set optcont.newtonOptions.eigsolver = EigArpack(0.1, :LM)

	br, = @time continuation(
		F_sh, dF_sh,
		deflationOp[1], par, (@lens _.l), optcont;
		plot = true, verbosity = 3,
		# tangentAlgo = BorderedPred(),
		# linearAlgo = MatrixBLS(),
		plotSolution = (x, p; kwargs...) -> (plotsol!(x; label="", kwargs...)),
		printSolution = (x, p) -> (n2 = norm(x), n8 = norm(x, 8)),
		# finaliseSolution = (z, tau, step, contResult; k...) -> 	(Base.display(contResult.eig[end].eigenvals) ;true),
		# callbackN = cb,
		normC = x -> norm(x, Inf))
###################################################################################################
# codim2 Fold continuation
optfold = @set optcont.detectBifurcation = 0
@set! optfold.newtonOptions.verbose = true
optfold = setproperties(optfold; pMin = -2., pMax= 2., dsmax = 0.1)
brfold = continuation(F_sh, dF_sh, br, 1, (@lens _.ν), optfold;
	verbosity = 3, plot = true, issymmetric = true,
	# bdlinsolver = MatrixBLS(),
	updateMinAugEveryStep = 1,
	)


###################################################################################################
using IncompleteLU
prec = ilu(L1 + I,τ = 0.15)
prec = lu(L1 + I)
ls = GMRESIterativeSolvers(reltol = 1e-5, N = Nx*Ny, Pl = prec)

function dF_sh2(du, u, p)
	@unpack l, ν, L1 = p
	return -L1 * du .+ (l .+ 2 .* ν .* u .- 3 .* u.^2) .* du
end

sol_hexa, _, flag = @time newton(
		F_sh,
		(u, p) -> (du -> dF_sh2(du, u, p)),
		vec(sol0), par,
		@set optnew.linsolver = ls)
	println("--> norm(sol) = ", norm(sol_hexa, Inf64))
	plotsol(sol_hexa)
###################################################################################################
# Automatic branch switching

br2, = continuation(jet..., br, 2, setproperties(optcont; ds = -0.001, detectBifurcation = 3, plotEveryStep = 5, maxSteps = 170);  nev = 30,
			plot = true, verbosity = 2,
			plotSolution = (x, p; kwargs...) -> (plotsol!(x; label="", kwargs...);plot!(br; subplot=1,plotfold=false)),
			printSolution = (x, p) -> norm(x),
			normC = x -> norm(x, Inf))

plot(br, br2...)
###################################################################################################
# Manual branch switching
bp2d = computeNormalForm(jet..., br, 11; verbose = true, nev = 80)
BK.nf(bp2d)


using ProgressMeter, LaTeXStrings
Nd = 140; L = 0.9
# sampling grid
X = LinRange(-L,L, Nd); Y = LinRange(-L,L, Nd); P = LinRange(-0.0014,0.0014, Nd+1)

# sample reduced equation on the grid for the first component
V1a = @showprogress [bp2d(Val(:reducedForm),[x1,y1], p1)[1] for p1 in P, x1 in X, y1 in Y]
Ind1 = findall( abs.(V1a) .<= 9e-4 * maximum(abs.(V1a)))
# intersect with second component
V2a = @showprogress [bp2d(Val(:reducedForm),[X[ii[2]],Y[ii[3]]], P[ii[1]])[2] for ii in Ind1]
Ind2 = findall( abs.(V2a) .<= 3e-3 * maximum(abs.(V2a)))

# get solutions
resp = Float64[]; resx = Vector{Float64}[]; resnrm = Float64[]
	@showprogress for k in Ind2
		ii = Ind1[k]
		push!(resp, P[ii[1]])
		push!(resnrm, sqrt(X[ii[2]]^2+Y[ii[3]]^2))
		push!(resx, [X[ii[2]], Y[ii[3]]])
	end

plot(
	scatter(1e4resp, map(x->x[1], resx), map(x->x[2], resx); label = "", markerstrokewidth=0, xlabel = L"10^4 \cdot \lambda", ylabel = L"x_1", zlabel = L"x_2", zcolor = resnrm, color = :viridis,colorbar=false),
	scatter(1e4resp, resnrm; label = "", markersize =2, markerstrokewidth=0, xlabel = L"10^4 \cdot \lambda", ylabel = L"\|x\|"))
###################################################################################################
function optionsCont(x,p,l; opt = optcont)
	if l <= 1
		return opt
	elseif l==2
		return setproperties(opt ;detectBifurcation = 0,ds = 0.001, a = 0.75)
	else
		return setproperties(opt ;detectBifurcation = 0,ds = 0.00051, dsmax = 0.01)
	end
end

diagram = bifurcationdiagram(jet..., br, 2, optionsCont;
	plot = true, verbosity = 0,
	# usedeflation = true,
	# δp = 0.005,
	# tangentAlgo = BorderedPred(),
	callbackN = cb,
	# linearAlgo = MatrixBLS(),
	plotSolution = (x, p; kwargs...) -> (plotsol!(x; label="", kwargs...)),
	printSolution = (x, p) -> norm(x),
	finaliseSolution = (z, tau, step, contResult) -> 	(Base.display(contResult.eig[end].eigenvals) ;true),
	normC = x -> norm(x, Inf)
	)

plot(diagram; code = (), legend = false, plotfold = false)
	plot!(br)

# BK.add!(diagram, br2, 2)

###################################################################################################
deflationOp = DeflationOperator(2.0, dot, 1.0, [sol_hexa])
optcontdf = @set optcont.newtonOptions.verbose = true
brdf,  = continuation(F_sh, dF_sh, par, (@lens _.l), setproperties(optcontdf; detectBifurcation = 0, plotEveryStep = 1),
	deflationOp;
	showplot = true, verbosity = 2,
	# tangentAlgo = BorderedPred(),
	# linearAlgo = MatrixBLS(),
	# plotSolution = (x, p; kwargs...) -> (plotsol!(x; label="", kwargs...)),
	maxIterDefOp = 50,
	maxBranches = 40,
	seekEveryStep = 5,
	printSolution = (x, p) -> norm(x),
	perturbSolution = (x,p,id) -> (x  .+ 0.1 .* rand(length(x))),
	# finaliseSolution = (z, tau, step, contResult) -> 	(Base.display(contResult.eig[end].eigenvals) ;true),
	# callbackN = cb,
	normN = x -> norm(x, Inf))

plot(brdf...)
