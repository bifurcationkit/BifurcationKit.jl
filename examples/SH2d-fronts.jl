using Revise
	using DiffEqOperators, Setfield, Parameters
	using PseudoArcLengthContinuation, LinearAlgebra, Plots, SparseArrays
	const PALC = PseudoArcLengthContinuation

heatmapsol(x) = heatmap(reshape(Array(x), Nx, Ny)', color=:viridis)

Nx = 151*1
	Ny = 100*1
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

optnew = PALC.NewtonPar(verbose = true, tol = 1e-8, maxIter = 20)
# optnew = PALC.NewtonPar(verbose = true, tol = 1e-8, maxIter = 20, eigsolver = EigArpack(0.5, :LM))
	sol_hexa, hist, flag = @time PALC.newton(F_sh, dF_sh, vec(sol0), par, optnew)
	println("--> norm(sol) = ", norm(sol_hexa, Inf64))
	heatmapsol(sol_hexa)

heatmapsol(0.2vec(sol_hexa) .* vec([exp(-(x+0lx)^2/25) for x in X, y in Y]))

###################################################################################################
# recherche de solutions
deflationOp = DeflationOperator(2.0, (x, y)->dot(x, y), 1.0, [sol_hexa])

optnew = @set optnew.maxIter = 250
outdef, _, flag, _ = @time PALC.newton(F_sh, dF_sh,
		# 0.4vec(sol_hexa) .* vec([exp(-1(x+1lx)^2/25) for x in X, y in Y]),
		0.4vec(sol_hexa) .* vec([1 .- exp(-1(x+0lx)^2/55) for x in X, y in Y]),
		par, optnew, deflationOp, normN = x -> norm(x, Inf))
	println("--> norm(sol) = ", norm(outdef))
	heatmapsol(outdef) |> display
	flag && push!(deflationOp, outdef)

heatmapsol(deflationOp[2])

heatmapsol(0.4vec(sol_hexa) .* vec([1 .- exp(-1(x+0lx)^2/55) for x in X, y in Y]))
###################################################################################################
optcont = ContinuationPar(dsmin = 0.0001, dsmax = 0.005, ds= -0.001, pMax = -0.095, pMin = -1.0, newtonOptions = setproperties(optnew; tol = 1e-9, maxIter = 15), maxSteps = 145, detectBifurcation = 2, nev = 40, detectFold = false, dsminBisection =1e-7, saveSolEveryNsteps = 4)
	optcont = @set optcont.newtonOptions.eigsolver = EigArpack(0.1, :LM)

	br, u1 = @time PALC.continuation(
		F_sh, dF_sh,
		deflationOp[1], par, (@lens _.l), optcont;
		plot = true, verbosity = 3,
		tangentAlgo = BorderedPred(),
		# linearAlgo = MatrixBLS(),
		plotSolution = (x, p; kwargs...) -> (heatmap!(X, Y, reshape(x, Nx, Ny)'; color=:viridis, label="", kwargs...);ylims!(-1,1,subplot=4);xlims!(-.5,.3,subplot=4)),
		printSolution = (x, p) -> norm(x),
		finaliseSolution = (z, tau, step, contResult) -> 	(Base.display(contResult.eig[end].eigenvals) ;true),
		normC = x -> norm(x, Inf))
###################################################################################################
using IncompleteLU
prec = ilu(L1 + I,τ=0.05)
prec = lu(L1 + I)
ls = GMRESIterativeSolvers(tol = 1e-5, N = Nx*Ny, Pl = prec)

function dF_sh2(du, u, p)
	@unpack l, ν, L1 = p
	return -L1 * du .+ (l .+ 2 .* ν .* u .- 3 .* u.^2) .* du
end

sol_hexa, _, flag = @time PALC.newton(
		F_sh,
		(u, p) -> (du -> dF_sh2(du, u, p)),
		vec(sol0), parSH,
		@set optnew.linsolver = ls)
	println("--> norm(sol) = ", norm(sol_hexa, Inf64))
	heatmapsol(sol_hexa)
