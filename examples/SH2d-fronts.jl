using Revise
	using DiffEqOperators
	using PseudoArcLengthContinuation, LinearAlgebra, Plots, SparseArrays
	const Cont = PseudoArcLengthContinuation

heatmapsol(x) = heatmap(reshape(Array(x), Nx, Ny)', color=:viridis)

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

Δ, D2x = Laplacian2D(Nx, Ny, lx, ly, :Neumann)
const L1 = (I + Δ)^2

function F_sh(u, l=-0.15, ν=1.3)
	return -L1 * u .+ (l .* u .+ ν .* u.^2 .- u.^3)
end

function dF_sh(u, l=-0.15, ν=1.3)
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

opt_new = Cont.NewtonPar(verbose = true, tol = 1e-9, maxIter = 20)
	sol_hexa, hist, flag = @time Cont.newton(
				x -> F_sh(x, -.1, 1.3),
				u -> dF_sh(u, -.1, 1.3),
				vec(sol0),
				opt_new)
	println("--> norm(sol) = ", norm(sol_hexa, Inf64))
	heatmapsol(sol_hexa)



heatmapsol(0.2vec(sol_hexa) .* vec([exp(-(x+0lx)^2/25) for x in X, y in Y]))


# using Arpack
# J0 = dF_sh(sol_hexa, -.1, 1.3)
# Arpack.eigs(J0, nev = 10, which = :LR, sigma = 0.)
#
# using KrylovKit
# @time KrylovKit.eigsolve(J0,10, :LR)
#
# using ArnoldiMethod
# decomp,_ = @time ArnoldiMethod.partialschur(x -> J0\x, nev = 10, which = LR(), tol=1e-6)
# partialeigen(decomp)
#
# using IterativeSolvers
# IterativeSolvers.ei


# 0.7*vec(sol1 .* exp.(vec(exp.(-0*X.^2 .- X'.^2/50.))))
# 0.3*sol_hexa .* vec(1 .-exp.(-0X.^2 .- (X.-lx)'.^2/(2*8^2))) p = -.175
###################################################################################################
# recherche de solutions
deflationOp = DeflationOperator(2.0, (x, y)->dot(x, y), 1.0, [sol_hexa])

opt_new.maxIter = 250
outdef, _, flag, _ = @time Cont.newtonDeflated(
				x -> F_sh(x, -.1, 1.3),
				u -> dF_sh(u, -.1, 1.3),
				0.4vec(sol_hexa) .* vec([exp(-1(x+0lx)^2/25) for x in X, y in Y]),

				opt_new, deflationOp, normN = x->norm(x, Inf64))
		println("--> norm(sol) = ", norm(outdef))
		heatmapsol(outdef) |> display
		flag && push!(deflationOp, outdef)

heatmapsol(deflationOp.roots[2])
###################################################################################################
opts_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.015, ds= -0.0015, pMax = -0.0, pMin = -1.0, theta = 0.5, plot_every_n_steps = 3, newtonOptions = opt_new, a = 0.5, detect_fold = true, detect_bifurcation = false)
	opts_cont.newtonOptions.tol = 1e-9
	opts_cont.newtonOptions.maxIter = 50
	opts_cont.maxSteps = 450
	opts_cont.computeEigenValues = true

	br, u1 = @time Cont.continuation(
		(x, p) -> F_sh(x, p, 1.3),
		(x, p) -> dF_sh(x, p, 1.3),
		deflationOp.roots[1],
		-0.1, verbosity = 2,
		opts_cont, plot = true,
		plotsolution = (x;kwargs...)->(heatmap!(X, Y, reshape(x, Nx, Ny)', color=:viridis, subplot=4, label="")),
		printsolution = x->norm(x))#norm(x)/N^2*lx*ly)

# heatmap!(X, Y, reshape(sol_hexa, Nx, Ny)', color=:viridis, subplot=4, xlabel="Solution at blue point")
branches = []
push!(branches, br)

# Cont.plotBranch(branches[1])
# Cont.plotBranch!(branches[7], ylabel="max U")

Cont.plotBranch(branches,xlabel="p", ylabel="||u||",label="")

# ls = vcat(fill(:dash,5),fill(:solid,5))
# plot(rand(10),linestyle = ls)
