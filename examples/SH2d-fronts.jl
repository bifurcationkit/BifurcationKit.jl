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

opt_new = Cont.NewtonPar(verbose = true, tol = 1e-9, maxIter = 20, eigsolver = eig_KrylovKit(tol = 1e-9, issymmetric = true))
	sol_hexa, hist, flag = @time Cont.newton(
				x -> F_sh(x, -.1, 1.3),
				u -> dF_sh(u, -.1, 1.3),
				vec(sol0),
				opt_new)
	println("--> norm(sol) = ", norm(sol_hexa, Inf64))
	heatmapsol(sol_hexa)

heatmapsol(0.2vec(sol_hexa) .* vec([exp(-(x+0lx)^2/25) for x in X, y in Y]))

###################################################################################################
# recherche de solutions
deflationOp = DeflationOperator(2.0, (x, y)->dot(x, y), 1.0, [sol_hexa])

opt_new.maxIter = 250
outdef, _, flag, _ = @time Cont.newtonDeflated(
				x -> F_sh(x, -.1, 1.3),
				u -> dF_sh(u, -.1, 1.3),
				0.4vec(sol_hexa) .* vec([exp(-1(x+0lx)^2/25) for x in X, y in Y]),

				opt_new, deflationOp, normN = x -> norm(x, Inf))
		println("--> norm(sol) = ", norm(outdef))
		heatmapsol(outdef) |> display
		flag && push!(deflationOp, outdef)

heatmapsol(deflationOp.roots[2])
###################################################################################################
opts_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.005, ds= -0.0015, pMax = -0.0, pMin = -1.0, newtonOptions = opt_new, theta = 0.6)
	opts_cont.newtonOptions.tol = 1e-9
	opts_cont.newtonOptions.maxIter = 20

	opts_cont.maxSteps = 141
	opts_cont.computeEigenValues = true
	opts_cont.nev = 30

	br, u1 = @time Cont.continuation(
		(x, p) ->  F_sh(x, p, 1.3),
		(x, p) -> dF_sh(x, p, 1.3),
		deflationOp.roots[3],
		-0.1, verbosity = 2,
		opts_cont, plot = true,
		# tangentalgo = BorderedPred(),
		plotsolution = (x;kwargs...)->(heatmap!(X, Y, reshape(x, Nx, Ny)', color=:viridis, subplot=4, label="");ylims!(-1,1,subplot=5);xlims!(-.5,.3,subplot=5)),
		printsolution = x -> norm(x),
		normC = x -> norm(x, Inf))

# 1 ds- 100
# 2 ds+ 110
# 3 ds- 130
# 4 ds- 90

# heatmap!(X, Y, reshape(sol_hexa, Nx, Ny)', color=:viridis, subplot=4, xlabel="Solution at blue point")
# branches = []
push!(branches, br)

# Cont.plotBranch(branches[1])
# Cont.plotBranch!(branches[7], ylabel="max U")

Cont.plotBranch(branches, false;xlabel="p", ylabel="||u||",label="")

# ls = vcat(fill(:dash,5),fill(:solid,5))
# plot(rand(10),linestyle = ls)


branches[2]

plotBranch!(br, label="", xlabel="l",marker=:d)
br


cstab = [st ? :green : :red for st in br.stability]
scatter!(br.branch[1,:],br.branch[2,:], color=cstab)


plot(br.branch[1,:],br.branch[2,:], color=cstab)
