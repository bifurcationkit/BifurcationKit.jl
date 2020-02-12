# example taken from Aragón, J. L., R. A. Barrio, T. E. Woolley, R. E. Baker, and P. K. Maini. “Nonlinear Effects on Turing Patterns: Time Oscillations and Chaos.” Physical Review E 86, no. 2 (August 8, 2012): 026201. https://doi.org/10.1103/PhysRevE.86.026201.
using Revise
	using DiffEqOperators, ForwardDiff, DifferentialEquations, Sundials
	using PseudoArcLengthContinuation, LinearAlgebra, Plots, SparseArrays, Parameters, Setfield
	const PALC = PseudoArcLengthContinuation

norminf = x -> norm(x, Inf)
f(u, v, p) = p.η * (      u + p.a * v - p.C * u * v - u * v^2)
g(u, v, p) = p.η * (p.H * u + p.b * v + p.C * u * v + u * v^2)

function Laplacian(N, lx, bc = :Dirichlet)
	hx = 2lx/N
	D2x = CenteredDifference(2, 2, hx, N)
	if bc == :Neumann
		Qx = Neumann0BC(hx)
	elseif bc == :Dirichlet
		Qx = Dirichlet0BC(typeof(hx))
	end
	D2xsp = sparse(D2x * Qx)[1]
end

function NL!(dest, u, p, t = 0.)
	N = div(length(u), 2)
	u1 =  @view (u[1:N])
	u2 =  @view (u[N+1:end])

	dest[1:N]     .= f.(u1, u2, Ref(p))
	dest[N+1:end] .= g.(u1, u2, Ref(p))

	return dest
end

function Fbr!(f, u, p, t = 0.)
	mul!(f, p.Δ, u)
	f .= f .+ NL(u, p)
end

function NL(u, p)
	out = similar(u)
	NL!(out, u, p)
	out
end

function Fbr(x, p, t = 0.)
	f = similar(x)
	Fbr!(f, x, p)
end

dNL(x, p, dx) = ForwardDiff.derivative(t -> NL(x .+ t .* dx, p), 0.)

function dFbr(x, p, dx)
	f = similar(dx)
	mul!(f, p.Δ, dx)

	nl = dNL(x, p, dx)
	f .= f .+ nl
end

Jbr(x, p) = sparse(ForwardDiff.jacobian(x -> Fbr(x, p), x))
# Jbr(sol0, par_br)

####################################################################################################
N = 100
	n = 2N
	lx = 3pi /2
	X = LinRange(-lx,lx, N)
	Δ = Laplacian(N, lx, :Neumann)
	D = 0.08
	par_br = (η = 1.0, a = -1., b = -3/2., H = 3.0, D = D, C = -0.6, Δ = blockdiag(D*Δ, Δ))

	u0 = cos.(2X)
	solc0 = vcat(u0, u0)
####################################################################################################
# eigls = DefaultEig()
eigls = EigArpack(0.5, :LM)
	optnewton = NewtonPar(eigsolver = eigls, verbose=true, maxIter = 3200, tol=1e-9)
	out, _, _ = @time newton(
		x -> Fbr(x, par_br),
		x -> Jbr(x, par_br),
		solc0, optnewton,
		normN = norminf)

		plot();plot!(X,out[1:N]);plot!(X,solc0[1:N], label = "sol0",line=:dash)


optcont = ContinuationPar(dsmax = 0.01, ds = -0.001, pMin = -1.8, detectBifurcation = 1, nev = 21, plotEveryNsteps = 50, newtonOptions = optnewton, maxSteps = 400)

	br, _ = @time continuation(
		(x, p) -> Fbr(x, @set par_br.C = p),
		(x, p) -> Jbr(x, @set par_br.C = p),
		solc0, -0.2,
		optcont;
		plot = true, verbosity = 2,
		printSolution = (x, p) -> norm(x, Inf),
		plotSolution = (x; kwargs...) -> plot!(x[1:end÷2];label="",ylabel ="u", kwargs...))
####################################################################################################
ind_hopf = 1
	hopfpt = HopfPoint(br, ind_hopf)

# we get the parameters from the Hopf point
hopfpoint = HopfPoint(br, ind_hopf)
C_hopf = hopfpoint.p[1]
ωH = hopfpoint.p[end] |> abs
# number of time slices
M = 61

# we get the parameters from the Hopf point
orbitguess = zeros(2N, M)
vec_hopf = geteigenvector(optnewton.eigsolver,
			br.eig[br.bifpoint[ind_hopf].idx][2],
			br.bifpoint[ind_hopf].ind_bif-1)
vec_hopf ./=  norm(vec_hopf)
# vec_hopf is the eigenvector for the eigenvalues iω
phase = []
	for ii=1:M
		t = (ii-1)/(M-1)
		orbitguess[:, ii] .= real.(hopfpoint.u .+
				2.1 * vec_hopf * exp(-2pi * complex(0, 1) * (t - .2750)))
		push!(phase, dot(orbitguess[:, ii] - hopfpoint.u,real.(vec_hopf)))
	end
plot(phase)
orbitguess_f = vcat(vec(orbitguess), 2pi/ωH) |> vec

poTrap = p -> PeriodicOrbitTrapProblem(
	x ->  Fbr(x, @set par_br.C = p),
	x ->  Jbr(x, @set par_br.C = p),
	real.(vec_hopf),
	hopfpoint.u,
	M)

poTrap(-0.9)(orbitguess_f) |> plot
PALC.plotPeriodicPOTrap(orbitguess_f, N, M)

deflationOp = DeflationOperator(2.0, (x,y) -> dot(x[1:end-1], y[1:end-1]),1.0, [zero(orbitguess_f)])
# deflationOp = DeflationOperator(2.0, (x,y) -> dot(x[1:end-1], y[1:end-1]),1.0, [outpo_f])

opt_po = NewtonPar(tol = 1e-10, verbose = true, maxIter = 120)

outpo_f, _, flag = @time PALC.newton(
		poTrap(-0.9),
		orbitguess_f, opt_po, :FullLU;
		normN = norminf)
	flag && printstyled(color=:red, "--> T = ", outpo_f[end], ", amplitude = ", PALC.amplitude(outpo_f,2*N, M),"\n")
	PALC.plotPeriodicPOTrap(outpo_f, N, M)

eig = EigKrylovKit(tol= 1e-10, x₀ = rand(2N), verbose = 2, dim = 40)
# eig = EigArpack()
eig = DefaultEig()
optcontpo = ContinuationPar(dsmin = 0.001, dsmax = 0.015, ds= 0.01, pMin = -1.8, maxSteps = 140, newtonOptions = (@set opt_po.eigsolver = eig), nev = 25, precisionStability = 1e-7, detectBifurcation = 0)
	br_po, _ , _ = @time continuationPOTrap(poTrap,
		outpo_f, -0.87,
		optcontpo; verbosity = 2,
		plot = true,
		# callbackN = (x, f, J, res, iteration, options; kwargs...) -> (println("--> amplitude = ", amplitude(x, n, M));true),
		plotSolution = (x;kwargs...) ->  heatmap!(reshape(x[1:end-1], 2*N, M)'; ylabel="time", color=:viridis, kwargs...),
		printSolution = (u, p) -> PALC.maximumPOTrap(u, N, M; ratio = 2),
		normC = norminf)

# branches = [br_po]
push!(branches, br_po)
plotBranch(vcat(branches, br), label = "");title!("")
####################################################################################################
# Period doubling
ind_pd = 1
vec_pd = geteigenvector(eig,
		br_po.eig[br_po.bifpoint[ind_pd].idx][2],
		br_po.bifpoint[ind_pd].ind_bif)

# orbitguess_f = br_po.bifpoint[1].u .+ 0.1 * real.(vec_pd)
# orbitguess_f[end] *= 2
#
# outpo_pd, _, _ = @time PALC.newton(
# 		poTrap(br_po.bifpoint[1].param),
# 		orbitguess_f, opt_po, :FullLU;
# 		normN = norminf)
# 	printstyled(color=:red, "--> T = ", outpo_pd[end], ", amplitude = ", PALC.amplitude(outpo_pd, 2N, M),"\n")
# 	PALC.plotPeriodicPOTrap(outpo_pd, N, M)
####################################################################################################
# shooting
orbitsection = Array(sol[:,[end]])
# orbitsection = orbitguess[:, 1]

initpo = vcat(vec(orbitsection), 3.)

PALC.plotPeriodicShooting(initpo[1:end-1], 1);title!("")


probSh = p -> ShootingProblem(u -> Fbr(u, p), p, prob_sp, ETDRK2(krylov=true),
		1, x -> PALC.sectionShooting(x, Array(sol[:,[end]]), p, Fbr); atol = 1e-14, rtol = 1e-14, dt = 0.1)

probSh = p -> ShootingProblem(u -> Fbr(u, p), p, prob, Rodas4P(),
		1, x -> sectionArray(x, Array(sol[:,[end]]), p))#; atol = 1e-12, rtol = 1e-10)


ls = GMRESIterativeSolvers(tol = 1e-7, N = length(initpo), maxiter = 50, verbose = false)
	# ls = GMRESKrylovKit{Float64}(verbose = 0, dim = 200, atol = 1e-9, rtol = 1e-5)
	optn = NewtonPar(verbose = true, tol = 1e-9,  maxIter = 120, linsolver = ls)
	# deflationOp = PALC.DeflationOperator(2.0, (x,y) -> dot(x[1:end-1], y[1:end-1]),1.0, [outpo])
	outposh, _, flag = @time newton(probSh(@set par_br.C = -0.86),
		initpo, optn;
		callbackN = (x, f, J, res, iteration; kwargs...) -> (@show x[end];true),
		normN = norminf)
	flag && printstyled(color=:red, "--> T = ", outposh[end], ", amplitude = ", PALC.getAmplitude(probSh(@set par_br.C = -0.86), outposh; ratio = 2),"\n")

	plot(initpo[1:end-1], label = "Init guess")
	plot!(outpo[1:end-1], label = "sol")

eig = EigKrylovKit(tol= 1e-12, x₀ = rand(2N), verbose = 2, dim = 40)
eig = DefaultEig()
optcontpo = ContinuationPar(dsmin = 0.0001, dsmax = 0.02, ds= 0.005, pMin = -1.8, maxSteps = 70, newtonOptions = (@set optn.eigsolver = eig), nev = 5, precisionStability = 1e-3, detectBifurcation = 1)
	br_po_sh, _ , _ = @time continuationPOShooting(
		p -> probSh(@set par_br.C = p),
		outposh, -0.86,
		optcontpo; verbosity = 2,
		plot = true,
		plotSolution = (x; kwargs...) -> PALC.plotPeriodicShooting!(x[1:end-1], 1; kwargs...),
		printSolution = (u, p) -> PALC.getMaximum(probSh(@set par_br.C = p), u; ratio = 2), normC = norminf)

# branches = [br_po_sh]
# push!(branches, br_po_sh)
# plotBranch(branches);title!("")

plotBranch(vcat(br_po_sh, br_po, br), label = "");title!("")

####################################################################################################
# shooting PD
prob_sp = SplitODEProblem(f1, f2, solc0, (0.0, 300.0), @set par_br.C = -1.32)

solpd = @time solve(prob_sp, ETDRK2(krylov=true); abstol=1e-14, reltol=1e-14, dt = 0.1)
	# heatmap(sol.t, X, sol[N,:], color=:viridis, xlim=(20,280.0))

plot(solpd.t, solpd[N÷2, :], xlim=(290,296.2))

solpd.t[end-100]

orbitsectionpd = Array(solpd[:,end-100])
initpo_pd = vcat(vec(orbitsectionpd), 6.2)
PALC.plotPeriodicShooting(initpo_pd[1:end-1], 1);title!("")

ls = GMRESIterativeSolvers(tol = 1e-7, N = length(initpo), maxiter = 50, verbose = false)
	# ls = GMRESKrylovKit{Float64}(verbose = 0, dim = 200, atol = 1e-9, rtol = 1e-5)
	optn = NewtonPar(verbose = true, tol = 1e-9,  maxIter = 120, linsolver = ls)
	# deflationOp = PALC.DeflationOperator(2.0, (x,y) -> dot(x[1:end-1], y[1:end-1]),1.0, [outpo])
	outposh_pd, _, flag = @time newton(x -> probSh(@set par_br.C = -1.32)(x),
		x -> (dx -> probSh(@set par_br.C = -1.32)(x, dx)),
		initpo_pd, optn;
		callbackN = (x, f, J, res, iteration; kwargs...) -> (@show x[end];true),
		normN = norminf)
	flag && printstyled(color=:red, "--> T = ", outposh_pd[end], ", amplitude = ", PALC.getAmplitude(probSh(@set par_br.C = -0.86), outposh_pd; ratio = 2),"\n")

	plot(initpo[1:end-1], label = "Init guess")
	plot!(outposh_pd[1:end-1], label = "sol")

optcontpo = ContinuationPar(dsmin = 0.0001, dsmax = 0.005, ds= 0.001, pMin = -1.8, maxSteps = 100, newtonOptions = (@set optn.eigsolver = eig), nev = 5, precisionStability = 1e-3, detectBifurcation = 1)
	br_po_sh_pd, _ , _ = @time continuationPOShooting(
		p -> probSh(@set par_br.C = p),
		outposh_pd, -1.32,
		optcontpo; verbosity = 2,
		plot = true,
		# callbackN = cb_ss,
		plotSolution = (x; kwargs...) -> PALC.plotPeriodicShooting!(x[1:end-1], 1; kwargs...),
		printSolution = (u, p) -> PALC.getMaximum(probSh(@set par_br.C = p), u; ratio = 2), normC = norminf)

plotBranch(vcat(br_po_sh, br, br_po_sh_pd), label = "");title!("")
