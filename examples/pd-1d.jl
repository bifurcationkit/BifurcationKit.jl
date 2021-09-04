# example taken from Aragón, J. L., R. A. Barrio, T. E. Woolley, R. E. Baker, and P. K. Maini. “Nonlinear Effects on Turing Patterns: Time Oscillations and Chaos.” Physical Review E 86, no. 2 (August 8, 2012): 026201. https://doi.org/10.1103/PhysRevE.86.026201.
using Revise
	using DiffEqOperators, ForwardDiff, DifferentialEquations, Sundials
	using BifurcationKit, LinearAlgebra, Plots, SparseArrays, Parameters, Setfield
	const BK = BifurcationKit

norminf(x) = norm(x, Inf)
f(u, v, p) = p.η * (      u + p.a * v - p.C * u * v - u * v^2)
g(u, v, p) = p.η * (p.H * u + p.b * v + p.C * u * v + u * v^2)

function Laplacian(N, lx, bc = :Dirichlet)
	hx = 2lx/N
	D2x = CenteredDifference(2, 2, hx, N)
	if bc == :Neumann
		Qx = Neumann0BC(hx)
	elseif bc == :Dirichlet
		Qx = Dirichlet0BC(typeof(hx))
	elseif bc == :Periodic
		Qx = PeriodicBC(typeof(hx))
	end
	D2xsp = sparse(D2x * Qx)[1] |> sparse
end

@views function NL!(dest, u, p, t = 0.)
	N = div(length(u), 2)
	u1 =  u[1:N]
	u2 =  u[N+1:end]

	dest[1:N]     .= f.(u1, u2, Ref(p))
	dest[N+1:end] .= g.(u1, u2, Ref(p))

	return dest
end

function Fbr!(f, u, p, t = 0.)
	mul!(f, p.Δ, u)
	f .= f .+ NL(u, p)
end

NL(u, p) = NL!(similar(u), u, p)
Fbr(x, p, t = 0.) = Fbr!(similar(x), x, p)
dNL(x, p, dx) = ForwardDiff.derivative(t -> NL(x .+ t .* dx, p), 0.)

function dFbr(x, p, dx)
	f = similar(dx)
	mul!(f, p.Δ, dx)

	nl = dNL(x, p, dx)
	f .= f .+ nl
end

Jbr(x, p) = sparse(ForwardDiff.jacobian(x -> Fbr(x, p), x))
jet = BK.getJet(Fbr, Jbr)

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

out, = @time newton(Fbr, Jbr, solc0, par_br, optnewton, normN = norminf)
	plot();plot!(X,out[1:N]);plot!(X,solc0[1:N], label = "sol0",line=:dash)


optcont = ContinuationPar(dsmax = 0.051, ds = -0.001, pMin = -1.8, detectBifurcation = 3, nev = 21, plotEveryStep = 50, newtonOptions = optnewton, maxSteps = 370, nInversion = 10, maxBisectionSteps = 25)

	br, = @time continuation(Fbr, Jbr, solc0, (@set par_br.C = -0.2), (@lens _.C), optcont;
		plot = true, verbosity = 3,
		recordFromSolution = (x, p) -> norm(x, Inf),
		plotSolution = (x, p; kwargs...) -> plot!(x[1:end÷2];label="",ylabel ="u", kwargs...))

plot(br)
####################################################################################################
# branching from Hopf bp using aBS
opt_po = NewtonPar(tol = 1e-10, verbose = true, maxIter = 120)

eig = EigKrylovKit(tol= 1e-10, x₀ = rand(2N), verbose = 2, dim = 40)
eig = DefaultEig()
optcontpo = ContinuationPar(dsmin = 0.001, dsmax = 0.015, ds= 0.01, pMin = -1.8, maxSteps = 60, newtonOptions = (@set opt_po.eigsolver = eig), nev = 25, precisionStability = 1e-7, detectBifurcation = 3, dsminBisection = 1e-6)
M = 100
	probFD = PeriodicOrbitTrapProblem(M = M)
	br_po, = @time continuation(
		# arguments for branch switching from the first
		# Hopf bifurcation point
		jet..., br, 1,
		# arguments for continuation
		optcontpo, probFD;
		# OPTIONAL parameters
		# we want to jump on the new branch at phopf + δp
		# ampfactor is a factor to increase the amplitude of the guess
		verbosity = 2,
		plot = true,
		linearPO = :FullSparseInplace,
		# callbackN = (x, f, J, res, iteration, options; kwargs...) -> (println("--> amplitude = ", amplitude(x, n, M));true),
		plotSolution = (x, p;kwargs...) ->  (heatmap!(reshape(x[1:end-1], 2*N, M)'; ylabel="time", color=:viridis, kwargs...);plot!(br, subplot=1)),
		recordFromSolution = (u, p) -> maximum(u[1:end-1]),#BK.maximumPOTrap(u, N, M; ratio = 2),
		normC = norminf)

plot(br, br_po, label = "")
####################################################################################################
# branching from Hopf PD using aBS

br_po_pd2, = @time BK.continuationPOTrapFromPD(
		# arguments for branch switching from the first
		# Hopf bifurcation point
		br_po_pd, 2, setproperties(br_po.contparams; detectBifurcation = 3, plotEveryStep = 1, ds = 0.01);
		# OPTIONAL parameters
		# we want to jump on the new branch at phopf + δp
		# ampfactor is a factor to increase the amplitude of the guess
		ampfactor = 0.1, δp = -0.01,
		verbosity = 3,
		usedeflation = false,
		plot = true,
		# linearPO = :FullSparseInplace,
		linearPO = :BorderedSparseInplace,
		callbackN = (x, f, J, res, iteration, itl, options; kwargs...) -> (println("--> amplitude = ", x[end]);true),
		plotSolution = (x, p;kwargs...) ->  (heatmap!(reshape(x[1:end-1], 2*N, M)'; ylabel="time", color=:viridis, kwargs...);plot!(br_po, br_po_pd, subplot=1)),
		recordFromSolution = (u, p) -> maximum(u[1:end-1]),#BK.maximumPOTrap(u, N, M; ratio = 2),
		normC = norminf)

plot(br, br_po, br_po_pd, br_po_pd2, label = "")
####################################################################################################
# Period doubling
ind_pd = 1
vec_pd = geteigenvector(eig,
		br_po.eig[br_po.specialpoint[ind_pd].idx][2],
		br_po.specialpoint[ind_pd].ind_ev)

orbitguess_f = br_po.specialpoint[1].x .+ 0.1 * real.(vec_pd)
orbitguess_f[end] *= 2
#
# outpo_pd, _, _ = @time BK.newton(
# 		poTrap(br_po.bifpoint[1].param),
# 		orbitguess_f, opt_po, :FullLU;
# 		normN = norminf)
# 	printstyled(color=:red, "--> T = ", outpo_pd[end], ", amplitude = ", BK.amplitude(outpo_pd, 2N, M),"\n")
# 	BK.plotPeriodicPOTrap(outpo_pd, N, M)
####################################################################################################
# shooting
par_br_hopf = @set par_br.C = -0.86
f1 = DiffEqArrayOperator(par_br.Δ)
f2 = NL!
prob_sp = SplitODEProblem(f1, f2, solc0, (0.0, 280.0), par_br_hopf)

sol = @time solve(prob_sp, ETDRK2(krylov=true); abstol=1e-14, reltol=1e-14, dt = 0.1)
orbitsection = Array(sol[:,[end]])
# orbitsection = orbitguess[:, 1]

initpo = vcat(vec(orbitsection), 3.)

BK.plotPeriodicShooting(initpo[1:end-1], 1);title!("")


probSh = ShootingProblem(Fbr, par_br_hopf, prob_sp, ETDRK2(krylov=true),
		[sol(280.0)]; abstol=1e-14, reltol=1e-14, dt = 0.1, parallel = true)

probSh(initpo, par_br_hopf)

ls = GMRESIterativeSolvers(reltol = 1e-7, N = length(initpo), maxiter = 50, verbose = false)
	# ls = GMRESKrylovKit(verbose = 0, dim = 200, atol = 1e-9, rtol = 1e-5)
	optn = NewtonPar(verbose = true, tol = 1e-9,  maxIter = 20, linsolver = ls)
	# deflationOp = BK.DeflationOperator(2 (x,y) -> dot(x[1:end-1], y[1:end-1]),1.0, [outpo])
	outposh, _, flag = @time newton(probSh, initpo, par_br_hopf, optn;
		callbackN = (x, f, J, res, iteration; kw...) -> (@show x[end];true),
		normN = norminf)
	flag && printstyled(color=:red, "--> T = ", outposh[end], ", amplitude = ", BK.getAmplitude(probSh, outposh, par_br_hopf; ratio = 2),"\n")

plot(initpo[1:end-1], label = "Init guess")
	plot!(outposh[1:end-1], label = "sol")

eig = EigKrylovKit(tol= 1e-12, x₀ = rand(2N), verbose = 2, dim = 40)
# eig = DefaultEig()
optcontpo = ContinuationPar(dsmin = 0.0001, dsmax = 0.01, ds= -0.005, pMin = -1.8, maxSteps = 50, newtonOptions = (@set optn.eigsolver = eig), nev = 10, precisionStability = 1e-2, detectBifurcation = 3)
	br_po_sh, = @time continuation(probSh, outposh, par_br_hopf, (@lens _.C), optcontpo;
		verbosity = 3,	plot = true,
		linearAlgo = MatrixFreeBLS(@set ls.N = probSh.M*n+2),
		finaliseSolution = (z, tau, step, contResult; kw...) ->
			(BK.haseigenvalues(contResult) && Base.display(contResult.eig[end].eigenvals) ;true),
		plotSolution = (x, p; kwargs...) -> BK.plotPeriodicShooting!(x[1:end-1], 1; kwargs...),
		recordFromSolution = (u, p) -> BK.getMaximum(probSh, u, (@set par_br_hopf.C = p); ratio = 2), normC = norminf)

# branches = [br_po_sh]
# push!(branches, br_po_sh)
# plot(branches...)

plot(br_po_sh, br, label = "")

####################################################################################################
# shooting Period Doubling
par_br_pd = @set par_br.C = -1.32
f1 = DiffEqArrayOperator(par_br.Δ)
f2 = NL!
prob_sp = SplitODEProblem(f1, f2, solc0, (0.0, 300.0), par_br_pd; abstol=1e-14, reltol=1e-14, dt = 0.1)
# solution close to the PD point.

solpd = @time solve(prob_sp, ETDRK2(krylov=true))
	# heatmap(sol.t, X, sol[1:N,:], color=:viridis, xlim=(20,280.0))

plot(solpd.t, solpd[N÷2, :], xlim=(290,296.2))

solpd.t[end-100]


orbitsectionpd = Array(solpd[:,end-100])
initpo_pd = vcat(vec(orbitsectionpd), 6.2)
BK.plotPeriodicShooting(initpo_pd[1:end-1], 1);title!("")

ls = GMRESIterativeSolvers(reltol = 1e-7, N = length(initpo_pd), maxiter = 50, verbose = false)
	# ls = GMRESKrylovKit(verbose = 0, dim = 200, atol = 1e-9, rtol = 1e-5)
	optn = NewtonPar(verbose = true, tol = 1e-9,  maxIter = 120, linsolver = ls)
	# deflationOp = BK.DeflationOperator(2 (x,y) -> dot(x[1:end-1], y[1:end-1]),1.0, [outpo])
	outposh_pd, _, flag = @time newton(probSh, initpo_pd, par_br_pd, optn;
		callback = (x, f, J, res, iteration, itlinear, options; kwargs...) -> (@show x[end];true),
		normN = norminf)
	flag && printstyled(color=:red, "--> T = ", outposh_pd[end], ", amplitude = ", BK.getAmplitude(probSh, outposh_pd, (@set par_br.C = -0.86); ratio = 2),"\n")

	plot(initpo[1:end-1], label = "Init guess")
	plot!(outposh_pd[1:end-1], label = "sol")

optcontpo = ContinuationPar(dsmin = 0.0001, dsmax = 0.005, ds= -0.001, pMin = -1.8, maxSteps = 500, newtonOptions = (@set optn.eigsolver = eig), nev = 10, precisionStability = 1e-2, detectBifurcation = 0)
	br_po_sh_pd, = @time continuation(
		probSh, outposh_pd, par_br_pd, (@lens _.C),
		optcontpo; verbosity = 3,
		plot = true,
		linearAlgo = MatrixFreeBLS(@set ls.N = probSh.M*n+2),
		finaliseSolution = (z, tau, step, contResult; k...) ->
			(Base.display(contResult.eig[end].eigenvals) ;println("--> T = ", z.u[end]);true),
		plotSolution = (x, p; kwargs...) -> (BK.plotPeriodicShooting!(x[1:end-1], 1; kwargs...); plot!(br_po_sh; subplot=1, legend=false)),
		recordFromSolution = (u, p; k...) -> BK.getMaximum(probSh, u, (@set par_br_pd.C = p); ratio = 2), normC = norminf)

plot(br_po_sh_pd, br, label = "");title!("")
