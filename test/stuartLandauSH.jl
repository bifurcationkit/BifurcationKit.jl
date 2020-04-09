# using Revise, Plots
using OrdinaryDiffEq, ForwardDiff, Test
	using PseudoArcLengthContinuation, LinearAlgebra, Parameters, Setfield
	const PALC = PseudoArcLengthContinuation

norminf = x -> norm(x, Inf)

function Fsl!(f, u, p, t)
	@unpack r, μ, ν, c3, c5 = p
	u1 = u[1]
	u2 = u[2]

	ua = u1^2 + u2^2

	f[1] = r * u1 - ν * u2 - ua * (c3 * u1 - μ * u2) - c5 * ua^2 * u1
	f[2] = r * u2 + ν * u1 - ua * (c3 * u2 + μ * u1) - c5 * ua^2 * u2

	return f
end

Fsl(x, p) = Fsl!(similar(x), x, p, 0.)

Fode(f, x, p, t) = Fsl!(f, x, p, t)
dFsl(x, dx, p) = ForwardDiff.derivative(t -> Fsl(x .+ t .* dx, p), 0.)
# JacOde(J, x, p, t) = copyto!(J, Jsl(x, p))

par_sl = (r = 0.5, μ = 0., ν = 1.0, c3 = 1.0, c5 = 0.0,)
u0 = [.001, .001]

function FslMono!(f, x, p, t)
	u = x[1:2]
	du = x[3:4]
	Fsl!(f[1:2], u, p, t)
	f[3:4] .= dFsl(u, du, p)
end

####################################################################################################
par_hopf = (@set par_sl.r = 0.1)

prob = ODEProblem(Fode, u0, (0., 100.), par_hopf)
probMono = ODEProblem(FslMono!, vcat(u0, u0), (0., 100.), par_hopf)
####################################################################################################
sol = @time solve(probMono, Tsit5(), abstol =1e-9, reltol=1e-6)
sol = @time solve(prob, Tsit5(), abstol =1e-9, reltol=1e-6)
# plot(sol[1,:], sol[2,:])
####################################################################################################
section(x) = x[1]
# standard simple shooting
M = 1
dM = 1
probSh = p -> PALC.ShootingProblem(u -> Fsl(u, p), p, prob, Rodas4(),
		1, section; rtol = 1e-9)

initpo = [0.3, 0., 6.]
res = @time probSh(par_hopf)(initpo)

# test of the differential of thew shooting method
_pb = probSh(par_hopf)

_dx = rand(3)
resAD = ForwardDiff.derivative(z -> _pb(initpo .+ z .* _dx), 0.)
resFD = (_pb(initpo .+ 1e-8 .* _dx) - _pb(initpo)) * 1e8
resAN = _pb(initpo, _dx; δ = 1e-8)
@test norm(resAN - resFD, Inf) < 1e-1
@test norm(resAN - resAD, Inf) < 1e-1


####################################################################################################
# we test this using Newton - Continuation

ls = GMRESIterativeSolvers(tol = 1e-5, N = length(initpo))
optn = NewtonPar(verbose = false, tol = 1e-9,  maxIter = 20, linsolver = ls)
deflationOp = PALC.DeflationOperator(2.0, (x,y) -> dot(x[1:end-1], y[1:end-1]),1.0, [zeros(3)])
outpo, _ = @time PALC.newton(probSh(par_hopf),
	initpo,
	optn,
	normN = norminf)

getPeriod(probSh(par_hopf), outpo)

opts_po_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.01, ds= -0.01, pMax = 4.0, maxSteps = 30, newtonOptions = @set optn.tol = 1e-7)#
	br_pok2, upo , _= @time PALC.continuationPOShooting(
		p -> probSh(@set par_hopf.r = p),
		outpo, par_hopf.r,
		opts_po_cont;
		tangentAlgo = BorderedPred(),
		verbosity = 0,
		plot = false,
		# plotSolution = (x, p; kwargs...) -> plot!(x[1:end-1]; kwargs...),
		printSolution = (u, p) -> norm(u[1:2]), normC = norminf)
####################################################################################################
# Single Poincaré Shooting with hyperplane parametrization
normals = [[-1., 0.]]
centers = [zeros(2)]

probPsh = p -> PALC.PoincareShootingProblem(u -> Fsl(u, p), p,
		prob, Rodas4(),
		probMono, Rodas4(),
		normals, centers; rtol = 1e-8)

hyper = probPsh(par_hopf).section

initpo_bar = PALC.R(hyper, [0,0.4], 1)

PALC.E(hyper, [1.0], 1)

initpo_bar = [0.4]

probPsh(par_hopf)(initpo_bar)

ls = GMRESIterativeSolvers(tol = 1e-7, N = length(initpo_bar), maxiter = 500, verbose = false)
	eil = DefaultEig()
	optn = NewtonPar(verbose = false, tol = 1e-8,  maxIter = 140, linsolver = ls, eigsolver = eil)
	deflationOp = PALC.DeflationOperator(2.0, (x,y) -> dot(x, y), 1.0, [zero(initpo_bar)])
	outpo, _ = @time PALC.newton(probPsh(par_hopf),
			# x -> (dx -> probPsh(par_hopf)(x, dx)),
			initpo_bar,
			optn; normN = norminf)
	println("--> Point on the orbit = ", PALC.E(hyper, outpo, 1))

getPeriod(probPsh(par_hopf), outpo)

opts_po_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.015, ds= -0.01, pMax = 4.0, maxSteps = 50, newtonOptions = (@set optn.tol = 1e-9), detectBifurcation = 0)
	br_pok2, upo , _= @time PALC.continuationPOShooting(
		p -> probPsh(@set par_hopf.r = p),
		outpo, par_hopf.r,
		opts_po_cont; verbosity = 0,
		tangentAlgo = BorderedPred(),
		plot = false,
		# plotSolution = (x, p;kwargs...) -> plot!(x; kwargs...),
		printSolution = (u, p) -> norm(u), normC = norminf)
####################################################################################################
# normals = [[-1., 0.], [1, -1]]
# centers = [zeros(2), zeros(2)]
# initpo_bar = [1.04, -1.04/√2]

normals = [[-1., 0.], [1, 0]]
centers = [zeros(2), zeros(2)]
initpo_bar = [0.2, -0.2]

probPsh = p -> PALC.PoincareShootingProblem(u -> Fsl(u, p, 0.), p, prob, Tsit5(), normals, centers; rtol = 1e-6)

hyper = probPsh(par_hopf).section

probPsh(par_hopf)(initpo_bar, verbose = true)

ls = GMRESIterativeSolvers(tol = 1e-5, N = length(initpo_bar), maxiter = 500, verbose = false)
	eil = EigArpack(v0 = rand(ls.N))
	optn = NewtonPar(verbose = false, tol = 1e-9,  maxIter = 140, linsolver = ls)
	deflationOp = PALC.DeflationOperator(2.0, (x,y) -> dot(x, y), 1.0, [zero(initpo_bar)])
	outpo, _ = @time PALC.newton(probPsh(par_hopf),
			initpo_bar, optn; normN = norminf)
println("--> Point on the orbit = ", PALC.E(hyper, [outpo[1]], 1), PALC.E(hyper, [outpo[2]], 2))

getPeriod(probPsh(par_hopf), outpo)

opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.025, ds= -0.01, pMax = 4.0, maxSteps = 50, newtonOptions = (@set optn.tol = 1e-9), detectBifurcation = 0)
	br_pok2, upo , _= @time PALC.continuationPOShooting(
		p -> probPsh(@set par_hopf.r = p),
		outpo, par_hopf.r,
		opts_po_cont; verbosity = 0,
		tangentAlgo = BorderedPred(),
		plot = false,
		# plotSolution = (x, p;kwargs...) -> plot!(x; kwargs...),
		printSolution = (u, p) -> norm(u), normC = norminf)

####################################################################################################
normals = [[-1., 0.], [1, 0], [0, 1]]
centers = [zeros(2), zeros(2), zeros(2)]
initpo = [[0., 0.4], [0, -.3], [0.3, 0]]

probPsh = p -> PALC.PoincareShootingProblem(u -> Fsl(u, p, 0.), p, prob, Tsit5(), normals, centers; rtol = 1e-6)

hyper = probPsh(par_hopf).section
initpo_bar = reduce(vcat, [PALC.R(hyper, initpo[ii], ii) for ii in eachindex(centers)])

probPsh(par_hopf)(initpo_bar, verbose = true)

ls = GMRESIterativeSolvers(tol = 1e-7, N = length(initpo_bar), maxiter = 10, verbose = false)
	optn = NewtonPar(verbose = false, tol = 1e-9,  maxIter = 50, linsolver = ls)
	outpo, _ = @time PALC.newton(probPsh(par_hopf),
		initpo_bar, optn; normN = norminf)

for ii=1:length(normals)
	@show PALC.E(hyper, [outpo[ii]], ii)
end

getPeriod(probPsh(par_hopf), outpo)

opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.025, ds= -0.005, pMax = 4.0, maxSteps = 50, newtonOptions = setproperties(optn; tol = 1e-9), detectBifurcation = 0)
	br_hpsh, upo , _= @time PALC.continuationPOShooting(
		p -> probPsh(@set par_hopf.r = p),
		outpo, par_hopf.r,
		opts_po_cont;
		verbosity = 0, plot = false,
		# plotSolution = (x, p;kwargs...) -> plot!(x, subplot=3),
		printSolution = (u, p) -> norm(u), normC = norminf)
