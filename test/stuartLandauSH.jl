# using Revise, Plots
using OrdinaryDiffEq, ForwardDiff, Test
	using BifurcationKit, LinearAlgebra, Parameters, Setfield
	const BK = BifurcationKit
	const FD = ForwardDiff

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
dFsl(x, dx, p) = FD.derivative(t -> Fsl(x .+ t .* dx, p), 0.)

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
####################################################################################################
# continuation
optconteq = ContinuationPar(ds = -0.01, detectBifurcation = 3, pMin = -0.5, nInversion = 4)
br, = continuation(Fsl, u0, par_hopf, (@lens _.r), optconteq)
####################################################################################################
prob = ODEProblem(Fode, u0, (0., 100.), par_hopf)
probMono = ODEProblem(FslMono!, vcat(u0, u0), (0., 100.), par_hopf)
####################################################################################################
sol = solve(probMono, KenCarp4(), abstol =1e-9, reltol=1e-6)
sol = solve(prob, KenCarp4(), abstol =1e-9, reltol=1e-6)
# plot(sol[1,:], sol[2,:])
####################################################################################################
section(x, T) = x[1] #* x[end]
# standard simple shooting
M = 1
dM = 1
_pb = ShootingProblem(Fsl, par_hopf, prob, KenCarp4(), 1, section; abstol =1e-10, reltol=1e-9)

initpo = [0.13, 0., 6.]
res = _pb(initpo, par_hopf)

# test of the differential of the shooting method

_dx = rand(3)
resAD = FD.derivative(z -> _pb(initpo .+ z .* _dx, par_hopf), 0.)
resFD = (_pb(initpo .+ 1e-8 .* _dx, par_hopf) - _pb(initpo, par_hopf)) .* 1e8
resAN = _pb(initpo, par_hopf, _dx; δ = 1e-8)
@test norm(resAN - resFD, Inf) < 4e-6
@test norm(resAN - resAD, Inf) < 4e-6
####################################################################################################
# test shooting interface M = 1
_pb = ShootingProblem(Fsl, par_hopf, prob, KenCarp4(), [initpo[1:end-1]]; abstol =1e-10, reltol=1e-9)
res = _pb(initpo, par_hopf)
res = _pb(initpo, par_hopf, initpo)

# test the jacobian of the functional in the case M=1
_Jad = FD.jacobian( x -> _pb(x, par_hopf), initpo)
_Jana = _pb(Val(:JacobianMatrix), initpo, par_hopf)
@test norm(_Jad - _Jana, Inf) < 1e-7

_pb2 = ShootingProblem(Fsl, par_hopf, prob, Rodas4(), probMono, Rodas4(autodiff=false), [initpo[1:end-1]]; abstol =1e-10, reltol=1e-9)
res = _pb2(initpo, par_hopf)
res = _pb2(initpo, par_hopf, initpo)
BK.isSimple(_pb2)

# we test this using Newton - Continuation
ls = GMRESIterativeSolvers(tol = 1e-5, N = length(initpo))
optn = NewtonPar(verbose = false, tol = 1e-9,  maxIter = 20, linsolver = ls)
deflationOp = BK.DeflationOperator(2.0, (x,y) -> dot(x[1:end-1], y[1:end-1]), 1.0, [zeros(3)])
outpo, _, flag,_ = newton(_pb,
	initpo, par_hopf,
	optn,
	normN = norminf)
	@test flag

BK.getPeriod(_pb, outpo, par_hopf)
BK.getAmplitude(_pb, outpo, par_hopf)
BK.getMaximum(_pb, outpo, par_hopf)
BK.getTrajectory(_pb, outpo, par_hopf)

opts_po_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.01, ds= -0.01, pMax = 4.0, maxSteps = 30, detectBifurcation = 2, nev = 2, newtonOptions = @set optn.tol = 1e-7)
	br_pok2, upo , _= continuation(
		_pb,
		outpo, par_hopf, (@lens _.r),
		opts_po_cont;
		tangentAlgo = BorderedPred(),
		verbosity = 0,
		plot = false,
		# plotSolution = (x, p; kwargs...) -> plot!(x[1:end-1]; kwargs...),
		printSolution = (u, p) -> norm(u[1:2]),
		normC = norminf)
# plot(br_pok2)
####################################################################################################
# test automatic branch switching
JFsl(x, p) = FD.jacobian(t -> Fsl(t, p), x)
d1Fsl(x, p, dx) = FD.derivative(t -> Fsl(x .+ t .* dx, p), 0.)
d2Fsl(x, p, dx1, dx2) = FD.derivative(t -> d1Fsl(x .+ t .* dx2, p, dx1), 0.)
d3Fsl(x, p, dx1, dx2, dx3) = FD.derivative(t -> d2Fsl(x .+ t .* dx3, p, dx1, dx2), 0.)
jet = (Fsl, JFsl, d2Fsl, d3Fsl)

br_pok2, = continuation(jet..., br, 1, opts_po_cont, ShootingProblem(1, par_hopf, prob, KenCarp4());printSolution = (u, p) -> norm(u[1:2]), normC = norminf)

# test matrix-free computation of floquet coefficients
eil = EigKrylovKit(dim = 2, x₀=rand(2))
opts_po_contMF = @set opts_po_cont.newtonOptions.eigsolver = eil
opts_po_contMF = @set opts_po_cont.detectBifurcation=0
br_pok2, = continuation(jet...,br,1, opts_po_contMF, ShootingProblem(1, par_hopf, prob, Rodas4());printSolution = (u, p) -> norm(u[1:2]), normC = norminf, plot=false)
####################################################################################################
# test shooting interface M > 1
_pb = ShootingProblem(Fsl, par_hopf, prob, KenCarp4(), [initpo[1:end-1],initpo[1:end-1],initpo[1:end-1]]; abstol =1e-10, reltol=1e-9)
initpo = [0.13, 0, 0, 0.13, 0, 0.13 , 6.3]
res = _pb(initpo, par_hopf)
res = _pb(initpo, par_hopf, initpo)
# test the jacobian of the functional in the case M=1
_Jad = FD.jacobian( x -> _pb(x, par_hopf), initpo)
_Jana = _pb(Val(:JacobianMatrix), initpo, par_hopf)
@test norm(_Jad - _Jana, Inf) < 1e-7
####################################################################################################
# Single Poincaré Shooting with hyperplane parametrization
normals = [[-1., 0.]]
centers = [zeros(2)]

probPsh = PoincareShootingProblem(2, par_hopf,prob, Rodas4(), probMono, Rodas4(); abstol=1e-10, reltol=1e-9)
probPsh = PoincareShootingProblem(2, par_hopf,prob, Rodas4(); rtol = abstol=1e-10, reltol=1e-9)

probPsh = PoincareShootingProblem(Fsl, par_hopf,
		prob, Rodas4(),
		probMono, Rodas4(),
		normals, centers; rtol = abstol =1e-10, reltol=1e-9)

hyper = probPsh.section

initpo_bar = BK.R(hyper, [0, 0.4], 1)

BK.E(hyper, [1.0,], 1)
initpo_bar = [0.4]

probPsh(initpo_bar, par_hopf)

# test of the analytical formula for jacobian of the functional
_Jad = BifurcationKit.finiteDifferences( x-> probPsh(x, par_hopf), initpo_bar)
_Jana = probPsh(Val(:JacobianMatrix), initpo_bar, par_hopf)
@test norm(_Jad - _Jana, Inf) < 1e-4

ls = GMRESIterativeSolvers(tol = 1e-7, N = length(initpo_bar), maxiter = 500, verbose = false)
	eil = EigKrylovKit(dim = 1, x₀=rand(1))
	optn = NewtonPar(verbose = false, tol = 1e-8,  maxIter = 140, linsolver = ls, eigsolver = eil)
	deflationOp = BK.DeflationOperator(2.0, dot, 1.0, [zero(initpo_bar)])
	outpo, _, flag, = newton(probPsh,
			initpo_bar, par_hopf,
			optn; normN = norminf)
	@test flag

BK.getPeriod(probPsh, outpo, par_hopf)
BK.getAmplitude(probPsh, outpo, par_hopf)
BK.getMaximum(probPsh, outpo, par_hopf)
BK.getTrajectory(probPsh, outpo, par_hopf)

probPsh = PoincareShootingProblem(Fsl, par_hopf,
		prob, Rodas4(),
		# probMono, Rodas4(autodiff=false),
		normals, centers; rtol = 1e-8)

opts_po_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.015, ds= -0.01, pMax = 4.0, maxSteps = 50, newtonOptions = setproperties(optn;tol = 1e-9, eigsolver = eil), detectBifurcation = 1)
	br_pok2, = continuation(
		probPsh, outpo, par_hopf, (@lens _.r),
		opts_po_cont; verbosity = 0,
		tangentAlgo = BorderedPred(),
		plot = false,
		# plotSolution = (x, p;kwargs...) -> plot!(x; kwargs...),
		printSolution = (u, p) -> norm(u), normC = norminf)
# plot(br_pok2)
####################################################################################################
# normals = [[-1., 0.], [1, -1]]
# centers = [zeros(2), zeros(2)]
# initpo_bar = [1.04, -1.04/√2]

normals = [[-1., 0.], [1, 0]]
centers = [zeros(2), zeros(2)]
initpo_bar = [0.2, -0.2]

probPsh = BK.PoincareShootingProblem(Fsl, par_hopf, prob, KenCarp4(), normals, centers; abstol=1e-10, reltol=1e-9)
# version with analytical jacobian
probPsh2 = BK.PoincareShootingProblem(Fsl, par_hopf, prob, KenCarp4(), normals, centers; abstol=1e-10, reltol=1e-9, δ = 0)

hyper = probPsh.section

# test of the analytical formula for jacobian of the functional
_Jad = BifurcationKit.finiteDifferences( x-> probPsh(x, par_hopf), initpo_bar)
# _Jphifd = BifurcationKit.finiteDifferences(x->probPsh.flow(x, par_hopf, Inf64), [0,0.4]; δ=1e-8)
_Jana = probPsh(Val(:JacobianMatrix), initpo_bar, par_hopf)
@test norm(_Jad - _Jana, Inf) < 1e-5

ls = GMRESIterativeSolvers(tol = 1e-5, N = length(initpo_bar), maxiter = 500, verbose = false)
	eil = EigKrylovKit(dim = 1, x₀=rand(1))
	optn = NewtonPar(verbose = false, tol = 1e-9,  maxIter = 140, linsolver = ls, eigsolver = eil)
	deflationOp = BK.DeflationOperator(2.0, dot, 1.0, [zero(initpo_bar)])
	outpo, = newton(probPsh2, initpo_bar, par_hopf, optn; normN = norminf)
	outpo, = newton(probPsh, initpo_bar, par_hopf, optn; normN = norminf)

getPeriod(probPsh, outpo, par_hopf)

opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.025, ds= -0.01, pMax = 4.0, maxSteps = 50, newtonOptions = (@set optn.tol = 1e-9), detectBifurcation = 1, nev = 2)
	br_pok2, = continuation(probPsh, outpo, par_hopf, (@lens _.r),
		opts_po_cont; verbosity = 0,
		tangentAlgo = BorderedPred(),
		plot = false,
		# plotSolution = (x, p;kwargs...) -> plot!(x; kwargs...),
		printSolution = (u, p) -> norm(u), normC = norminf)

####################################################################################################
normals = [[-1., 0.], [1, 0], [0, 1]]
centers = [zeros(2), zeros(2), zeros(2)]
initpo = [[0., 0.4], [0, -.3], [0.3, 0]]

probPsh = PoincareShootingProblem(Fsl, par_hopf, prob, KenCarp4(), normals, centers; abstol=1e-10, reltol=1e-9)

hyper = probPsh.section
initpo_bar = reduce(vcat, [BK.R(hyper, initpo[ii], ii) for ii in eachindex(centers)])

probPsh(initpo_bar, par_hopf; verbose = true)

# test of the analytical formula for jacobian of the functional
_Jad = BifurcationKit.finiteDifferences( x-> probPsh(x, par_hopf), initpo_bar)
_Jana = probPsh(Val(:JacobianMatrix), initpo_bar, par_hopf)
@test norm(_Jad - _Jana, Inf) < 1e-5

ls = GMRESIterativeSolvers(tol = 1e-7, N = length(initpo_bar), maxiter = 10, verbose = false)
	optn = NewtonPar(verbose = false, tol = 1e-9,  maxIter = 50, linsolver = ls, eigsolver = eil)
	outpo, = newton(probPsh, initpo_bar, par_hopf, optn; normN = norminf)

for ii=1:length(normals)
	@show BK.E(hyper, [outpo[ii]], ii)
end

getPeriod(probPsh, outpo, par_hopf)

opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.025, ds= -0.005, pMax = 4.0, maxSteps = 10, newtonOptions = setproperties(optn; tol = 1e-8), detectBifurcation = 1)
	br_hpsh, upo , _= continuation(probPsh, outpo, par_hopf, (@lens _.r),
		opts_po_cont; normC = norminf)
# plot(br_hpsh)
####################################################################################################
# test automatic branch switching with most possible options
# calls with analytical jacobians
br_psh, = continuation(jet..., br,1, (@set opts_po_cont.ds = 0.005), PoincareShootingProblem(1, par_hopf, prob, KenCarp4(), probMono, KenCarp4(); abstol=1e-10, reltol=1e-9); normC = norminf)


opts_po_cont = @set opts_po_cont.detectBifurcation = 0
for M in [1,2], linearPO in [:autodiffMF, :MatrixFree, :autodiffDense, :FiniteDifferencesDense]
	@show M, linearPO
	br_psh, = continuation(jet..., br, 1, (@set opts_po_cont.ds = 0.005), PoincareShootingProblem(M, par_hopf, prob, Rodas4P(); abstol=1e-10, reltol=1e-9, parallel = true); normC = norminf, updateSectionEveryStep = 2, linearPO = linearPO == :autodiffMF ? :FiniteDifferencesDense : linearPO, verbosity = 0)

	br_ssh, = continuation(jet..., br, 1, (@set opts_po_cont.ds = 0.005), ShootingProblem(M, par_hopf, prob, Rodas4P(); abstol=1e-10, reltol=1e-9, parallel = true); normC = norminf, updateSectionEveryStep = 2, linearPO = linearPO, verbosity = 0)
end

br_psh, = continuation(jet..., br,1, (@set opts_po_cont.ds = 0.005), PoincareShootingProblem(2, par_hopf, prob, KenCarp4(), probMono, KenCarp4(); abstol=1e-10, reltol=1e-9); normC = norminf)

# plot(br_hpsh, vars = (:step, :p))
# plot(br_hpsh)
