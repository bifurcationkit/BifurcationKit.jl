# using Revise, Plots
using OrdinaryDiffEq, ForwardDiff, Test
	using BifurcationKit, LinearAlgebra, Parameters, Setfield
	const BK = BifurcationKit
	const FD = ForwardDiff

norminf(x) = norm(x, Inf)

function Fsl!(f, u, p, t = 0)
	@unpack r, μ, ν, c3, c5 = p
	u1 = u[1]
	u2 = u[2]

	ua = u1^2 + u2^2

	f[1] = r * u1 - ν * u2 - ua * (c3 * u1 - μ * u2) - c5 * ua^2 * u1
	f[2] = r * u2 + ν * u1 - ua * (c3 * u2 + μ * u1) - c5 * ua^2 * u2
	return f
end

Fsl(x, p) = Fsl!(similar(x), x, p)
dFsl(x, dx, p) = FD.derivative(t -> Fsl(x .+ t .* dx, p), 0.)

par_sl = (r = 0.5, μ = 0., ν = 1.0, c3 = 1.0, c5 = 0.0,)
par_hopf = (@set par_sl.r = 0.1)
u0 = [.001, .001]

prob_vf = BifurcationKit.BifurcationProblem(Fsl, u0, par_hopf, (@lens _.r))

function FslMono!(f, x, p, t)
	u = x[1:2]
	du = x[3:4]
	Fsl!(f[1:2], u, p, t)
	f[3:4] .= dFsl(u, du, p)
end
####################################################################################################
# continuation
optconteq = ContinuationPar(ds = -0.01, detectBifurcation = 3, pMin = -0.5, nInversion = 4)
br = continuation(prob_vf, PALC(), optconteq)
####################################################################################################
prob = ODEProblem(Fsl!, u0, (0., 100.), par_hopf)
probMono = ODEProblem(FslMono!, vcat(u0, u0), (0., 100.), par_hopf)
####################################################################################################
sol = solve(probMono, KenCarp4(), abstol=1e-9, reltol=1e-6)
sol = solve(prob, KenCarp4(), abstol=1e-9, reltol=1e-6)
# plot(sol[1,:], sol[2,:])

# test generation of initial guess from ODESolution
generateCIProblem(PeriodicOrbitTrapProblem(M = 10), prob_vf, sol, 1.)
generateCIProblem(PeriodicOrbitOCollProblem(10, 2), prob_vf, sol, 1.)
generateCIProblem(ShootingProblem(M=10), prob_vf, prob, sol, 1.)
generateCIProblem(PoincareShootingProblem(M=10), prob_vf, prob, sol, 1.)
####################################################################################################
section(x, T) = x[1] #* x[end]
section(x, T, dx, dT) = dx[1] #* x[end]
# standard simple shooting
M = 1
dM = 1
_pb = ShootingProblem(prob, KenCarp4(), 1, section; abstol =1e-10, reltol=1e-9)

initpo = [0.13, 0., 6.]
res = _pb(initpo, par_hopf)

# test of the differential of the shooting method

_dx = rand(3)
resAD = FD.derivative(z -> _pb(initpo .+ z .* _dx, par_hopf), 0.)
resFD = (_pb(initpo .+ 1e-8 .* _dx, par_hopf) - _pb(initpo, par_hopf)) .* 1e8
resAN = _pb(initpo, par_hopf, _dx; δ = 1e-8)
@test norm(resAN - resFD, Inf) < 5e-6
@test norm(resAN - resAD, Inf) < 5e-6
####################################################################################################
# test shooting interface M = 1
_pb = ShootingProblem(prob, Rodas4(), [initpo[1:end-1]]; abstol =1e-10, reltol=1e-9, lens = (@lens _.r))
res = _pb(initpo, par_hopf)
res = _pb(initpo, par_hopf, initpo)
@test _pb.flow.prob.p == _pb.par

# test the jacobian of the functional in the case M=1
_Jad = FD.jacobian( x -> _pb(x, par_hopf), initpo)
_Jana = _pb(Val(:JacobianMatrix), initpo, par_hopf)
@test norm(_Jad - _Jana, Inf) < 1e-7

_pb2 = ShootingProblem(prob, Rodas4(), probMono, Rodas4(autodiff=false), [initpo[1:end-1]]; abstol = 1e-10, reltol = 1e-9)
res = _pb2(initpo, par_hopf)
res = _pb2(initpo, par_hopf, initpo)
@test BK.isSimple(_pb2)
@test _pb2.flow.prob.p == _pb2.par

# we test this using Newton - Continuation
optn = NewtonPar(verbose = false, tol = 1e-9,  maxIter = 20)
# deflationOp = BK.DeflationOperator(2, (x,y) -> dot(x[1:end-1], y[1:end-1]), 1.0, [zeros(3)])
outpo = newton(_pb, initpo, optn; normN = norminf)
@test BK.converged(outpo)
@test outpo.prob.prob.jacobian == :autodiffDense

BK.getPeriod(_pb, outpo.u, par_hopf)
BK.getAmplitude(_pb, outpo.u, par_hopf)
BK.getMaximum(_pb, outpo.u, par_hopf)
BK.getPeriodicOrbit(_pb, outpo.u, par_hopf)

opts_po_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.01, ds= -0.01, pMax = 4.0, maxSteps = 30, detectBifurcation = 2, nev = 2, newtonOptions = (@set optn.tol = 1e-7), tolStability = 1e-5)
	br_pok2 = continuation(_pb, outpo.u, PALC(tangent = Bordered()),
		opts_po_cont;
		verbosity = 0, plot = false,
		normC = norminf)
@test br_pok2.prob isa BK.WrapPOSh
@test br_pok2.prob.prob.jacobian == :autodiffDense
@test br_pok2.period[1] ≈ 2pi rtol = 1e-7
# plot(br_pok2)
####################################################################################################
# test automatic branch switching
br_pok2 = continuation(br, 1, opts_po_cont, ShootingProblem(1, prob, KenCarp4();  abstol = 1e-10, reltol = 1e-9, lens = (@lens _.r)); normC = norminf, verbosity = 0)

@test br_pok2.prob.prob.jacobian == :autodiffDense
@test br_pok2.prob isa BK.WrapPOSh
@test br_pok2.period[1] ≈ 2pi rtol = 1e-7

# idem with deflation
br_pok2 = continuation(br, 1, opts_po_cont, ShootingProblem(1, prob, KenCarp4();  abstol = 1e-10, reltol = 1e-9, lens = (@lens _.r)); normC = norminf, usedeflation = true)
@test br_pok2.prob.prob.jacobian == :autodiffDense
@test br_pok2.prob isa BK.WrapPOSh
@test br_pok2.period[1] ≈ 2pi rtol = 1e-7

# test matrix-free computation of floquet coefficients
eil = EigKrylovKit(dim = 2, x₀=rand(2))
opts_po_contMF = @set opts_po_cont.newtonOptions.eigsolver = eil
opts_po_contMF = @set opts_po_cont.detectBifurcation = 0
br_pok2 = continuation(br,1, opts_po_contMF, ShootingProblem(1, prob, Rodas4(); abstol = 1e-10, reltol = 1e-9, lens = (@lens _.r)); normC = norminf)
@test br_pok2.prob.prob.jacobian == :autodiffDense
@test br_pok2.prob isa BK.WrapPOSh
@test br_pok2.period[1] ≈ 2pi rtol = 1e-7

# case with 2 sections
br_pok2_s2 = continuation(br, 1, (@set opts_po_cont.newtonOptions.verbose = false), ShootingProblem(2, prob, KenCarp4();  abstol = 1e-10, reltol = 1e-9, lens = (@lens _.r)); normC = norminf)
@test br_pok2_s2.prob.prob.jacobian == :autodiffDense
@test br_pok2_s2.prob isa BK.WrapPOSh
@test br_pok2_s2.period[1] ≈ 2pi rtol = 1e-7
####################################################################################################
# test shooting interface M > 1
_pb = ShootingProblem(prob, KenCarp4(), [initpo[1:end-1],initpo[1:end-1],initpo[1:end-1]]; abstol =1e-10, reltol=1e-9)
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

probPsh = PoincareShootingProblem(2, prob, Rodas4(), probMono, Rodas4(); abstol=1e-10, reltol=1e-9, jacobian = :autodiffDenseAnalytical)
@test probPsh.par == probPsh.flow.prob1.p

probPsh = PoincareShootingProblem(2, prob, Rodas4(); rtol = abstol=1e-10, reltol=1e-9, jacobian = :autodiffDenseAnalytical)
@test probPsh.par == probPsh.flow.prob.p

probPsh = PoincareShootingProblem(prob, Rodas4(),
		probMono, Rodas4(),
		normals, centers; abstol = 1e-10, reltol = 1e-9,
		jacobian = :autodiffDenseAnalytical)

initpo_bar = BK.R(probPsh, [0, 0.4], 1)

BK.E(probPsh, [1.0,], 1)
initpo_bar = [0.4]

probPsh(initpo_bar, par_hopf)

# test of the analytical formula for jacobian of the functional
_Jad = BifurcationKit.finiteDifferences( x -> probPsh(x, par_hopf), initpo_bar)
_Jana = probPsh(Val(:JacobianMatrix), initpo_bar, par_hopf)
@test norm(_Jad - _Jana, Inf) < 1e-4

ls = DefaultLS()
	eil = EigKrylovKit(dim = 1, x₀ = rand(1))
	optn = NewtonPar(verbose = false, tol = 1e-8,  maxIter = 140, linsolver = ls, eigsolver = eil)
	deflationOp = BK.DeflationOperator(2, dot, 1.0, [zero(initpo_bar)])
	outpo = newton(probPsh, initpo_bar, optn; normN = norminf)
	@test BK.converged(outpo)

BK.getPeriod(probPsh, outpo.u, par_hopf)
BK.getAmplitude(probPsh, outpo.u, par_hopf)
BK.getMaximum(probPsh, outpo.u, par_hopf)
BK.getPeriodicOrbit(probPsh, outpo.u, par_hopf)

probPsh = PoincareShootingProblem(prob, Rodas4(),
		# probMono, Rodas4(autodiff=false),
		normals, centers; abstol = 1e-10, reltol = 1e-9,
		jacobian = :autodiffDenseAnalytical)

probPsh(outpo.u, par_hopf)
probPsh(outpo.u, par_hopf, outpo.u)
probPsh([0.30429879744900434], par_hopf)
probPsh([0.30429879744900434], (r = 0.09243096156871472, μ = 0.0, ν = 1.0, c3 = 1.0, c5 = 0.0))
BK.evolve(probPsh.flow,[0.0, 0.30429879744900434], (r = 0.094243096156871472, μ = 0.0, ν = 1.0, c3 = 1.0, c5 = 0.0), Inf64) # this gives an error in DiffEqBase

opts_po_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.015, ds= 0.01, pMax = 4.0, maxSteps = 30, newtonOptions = setproperties(optn; tol = 1e-7, eigsolver = eil), detectBifurcation = 0)
	# br_pok2, = continuation(
	# 	probPsh, outpo, par_hopf, (@lens _.r),
	# 	opts_po_cont; verbosity = 2,
	# 	tangentAlgo = BorderedPred(),
	# 	jacobianPO = :autodiffDenseAnalytical,
	# 	plot = false, normC = norminf)
# plot(br_pok2)
####################################################################################################
# normals = [[-1., 0.], [1, -1]]
# centers = [zeros(2), zeros(2)]
# initpo_bar = [1.04, -1.04/√2]

normals = [[-1., 0.], [1, 0]]
centers = [zeros(2), zeros(2)]
initpo_bar = [0.2, -0.2]

probPsh = PoincareShootingProblem(prob, KenCarp4(), normals, centers; abstol=1e-10, reltol=1e-9, lens = (@lens _.r), jacobian = :autodiffDenseAnalytical)
# version with analytical jacobian
probPsh2 = PoincareShootingProblem(prob, KenCarp4(), normals, centers; abstol=1e-10, reltol=1e-9, δ = 0, lens = (@lens _.r), jacobian = :autodiffDenseAnalytical)

# test of the analytical formula for jacobian of the functional
_Jad = BifurcationKit.finiteDifferences( x-> probPsh(x, par_hopf), initpo_bar)
# _Jphifd = BifurcationKit.finiteDifferences(x->probPsh.flow(x, par_hopf, Inf64), [0,0.4]; δ=1e-8)
_Jana = probPsh(Val(:JacobianMatrix), initpo_bar, par_hopf)
@test norm(_Jad - _Jana, Inf) < 1e-5

ls = DefaultLS()
	eil = EigKrylovKit(dim = 1, x₀=rand(1))
	optn = NewtonPar(verbose = false, tol = 1e-9,  maxIter = 140, linsolver = ls, eigsolver = eil)
	deflationOp = DeflationOperator(2.0, dot, 1.0, [zero(initpo_bar)])

	outpo = newton(probPsh2, initpo_bar, optn; normN = norminf)
	@test BK.converged(outpo)

	outpo = newton(probPsh, initpo_bar, optn; normN = norminf)
	@test BK.converged(outpo)

getPeriod(probPsh, outpo.u, par_hopf)

opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.025, ds= -0.01, pMax = 4.0, maxSteps = 50, newtonOptions = (@set optn.tol = 1e-9), detectBifurcation = 3, nev = 2)
	br_pok2 = continuation(probPsh, outpo.u, PALC(tangent = Bordered()),
		opts_po_cont; verbosity = 0,
		plot = false, normC = norminf)
@test br_pok2.prob isa BK.WrapPOSh
@test br_pok2.period[1] ≈ 2pi rtol = 1e-7
####################################################################################################
normals = [[-1., 0.], [1, 0], [0, 1]]
centers = [zeros(2), zeros(2), zeros(2)]
initpo = [[0., 0.4], [0, -.3], [0.3, 0]]

probPsh = PoincareShootingProblem(prob, KenCarp4(), normals, centers; abstol=1e-10, reltol=1e-9, lens = (@lens _.r), jacobian = :autodiffDenseAnalytical)

initpo_bar = reduce(vcat, [BK.R(probPsh, initpo[ii], ii) for ii in eachindex(centers)])
# same with projection function
initpo_bar = reduce(vcat, BK.projection(probPsh, initpo))

# test of the other projection function
BK.projection(probPsh, reduce(hcat, initpo)')

probPsh(initpo_bar, par_hopf; verbose = true)

# test of the analytical formula for jacobian of the functional
_Jad = BifurcationKit.finiteDifferences( x -> probPsh(x, par_hopf), initpo_bar)
_Jana = probPsh(Val(:JacobianMatrix), initpo_bar, par_hopf)
@test norm(_Jad - _Jana, Inf) < 1e-5


outpo = newton(probPsh, initpo_bar, optn; normN = norminf)
BK.converged(outpo)

for ii=1:length(normals)
	@show BK.E(probPsh	, [outpo.u[ii]], ii)
end

getPeriod(probPsh, outpo.u, par_hopf)

opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.025, ds= -0.005, pMax = 4.0, maxSteps = 10, newtonOptions = setproperties(optn; tol = 1e-8), detectBifurcation = 3)
	br_hpsh = continuation(probPsh, outpo.u, PALC(),
		opts_po_cont; normC = norminf)

@test br_hpsh.prob isa BK.WrapPOSh
@test br_hpsh.period[1] ≈ 2pi rtol = 1e-7
# plot(br_hpsh)
####################################################################################################
# test automatic branch switching with most possible options
# calls with analytical jacobians
br_psh = continuation(br, 1, (@set opts_po_cont.ds = 0.005), PoincareShootingProblem(2, prob, KenCarp4(); abstol=1e-10, reltol=1e-9, lens = @lens _.r); normC = norminf)
@test br_hpsh.prob isa BK.WrapPOSh
@test br_hpsh.period[1] ≈ 2pi rtol = 1e-7

# test Iterative Floquet eigen solver
@set! opts_po_cont.newtonOptions.eigsolver.dim = 20
@set! opts_po_cont.newtonOptions.eigsolver.x₀ = rand(2)
br_sh = continuation(br, 1, ContinuationPar(opts_po_cont; ds = 0.005, saveSolEveryStep = 1), ShootingProblem(2, prob, KenCarp4(); abstol=1e-10, reltol=1e-9, lens = @lens _.r); normC = norminf)
@test br_hpsh.prob isa BK.WrapPOSh
@test br_hpsh.period[1] ≈ 2pi rtol = 1e-7

# test MonodromyQaD
# BK.MonodromyQaD(br_sh.functional, br.sol)

ls = GMRESIterativeSolvers(reltol = 1e-7, N = length(initpo_bar), maxiter = 500, verbose = false)
@set! opts_po_cont.detectBifurcation = 0
@set! opts_po_cont.newtonOptions.linsolver = ls

for M in (1,2), jacobianPO in (:autodiffMF, :MatrixFree, :autodiffDenseAnalytical, :FiniteDifferencesDense)
	@info M, jacobianPO, "PS"

	# specific to Poincaré Shooting
	jacPO = jacobianPO == :autodiffMF ? :FiniteDifferencesDense : jacobianPO
	_parallel = jacPO == :MatrixFree ? false : true

	br_psh = continuation(br, 1,(@set opts_po_cont.ds = 0.005), PoincareShootingProblem(M, prob, Rodas4P(); abstol=1e-10, reltol=1e-9, parallel = _parallel, jacobian = jacPO); normC = norminf, updateSectionEveryStep = 2, linearAlgo = BorderingBLS(solver = (@set ls.N = M), checkPrecision = false), verbosity = 0)

	br_ssh = continuation(br, 1, (@set opts_po_cont.ds = 0.005),
	ShootingProblem(M, prob, Rodas4P(); abstol=1e-10, reltol=1e-9, parallel = _parallel, jacobian = jacPO); normC = norminf, updateSectionEveryStep = 2,
	linearAlgo = BorderingBLS(solver = (@set ls.N = 2M + 1), checkPrecision = false), verbosity = 0)
end
