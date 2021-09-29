# using Revise, Plots
using Test
	using BifurcationKit, Parameters, Setfield, LinearAlgebra, ForwardDiff, SparseArrays
	const BK = BifurcationKit
##################################################################
# The goal of these tests is to test all combinaisons of options
##################################################################

norminf = x -> norm(x, Inf)

function Fsl!(f, u, p, t)
	@unpack r, μ, ν, c3 = p
	u1 = u[1]
	u2 = u[2]

	ua = u1^2 + u2^2

	f[1] = r * u1 - ν * u2 - ua * (c3 * u1 - μ * u2)
	f[2] = r * u2 + ν * u1 - ua * (c3 * u2 + μ * u1)
	return f
end

Fsl(x, p) = Fsl!(similar(x), x, p, 0.)

####################################################################################################
par_sl = (r = 0.5, μ = 0., ν = 1.0, c3 = 1.0)
u0 = [.001, .001]
par_hopf = (@set par_sl.r = 0.1)
####################################################################################################
# continuation, Hopf bifurcation point detection
optconteq = ContinuationPar(ds = -0.01, detectBifurcation = 3, pMin = -0.5, nInversion = 8)
br, = continuation(Fsl, u0, par_hopf, (@lens _.r), optconteq)
####################################################################################################
poTrap = PeriodicOrbitTrapProblem(
	Fsl, (x, p) -> sparse(ForwardDiff.jacobian(z -> Fsl(z, p), x)), # we put sparse to try the different linear solvers
	[1., 0.],
	zeros(2),
	20, 2)

BK.hasHessian(poTrap)
show(poTrap)

# guess for the periodic orbit
orbitguess_f = reduce(vcat, [√(par_hopf.r) .* [cos(θ), sin(θ)] for θ in LinRange(0, 2pi, poTrap.M)])
	push!(orbitguess_f, 2pi)
optn_po = NewtonPar()
opts_po_cont = ContinuationPar(dsmax = 0.02, ds = 0.001, pMax = 2.2, maxSteps = 25, newtonOptions = optn_po)

lsdef = DefaultLS()
lsit = GMRESKrylovKit()
for (ind, linearPO) in enumerate([:Dense, :FullLU, :BorderedLU, :FullSparseInplace, :BorderedSparseInplace, :FullMatrixFree, :BorderedMatrixFree])
	@show linearPO, ind
	_ls = ind > 5 ? lsit : lsdef
	outpo_f, _, flag = newton(poTrap,
		orbitguess_f, par_hopf, (@set optn_po.linsolver = _ls);
		linearPO = linearPO,
		normN = norminf)
	@test flag

	br_po, = continuation(poTrap, outpo_f,
		par_hopf, (@lens _.r),	(@set opts_po_cont.newtonOptions.linsolver = _ls), linearPO = linearPO;
		verbosity = 0,	plot = false,
		# plotSolution = (x, p; kwargs...) -> BK.plotPeriodicPOTrap(x, poTrap.M, 2, 1; ratio = 2, kwargs...),
		printSolution = (u, p) -> BK.getAmplitude(poTrap, u, par_hopf; ratio = 1), normC = norminf)
end

outpo_f, = newton(poTrap, orbitguess_f, par_hopf, optn_po; linearPO = :Dense)
outpo = reshape(outpo_f[1:end-1], 2, poTrap.M)

# computation of the Jacobian at out_pof
_J1 = poTrap(Val(:JacFullSparse), outpo_f, par_hopf)
_Jfd = ForwardDiff.jacobian(z-> poTrap(z,par_hopf), outpo_f)

# test of the jacobian againt automatic differentiation
@test norm(_Jfd - Array(_J1), Inf) < 1e-7

####################################################################################################
# tests for constructor of Floquet routines
BK.checkFloquetOptions(EigArpack())
BK.checkFloquetOptions(EigArnoldiMethod())
BK.checkFloquetOptions(EigKrylovKit())
FloquetQaD(EigKrylovKit()) |> FloquetQaD
