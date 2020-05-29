module PseudoArcLengthContinuation
	using Plots, JLD2, Printf, Dates, LinearMaps, BlockArrays
	using Setfield: setproperties, @set, Lens, get, set, @lens
	using Parameters: @with_kw, @unpack
	using DocStringExtensions

	include("BorderedArrays.jl")
	include("LinearSolver.jl")
	include("EigSolver.jl")
	include("LinearBorderSolver.jl")
	include("Preconditioner.jl")
	include("Newton.jl")
	include("Continuation.jl")
	include("Bifurcations.jl")
	include("Predictor.jl")
	include("NormalForms.jl")
	include("DeflationOperator.jl")
	include("BorderedProblem.jl")
	include("Plotting.jl")
	include("Utils.jl")

	include("codim2/codim2.jl")
	include("codim2/FoldCont.jl")
	include("codim2/HopfCont.jl")

	include("periodicorbit/PeriodicOrbits.jl")
	include("periodicorbit/PeriodicOrbitUtils.jl")
	include("periodicorbit/Flow.jl")
	include("periodicorbit/StandardShooting.jl")
	include("periodicorbit/PoincareShooting.jl")
	include("periodicorbit/PeriodicOrbitFD.jl")
	include("periodicorbit/FloquetQaD.jl")

	# linear solvers
	export DefaultLS, GMRESIterativeSolvers, GMRESIterativeSolvers!, GMRESKrylovKit,
			DefaultEig, EigArpack, EigIterativeSolvers, EigKrylovKit, EigArnoldiMethod, geteigenvector, AbstractEigenSolver

	# bordered nonlinear problems
	export BorderedProblem, JacobianBorderedProblem, LinearSolverBorderedProblem

	# preconditioner based on deflation
	export PrecPartialSchurKrylovKit, PrecPartialSchurArnoldiMethod

	# bordered linear problems
	export MatrixBLS, BorderingBLS, MatrixFreeBLS, BorderedArray

	# nonlinear deflation
	export DeflationOperator, DeflatedProblem, DeflatedLinearSolver, scalardM

	# predictors for continuation
	export SecantPred, BorderedPred, NaturalPred

	# newton methods
	export NewtonPar, newton, newtonDeflated, newtonPALC, newtonFold, newtonHopf, newtonBordered

	# continuation methods
	export ContinuationPar, ContResult, continuation, continuation!, continuationFold, continuationHopf, continuationPOTrap, continuationBordered

	# iterators for continuation
	export PALCIterable, iterate, PALCStateVariables, solution, getx, getp

	# codim2 Fold continuation
	export FoldPoint, FoldProblemMinimallyAugmented, FoldLinearSolveMinAug, foldPoint

	# codim2 Hopf continuation
	export HopfPoint, HopfProblemMinimallyAugmented, HopfLinearSolveMinAug

	# normal form
	export computeNormalForm

	# Periodic orbit computation
	export getPeriod, getAmplitude, getMaximum, getTrajectory

	# Periodic orbit computation based on Trapeze method
	export PeriodicOrbitTrapProblem, continuationPOTrap, continuationPOTrapBPFromPO

	# Periodic orbit computation based on Shooting
	export Flow, ShootingProblem, PoincareShootingProblem, continuationPOShooting, AbstractShootingProblem

	# Floquet multipliers computation
	export FloquetQaDTrap, FloquetQaDShooting

	# guess for periodic orbit from Hopf bifurcation point
	export guessFromHopf
end
