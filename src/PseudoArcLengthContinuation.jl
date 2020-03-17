module PseudoArcLengthContinuation
	using Plots, JLD2, Printf, Dates, LinearMaps, BlockArrays
	using Setfield: setproperties, @set
	using Parameters: @with_kw, @unpack

	include("BorderedArrays.jl")
	include("LinearSolver.jl")
	include("EigSolver.jl")
	include("LinearBorderSolver.jl")
	include("Preconditioner.jl")
	include("Newton.jl")
	include("Continuation.jl")
	include("Bifurcations.jl")
	include("Predictor.jl")
	include("DeflationOperator.jl")
	include("BorderedProblem.jl")
	include("Plotting.jl")
	include("Utils.jl")
	include("FoldCont.jl")
	include("HopfCont.jl")

	include("periodicorbit/PeriodicOrbitUtils.jl")
	include("periodicorbit/Shooting.jl")
	include("periodicorbit/PeriodicOrbitFD.jl")
	include("periodicorbit/FloquetQaD.jl")

	export	DefaultLS, GMRESIterativeSolvers, GMRESIterativeSolvers!, GMRESKrylovKit,
			DefaultEig, EigArpack, EigIterativeSolvers, EigKrylovKit, EigArnoldiMethod, geteigenvector, AbstractEigenSolver
	export	BorderedProblem, JacobianBorderedProblem, LinearSolverBorderedProblem
	export	PrecPartialSchurKrylovKit, PrecPartialSchurArnoldiMethod
	export	MatrixBLS, BorderingBLS, MatrixFreeBLS, BorderedArray
	export	DeflationOperator, DeflatedProblem, DeflatedLinearSolver, scalardM
	export	SecantPred, BorderedPred, NaturalPred
	export	NewtonPar, newton, newtonDeflated, newtonPALC, newtonFold, newtonHopf, newtonBordered
	export	ContinuationPar, ContResult, continuation, continuation!, continuationFold, continuationHopf, continuationPOTrap, continuationBordered
	export	PALCIterable, iterate, PALCStateVariables, solution, getu, getp
	export	FoldPoint, FoldProblemMinimallyAugmented, FoldLinearSolveMinAug, foldPoint
	export	HopfPoint, HopfProblemMinimallyAugmented, HopfLinearSolveMinAug
	export	PeriodicOrbitTrapProblem, continuationPOTrap
	export	Flow, ShootingProblem, PoincareShootingProblem, continuationPOShooting, getPeriod, AbstractShootingProblem, extractPeriodShooting
	export	FloquetQaDTrap, FloquetQaDShooting
	export	plotBranch, plotBranch!, guessFromHopf
end
