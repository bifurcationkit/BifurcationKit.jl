module BifurcationKit
	using Plots, JLD2, Printf, Dates, LinearMaps, BlockArrays, RecipesBase, StructArrays
	using Setfield: setproperties, @set, Lens, get, set, @lens
	using Parameters: @with_kw, @unpack, @with_kw_noshow
	using RecursiveArrayTools: VectorOfArray
	using DocStringExtensions
	using DataStructures: CircularBuffer

	include("BorderedArrays.jl")
	include("LinearSolver.jl")
	include("EigSolver.jl")
	include("LinearBorderSolver.jl")
	include("Preconditioner.jl")
	include("Newton.jl")
	include("ContParameters.jl")
	include("Results.jl")
	include("Continuation.jl")
	include("Bifurcations.jl")
	include("Predictor.jl")

	include("DeflationOperator.jl")
	include("BorderedProblem.jl")

	include("Utils.jl")

	include("codim2/codim2.jl")
	include("codim2/FoldCont.jl")
	include("codim2/HopfCont.jl")

	include("BifurcationPoints.jl")

	include("bifdiagram/BranchSwitching.jl")
	include("NormalForms.jl")
	include("bifdiagram/BifurcationDiagram.jl")

	include("DeflatedContinuation.jl")

	include("periodicorbit/Sections.jl")
	include("periodicorbit/PeriodicOrbits.jl")
	include("periodicorbit/PeriodicOrbitUtils.jl")
	include("periodicorbit/Flow.jl")
	include("periodicorbit/StandardShooting.jl")
	include("periodicorbit/PoincareShooting.jl")
	include("periodicorbit/PeriodicOrbitFD.jl")
	include("periodicorbit/FloquetQaD.jl")

	include("Plotting.jl")

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
	export SecantPred, BorderedPred, NaturalPred, MultiplePred, PolynomialPred

	# newton methods
	export NewtonPar, newton, newtonDeflated, newtonPALC, newtonFold, newtonHopf, newtonBordered

	# continuation methods
	export ContinuationPar, ContResult, GenericBifPoint, continuation, continuation!, continuationFold, continuationHopf, continuationPOTrap, continuationBordered, eigenvec, eigenvals

	# iterators for continuation
	export ContIterable, iterate, ContState, solution, getx, getp

	# codim2 Fold continuation
	export FoldPoint, FoldProblemMinimallyAugmented, FoldLinearSolveMinAug, foldPoint

	# codim2 Hopf continuation
	export HopfPoint, HopfProblemMinimallyAugmented, HopfLinearSolveMinAug

	# normal form
	export computeNormalForm, predictor

	# automatic bifurcation diagram
	export bifurcationdiagram, bifurcationdiagram!, Branch, BifDiagNode, getBranch, getBranchesFromBP

	# Periodic orbit computation
	export getPeriod, getAmplitude, getMaximum, getTrajectory, sectionSS, sectionPS

	# Periodic orbit computation based on Trapeze method
	export PeriodicOrbitTrapProblem, continuationPOTrap, continuationPOTrapBPFromPO

	# Periodic orbit computation based on Shooting
	export Flow, ShootingProblem, PoincareShootingProblem, continuationPOShooting, AbstractShootingProblem, SectionPS, SectionSS

	# Floquet multipliers computation
	export FloquetQaDTrap, FloquetQaDShooting

	# guess for periodic orbit from Hopf bifurcation point
	export guessFromHopf
end
