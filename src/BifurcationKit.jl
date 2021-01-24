module BifurcationKit
	using Printf, Dates, LinearMaps, BlockArrays, RecipesBase, StructArrays, Requires
	using Setfield: setproperties, @set, Lens, get, set, @lens
	using Parameters: @with_kw, @unpack, @with_kw_noshow
	using RecursiveArrayTools: VectorOfArray
	using DocStringExtensions
	using DataStructures: CircularBuffer
	using ForwardDiff


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
	include("codim2/MinAugFold.jl")
	include("codim2/MinAugHopf.jl")

	include("BifurcationPoints.jl")

	include("bifdiagram/BranchSwitching.jl")
	include("NormalForms.jl")
	include("bifdiagram/BifurcationDiagram.jl")

	include("DeflatedContinuation.jl")

	include("periodicorbit/Sections.jl")
	include("periodicorbit/PeriodicOrbits.jl")
	include("periodicorbit/PeriodicOrbitTrapeze.jl")
	# include("periodicorbit/PeriodicOrbitMIRK.jl")
	# include("periodicorbit/PeriodicOrbitFDAdapt.jl")
	include("periodicorbit/PeriodicOrbitUtils.jl")
	include("periodicorbit/Flow.jl")
	include("periodicorbit/StandardShooting.jl")
	include("periodicorbit/PoincareShooting.jl")
	include("periodicorbit/Floquet.jl")

	include("plotting/Recipes.jl")
	include("Diffeqwrap.jl")

	using Requires

	function __init__()
		@require Plots="91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin
			using .Plots
			include("plotting/PlotCont.jl")
		end

		@require JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819" begin
			using .JLD2
			"""
			Save solution / data in JLD2 file
			- `filename` is for example "example.jld2"
			- `sol` is the solution
			- `p` is the parameter
			- `i` is the index of the solution to be saved
			"""
			function saveToFile(filename, sol, p, i::Int64, br::ContResult)
				try
					# create a group in the JLD format
					jldopen(filename*".jld2", "a+") do file
						mygroup = JLD2.Group(file, "sol-$i")
						mygroup["sol"] = sol
						mygroup["param"] = p
					end

					jldopen(filename*"-branch.jld2", "w") do file
						file["branch"] = br
					end
				catch
					@error "Could not save branch in the jld2 file"
				end
			end
		end

	end


	# linear solvers
	export DefaultLS, GMRESIterativeSolvers, GMRESIterativeSolvers!, GMRESKrylovKit,
			DefaultEig, EigArpack, EigIterativeSolvers, EigKrylovKit, EigArnoldiMethod, geteigenvector, AbstractEigenSolver

	# bordered nonlinear problems
	export BorderedProblem, JacobianBorderedProblem, LinearSolverBorderedProblem

	# preconditioner based on deflation
	export PrecPartialSchurKrylovKit, PrecPartialSchurArnoldiMethod

	# bordered linear problems
	export MatrixBLS, BorderingBLS, MatrixFreeBLS, LSFromBLS, BorderedArray

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
	export FloquetQaD

	# guess for periodic orbit from Hopf bifurcation point
	export guessFromHopf
end
