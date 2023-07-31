module BifurcationKit
	using Printf, Dates, LinearMaps, BlockArrays, RecipesBase, StructArrays, Requires
	using Reexport
	@reexport using Setfield: setproperties, @set, @set!, Lens, get, set, @lens
	using Parameters: @with_kw, @unpack, @with_kw_noshow
	using RecursiveArrayTools: VectorOfArray, ArrayPartition
	using DocStringExtensions
	using DataStructures: CircularBuffer
	using ForwardDiff

	include("Problems.jl")
	include("jacobianTypes.jl")

	# we put this here to be used in LinearBorderSolver and Continuation
	abstract type AbstractContinuationAlgorithm end
	abstract type AbstractContinuationIterable{kind} end
	abstract type AbstractContinuationState{Tv} end

	include("ContKind.jl")

	include("BorderedArrays.jl")
	include("LinearSolver.jl")
	include("EigSolver.jl")
	include("LinearBorderSolver.jl")
	include("Preconditioner.jl")
	include("Newton.jl")
	include("ContParameters.jl")
	include("Results.jl")

	include("events/Event.jl")

	include("DeflationOperator.jl")

	# continuation
	include("Continuation.jl")
	include("IteratorUtils.jl")

	# events
	include("events/EventDetection.jl")
	include("events/BifurcationDetection.jl")

	include("Bifurcations.jl")

	# continuers
	include("continuation/Contbase.jl")
	include("continuation/Natural.jl")
	include("continuation/Palc.jl")
	include("continuation/Multiple.jl")
	include("continuation/MoorePenrose.jl")
	include("DeflatedContinuation.jl")

	# wip
	include("BorderedProblem.jl")

	include("Utils.jl")

	# generic codim 2
	include("codim2/codim2.jl")
	include("codim2/MinAugFold.jl")
	include("codim2/MinAugHopf.jl")
	include("codim2/MinAugBT.jl")

	include("BifurcationPoints.jl")

	include("bifdiagram/BranchSwitching.jl")
	include("NormalForms.jl")
	include("codim2/BifurcationPoints.jl")
	include("codim2/NormalForms.jl")
	include("bifdiagram/BifurcationDiagram.jl")

	# periodic orbit problems
	include("periodicorbit/Sections.jl")
	include("periodicorbit/PeriodicOrbits.jl")
	include("periodicorbit/PeriodicOrbitTrapeze.jl")
	include("periodicorbit/PeriodicOrbitCollocation.jl")
	include("periodicorbit/Flow.jl")
	include("periodicorbit/FlowDE.jl")
	include("periodicorbit/StandardShooting.jl")
	include("periodicorbit/PoincareShooting.jl")
	include("periodicorbit/ShootingDE.jl")
	include("periodicorbit/Floquet.jl")
	include("periodicorbit/BifurcationPoints.jl")
	include("periodicorbit/PeriodicOrbitUtils.jl")

	include("periodicorbit/PoincareRM.jl")
	include("periodicorbit/NormalForms.jl")

	# periodic orbit codim 2
	include("periodicorbit/codim2/utils.jl")
	include("periodicorbit/codim2/codim2.jl")
	# include("periodicorbit/codim2/PeriodicOrbitTrapeze.jl")
	include("periodicorbit/codim2/PeriodicOrbitCollocation.jl")
	include("periodicorbit/codim2/StandardShooting.jl")
	include("periodicorbit/codim2/MinAugPD.jl")
	include("periodicorbit/codim2/MinAugNS.jl")

	# wave problem
	include("wave/WaveProblem.jl")
	include("wave/EigSolver.jl")

	# plotting
	include("plotting/Utils.jl")

	# wrappers for SciML
	include("Diffeqwrap.jl")

	using Requires

	function __init__()
		# if Plots.jl is available, then we allow plotting of solutions
		@require Plots="91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin
			using .Plots
			include("plotting/RecipesPlots.jl")
			get_plot_backend() = BK_Plots()
		end
		@require AbstractPlotting="537997a7-5e4e-5d89-9595-2241ea00577e" begin
			using .AbstractPlotting: @recipe, layoutscene, Figure, Axis, lines!
			include("plotting/RecipesMakie.jl")
		end

		@require GLMakie="e9467ef8-e4e7-5192-8a1a-b1aee30e663a" begin
			@info "Loading GLMakie code in BifurcationKit"
			using .GLMakie: @recipe, Figure, Axis, lines!, PointBased, Point2f0, scatter!
			include("plotting/RecipesMakie.jl")
			get_plot_backend() = BK_Makie()
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
			function saveToFile(iter::AbstractContinuationIterable, sol, p, i::Int64, br::ContResult)
				if iter.contParams.saveToFile == false; return nothing; end
				filename = iter.filename
				# this allows to save two branches forward/backward in case
				# bothside = true is passed to continuation
				fd = iter.contParams.ds >=0 ? "fw" : "bw"

				# create a group in the JLD format
				jldopen(filename*".jld2", "a+") do file
					if haskey(file, "sol-$fd-$i")
						delete!(file, "sol-$fd-$i")
					end
					mygroup = JLD2.Group(file, "sol-$fd-$i")
					mygroup["sol"] = sol
					mygroup["param"] = p
				end

				jldopen(filename*"-branch.jld2", "a+") do file
					if haskey(file, "branch"*fd)
						delete!(file, "branch"*fd)
					end
					file["branch"*fd] = br
				end
			end

			# final save of branch, in case bothsided = true is used
			function saveToFile(iter::AbstractContinuationIterable, br::ContResult)
				if iter.contParams.saveToFile == false; return nothing; end
				filename = iter.filename

				jldopen(filename*"-branch.jld2", "a+") do file
					if haskey(file, "branchfw")
						delete!(file, "branchfw")
					end
					if haskey(file, "branchbw")
						delete!(file, "branchbw")
					end
					if haskey(file, "branch")
						delete!(file, "branch")
					end
					file["branch"] = br
				end
			end
		end
	end

	# linear solvers
	export DefaultLS, GMRESIterativeSolvers, GMRESKrylovKit,
			DefaultEig, EigArpack, EigIterativeSolvers, EigKrylovKit, EigArnoldiMethod, geteigenvector, AbstractEigenSolver

	# Problems
	export BifurcationProblem, BifFunction, getLens, getParams, reMake

	# bordered nonlinear problems
	export BorderedProblem, JacobianBorderedProblem, LinearSolverBorderedProblem

	# preconditioner based on deflation
	export PrecPartialSchurKrylovKit, PrecPartialSchurArnoldiMethod

	# bordered linear problems
	export MatrixBLS, BorderingBLS, MatrixFreeBLS, LSFromBLS, BorderedArray

	# nonlinear deflation
	export DeflationOperator, DeflatedProblem, DefProbCustomLinearSolver, scalardM

	# predictors for continuation
	export Natural, PALC, Multiple, Secant, Bordered, DefCont, Polynomial, MoorePenrose, MoorePenroseLS

	# newton methods
	export NewtonPar, newton, newtonDeflated, newtonPALC, newtonFold, newtonHopf, newtonBordered, NonLinearSolution

	# continuation methods
	export ContinuationPar, ContResult, GenericBifPoint, continuation, continuation!, continuationFold, continuationHopf, continuationPOTrap, continuationBordered, eigenvec, eigenvals, getSolx, getSolp, bifurcation_points, SpecialPoint

	# events
	export ContinuousEvent, DiscreteEvent, PairOfEvents, SetOfEvents, SaveAtEvent, FoldDetectEvent, BifDetectEvent

	# iterators for continuation
	export ContIterable, iterate, ContState, solution, getx, getp

	# codim2 Fold continuation
	export FoldPoint, FoldProblemMinimallyAugmented, FoldLinearSolveMinAug, foldPoint

	# codim2 Hopf continuation
	export HopfPoint, HopfProblemMinimallyAugmented, HopfLinearSolveMinAug

	# normal form
	export getNormalForm, hopfNormalForm, predictor

	# automatic bifurcation diagram
	export bifurcationdiagram, bifurcationdiagram!, Branch, BifDiagNode, getBranch, getBranchesFromBP

	# Periodic orbit computation
	export generateSolution, getPeriod, getAmplitude, getMaximum, getPeriodicOrbit, sectionSS, sectionPS, guessFromHopf, generateCIProblem

	# Periodic orbit computation based on Trapeze method
	export PeriodicOrbitTrapProblem, continuationPOTrap, continuationPOTrapBPFromPO

	# Periodic orbit computation based on Shooting
	export Flow, ShootingProblem, PoincareShootingProblem, continuationPOShooting, AbstractShootingProblem, SectionPS, SectionSS

	# Periodic orbit computation based on Collocation
	export PeriodicOrbitOCollProblem

	# Floquet multipliers computation
	export FloquetQaD

	# waves
	export TWProblem
end
