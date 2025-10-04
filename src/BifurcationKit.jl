module BifurcationKit
    using Printf, Dates
    import BlockArrays, StructArrays, LinearMaps
    using Reexport
    @reexport using Accessors: setproperties, @set, @reset, PropertyLens, getall, set, @optic, IndexLens, ComposedOptic
    @reexport using ArnoldiMethod: LM, LR, LI, SR, SI
    using Parameters: @with_kw, @with_kw_noshow
    using PreallocationTools: DiffCache, get_tmp
    using DocStringExtensions
    import DataStructures # used for Polynomial predictor
    using ForwardDiff
    import Random: randn!
    import LinearAlgebra as LA

    include("Accessors.jl")
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
    include("continuation/AutoSwitch.jl")
    include("DeflatedContinuation.jl")
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
    include("periodicorbit/cop.jl")
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
    include("periodicorbit/codim2/BifurcationPoints.jl")
    include("periodicorbit/codim2/NormalForms.jl")

    # wave problem
    include("wave/WaveProblem.jl")
    include("wave/EigSolver.jl")

    # plotting
    include("plotting/Utils.jl")

    # wrappers for SciML
    include("Diffeqwrap.jl")

    function save_to_file end

    # linear solvers
    export norminf
    
    export DefaultLS, GMRESIterativeSolvers, GMRESKrylovKit, KrylovLS, KrylovLSInplace
    export DefaultEig, EigArpack, EigKrylovKit, EigArnoldiMethod, geteigenvector, AbstractEigenSolver

    # Problems
    export BifurcationProblem, BifFunction, getlens, getparams, re_make, ODEBifProblem, DAEBifProblem

    # bordered nonlinear problems
    # export BorderedProblem, JacobianBorderedProblem, LinearSolverBorderedProblem, newtonBordered, continuationBordered

    # preconditioner based on deflation
    export PrecPartialSchurKrylovKit, PrecPartialSchurArnoldiMethod

    export BorderedArray, zerovector

    # bordered linear problems
    export MatrixBLS, BorderingBLS, MatrixFreeBLS, LSFromBLS

    # nonlinear deflation
    export DeflationOperator, DeflatedProblem

    # predictors for continuation
    export Natural, PALC, Multiple, Secant, Bordered, DefCont, Polynomial, MoorePenrose, MoorePenroseLS, AutoSwitch

    # newton methods
    export NewtonPar, Newton, newton, newton_palc, newton_hopf, NonLinearSolution

    # continuation methods
    export ContinuationPar, ContResult, continuation, continuation!, continuation_fold, continuation_hopf, continuation_potrap, eigenvec, eigenvals, get_solx, get_solp, bifurcation_points, SpecialPoint

    # events
    export ContinuousEvent, DiscreteEvent, PairOfEvents, SetOfEvents, SaveAtEvent, FoldDetectEvent, BifDetectEvent

    # iterators for continuation
    export ContIterable, iterate, ContState, getsolution, getx, getp, getpreviousx, getpreviousp, gettangent, getpredictor, get_previous_solution

    # codim2 Fold continuation
    export foldpoint, FoldProblemMinimallyAugmented, FoldLinearSolverMinAug

    # codim2 Hopf continuation
    export HopfPoint, HopfProblemMinimallyAugmented, HopfLinearSolverMinAug

    # normal form
    export get_normal_form, hopf_normal_form, predictor

    # automatic bifurcation diagram
    export bifurcationdiagram, bifurcationdiagram!, Branch, BifDiagNode, get_branch, get_branches_from_BP

    # Periodic orbit computation
    export generate_solution, getperiod, get_periodic_orbit, guess_from_hopf, generate_ci_problem

    # Periodic orbit computation based on Trapeze method
    export PeriodicOrbitTrapProblem, continuation_potrap

    # Periodic orbit computation based on Shooting
    export Flow, ShootingProblem, PoincareShootingProblem, AbstractShootingProblem, SectionPS, SectionSS

    # Periodic orbit computation based on Collocation
    export PeriodicOrbitOCollProblem, COPBLS, COPLS

    # Floquet multipliers computation
    export FloquetQaD, FloquetColl, FloquetGEV

    # waves
    export TWProblem
end
