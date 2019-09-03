module PseudoArcLengthContinuation
	using Parameters, Plots, JLD2, Printf, Dates, LinearMaps, Setfield, BlockArrays

	include("predictor.jl")
	include("LinearSolver.jl")
	include("EigSolver.jl")
	include("BorderedArrays.jl")
	include("LinearBorderSolver.jl")
	include("DeflationOperator.jl")
	include("Newton.jl")
	include("utils.jl")
	include("FoldCont.jl")
	include("HopfCont.jl")
	include("periodicorbit/PeriodicOrbit.jl")

	export	ContinuationPar, ContResult, continuation, continuationFold, continuationHopf, BorderedArray
	export	SecantPred, BorderedPred, NaturalPred
	export	NewtonPar, newton, newtonDeflated, newtonPArcLength, newtonFold, newtonHopf
	export	DeflationOperator, DeflatedProblem, DeflatedLinearSolver, scalardM
	export	Default, GMRES_IterativeSolvers, GMRES_KrylovKit,
			Default_eig, Default_eig_sp, eig_IterativeSolvers, eig_KrylovKit, eig_MF_KrylovKit, getEigenVector
	export	FoldPoint, FoldProblemMinimallyAugmented, FoldLinearSolveMinAug, foldPoint
	export	HopfPoint, HopfProblemMinimallyAugmented, HopfLinearSolveMinAug
	export	ShootingProblemTrap, ShootingProblemBE, ShootingProblemMid, PeriodicOrbitLinearSolverMid, PeriodicOrbitTrap
	export plotBranch, plotBranch!


	################################################################################################
	"""
		continuation(F, J, u0, p0::Real, contParams::ContinuationPar; plot = false, normC = norm, printsolution = norm, plotsolution::Function = (x; kwargs...)->nothing, finaliseSolution::Function = (z, tau, step, contResult) -> true, linearalgo = :bordering, verbosity = 2)

	Compute the continuation curve associated to the functional `F` and its jacobian `J`. The parameters are as follows
	- `F = (x, p) -> F(x, p)` where `p` is the parameter for the continuation
	- `J = (x, p) -> d_xF(x, p)` its associated jacobian. It can be a matrix, a function or a callable struct.
	- `u0` initial guess
	- `contParams` parameters for continuation, with struct `ContinuationPar`
	- `plot = false` whether to plot the solution while computing
	- `printsolution::Function = norm` function used to plot in the continuation curve, e.g. `norm` or `x -> x[1]`
	- `plotsolution::Function = (x; kwargs...) -> nothing` function implementing the plot of the solution.
	- `finaliseSolution::Function = (z, tau, step, contResult) -> true` Function called at the end of each continuation step. Can be used to alter the continuation procedure (stop it by returning false), saving personal data, plotting...
	- `tangentalgo = SecantPred()` controls the algorithm use to predict the tangent along the curve of solutions or the corrector. Can be `NaturalPred`, `SecantPred` or `BorderedPred`.
	- `linearalgo = :bordering`. Must belong to `[:bordering, :full]`. Used to control the way the extended linear system associated to the continuation problem is solved.
	- `verbosity` controls the amount of information printed during the continuation process.
	- `normC = norm` norm used in the different Newton solves

	The function outputs
	- `contres::ContResult` structure which contains the computed branch
	- `u::BorderedArray` the last solution computed on the branch

	# Method

	## Bordered system of equations

	The pseudo arclength continuation method solves the equation ``F(x,p) = 0`` (or dimension N) together with the pseudo-arclength constraint ``N(x, p) = \\frac{\\theta}{length(x)} \\langle x - x_0, \\tau_0\\rangle + (1 - \\theta)\\cdot(p - p_0)\\cdot dp_0 - ds = 0``. In practice, the curve is parametrised by ``s`` so that ``(x(s),p(s))`` is a curve of solutions to ``F(x,p)``. This formulation allows to pass turning points (where the implicit theorem fails). In the previous formula, ``(x_0, p_0)`` is a solution for a given ``s_0``, ``(\\tau_0, dp_0)`` is the tangent to the curve at ``s_0``. Hence, to compute the curve of solutions, we need solve an equation of dimension N+1 which is called a Bordered system.

	!!! warning "Parameter `theta`"
	    The parameter `theta` in the struct `ContinuationPar`is very important. It should be tuned for the continuation to work properly especially in the case of large problems where the ``\\langle x - x_0, \\tau_0\\rangle`` component in the constraint might be favoured too much.

	The parameter ds is adjusted internally depending on the number of Newton iterations and other factors. See the function `stepSizeControl` for more information. An important parameter to adjust the magnitude of this adaptation is the parameter `a` in the struct `ContinuationPar`.

	## Algorithm

	The algorithm works as follows:
	0. Start from a known solution ``(x_0,p_0,\\tau_0,dp_0)``
	1. **Predictor** set ``(x_1,p_1) = (x_0,p_0) + ds\\cdot (\\tau_0,dp_0)``
	2. **Corrector** solve ``F(x,p)=0,\\ N(x,p)=0`` with a (Bordered) Newton Solver.
	3. **New tangent** Compute ``(\\tau_1,dp_1)``, set ``(x_0,p_0,\\tau_0,dp_0)=(x_1,p_1,\\tau_1,dp_1)`` and return to step 2

	## Natural continuation

	We speak of *natural* continuation when we do not consider the constraint ``N(x,p)=0``. Knowing ``(x_0,p_0)``, we use ``x_0`` as a guess for solving ``F(x,p_1)=0`` with ``p_1`` close to ``p_0``. Again, this will fail at Turning points but it can be faster to compute than the constrained case. This is set by the option `tangentalgo = NaturalPred()` in `continuation`.

	## Tangent computation (step 4)
	There are various ways to compute ``(\\tau_1,p_1)``. The first one is called secant and is parametrised by the option `tangentalgo = SecantPred()` in `continuation`. It is computed by ``(\\tau_1,p_1) = (z_1,p_1) - (z_0,p_0)`` and normalised by the norm ``\\|u,p\\|^2_\\theta = \\frac{\\theta}{length(u)} \\langle u,u\\rangle + (1 - \\theta)\\cdot p^2``. Another method is use computing ``(\\tau_1,p_1)`` by solving a bordered linear system, see the function `getTangentBordered` for more information ; it is set by the option `tangentalgo = BorderedPred()`.

	## Bordered linear solver

	When solving the Bordered system ``F(x,p) = 0,\\ N(x, p)=0``, one faces the issue of solving the Bordered linear system ``\\begin{bmatrix} J & a    ; b^T & c\\end{bmatrix}\\begin{bmatrix}X ;  y\\end{bmatrix} =\\begin{bmatrix}R ; n\\end{bmatrix}``. This can be solved in many ways via bordering (which requires two Jacobian inverses) or by forming the bordered matrix (which works well for sparse matrices). The choice of method is set by the argument `linearalgo`. Have a look at the function `linearBorderedSolver` for more information.
	"""
	function continuation(Fhandle,
						Jhandle,
						u0,
						p0::T,
						contParams::ContinuationPar{T, S, E};
						tangentalgo = SecantPred(),
						linearalgo   = :bordering,
						plot = false,
						printsolution = norm,
						normC = norm,
						plotsolution = (x;kwargs...) -> nothing,
						finaliseSolution = (z, tau, step, contResult) -> true,
						verbosity = 2) where {T, S <: LinearSolver, E <: EigenSolver}
		################################################################################################
		## Get parameters
		@unpack pMin, pMax, maxSteps, newtonOptions = contParams
		epsi = contParams.finDiffEps

		check!(contParams)

		# Filename to save the computations
		filename = "branch-" * string(Dates.now())

		(verbosity > 0) && printstyled("#"^50*"\n*********** ArcLengthContinuationNewton *************\n\n", bold = true, color = :red)

		# Converge initial guess
		(verbosity > 0) && printstyled("*********** CONVERGE INITIAL GUESS *************", bold = true, color = :magenta)
		u0, fval, isconverged, it_number = newton(x -> Fhandle(x, p0),
												x -> Jhandle(x, p0),
												u0, newtonOptions, normN = normC)
		@assert isconverged "Newton failed to converge initial guess"
		(verbosity > 0) && (print("\n--> convergence of initial guess = ");printstyled("OK\n", color=:green))
		(verbosity > 0) && println("--> p = $(p0), initial step")

		# Save data and hold general information
		if contParams.computeEigenValues
			# Eigen elements computation
			evsol =  newtonOptions.eigsolve(Jhandle(u0, p0), contParams.nev)
			contRes = initContRes(VectorOfArray([vcat(p0, printsolution(u0), it_number, contParams.ds)]), u0, evsol, contParams)
		else
			contRes = initContRes(VectorOfArray([vcat(p0, printsolution(u0), it_number, contParams.ds)]), u0, 0, contParams)
		end

		(verbosity > 0) && printstyled("\n******* COMPUTING INITIAL TANGENT *************", bold = true, color = :magenta)
		u_pred, fval, isconverged, it_number = newton(x -> Fhandle(x, p0 + contParams.ds / T(50)),
													x -> Jhandle(x, p0 + contParams.ds / T(50)),
													u0, newtonOptions, normN = normC)
		@assert isconverged "Newton failed to converge for the computation of the initial tangent"
		(verbosity > 0) && (print("\n--> convergence of initial guess = ");printstyled("OK\n\n", color=:green))
		(verbosity > 0) && println("--> p = $(p0 + contParams.ds/50), initial step (bis)")

		duds = copy(u_pred)
		axpby!(-T(1)/ (contParams.ds / T(50)), u0, T(1)/ (contParams.ds / T(50)), duds)
		# duds = (u_pred - u0) / (contParams.ds / T(50));
		dpds = T(1)
		α = normtheta(duds, dpds, contParams.theta)
		@assert typeof(α) == T
		@assert α > 0 "Error, α = 0, cannot scale first tangent vector"
		rmul!(duds, T(1) / α); dpds = dpds / α

		## Initialise continuation
		step = 0
		continuationFailed = false
		# number of iterations for newton correction
		it_number = 0

		# Variables to hold the predictor
		z_pred   = BorderedArray(copy(u0), p0)
		tau_pred = BorderedArray(copy(u0), p0)
		tau_new  = BorderedArray(copy(u0), p0)

		z_old     = BorderedArray(copy(u_pred), p0)
		tau_old   = BorderedArray(copy(duds), dpds)

		(verbosity > 0) && println("--> Start continuation from p = ", z_old.p)

		## Main continuation loop
		while (step < maxSteps) & ~continuationFailed & (z_old.p < contParams.pMax) & (z_old.p > contParams.pMin)
			# Predictor: z_pred
			getPredictor!(z_pred, z_old, tau_old, contParams, tangentalgo)
			(verbosity > 0) && println("########################################################################")
			(verbosity > 0) && @printf("Start of Continuation Step %d: Parameter: p1 = %2.4e --> %2.4e\n", step, z_old.p, z_pred.p)
			(length(contRes.branch[4, :])>1 && (verbosity > 0)) && @printf("Step size  = %2.4e --> %2.4e\n", contRes.branch[4, end-1], contParams.ds)

			# Corrector, ie newton correction
			z_new, fval, isconverged, it_number  = corrector(Fhandle, Jhandle,
					z_old, tau_old, z_pred, contParams, tangentalgo,
					linearalgo, normC = normC)

			# Successful step
			if isconverged
				(verbosity > 0) && printstyled("--> Step Converged in $it_number Nonlinear Solver Iterations!\n", color=:green)

				# get predictor
				getTangent!(tau_new, z_new, z_old, tau_old, Fhandle, Jhandle, contParams, tangentalgo, verbosity)

				# Output
				push!(contRes.branch, vcat(z_new.p, printsolution(z_new.u), it_number, contParams.ds))

				# Detection of codim 1 bifurcation points
				# This should be there before the old z is re-written
				if contParams.detect_fold || contParams.detect_bifurcation
					detectBifucation(contParams, contRes, z_old, tau_old, printsolution, verbosity)
				end

				copyto!(z_old, z_new)
				copyto!(tau_old, tau_new)

				if contParams.computeEigenValues
					# number of eigenvalues to be computed
					res = computeEigenvalues(contParams, contRes, Jhandle(z_old.u, z_old.p),  step)
					push!(contRes.stability, mapreduce(x->real(x)<0, *, contRes.eig[end][1]))
					(verbosity > 0) && printstyled(color=:green,"--> Computed ",contParams.nev," eigenvalues in ",res[end]," iterations\n")
				end

		  		# Plotting
				(plot && mod(step, contParams.plot_every_n_steps) == 0 ) && plotBranchCont(contRes, z_old, contParams, plotsolution)

				# Saving Solution
				if contParams.save
					(verbosity > 0) && printstyled("--> Solving solution in file\n", color=:green)
					saveSolution(filename, z_old.u, z_old.p, step, contRes, contParams)
				end

				# Call user saved finaliseSolution function. If returns false, stop continuation
				!finaliseSolution(z_old, tau_old, step, contRes) && (step = maxSteps)
			else
				(verbosity > 0) && printstyled("Newton correction failed\n", color=:red)
				(verbosity > 0) && println("--> Newton Residuals history = ", fval)
			end

			step += 1
			continuationFailed = stepSizeControl(contParams, isconverged, it_number, tau_old, contRes.branch, verbosity)
		end # while
		plot && plotBranchCont(contRes, z_old, contParams, plotsolution)

		# we remove the initial guesses that are meaningless
		popfirst!(contRes.bifpoint)
		return contRes, z_old, tau_old
	end

	continuation(Fhandle::Function, u0, p0::T, contParams::ContinuationPar{T, S, E}; kwargs...) where {T, S <: LinearSolver, E <: EigenSolver} = continuation(Fhandle, (u0, p) -> finiteDifferences(u -> Fhandle(u, p), u0), u0, p0, contParams; kwargs...)
end
