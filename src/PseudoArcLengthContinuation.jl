module PseudoArcLengthContinuation
	using Parameters, Plots, JLD2, Printf, Dates, LinearMaps, Setfield, BlockArrays

	include("Predictor.jl")
	include("LinearSolver.jl")
	include("EigSolver.jl")
	include("BorderedArrays.jl")
	include("LinearBorderSolver.jl")
	include("DeflationOperator.jl")
	include("Newton.jl")
	include("Utils.jl")
	include("FoldCont.jl")
	include("HopfCont.jl")
	include("periodicorbit/PeriodicOrbit.jl")

	export	ContinuationPar, ContResult, continuation, continuationFold, continuationHopf, BorderedArray
	export	SecantPred, BorderedPred, NaturalPred
	export	MatrixBLS, BorderingBLS, MatrixFreeBLS
	export	NewtonPar, newton, newtonDeflated, newtonPArcLength, newtonFold, newtonHopf
	export	DeflationOperator, DeflatedProblem, DeflatedLinearSolver, scalardM
	export	DefaultLS, GMRES_IterativeSolvers, GMRES_KrylovKit,
			DefaultEig, DefaultEigSparse, eig_IterativeSolvers, eig_KrylovKit, eig_MF_KrylovKit, getEigenVector
	export	FoldPoint, FoldProblemMinimallyAugmented, FoldLinearSolveMinAug, foldPoint
	export	HopfPoint, HopfProblemMinimallyAugmented, HopfLinearSolveMinAug
	export	ShootingProblemTrap, ShootingProblemBE, ShootingProblemMid, PeriodicOrbitLinearSolverMid, PeriodicOrbitTrap
	export	plotBranch, plotBranch!


	################################################################################################
	"""
		continuation(F, J, x0, p0::Real, contParams::ContinuationPar; plot = false, normC = norm, printsolution = norm, plotsolution::Function = (x; kwargs...)->nothing, finaliseSolution::Function = (z, tau, step, contResult) -> true, linearalgo = BorderingBLS(), tangentalgo = SecantPred(), verbosity = 0)

	Compute the continuation curve associated to the functional `F` and its jacobian `J`. The parameters are as follows
	- `F = (x, p) -> F(x, p)` where `p` is the parameter for the continuation
	- `J = (x, p) -> d_xF(x, p)` its associated jacobian. It can be a matrix, a function or a callable struct.
	- `x0` initial guess
	- `contParams` parameters for continuation, with struct `ContinuationPar`
	- `plot = false` whether to plot the solution while computing
	- `printsolution::Function = norm` function used to plot in the continuation curve. It is also used in the way results are saved. It could be `norm` or `x -> x[1]`. This is also useful when saving several huge vectors is not possible for memory reasons for example (on GPU...).
	- `plotsolution::Function = (x; kwargs...) -> nothing` function implementing the plot of the solution.
	- `finaliseSolution::Function = (z, tau, step, contResult) -> true` Function called at the end of each continuation step. Can be used to alter the continuation procedure (stop it by returning false), saving personal data, plotting...
	- `tangentalgo = SecantPred()` controls the algorithm use to predict the tangent along the curve of solutions or the corrector. Can be `NaturalPred`, `SecantPred` or `BorderedPred`.
	- `linearalgo = BorderingBLS()`. Must belong to `[MatrixBLS(), BorderingBLS(), MatrixFreeBLS()]`. Used to control the way the extended linear system associated to the continuation problem is solved.
	- `verbosity ∈ {0,1,2}` controls the amount of information printed during the continuation process.
	- `normC = norm` norm used in the different Newton solves

	The function outputs
	- `contres::ContResult` structure which contains the computed branch
	- `u::BorderedArray` the last solution computed on the branch

	# Method

	## Bordered system of equations

	The pseudo-arclength continuation method solves the equation ``F(x, p) = 0`` (of dimension N) together with the pseudo-arclength constraint ``N(x, p) = \\frac{\\theta}{length(x)} \\langle x - x_0, \\tau_0\\rangle + (1 - \\theta)\\cdot(p - p_0)\\cdot dp_0 - ds = 0``. In practice, the curve ``\\gamma`` is parametrised by ``s`` so that ``\\gamma(s) = (x(s), p(s))`` is a curve of solutions to ``F(x, p)``. This formulation allows to pass turning points (where the implicit theorem fails). In the previous formula, ``(x_0, p_0)`` is a solution for a given ``s_0``, ``(\\tau_0, dp_0)`` is the tangent to the curve at ``s_0``. Hence, to compute the curve of solutions, we need solve an equation of dimension N+1 which is called a Bordered system.

	!!! warning "Parameter `theta`"
	    The parameter `theta` in the struct `ContinuationPar`is very important. It should be tuned for the continuation to work properly especially in the case of large problems where the ``\\langle x - x_0, \\tau_0\\rangle`` component in the constraint might be favoured too much.

	The parameter ds is adjusted internally depending on the number of Newton iterations and other factors. See the function `stepSizeControl` for more information. An important parameter to adjust the magnitude of this adaptation is the parameter `a` in the struct `ContinuationPar`.

	## Algorithm

	The algorithm works as follows:
	0. Start from a known solution ``(x_0, p_0,\\tau_0 ,dp_0)``
	1. **Predictor** set ``(x_1, p_1) = (x_0, p_0) + ds\\cdot (\\tau_0, dp_0)``
	2. **Corrector** solve ``F(x, p)=0,\\ N(x, p)=0`` with a (Bordered) Newton Solver with guess ``(x_1, p_1)``.
	3. **New tangent** Compute ``(\\tau_1, dp_1)``, set ``(x_0, p_0, \\tau_0, dp_0) = (x_1, p_1, \\tau_1, dp_1)`` and return to step 2

	## Natural continuation

	We speak of *natural* continuation when we do not consider the constraint ``N(x, p)=0``. Knowing ``(x_0, p_0)``, we use ``x_0`` as a guess for solvin g ``F(x,p_1)=0`` with ``p_1`` close to ``p_0``. Again, this will fail at Turning points but it can be faster to compute than the constrained case. This is set by the option `tangentalgo = NaturalPred()` in `continuation`.

	## Tangent computation (step 4)
	There are various ways to compute ``(\\tau_1, p_1)``. The first one is called secant and is parametrised by the option `tangentalgo = SecantPred()`. It is computed by ``(\\tau_1, p_1) = (z_1, p_1) - (z_0, p_0)`` and normalised by the norm ``\\|(u, p)\\|^2_\\theta = \\frac{\\theta}{length(u)} \\langle u,u\\rangle + (1 - \\theta)\\cdot p^2``. Another method is to compute ``(\\tau_1, p_1)`` by solving a bordered linear system, see the function `getTangent!` for more information ; it is set by the option `tangentalgo = BorderedPred()`.

	## Bordered linear solver

	When solving the Bordered system ``F(x, p) = 0,\\ N(x, p)=0``, one faces the issue of solving the Bordered linear system ``\\begin{bmatrix} J & a    ; b^T & c\\end{bmatrix}\\begin{bmatrix}X ;  y\\end{bmatrix} =\\begin{bmatrix}R ; n\\end{bmatrix}``. This can be solved in many ways via bordering (which requires two Jacobian inverses), by forming the bordered matrix (which works well for sparse matrices) or by using a full Matrix Free formulation. The choice of method is set by the argument `linearalgo`. Have a look at the struct `linearBorderedSolver` for more information.
	"""
	function continuation(Fhandle,
						Jhandle,
						x0,
						p0::T,
						contParams::ContinuationPar{T, S, E};
						tangentalgo = SecantPred(),
						linearalgo  = BorderingBLS(),
						plot = false,
						printsolution = norm,
						normC = norm,
						plotsolution = (x;kwargs...) -> nothing,
						finaliseSolution = (z, tau, step, contResult) -> true,
						verbosity = 0) where {T, S <: AbstractLinearSolver, E <: AbstractEigenSolver}
		################################################################################################
		(verbosity > 0) && printstyled("#"^50*"\n*********** ArcLengthContinuationNewton *************\n\n", bold = true, color = :red)

		# Get parameters
		@unpack pMin, pMax, maxSteps, newtonOptions = contParams
		epsi = contParams.finDiffEps

		# Check the logic of the parameters
		check!(contParams)

		# Create a bordered linear solver
		linearalgo = @set linearalgo.solver = contParams.newtonOptions.linsolver

		# Filename to save the computations
		filename = "branch-" * string(Dates.now())

		# Converge initial guess
		(verbosity > 0) && printstyled("*********** CONVERGE INITIAL GUESS *************", bold = true, color = :magenta)
		u0, fval, isconverged, it_number = newton(x -> Fhandle(x, p0),
												x -> Jhandle(x, p0),
												x0, newtonOptions, normN = normC)
		@assert isconverged "Newton failed to converge initial guess"
		(verbosity > 0) && (print("\n--> convergence of initial guess = ");printstyled("OK\n", color=:green))
		(verbosity > 0) && println("--> p = $(p0), initial step")

		# Save data and hold general information
		if contParams.computeEigenValues
			# Eigen elements computation
			evsol =  newtonOptions.eigsolver(Jhandle(u0, p0), contParams.nev)
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
		# Initialise continuation
		step = 0
		continuationFailed = false

		# number of iterations for newton correction
		it_number = 0

		# Variables to hold the predictor
		z_pred   = BorderedArray(copy(u0), p0)		# current solution
		tau_pred = BorderedArray(copy(u0), p0)		# tangent predictor
		tau_new  = BorderedArray(copy(u0), p0)		# new tangent

		z_old     = BorderedArray(copy(u_pred), p0)	# previous solution
		tau_old   = BorderedArray(copy(duds), dpds)	# previous tangent

		(verbosity > 0) && println("--> Start continuation from p = ", z_old.p)

		# Main continuation loop
		while (step < maxSteps) & ~continuationFailed & (z_old.p < contParams.pMax) & (z_old.p > contParams.pMin)
			# Predictor: z_pred
			getPredictor!(z_pred, z_old, tau_old, contParams, tangentalgo)
			(verbosity > 0) && println("########################################################################")
			(verbosity > 0) && @printf("Start of Continuation Step %d: Parameter: p1 = %2.4e --> %2.4e\n", step, z_old.p, z_pred.p)
			(length(contRes) > 1 && (verbosity > 0)) && @printf("Step size  = %2.4e --> %2.4e\n", contRes.branch[4, end-1], contParams.ds)

			# Corrector, ie newton correction
			z_new, fval, isconverged, it_number  = corrector(Fhandle, Jhandle,
					z_old, tau_old, z_pred, contParams, tangentalgo,
					linearalgo, normN = normC)

			# Successful step
			if isconverged
				(verbosity > 0) && printstyled("--> Step Converged in $it_number Nonlinear Solver Iterations!\n", color=:green)

				# Get predictor, it only modifies tau_new
				getTangent!(tau_new, z_new, z_old, tau_old, Fhandle, Jhandle, contParams, tangentalgo, verbosity, linearalgo)

				# Output
				push!(contRes.branch, vcat(z_new.p, printsolution(z_new.u), it_number, contParams.ds))

				# Eigenvalues computation
				if contParams.computeEigenValues
					res = computeEigenvalues(contParams, contRes, Jhandle(z_new.u, z_new.p),  step)
					# assess stability of the point
					push!(contRes.stability, mapreduce(x->real(x)<0, *, contRes.eig[end][1]))
					(verbosity > 0) && printstyled(color=:green,"--> Computed ", contParams.nev, " eigenvalues in ", res[end], " iterations, #unstable = ", sum(real.(res[1]) .> 0),"\n")
				end

				# # Detection of codim 1 bifurcation points
				if contParams.detect_fold || contParams.detect_bifurcation
					detectBifucation(contParams, contRes, z_old, tau_old, normC, printsolution, verbosity)
				end

		  		# Plotting
				(plot && mod(step, contParams.plot_every_n_steps) == 0 ) && plotBranchCont(contRes, z_new, contParams, plotsolution)

				# Saving Solution
				if contParams.save
					(verbosity > 0) && printstyled("--> Solving solution in file\n", color=:green)
					saveSolution(filename, z_new.u, z_new.p, step, contRes, contParams)
				end

				# Call user saved finaliseSolution function. If returns false, stop continuation
				!finaliseSolution(z_new, tau_new, step, contRes) && (step = maxSteps)

				# update current solution
				copyto!(z_old, z_new)
				copyto!(tau_old, tau_new)
			else
				(verbosity > 0) && printstyled("Newton correction failed\n", color=:red)
				(verbosity > 0) && println("--> Newton Residuals history = ", fval)
			end

			step += 1
			continuationFailed = stepSizeControl(contParams, isconverged, it_number, tau_old, contRes.branch, verbosity)
		end # End while
		plot && plotBranchCont(contRes, z_old, contParams, plotsolution)

		# We remove the initial guesses which are meaningless
		popfirst!(contRes.bifpoint)

		# return current solution in case the corrector did not converge
		return contRes, z_old, tau_old
	end

	continuation(Fhandle::Function, u0, p0::T, contParams::ContinuationPar{T, S, E}; kwargs...) where {T, S <: AbstractLinearSolver, E <: AbstractEigenSolver} = continuation(Fhandle, (u0, p) -> finiteDifferences(u -> Fhandle(u, p), u0), u0, p0, contParams; kwargs...)
end
