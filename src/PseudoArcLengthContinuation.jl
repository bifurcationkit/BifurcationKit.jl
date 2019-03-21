module PseudoArcLengthContinuation
	using Parameters, Plots, JLD2, Printf, Dates, LinearMaps, Setfield, BlockArrays

	include("LinearSolver.jl")
	include("EigSolver.jl")
	include("LinearBorderSolver.jl")
	include("DeflationOperator.jl")
	include("Newton.jl")
	include("utils.jl")
	include("FoldCont.jl")
	include("HopfCont.jl")
	include("periodicorbit/PeriodicOrbit.jl")

	export	ContinuationPar, ContResult, continuation, continuationFold, continuationHopf, BorderedVector
	export 	NewtonPar, newton, newtonDeflated, newtonPArcLength, newtonFold, newtonHopf
	export  DeflationOperator, DeflatedProblem, DeflatedLinearSolver, scalardM
	export	Default, GMRES_IterativeSolvers, GMRES_KrylovKit,
			Default_eig, Default_eig_sp, eig_IterativeSolvers, eig_KrylovKit, eig_MF_KrylovKit, getEigenVector
	export	FoldPoint, FoldProblemMinimallyAugmented, FoldLinearSolveMinAug, foldPoint
	export	HopfPoint, HopfProblemMinimallyAugmented, HopfLinearSolveMinAug
	export  ShootingProblemTrap, ShootingProblemBE, ShootingProblemMid, PeriodicOrbitLinearSolverMid, PeriodicOrbitTrap
	export plotBranch, plotBranch!

	################################################################################################
	# equation of the arc length constraint
	@inline function arcLengthEq(u, p, du, dp, xi, ds)
		return dottheta(u, du, p, dp, xi) - ds
	end
	################################################################################################
	function corrector(Fhandle, Jhandle, z_old::M, tau_old::M, z_pred::M, contparams, linearalgo = :bordered; normC::Function = norm) where {T, vectype, M<:BorderedVector{vectype, T}}
		if contparams.natural
			res = newton(u -> Fhandle(u, z_pred.p), u -> Jhandle(u, z_pred.p), z_pred.u, contparams.newtonOptions, normN = normC)
			return BorderedVector(res[1], z_pred.p), res[2], res[3], res[4]
		else
			return newtonPseudoArcLength(Fhandle, Jhandle,
								z_old, tau_old, z_pred,
								contparams; linearalgo = linearalgo, normN = normC)
		end
	end
	################################################################################################
	function getPredictor!(z_pred::M, z_old::M, tau::M, contparams) where {T, vectype, M<:BorderedVector{vectype, T}}
		# we perform z_pred = z_old + contparams.ds * tau
		copyto!(z_pred, z_old)
		axpy!(contparams.ds, tau, z_pred)
	end
	################################################################################################
	function getTangentSecant!(tau_new::M, z_new::M, z_old::M, contparams, verbosity) where {T, vectype, M<:BorderedVector{vectype, T}}
		(verbosity > 0) && println("--> predictor = Secant")
		# secant predictor: tau = z_new - z_old; tau *= sign(ds) / normtheta(tau)
		copyto!(tau_new, z_new)
		minus!(tau_new, z_old)
		α = sign(contparams.ds) / normtheta(tau_new, contparams.theta)
		rmul!(tau_new, α)
	end
	################################################################################################
	function getTangentBordered!(tau_new::M, z_new::M, z_old::M, tau_old::M, F, J, contparams, verbosity) where {T, vectype, M<:BorderedVector{vectype, T}}
		(verbosity > 0) && println("--> predictor = Tangent")
		# tangent predictor
		epsi = contparams.finDiffEps
		dFdl = (F(z_old.u, z_old.p + epsi) - F(z_old.u, z_old.p)) / epsi

		# tau = getTangent(J(z_old.u, z_old.p), dFdl, tau_old, contparams.theta, contparams.newtonOptions.linsolve)
		tauu, taup, it = linearBorderedSolver( J(z_old.u, z_old.p), dFdl,
				BorderedVector(tau_old.u * contparams.theta / length(tau_old.u),
				 				tau_old.p * (1 - contparams.theta)),
								0 * z_old.u, 1.0, contparams.theta,
								contparams.newtonOptions.linsolve)
		tau = BorderedVector(tauu, taup)
		b = sign((tau.p) * convert(T, z_new.p - z_old.p))
		α = b * sign(contparams.ds) / normtheta(tau, contparams.theta)
		# tau_new = α * tau
		copyto!(tau_new, tau)
		rmul!(tau_new, α)
	end
	################################################################################################
	function arcLengthScaling(contparams, tau::M, verbosity) where {T, vectype, M<:BorderedVector{vectype, T}}
		g = abs(tau.p * contparams.theta)
		(verbosity > 0) && print("Theta changes from $(contparams.theta) to ")
		if (g > contparams.gMax)
			contparams.theta = contparams.gGoal / tau.p * sqrt( abs(1.0 - g*g) / abs(1.0 - tau.p^2) )
		    if (contparams.theta < contparams.thetaMin)
		      contparams.theta = contparams.thetaMin;
		  end
		end
		print("$(contparams.theta)\n")
		@show g
	end
	################################################################################################
	function stepSizeControl(contparams, converged::Bool, it_number::Int64, tau::M, branch, verbosity) where {T, vectype, M<:BorderedVector{vectype, T}}
		if converged == false
			(verbosity > 0) && abs(contparams.ds) <= contparams.dsmin && (printstyled("*"^80*"\nFailure to converge with given tolerances\n"*"*"^80, color=:red);return true)
			contparams.ds = sign(contparams.ds) * max(abs(contparams.ds) / 2, contparams.dsmin);
			(verbosity > 0) && printstyled("Halving continuation step, ds=$(contparams.ds)\n", color=:red)
		else
			if (length(branch)>1)
				# control to have the same number of Newton iterations
				Nmax = contparams.newtonOptions.maxIter
				factor = (Nmax - it_number) / Nmax
				contparams.ds *= 1 + contparams.a * factor^2
				(verbosity > 0) && @show 1 + contparams.a * factor^2
			end

		end

		# control step to stay between bounds
		if abs(contparams.ds) < contparams.dsmin
			contparams.ds = sign(contparams.ds) * contparams.dsmin
		end

		if abs(contparams.ds) > contparams.dsmax
			contparams.ds = sign(contparams.ds) * contparams.dsmax
		end

		contparams.doArcLengthScaling && arcLengthScaling(contparams, tau, verbosity)
		@assert abs(contparams.ds) >= contparams.dsmin
		return false
	end
	################################################################################################
	function computeEigenvalues(contparams, contResult, J, step)
		nev_ = max(sum( real.(contResult.eig[end][1]) .> 0) + 2, contparams.nev)
		eig_elements = contparams.newtonOptions.eigsolve(J, contparams.nev)
		(mod(step, contparams.save_eig_every_n_steps) == 0 ) && push!(contResult.eig, (eig_elements[1], eig_elements[2], step + 1))
		eig_elements
	end
	################################################################################################
	"""
		continuation(F::Function, J, u0, p0::Real, contParams::ContinuationPar; plot = false, normC = norm, printsolution = norm, plotsolution::Function = (x;kwargs...)->nothing, finaliseSolution::Function = (x, y)-> nothing, linearalgo   = :bordering, verbosity = 2)

	Compute the continuation curve associated to the functional `F` and its jacobian `J`. The parameters are as follows
	- `F = (x, p) -> F(x, p)` where `p` is the parameter for the continuation
	- `J = (x, p) -> d_xF(x, p)` its associated jacobian
	- `u0` initial guess
	- `contParams` parameters for continuation, with type `ContinuationPar`
	- `plot = false` whether to plot the solution while computing
	- `printsolution = norm` function used to plot in the continuation curve, e.g. `norm` or `x -> x[1]`
	- `plotsolution::Function = (x; kwargs...)->nothing` function implementing the plotting of the solution.
	- `finaliseSolution::Function = (z, tau, step, contResult) -> true` Function called at the end of each continuation step. Can be used to alter the continuation step (stop it by returning false) or saving personal data...
	- `linearalgo   = :bordering`. Must belong to `[:bordering, :full]`
	- `verbosity` controls the amount of information printed during the continuation process.
	- 'normC = norm' norm to be used in the different Newton solves

	The function outputs
	- `contres::ContResult` structure which contains the computed branch
	- `u::BorderedVector` the last solution computed on the branch

	# Method

	## Bordered system of equations

	The pseudo arclength continuation method solve the equation ``F(x,p) = 0`` (or dimension N) together with the pseudo-arclength consraint ``N(x, p) = \\frac{\\theta}{length(u)} \\langle x - x_0, \\tau_0\\rangle + (1 - \\theta)\\cdot(p - p_0)\\cdot dp_0 - ds = 0``. In practice, the curve is parametrised by ``s`` so that ``(x(s),p(s))`` is a curve of solution to ``F(x,p)``. This formulation allows to pass turning points (where the implicit theorem fails). In the previous formula, ``(x_0, p_0)`` is a solution for a given ``s_0``, ``(\\tau_0, dp_0)`` is the tangent to the curve at ``s_0``. Hence, to compute the solution curve, we solve an equation of dimension N+1 which is called a Bordered system.

	!!! warning "Parameter `theta`"
	    The parameter `theta` in the type `ContinuationPar`is very important. It should be tuned for the continuation to work properly especially in the case of large problems in which cases the ``\\langle x - x_0, \\tau_0\\rangle`` component in the constraint might be favoured too much.

	The parameter ds is adjusted internally depending on the number of Newton iterations and other factors. See the function `stepSizeControl` for more information. An important parameter to adjust the magnitude of this adaptation is the parameter `a` in the type `ContinuationPar`.

	## Algorithm

	The algorithm works as follows:
	0. Start from a known solution ``(x_0,p_0,\\tau_0,dp_0)``
	1. **Predictor** set ``(x_1,p_1) = (x_0,p_0) + ds\\cdot (\\tau_0,dp_0)``
	2. **Corrector** solve ``F(x,p)=0,\\ N(x,p)=0`` with a (Bordered) Newton Solver.
	3. **New tangent** Compute ``(\\tau_1,dp_1)``, set ``(x_0,p_0,\\tau_0,dp_0)=(x_1,p_1,\\tau_1,dp_1)`` and return to step 2

	## Natural continuation

	We speak of *natural* continuation when we do not consider the constraint ``N(x,p)=0``. Knowing ``(x_0,p_0)``, we use ``x_0`` as a guess for solving ``F(x,p_1)=0`` with ``p_1`` close to ``p_0``. Again, this will fail at Turning Point but it can be faster to compute than the constrained case. This is set by the field `natural` in the type ContinuationPar`

	## Tangent computation (step 4)
	There are various ways to compute ``(\\tau_1,p_1)``. The first one is called secant and is parametrised by the field `secant` in the type `ContinuationPar`. It is computed by ``(\\tau_1,p_1) = (z_1,p_1) - (z_0,p_0)`` and normalised by the norm ``\\|u,p\\|^2_\\theta = \\frac{\\theta}{length(u)} \\langle u,u\\rangle + (1 - \\theta)\\cdot p^2``. If `secant` is set to `false`, another method is use computing ``(\\tau_1,p_1)`` by solving a bordered linear system, see the function `getTangentBordered` for more information.

	## Bordered linear solver

	When solving the Bordered system ``F(x,p) = 0,\\ N(x, p)=0``, one faces the issue of solving the Bordered linear system ``\\begin{bmatrix} J & a    ; b^T & c\\end{bmatrix}\\begin{bmatrix}X ;  y\\end{bmatrix} =\\begin{bmatrix}R ; n\\end{bmatrix}``. This can be solved in many ways via bordering (which requires two Jacobian inverses) or by forming the bordered matrix (which works well for sparse matrices). The choice of method is set by the argument `linearalgo`. Have a look at the function `linearBorderedSolver` for more information.
	"""
	function continuation(Fhandle,
						Jhandle,
						u0,
						p0::Real,
						contParams::ContinuationPar{T, S, E};
						linearalgo   = :bordering,
						plot = false,
						printsolution = norm,
						normC = norm,
						plotsolution = (x;kwargs...) -> nothing,
						finaliseSolution = (z, tau, step, contResult) -> true,
						verbosity = 2) where {T, S <: LinearSolver, E <: EigenSolver}
		################################################################################################
		## Rename parameters
		pMin          = contParams.pMin
		pMax          = contParams.pMax
		maxSteps      = contParams.maxSteps
		epsi          = contParams.finDiffEps
		newtonOptions = contParams.newtonOptions

		# if we chose a natural continuation, we disable to computation of the tangent by a Bordered system and turn to finite differences.
		if contParams.natural
			contParams.secant = true
		end

		if contParams.detect_bifurcation
			contParams.computeEigenValues = true
		end

		# filename to save the computations
		filename = "branch-"*string(Dates.now())

		(verbosity > 0) && printstyled("#"^50*"\n*********** ArcLengthContinuationNewton *************\n\n", bold=true, color=:red)
		## Converge initial guess
		(verbosity > 0) && printstyled("*********** CONVERGE INITIAL GUESS *************", bold=true, color=:magenta)
		u0, fval, exitflag, it_number = newton(x -> Fhandle(x, p0),  u -> Jhandle(u, p0), u0, newtonOptions, normN = normC)
		!exitflag && error("Newton failed to converge initial guess")
		(verbosity > 0) && (print("\n--> convergence of initial guess = ");printstyled("OK\n", color=:green))
		(verbosity > 0) && println("--> p = $(p0), initial step")

		# save data and hold general information
		if contParams.computeEigenValues
			# eigen elements computation
			evsol =  newtonOptions.eigsolve(Jhandle(u0, p0), contParams.nev)
			contRes = ContResult{T, typeof(u0), typeof(evsol[2])}(
				branch = VectorOfArray([vcat(p0, printsolution(u0), it_number, contParams.ds)]),
				bifpoint = [(:none, 0, T(0.), T(0.), u0, u0, 0)],
				n_imag = [0],
				n_unstable = [0],
				eig = [(evsol[1], evsol[2], 0)] )
			# whether the current solution is stable
			contRes.stability[1] = mapreduce(x->real(x)<0, *, evsol[1])
			contRes.n_unstable[1] = mapreduce(x->round(real(x), digits=6) > 0, +, evsol[1])
			if length(evsol[1][1:contRes.n_unstable[1]])>0
				contRes.n_imag[1] = mapreduce(x->round(imag(x), digits=6) > 0, +, evsol[1][1:contRes.n_unstable[1]])
			else
				contRes.n_imag[1] = 0
			end
		else
			contRes = ContResult{T, typeof(u0), Array{Complex{T}, 2}}(
					branch = VectorOfArray([vcat(p0, printsolution(u0), it_number, contParams.ds)]),
					bifpoint = [(:none, 0, T(0.), T(0.), u0, u0, 0)],
					n_imag = [0],
					n_unstable = [0],
					eig = [([Complex{T}(1)], zeros(Complex{T}, 2, 2), 0)])
		end
		finaliseSolution(u0, u0, 0, contRes)


		(verbosity > 0) && printstyled("\n******* COMPUTING INITIAL TANGENT *************", bold=true, color=:magenta)
		u_pred, fval, exitflag, it_number = newton(x -> Fhandle(x, p0 + contParams.ds / T(50)),u -> Jhandle(u, p0 + contParams.ds / T(50)), u0, newtonOptions, normN = normC)
		!exitflag && error("Newton failed to converge for the computation of the initial tangent")
		(verbosity > 0) && (print("\n--> convergence of initial guess = ");printstyled("OK\n\n", color=:green))
		(verbosity > 0) && println("--> p = $(p0 + contParams.ds/50), initial step (bis)")
		finaliseSolution(u_pred, u_pred, 1, contRes)

		duds = (u_pred - u0) / (contParams.ds / T(50));	dpds = T(1.0)
		α = normtheta(duds, dpds, contParams.theta)
		@assert α > 0 "Error, α = 0, cannot scale first tangent vector"
		duds = duds / α; dpds = dpds / α

		## Initialise continuation
		step = 0
		continuationFailed = false
		# number of iterations for newton correction
		it_number = 0

		# variables to hold the predictor
		z_pred   = BorderedVector(copy(u0), p0)
		tau_pred = BorderedVector(copy(u0), p0)
		tau_new  = BorderedVector(copy(u0), p0)

		z_old     = BorderedVector(copy(u_pred), p0)
		tau_old   = BorderedVector(copy(duds), dpds)

		(verbosity > 0) && println("--> Start continuation from p = ", z_old.p)
		## Main continuation loop
		while (step < maxSteps) & ~continuationFailed & (z_old.p < contParams.pMax) & (z_old.p > contParams.pMin)
			# predictor: z_pred
			getPredictor!(z_pred, z_old, tau_old, contParams)
			(verbosity > 0) && println("########################################################################")
			(verbosity > 0) && @printf("Start of Continuation Step %d : Parameter: p1 = %2.4e from %2.4e\n", step, z_pred.p, z_old.p)
			(length(contRes.branch[4, :])>1 && (verbosity > 0)) && @printf("Current step size  = %2.4e   Previous step size = %2.4e\n", contParams.ds, contRes.branch[4, end-1])

		  	# Corrector
			z_new, fval, exitflag, it_number  = corrector(Fhandle, Jhandle,
					 z_old, tau_old, z_pred, contParams,
					 linearalgo, normC = normC)

			# Successful step
		  	if exitflag == true
				(verbosity > 0) && printstyled("--> Step Converged in $it_number Nonlinear Solver Iterations!\n", color=:green)
				# get predictor
				if contParams.secant
					getTangentSecant!(tau_new, z_new, z_old, contParams, verbosity)
				else
					getTangentBordered!(tau_new, z_new, z_old, tau_old, Fhandle, Jhandle, contParams, verbosity)
				end

		  		# Output
				push!(contRes.branch, vcat(z_new.p, printsolution(z_new.u), it_number, contParams.ds))

				# Detection of codim 1 bifurcation points
				# this should be there before the old z is re-written
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

				# call user defined finaliseSolution function. If returns false, stop continuation
				!finaliseSolution(z_old.u, tau_old.u, step, contRes) && (step = maxSteps)
			else
				(verbosity > 0) && printstyled("Newton correction failed\n", color=:red)
				(verbosity > 0) && println("--> Newton Residuals history = ", fval)
		  	end

			step += 1
			continuationFailed = stepSizeControl(contParams, exitflag, it_number, tau_old, contRes.branch, verbosity)
	  end # while
	  plot && plotBranchCont(contRes, z_old, contParams, plotsolution)

	  # we remove the initial guesses that are meaningless
	  popfirst!(contRes.bifpoint)
	  return contRes, z_old, tau_old
	end

	continuation(Fhandle::Function, u0, p0::Real, contParams::ContinuationPar{T, S, E}; kwargs...) where {T, S <: LinearSolver, E <: EigenSolver} = continuation(Fhandle, (u0, p)->finiteDifferences(u->Fhandle(u, p), u0), u0, p0, contParams; kwargs...)

end
