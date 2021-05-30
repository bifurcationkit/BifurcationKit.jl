"""
	options = ContinuationPar(dsmin = 1e-4,...)

Returns a variable containing parameters to affect the `continuation` algorithm used to solve `F(x,p) = 0`.

# Arguments
- `dsmin, dsmax` are the minimum, maximum arclength allowed value. It controls the density of points in the computed branch of solutions.
- `ds` is the initial arclength.
- `theta` is a parameter in the arclength constraint. It is very **important** to tune it. See the docs of [`continuation`](@ref).
- `pMin, pMax` allowed parameter range for `p`
- `maxSteps` maximum number of continuation steps
- `newtonOptions::NewtonPar`: options for the Newton algorithm
- `saveToFile = false`: save to file. A name is automatically generated.
- `saveSolEveryStep::Int64 = 0` at which continuation steps do we save the current solution`
- `plotEveryStep = 3`

## Handling eigen elements, their computation is triggered by the argument `detectBifurcation` (see below)
- `nev = 3` number of eigenvalues to be computed. It is automatically increased to have at least `nev` unstable eigenvalues. To be set for proper  bifurcation detection. See [Detection of bifurcation points](@ref) for more informations.
- `saveEigEveryStep = 1`	record eigen vectors every specified steps. **Important** for memory limited resource, *e.g.* GPU.
- `saveEigenvectors	= true`	**Important** for memory limited resource, *e.g.* GPU.

## Handling bifurcation detection
- `precisionStability = 1e-10` lower bound on the real part of the eigenvalues to test for stability of equilibria and periodic orbits
- `detectFold = true` detect Fold bifurcations? It is a useful option although the detection of Fold is cheap. Indeed, it may happen that there is a lot of Fold points and this can saturate the memory in memory limited devices (e.g. on GPU)
- `detectBifurcation::Int` ∈ {0, 1, 2, 3} If set to 0, nothing is done. If set to 1, the eigen-elements are computed. If set to 2, the bifurcations points are detected during the continuation run, but not located precisely. If set to 3, a bisection algorithm is used to locate the bifurcations points (slower). The possibility to switch off detection is a useful option. Indeed, it may happen that there are a lot of bifurcation points and this can saturate the memory of memory limited devices (e.g. on GPU)
- `dsminBisection` dsmin for the bisection algorithm for locating bifurcation points
- `nInversion` number of sign inversions in bisection algorithm
- `maxBisectionSteps` maximum number of bisection steps
- `tolBisectionEigenvalue` tolerance on real part of eigenvalue to detect bifurcation points in the bisection steps

## Handling `ds` adaptation (see [`continuation`](@ref) for more information)
- `a  = 0.5` aggressiveness factor. It is used to adapt `ds` in order to have a number of newton iterations per continuation step roughly constant. The higher `a` is, the larger the step size `ds` is changed at each continuation step.
- `thetaMin = 1.0e-3` minimum value of `theta`
- `doArcLengthScaling` trigger further adaptation of `theta`

## Handling event detection
- `detectEvent::Int` ∈ {0, 1, 2} If set to 0, nothing is done. If set to 1, the event locations are seek during the continuation run, but not located precisely. If set to 2, a bisection algorithm is used to locate the event (slower).
- `tolParamBisectionEvent` tolerance on parameter to locate event

## Misc
- `finDiffEps::T  = 1e-9` ε used in finite differences computations
- `detectLoop` [WORK IN PROGRESS] detect loops in the branch and stop the continuation

!!! tip "Mutating"
    For performance reasons, we decided to use an immutable structure to hold the parameters. One can use the package `Setfield.jl` to drastically simplify the mutation of different fields. See tutorials for more examples.
"""
@with_kw struct ContinuationPar{T, S <: AbstractLinearSolver, E <: AbstractEigenSolver}
	# parameters for arclength continuation
	dsmin::T	= 1e-3
	dsmax::T	= 1e-1
	@assert dsmax >= dsmin "You must provide a valid interval (ordered) for ds"
	ds::T		= 1e-2;		@assert dsmax >= abs(ds);	@assert abs(ds) >= dsmin
	@assert dsmin > 0 "The interval for ds must be positive"
	@assert dsmax > 0

	# parameters for scaling arclength step size
	theta::T					= 0.5 		# parameter in the dot product used for the extended system
	doArcLengthScaling::Bool  	= false
	gGoal::T					= 0.5
	gMax::T						= 0.8
	thetaMin::T					= 1.0e-3
	a::T						= 0.5  		# aggressiveness factor
	tangentFactorExponent::T 	= 1.5

	# parameters bound
	pMin::T	= -1.0
	pMax::T	=  1.0; 			@assert pMax >= pMin

	# maximum number of continuation steps
	maxSteps::Int64  = 100

	# Newton solver parameters
	finDiffEps::T  = 1e-9 					# constant for finite differences
	newtonOptions::NewtonPar{T, S, E} = NewtonPar()
	η::T = 150.								# parameter to estimate tangent at first point

	saveToFile::Bool = false 				# save to file?
	saveSolEveryStep::Int64 = 0				# what steps do we save the current solution

	# parameters for eigenvalues
	nev::Int64 = 3 							# number of eigenvalues
	saveEigEveryStep::Int64 = 1				# what steps do we keep the eigenvectors
	saveEigenvectors::Bool	= true			# useful options because if puts a high memory pressure

	plotEveryStep::Int64 = 10

	# handling bifurcation points
	precisionStability::T = 1e-10			# lower bound for stability of equilibria and periodic orbits
	detectFold::Bool = true 				# detect fold points?
	detectBifurcation::Int64 = 0			# detect bifurcation points?
	dsminBisection::T = 1e-16				# dsmin for the bisection algorithm when locating bifurcation points
	nInversion::Int64 = 2					# number of sign inversions in bisection algorithm
	maxBisectionSteps::Int64 = 15			# maximum number of bisection steps
	tolBisectionEigenvalue::T = 1e-16 		# tolerance on real part of eigenvalue to detect bifurcation points in the bisection steps. Must be small otherwise Shooting and friends will fail detecting bifurcations.

	# handling event detection
	detectEvent::Int64 = 0				# event location
	tolParamBisectionEvent::T = 1e-16	# tolerance on value of parameter

	@assert iseven(nInversion) "The option `nInversion` number must be odd"
	@assert detectBifurcation <= 3 "The option `detectBifurcation` must belong to {0,1,2,3}"
	@assert detectEvent <= 2 "The option `detectEvent` must belong to {0,1,2}"
	@assert (detectBifurcation > 1 && detectEvent == 0) || (detectBifurcation <= 1 && detectEvent >= 0)  "One of these options must be disabled detectBifurcation = $detectBifurcation and detectEvent = $detectEvent"
	@assert tolBisectionEigenvalue >= 0 "The option `tolBisectionEigenvalue` must be positive"
	detectLoop::Bool = false				# detect if the branch loops
end

@inline computeEigenElements(cp::ContinuationPar) = cp.detectBifurcation > 0
@inline computeEigenvalues(cp::ContinuationPar) = cp.detectBifurcation > 0
@inline computeEigenvectors(cp::ContinuationPar) = computeEigenvalues(cp) * cp.saveEigenvectors
