abstract type AbstractPredictor end
abstract type AbstractTangentComputation end

initialize!(state::AbstractContinuationState,
				iter::AbstractContinuationIterable) = initialize!(state, iter, getAlg(iter))

# compute the predictor given the tangent already contained in state
getPredictor!(state::AbstractContinuationState,
				iter::AbstractContinuationIterable) = getPredictor!(state, iter, getAlg(iter))

# update the predictor given the tangent already
# this is used in Bifurcation / Event detection using bisection
updatePredictor!(state::AbstractContinuationState,
				iter::AbstractContinuationIterable) = updatePredictor!(state, iter, getAlg(iter))

# default method
updatePredictor!(state::AbstractContinuationState,
				iter::AbstractContinuationIterable,
				alg::AbstractContinuationAlgorithm) = getPredictor!(state, iter, alg)

corrector!(state::AbstractContinuationState,
			iter::AbstractContinuationIterable; kwargs...) = corrector!(state, iter, getAlg(iter); kwargs...)

stepSizeControl!(state::AbstractContinuationState,
			iter::AbstractContinuationIterable; kwargs...) = stepSizeControl!(state, iter, getAlg(iter); kwargs...)

# used to reset the predictor when locating bifurcations
Base.empty!(alg::AbstractContinuationAlgorithm) = alg
Base.empty!(alg::AbstractTangentComputation) = alg
# Base.copy(::AbstractContinuationAlgorithm) = throw("Not defined. Please define a copy method for your continuation algorithm")

# we need to be able to reset / empty the predictors when locating bifurcation points, when doing automatic branch switching
function Base.empty(alg::AbstractContinuationAlgorithm)
	alg2 = deepcopy(alg)
	empty!(alg2)
	alg2
end

# this is called during initialisation of the continuation method. It can be used to adjust the algo.
update(alg::AbstractContinuationAlgorithm, ::ContinuationPar, _) = alg

# helper functions to update ::ContState when calling the corrector
function _updatefieldButNotSol!(state::AbstractContinuationState,
							sol::NonLinearSolution)
	state.converged = sol.converged
	state.itnewton = sol.itnewton
	state.itlinear = sol.itlineartot
	# record previous solution
	if converged(sol)
		copyto!(state.z_old, state.z)
	end
end
####################################################################################################
function stepSizeControl!(state::AbstractContinuationState,
						iter::AbstractContinuationIterable,
						::AbstractContinuationAlgorithm)
	if ~state.stopcontinuation && stepsizecontrol(state)
		_stepSizeControl!(state, getContParams(iter), iter.verbosity)
	end
end

function _stepSizeControl!(state, contparams::ContinuationPar, verbosity)
	ds = state.ds
	if converged(state) == false
		if  abs(ds) <= contparams.dsmin
			@error "Failure to converge with given tolerances."
			# we stop the continuation
			state.stopcontinuation = true
			return
		end
		dsnew = sign(ds) * max(abs(ds) / 2, contparams.dsmin);
		(verbosity > 0) && printstyled("Halving continuation step, ds=$(dsnew)\n", color=:red)

	else
		# control to have the same number of Newton iterations
		Nmax = contparams.newtonOptions.maxIter
		factor = (Nmax - state.itnewton) / Nmax
		dsnew = ds * (1 + contparams.a * factor^2)
	end

	dsnew = clampDs(dsnew, contparams)

	# we do not stop the continuation
	state.ds = dsnew
	state.stopcontinuation = false
	return
end
