"""
	Multiple Tangent continiation algorithm.

$(TYPEDFIELDS)

# Constructor(s)

	Multiple(pred, x0, α, n)

	Multiple(x0, α, n)
"""
@with_kw mutable struct Multiple{T <: Real, Tvec, Tpred <: PALC} <: AbstractContinuationAlgorithm
	"Tangent predictor used"
	alg::Tpred = PALC()

	"Save the current tangent"
	τ::Tvec

	"Damping in Newton iterations, 0 < α < 1"
	α::T

	"Number of predictors"
	nb::Int64

	"Index of the largest converged predictor"
	currentind::Int64 = 0

	"Index for lookup in residual history"
	pmimax::Int64 = 1

	"Maximum index for lookup in residual history"
	imax::Int64 = 4

	"Factor to increase ds upon successful step"
	dsfact::T = 1.5
end
Multiple(alg, x0, α::T, nb) where T = Multiple(alg = alg, τ = BorderedArray(x0, T(0)), α = α, nb = nb)
Multiple(x0, α, nb) = Multiple(PALC(), x0, α, nb)
Base.empty!(alg::Multiple) = (alg.currentind = 1; alg.pmimax = 1)
getLinsolver(alg::Multiple) = getLinsolver(alg.alg)

# callback for newton
function (algcont::Multiple)(state; kwargs...)
	resHist = get(state, :resHist, nothing)
	iteration = get(state, :iteration, 0)
	if algcont.currentind > 1
		return iteration - algcont.pmimax > 0 ? resHist[end] <= algcont.α * resHist[end-algcont.pmimax] : true
	end
	return true
end

function initialize!(state::AbstractContinuationState,
						iter::AbstractContinuationIterable,
						algo::Multiple, nrm = false)
	return initialize!(state, iter, algo.alg, nrm)
end

function getPredictor!(state::AbstractContinuationState,
	 					iter::AbstractContinuationIterable,
						algo::Multiple,
						nrm = false)
	# we just compute the tangent
	getPredictor!(state, iter, algo.alg, nrm)
	return nothing
end

function corrector!(_state::AbstractContinuationState, it::AbstractContinuationIterable,
		algo::Multiple, linearalgo = MatrixFreeBLS(); kwargs...)
	verbose = it.verbosity
	# we create a copy of the continuation cache
	state = copy(_state)
	@unpack ds = state
	(verbose > 1) && printstyled(color=:magenta, "──"^35*"\n   ┌─MultiplePred tangent predictor\n")
	# we combine the callbacks for the newton iterations
	cb = (state; k...) -> callback(state; k...) & alg(state; k...)
	# note that z_pred already contains ds * τ, hence ii=0 corresponds to this case
	for ii in algo.nb:-1:1
		(verbose > 1) && printstyled(color=:magenta, "   ├─ i = $ii, s(i) = $(ii*ds), converged = [")
		algo.currentind = ii # record the current index
		zpred = _copy(state.z_pred)
		axpy!(ii * ds, algo.τ, zpred)
		copyto!(state.z_pred, zpred)
		# we restore the original callback if it reaches the usual case ii == 0
		corrector!(state, it, algo.alg; kwargs...)
		if verbose > 1
			if converged(state)
				printstyled("YES", color=:green)
			else
				printstyled(" NO", color=:red)
			end
			printstyled("]\n", color=:magenta)
		end
		if converged(state) || ii == 1 # for i==1, we return the result anyway
			copyto!(_state, state)
			return true
		end
	end
	return true
end

function stepSizeControl(ds, θ, contparams::ContinuationPar, converged::Bool, it_newton_number::Int, tau::M, alg::Multiple, verbosity) where {T, vectype, M <: BorderedArray{vectype, T}}
	if converged == false
		dsnew = ds
		if abs(ds) < (1 + alg.nb) * contparams.dsmin
			if alg.pmimax < alg.imax
				(verbosity > 0) && printstyled("--> Increase pmimax\n", color=:red)
				alg.pmimax += 1
			else
				(verbosity > 0) && @error "Failure to converge with given tolerances"
				# we stop the continuation
				return ds, θ, true
			end
		else
			@error "--> Decrease ds"
			dsnew = ds / (1 + alg.nb)
			(verbosity > 0) && printstyled("Halving continuation step, ds = $(dsnew)\n", color=:red)
		end
	else # the newton correction has converged
		dsnew = ds
		if alg.currentind == alg.nb && abs(ds) * alg.dsfact <= contparams.dsmax
			(verbosity > 0) && @show dsnew
			# println("--> Increase ds")
			dsnew = ds *  alg.dsfact
		end
	end

	dsnew = clampDs(dsnew, contparams)

	return dsnew, θ, false
end
