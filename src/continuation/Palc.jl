"""
$(TYPEDEF)

$(TYPEDFIELDS)

This parametric type allows to define a new dot product from the one saved in `dt::dot`. More precisely:

	dt(u1, u2, p1::T, p2::T, theta::T) where {T <: Real}

computes, the weigthed dot product ``\\langle (u_1,p_1), (u_2,p_2)\\rangle_\\theta = \\theta \\Re \\langle u_1,u_2\\rangle  +(1-\\theta)p_1p_2`` where ``u_i\\in\\mathbb R^N``. The ``\\Re`` factor is put to ensure a real valued result despite possible complex valued arguments.

!!! info "Info"
    This is used in the pseudo-arclength constraint with the dot product ``\\frac{1}{N} \\langle u_1, u_2\\rangle,\\quad u_i\\in\\mathbb R^N``
"""
struct DotTheta{Tdot, Ta}
	"dot product used in pseudo-arclength constraint"
	dot::Tdot
	"Linear operator associated with dot product, i.e. dot(x, y) = <x, Ay>, where <,> is the standard dot product on R^N. You must provide an inplace function which evaluates A. For example `x -> rmul!(x, 1/length(x))`."
	apply!::Ta
end

DotTheta() = DotTheta( (x, y) -> dot(x, y) / length(x), x -> rmul!(x, 1/length(x))   )
DotTheta(dt) = DotTheta(dt, nothing)

# we restrict the type of the parameters because for complex problems, we still want the parameter to be real
(dt::DotTheta)(u1, u2, p1::T, p2::T, θ::T) where {T <: Real} = real(dt.dot(u1, u2) * θ + p1 * p2 * (one(T) - θ))

# Implementation of the norm associated to DotTheta
# we restrict the type of the parameters because for complex problems, we still want the parameter to be real
(dt::DotTheta)(u, p::T, θ::T) where T = sqrt(dt(u, u, p, p, θ))

(dt::DotTheta)(a::BorderedArray{vec, T}, b::BorderedArray{vec, T}, θ::T) where {vec, T} = dt(a.u, b.u, a.p, b.p, θ)
(dt::DotTheta)(a::BorderedArray{vec, T}, θ::T) where {vec, T} = dt(a.u, a.p, θ)
####################################################################################################
# equation of the arc length constraint
arcLengthEq(dt::DotTheta, u, p, du, dp, θ, ds) = dt(u, du, p, dp, θ) - ds
####################################################################################################
"""
$(TYPEDEF)

Pseudo-arclength continuation algorithm.

Additional information is available on the [website](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/PALC/).

# Fields

$(TYPEDFIELDS)

# Associated methods
"""
@with_kw struct PALC{Ttang <: AbstractTangentComputation, Tbls <: AbstractLinearSolver, T} <: AbstractContinuationAlgorithm
	"Tangent predictor, must be a subtype of `AbstractTangentComputation`. For example `Secant()` or `Bordered()`"
	tangent::Ttang = Secant()
	"[internal]"
	bothside::Bool = false
	"Bordered linear solver used to invert the jacobian of the newton bordered problem. It is also used to compute the tangent for the predictor `Bordered()`"
	bls::Tbls = MatrixBLS()
	# parameters for scaling arclength step size
	doArcLengthScaling::Bool  	= false
	gGoal::T					= 0.5
	gMax::T						= 0.8
	θMin::T						= 1.0e-3

	@assert ~(predictor isa ConstantPredictor) "You cannot use a constant predictor with PALC"
end
getLinsolver(alg::PALC) = alg.bls
getPredictor(alg::PALC) = alg.tangent

function Base.empty!(alg::PALC)
	empty!(alg.tangent)
	alg
end

function update(alg::PALC, contParams::ContinuationPar, linearAlgo)
	if isnothing(linearAlgo)
		if isnothing(alg.bls.solver)
			return @set alg.bls.solver = contParams.newtonOptions.linsolver
		end
	else
		return @set alg.bls = linearAlgo
	end
	alg
end

function initialize!(state::AbstractContinuationState,
						iter::AbstractContinuationIterable,
						alg::PALC,
						nrm = false)
	# for the initialisation step, we do not use a Bordered predictor which fails at bifurcation points
	getTangent!(state, iter, Secant())
	# we want to start at (u0, p0), not at (u1, p1)
	copyto!(state.z, state.z_old)
	# then update the predictor state.z_pred
	addTangent!(state::AbstractContinuationState, nrm)
end

function getPredictor!(state::AbstractContinuationState,
						iter::AbstractContinuationIterable,
						alg::PALC,
						nrm = false)
	# we first compute the tangent
	getTangent!(state, iter, alg.tangent)
	# then update the predictor state.z_pred
	addTangent!(state::AbstractContinuationState, nrm)
end

# this function only mutates z_pred
# the nrm argument allows to just the increment z_pred.p by ds
function addTangent!(state::AbstractContinuationState, nrm = false)
	# we perform z_pred = z + ds * τ
	copyto!(state.z_pred, state.z)
	ds = state.ds
	ρ = nrm ? ds / state.τ.p : ds
	axpy!(ρ, state.τ, state.z_pred)
end

updatePredictor!(state::AbstractContinuationState,
						iter::AbstractContinuationIterable,
						alg::PALC,
						nrm = false) = addTangent!(state, nrm)

function corrector!(state::AbstractContinuationState, it::AbstractContinuationIterable, alg::PALC; kwargs...)
	if state.z_pred.p <= it.contParams.pMin || state.z_pred.p >= it.contParams.pMax
		state.z_pred.p = clampPredp(state.z_pred.p, it)
		return corrector!(state, it, Natural(); kwargs...)
	end
	sol = newtonPALC(it, state; linearbdalgo = alg.bls, normN = it.normC, callback = it.callbackN, kwargs...)

	# update fields
	_updatefieldButNotSol!(state, sol)

	# update solution
	if converged(sol)
	copyto!(state.z, sol.u)
	end

	return true
end

###############################################
"""
	Secant Tangent predictor
"""
struct Secant <: AbstractTangentComputation end

# This function is used for initialisation in iterateFromTwoPoints
function _secantComputation!(τ::M, z₁::M, z₀::M, it::AbstractContinuationIterable, ds, θ, verbosity) where {T, vectype, M <: BorderedArray{vectype, T}}
	(verbosity > 0) && println("Predictor:  Secant")
	# secant predictor: τ = z₁ - z₀; tau *= sign(ds) / normtheta(tau)
	copyto!(τ, z₁)
	minus!(τ, z₀)
	α = sign(ds) / it.dotθ(τ, θ)
	rmul!(τ, α)
end

getTangent!(state::AbstractContinuationState,
			iter::AbstractContinuationIterable,
			algo::Secant) = _secantComputation!(state.τ, state.z, state.z_old, iter, state.ds, state.θ, iter.verbosity)
###############################################
"""
	Bordered Tangent predictor
"""
struct Bordered <: AbstractTangentComputation end

# tangent computation using Bordered system
# τ is the tangent prediction found by solving
# ┌                           ┐┌  ┐   ┌   ┐
# │      J            dFdl    ││τu│ = │ 0 │
# │  θ/N ⋅ τ.u     (1-θ)⋅τ.p  ││τp│   │ 1 │
# └                           ┘└  ┘   └   ┘
# it is updated inplace
function getTangent!(state::AbstractContinuationState,
					it::AbstractContinuationIterable,
					tgtalgo::Bordered)
	(it.verbosity > 0) && println("Predictor: Bordered")
	ϵ = it.contParams.finDiffEps
	τ = state.τ
	θ = state.θ
	T = eltype(it)

	# dFdl = (F(z.u, z.p + ϵ) - F(z.u, z.p)) / ϵ
	dFdl = residual(it.prob, state.z.u, setParam(it, state.z.p + ϵ))
	minus!(dFdl, residual(it.prob, state.z.u, setParam(it, state.z.p)))
	rmul!(dFdl, 1/ϵ)

	# compute jacobian
	J = jacobian(it.prob, state.z.u, setParam(it, state.z.p))

	# extract tangent as solution of the above bordered linear system
	τu, τp, flag, itl = getLinsolver(it)( it, state,
										J, dFdl,
										0*state.z.u, one(T)) # Right-hand side

	~flag && @warn "Linear solver failed to converge in tangent computation with type ::BorderedPred"

	# we scale τ in order to have ||τ||_θ = 1 and sign <τ, τold> = 1
	α = one(T) / sqrt(it.dotθ(τu, τu, τp, τp, θ))
	α *= sign(it.dotθ(τ.u, τu, τ.p, τp, θ))

	copyto!(τ.u, τu)
	τ.p = τp
	rmul!(τ, α)
end
####################################################################################################
"""
	Polynomial Tangent predictor

$(TYPEDFIELDS)

# Constructor(s)

	Polynomial(pred, n, k, v0)

	Polynomial(n, k, v0)

- `n` order of the polynomial
- `k` length of the last solutions vector used for the polynomial fit
- `v0` example of solution to be stored. It is only used to get the `eltype` of the tangent!!
"""
mutable struct Polynomial{T <: Real, Tvec, Ttg <: AbstractTangentComputation} <: AbstractTangentComputation
	"Order of the polynomial"
	n::Int64

	"Length of the last solutions vector used for the polynomial fit"
	k::Int64

	"Matrix for the interpolation"
	A::Matrix{T}

	"Algo for tangent when polynomial predictor is not possible"
	tangent::Ttg

	"Vector of solutions"
	solutions::CircularBuffer{Tvec}

	"Vector of parameters"
	parameters::CircularBuffer{T}

	"Vector of arclengths"
	arclengths::CircularBuffer{T}

	"Coefficients for the polynomials for the solution"
	coeffsSol::Vector{Tvec}

	"Coefficients for the polynomials for the parameter"
	coeffsPar::Vector{T}

	"Update the predictor by adding the last point (x, p)? This can be disabled in order to just use the polynomial prediction. It is useful when the predictor is called mutiple times during bifurcation detection using bisection."
	update::Bool
end

function Polynomial(pred, n, k, v0)
	@assert n<k "k must be larger than the degree of the polynomial"
	Polynomial(n,k,zeros(eltype(v0), k, n+1), pred,
		CircularBuffer{typeof(v0)}(k),  # solutions
		CircularBuffer{eltype(v0)}(k),  # parameters
		CircularBuffer{eltype(v0)}(k),  # arclengths
		Vector{typeof(v0)}(undef, n+1), # coeffsSol
		Vector{eltype(v0)}(undef, n+1), # coeffsPar
		true)
end
Polynomial(n, k, v0) = Polynomial(Secant(), n, k, v0)

isready(ppd::Polynomial) = length(ppd.solutions) >= ppd.k

function Base.empty!(ppd::Polynomial)
	empty!(ppd.solutions); empty!(ppd.parameters); empty!(ppd.arclengths);
end

function getStats(polypred::Polynomial)
	Sbar = sum(polypred.arclengths) / length(polypred.arclengths)
	σ = sqrt(sum(x->(x-Sbar)^2, polypred.arclengths ) / length(polypred.arclengths))
	# return 0,1
	return Sbar, σ
end

function (polypred::Polynomial)(ds::T) where T
	sbar, σ = getStats(polypred)
	s = polypred.arclengths[end] + ds
	snorm = (s-sbar)/σ
	# vector of powers of snorm
	S = Vector{T}(undef, polypred.n+1); S[1] = T(1)
	for jj = 1:polypred.n; S[jj+1] = S[jj] * snorm; end
	p = sum(S .* polypred.coeffsPar)
	x = sum(S .* polypred.coeffsSol)
	return x, p
end

function updatePred!(polypred::Polynomial)
	Sbar, σ = getStats(polypred)
	# re-scale the previous arclengths so that the Vandermond matrix is well conditioned
	Ss = (polypred.arclengths .- Sbar) ./ σ
	# construction of the Vandermond Matrix
	polypred.A[:, 1] .= 1
	for jj in 1:polypred.n; polypred.A[:, jj+1] .= polypred.A[:, jj] .* Ss; end
	# invert linear system for least square fitting
	B = (polypred.A' * polypred.A) \ polypred.A'
	mul!(polypred.coeffsSol, B, polypred.solutions)
	mul!(polypred.coeffsPar, B, polypred.parameters)
	return true
end

function getTangent!(state::AbstractContinuationState,
					it::AbstractContinuationIterable,
					polypred::Polynomial)
	(it.verbosity > 0) && println("Predictor: Polynomial")
	ds = state.ds
	# do we update the predictor with last converged point?
	if polypred.update
		if length(polypred.arclengths) == 0
			push!(polypred.arclengths, ds)
		else
			push!(polypred.arclengths, polypred.arclengths[end]+ds)
		end
		push!(polypred.solutions, state.z.u)
		push!(polypred.parameters, state.z.p)
	end

	if ~isready(polypred) || ~polypred.update
		return getTangent!(state, it, polypred.tangent)
	else
		return polypred.update ? updatePred!(polypred) : true
	end
end


####################################################################################################
function arcLengthScaling(θ, alg, τ::M, verbosity) where {M <: BorderedArray}
	# the arclength scaling algorithm is based on Salinger, Andrew G, Nawaf M Bou-Rabee,
	# Elizabeth A Burroughs, Roger P Pawlowski, Richard B Lehoucq, Louis Romero, and Edward D
	# Wilkes. “LOCA 1.0 Library of Continuation Algorithms: Theory and Implementation Manual,
	# ” March 1, 2002. https://doi.org/10.2172/800778.
	thetanew = θ
	g = abs(τ.p * θ)
	(verbosity > 0) && print("Theta changes from $(θ) to ")
	if (g > alg.gMax)
		thetanew = alg.gGoal / τ.p * sqrt( abs(1.0 - g^2) / abs(1.0 - τ.p^2) )
		if (thetanew < alg.thetaMin)
		  thetanew = alg.thetaMin;
		end
	end
	(verbosity > 0) && print("$(thetanew)\n")
	return thetanew
end

####################################################################################################
"""
This is the classical Newton-Krylov solver used to solve `F(x, p) = 0` together
with the scalar condition `n(x, p) ≡ θ ⋅ <x - x0, τx> + (1-θ) ⋅ (p - p0) * τp - n0 = 0`. This makes a problem of dimension N + 1.

The initial guess for the newton method is located in `state.z_pred`
"""
function newtonPALC(iter::AbstractContinuationIterable,
					state::AbstractContinuationState;
					normN = norm,
					callback = cbDefault,
					kwargs...)
	prob = iter.prob
	par = getParams(prob)
	paramlens = getLens(iter)
	contparams = getContParams(iter)
	dotθ = iter.dotθ
	T = eltype(iter)

	z0 = getSolution(state)
	τ0 = state.τ
	@unpack z_pred, ds, θ = state

	@unpack tol, maxIter, verbose, α, αmin, linesearch = contparams.newtonOptions
	@unpack finDiffEps, pMin, pMax = contparams
	linsolver = getLinsolver(iter)

	# we record the damping parameter
	α0 = α

	# N = θ⋅dot(x - z0.u, τ0.u) + (1 - θ)⋅(p - z0.p)⋅τ0.p - ds
	N(u, _p) = arcLengthEq(dotθ, minus(u, z0.u), _p - z0.p, τ0.u, τ0.p, θ, ds)
	normAC(resf, resn) = max(normN(resf), abs(resn))

	# Initialise iterations
	x = _copy(z_pred.u)
	p = z_pred.p
	x_pred = _copy(x)

	res_f = residual(prob, x, set(par, paramlens, p));  res_n = N(x, p)

	dX = _copy(res_f)
	dp = zero(T)
	up = zero(T)

	# dFdp = (F(x, p + finDiffEps) - res_f) / finDiffEps
	dFdp = _copy(residual(prob, x, set(par, paramlens, p + finDiffEps)))
	minus!(dFdp, res_f)						# dFdp = dFdp - res_f
	rmul!(dFdp, one(T) / finDiffEps)

	res     = normAC(res_f, res_n)
	resHist = [res]
	it = 0
	itlineartot = 0

	verbose && displayIteration(it, res)
	line_step = true

	compute = callback((;x, res_f, res, it, contparams, p, resHist, options = (;linsolver)); fromNewton = false, kwargs...)

	while (res > tol) && (it < maxIter) && line_step && compute
		# dFdp = (F(x, p + ϵ) - F(x, p)) / ϵ)
		copyto!(dFdp, residual(prob, x, set(par, paramlens, p + finDiffEps)))
			minus!(dFdp, res_f); rmul!(dFdp, one(T) / finDiffEps)

		# compute jacobian
		J = jacobian(prob, x, set(par, paramlens, p))
		# solve linear system
		# ┌            ┐┌  ┐   ┌     ┐
		# │ J     dFdp ││u │ = │res_f│
		# │ τ0.u  τ0.p ││up│   │res_n│
		# └            ┘└  ┘   └     ┘
		u, up, flag, itlinear = linsolver(iter, state, J, dFdp, res_f, res_n)
		itlineartot += sum(itlinear)

		if linesearch
			line_step = false
			while !line_step && (α > αmin)
				# x_pred = x - α * u
				copyto!(x_pred, x); axpy!(-α, u, x_pred)

				p_pred = p - α * up
				copyto!(res_f, residual(prob, x_pred, set(par, paramlens, p_pred)))

				res_n  = N(x_pred, p_pred)
				res = normAC(res_f, res_n)

				if res < resHist[end]
					if (res < resHist[end] / 4) && (α < 1)
						α *= 2
					end
					line_step = true
					copyto!(x, x_pred)

					# p = p_pred
					p  = clamp(p_pred, pMin, pMax)
				else
					α /= 2
				end
			end
			# we put back the initial value
			α = α0
		else
			minus!(x, u)
			p = clamp(p - up, pMin, pMax)

			copyto!(res_f, residual(prob, x, set(par, paramlens, p)))

			res_n  = N(x, p); res = normAC(res_f, res_n)
		end

		push!(resHist, res)
		it += 1

		verbose && displayIteration(it, res, itlinear)

		# shall we break the loop?
		compute = callback((;x, res_f, J, res, it, itlinear, contparams, z0, p, resHist, options = (;linsolver)); fromNewton = false, kwargs...)
	end
	verbose && displayIteration(it, res, 0, true) # display last line of the table
	flag = (resHist[end] < tol) & callback((;x, res_f, res, it, contparams, p, resHist, options = (;linsolver)); fromNewton = false, kwargs...)
	return NonLinearSolution(BorderedArray(x, p), prob, resHist, flag, it, itlineartot)
end
