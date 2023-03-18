@enum MoorePenroseLS direct=1 pInv=2 iterative=3
"""
	Moore-Penrose predictor / corrector

Moore-Penrose continuation algorithm.

Additional information is available on the [website](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/MooreSpence/).

# Constructors

`alg = MoorePenrose()`

`alg = MoorePenrose(tangent = PALC())`

# Fields

$(TYPEDFIELDS)
"""
struct MoorePenrose{T, Tls <: AbstractLinearSolver} <: AbstractContinuationAlgorithm
	"Tangent predictor, example `PALC()`"
	tangent::T
	"Use a direct linear solver. Can be BifurcationKit.direct, BifurcationKit.pInv or BifurcationKit.iterative"
	method::MoorePenroseLS
	"(Bordered) linear solver"
	ls::Tls
end
# important for bisection algorithm, switch on / off internal adaptive behavior
internalAdaptation!(alg::MoorePenrose, swch::Bool) = internalAdaptation!(alg.tangent, swch)

"""
$(SIGNATURES)
"""
function MoorePenrose(;tangent = PALC(), method = direct, ls = nothing)
	if ~(method == iterative)
		ls = isnothing(ls) ? DefaultLS() : ls
	else
		if isnothing(ls)
			if tangent isa PALC
				ls = tangent.bls
			else
				ls = MatrixBLS()
			end
		end
	end
	return MoorePenrose(tangent, method, ls)
end

getPredictor(alg::MoorePenrose) = getPredictor(alg.tangent)
getLinsolver(alg::MoorePenrose) = getLinsolver(alg.tangent)

function Base.empty!(alg::MoorePenrose)
	empty!(alg.tangent)
	alg
end

function update(alg0::MoorePenrose, contParams::ContinuationPar, linearAlgo)
	tgt = update(alg0.tangent, contParams, linearAlgo)
	alg = @set alg0.tangent = tgt
	if isnothing(linearAlgo)
		if hasproperty(alg.ls, :solver) && isnothing(alg.ls.solver)
			return @set alg.ls.solver = contParams.newtonOptions.linsolver
		end
	else
		return @set alg.ls = linearAlgo
	end
	alg
end


initialize!(state::AbstractContinuationState,
						iter::AbstractContinuationIterable,
						alg::MoorePenrose, nrm = false) = initialize!(state, iter, alg.tangent, nrm)

function getPredictor!(state::AbstractContinuationState,
						iter::AbstractContinuationIterable,
						alg::MoorePenrose, nrm = false)
	(iter.verbosity > 0) && println("Predictor:  MoorePenrose")
	# we just compute the tangent
	getPredictor!(state, iter, alg.tangent, nrm)
end

updatePredictor!(state::AbstractContinuationState,
						iter::AbstractContinuationIterable,
						alg::MoorePenrose,
						nrm = false) = updatePredictor!(state, iter, alg.tangent, nrm)

# corrector based on natural formulation
function corrector!(state::AbstractContinuationState,
				it::AbstractContinuationIterable,
				algo::MoorePenrose;
				kwargs...)
	if state.z_pred.p <= it.contParams.pMin || state.z_pred.p >= it.contParams.pMax
		state.z_pred.p = clampPredp(state.z_pred.p, it)
		return corrector!(state, it, Natural(); kwargs...)
	end
	sol = newtonMoorePenrose(it, state; normN = it.normC, callback = it.callbackN, kwargs...)

	# update fields
	_updatefieldButNotSol!(state, sol)

	# update solution
	if converged(sol)
		copyto!(state.z, sol.u)
	end

	return true
end

function newtonMoorePenrose(iter::AbstractContinuationIterable,
					state::AbstractContinuationState;
					normN = norm,
					callback = cbDefault, kwargs...)
	prob = iter.prob
	par = getParams(prob)
	paramlens = getLens(iter)
	contparams = getContParams(iter)
	dotθ = iter.dotθ
	T = eltype(iter)

	@unpack method = iter.alg

	z0 = getSolution(state)
	τ0 = state.τ
	z_pred = state.z_pred
	ds = state.ds; θ = state.θ

	@unpack tol, maxIter, verbose = contparams.newtonOptions
	@unpack finDiffEps, pMin, pMax = contparams
	linsolver = iter.alg.ls

	# Initialise iterations
	x = _copy(z_pred.u)
	p = z_pred.p
	x_pred = _copy(x)

	# Initialise residuals
	res_f = residual(prob, x, set(par, paramlens, p))

	dX = _copy(res_f) # copy(res_f)
	# dFdp = (F(x, p + finDiffEps) - res_f) / finDiffEps
	dFdp = _copy(residual(prob, x, set(par, paramlens, p + finDiffEps)))
	minus!(dFdp, res_f); rmul!(dFdp, T(1) / finDiffEps)

	res     = normN(res_f)
	resHist = [res]

	# iterations count
	it = 0

	# total number of linear iterations
	itlinear = 0
	itlineartot = 0

	verbose && displayIteration(it, res)
	line_step = true

	compute = callback((;x, res_f, res, it, contparams, p, resHist, z0); fromNewton = false, kwargs...)

	X = BorderedArray(x, p)
	if linsolver isa AbstractIterativeLinearSolver || (method == iterative)
		ϕ = _copy(τ0)
		rmul!(ϕ,  T(1) / norm(ϕ))
	end

	while (res > tol) && (it < maxIter) && line_step && compute
		it += 1
		# dFdp = (F(x, p + ϵ) - F(x, p)) / ϵ)
		copyto!(dFdp, residual(prob, x, set(par, paramlens, p + finDiffEps)))
		minus!(dFdp, res_f); rmul!(dFdp, T(1) / finDiffEps)

		# compute jacobian
		J = jacobian(prob, x, set(par, paramlens, p))
		if method == direct || method == pInv
			@debug "Moore-Penrose direct/pInv"
			Jb = hcat(J, dFdp)
			if method == direct
				dx, flag, converged = linsolver(Jb, res_f)
			else
				# pinv(Array(Jb)) * res_f seems to work better than the following
				dx = LinearAlgebra.pinv(Array(Jb)) * res_f; flag = true;
			end
			x .-= @view dx[1:end-1]
			p -= dx[end]
			itlinear = 1
		else
			@debug "Moore-Penrose Iterative"
			# A = hcat(J, dFdp); A = vcat(A, ϕ')
			# X .= X .- A \ vcat(res_f, 0)
			# x .= X[1:end-1]; p = X[end]
			du, dup, flag, itlinear1 = linsolver(J, dFdp, ϕ.u, ϕ.p, res_f, zero(T), one(T), one(T))
			minus!(x, du)
			p -= dup
			verbose && displayIteration(it, nothing, itlinear1)
		end

		p = clamp(p, pMin, pMax)
		res_f .= residual(prob, x, set(par, paramlens, p))
		res = normN(res_f)

		if method == iterative
			# compute jacobian
			J = jacobian(prob, x, set(par, paramlens, p))
			copyto!(dFdp, residual(prob, x, set(par, paramlens, p + finDiffEps)))
			minus!(dFdp, res_f); rmul!(dFdp, T(1) / finDiffEps)
			# A = hcat(J, dFdp); A = vcat(A, ϕ')
			# ϕ .= A \ vcat(zero(x),1)
			u, up, flag, itlinear2 = linsolver(J, dFdp, ϕ.u, ϕ.p, zero(x), one(T), one(T), one(T))
			~flag && @debug "Linear solver for (J-iω) did not converge."
			ϕ.u .= u; ϕ.p = up
			# rmul!(ϕ,  T(1) / norm(ϕ))
			itlinear = (itlinear1 .+ itlinear2)
		end
		push!(resHist, res)

		verbose && displayIteration(it, res, itlinear)

		# shall we break the loop?
		compute = callback((;x, res_f, J, res, it, itlinear, contparams, p, resHist, z0); fromNewton = false, kwargs...)
	end
	verbose && displayIteration(it, res, 0, true) # display last line of the table
	flag = (resHist[end] < tol) & callback((;x, res_f, nothing, res, it, contparams, p, resHist, z0); fromNewton = false, kwargs...)
	return NonLinearSolution(BorderedArray(x, p), prob, resHist, flag, it, itlineartot)
end
