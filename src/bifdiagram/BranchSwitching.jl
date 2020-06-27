import Base: getproperty, propertynames
"""
A Branch is a structure which encapsulates the result of a continuation run on a branch bifurcating from a bifurcation point.
"""
struct Branch{T <: Union{ContResult, Vector{ContResult}}, Tbp}
	γ::T
	bp::Tbp
end

from(br::Branch) = br.bp
from(br::Vector{Branch}) = from(br[1])
show(io::IO, br::Branch{T, Tbp}) where {T <: ContResult, Tbp} = show(io, br.γ, " from $(type(br.bp)) bifurcation point.")
show(io::IO, br::Branch{T, Tbp}) where {T <: Vector{ContResult}, Tbp} = for γ in br.γ; println(io,"\n"); show(io, γ, " from $(type(br.bp)) bifurcation point."); end

# extend the getproperty for easy manipulation of a Branch
# for example, it allows to use the plot recipe for ContResult as is
function getproperty(br::Branch, s::Symbol)
	if s in (:γ, :bp)
		getfield(br, s)
	else
		getproperty(br.γ, s)
	end
end

propertynames(br::Branch) = ((:γ, :bp)..., propertynames(br.γ)...)
####################################################################################################
"""
This function is the analog of [`continuation`](@ref) when the two first points on the branch are passed (instead of a single one). Hence `x0` is the first point with parameter `par0` and `x1` is the second point with parameter `set(par0, lens, p1)`.
"""
function continuation(Fhandle, Jhandle, x0::Tv, par0, x1::Tv, p1::Real, lens::Lens, contParams::ContinuationPar; linearAlgo = BorderingBLS(), kwargs...) where {Tv}
	# Create a bordered linear solver using the newton linear solver provided by the user
	_linearAlgo = @set linearAlgo.solver = contParams.newtonOptions.linsolver

	# create an iterable
	it = PALCIterable(Fhandle, Jhandle, x0, par0, lens, contParams, _linearAlgo; kwargs...)
	@warn "J'ai du inverser les deux points"
	return continuation(it, x1, p1, x0, get(par0, lens))
end

function continuation(it::PALCIterable, x0, p0::Real, x1, p1::Real)
	## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	# The result type of this method
	# is not known at compile time so we
	# need a function barrier to resolve it
	#############################################

	# we compute the cache for the continuation, i.e. state::PALCStateVariables
	# In this call, we also compute the initial point on the branch (and its stability) and the initial tangent
	state, _ = iterate(it, x0, p0, x1, p1)

	# variable to hold the result from continuation, i.e. a branch
	contRes = ContResult(it, state)

	# perform the continuation
	return continuation!(it, state, contRes)
end


"""
$(SIGNATURES)

Automatic branch switching at branch points based on a computation of the normal form. More information is provided in [Branch switching](@ref). An example of use is provided in [A generalized Bratu–Gelfand problem in two dimensions](@ref).

# Arguments
- `F, dF, d2F, d3F`: function `(x,p) -> F(x,p)` and its differentials `(x,p,dx) -> d1F(x,p,dx)`, `(x,p,dx1,dx2) -> d2F(x,p,dx1,dx2)`...
- `br` branch result from a call to [`continuation`](@ref)
- `ind_bif` index of the bifurcation point in `br` from which you want to branch from
- `optionsCont` options for the call to [`continuation`](@ref)

# Optional arguments
- `Jt` associated jacobian transpose, it should be implemented in an efficient manner. For matrix-free methods, `transpose` is not readily available and the user must provide a dedicated method. In the case of sparse based jacobian, `Jt` should not be passed as it is computed internally more efficiently, i.e. it avoid recomputing the jacobian as it would be if you pass `Jt = (x, p) -> transpose(dF(x, p))`.
- `δ` used internally to compute derivatives w.r.t the parameter `p`.
- `δp` used to specify a particular guess for the parameter on the bifurcated branch which is otherwise determined by `optionsCont.ds`. This allows to use a step larger than `optionsCont.dsmax`.
- `ampfactor = 1` factor which alter the amplitude of the bifurcated solution. Useful to magnify the bifurcated solution when the bifurcated branch is very steep.
- `nev` number of eigenvalues to be computed to get the right eigenvector
- `issymmetric` whether the Jacobian is Symmetric, avoid computing the left eigenvectors.
- `usedeflation = true` whether to use nonlinear deflation (see [Deflated problems](@ref)) to help finding the guess on the bifurcated branch
- `kwargs` optional arguments to be passed to [`continuation`](@ref), the regular `continuation` one.
"""
function continuation(F, dF, d2F, d3F, br::ContResult, ind_bif::Int, optionsCont::ContinuationPar ; Jt = nothing, δ = 1e-8, δp = nothing, ampfactor = 1, nev = optionsCont.nev, issymmetric = false, usedeflation = false, Teigvec = vectortype(br), kwargs...)
	# The usual branch switching algorithm is described in Keller. Numerical solution of bifurcation and nonlinear eigenvalue problems. We do not use this one but compute the Lyapunov-Schmidt decomposition instead and solve the polynomial equation instead.

	if kerneldim(br, ind_bif) > 1
		@info "kernel dimension = $(kerneldim(br, ind_bif))"
		return multicontinuation(F, dF, d2F, d3F, br, ind_bif, optionsCont; Jt = Jt, δ = δ, δp = δp, ampfactor = ampfactor, nev = nev, issymmetric = issymmetric, usedeflation = usedeflation, kwargs...)
	end

	verbose = get(kwargs, :verbosity, 0) > 0 ? true : false

	@assert br.type == :Equilibrium "Error! This bifurcation type is not handled.\n Branch point from $(br.type)"
	@assert br.bifpoint[ind_bif].type == :bp "Error! This bifurcation type is not handled.\n Branch point from $(br.bifpoint[ind_bif].type)"

	# compute predictor for point on new branch
	ds = isnothing(δp) ? optionsCont.ds : δp
	Ty = typeof(ds)

	# compute the normal form of the branch point
	bifpoint = computeNormalForm1d(F, dF, d2F, d3F, br, ind_bif; Jt = Jt, δ = δ, nev = nev, verbose = verbose, issymmetric = issymmetric, Teigvec = Teigvec)

	# compute predictor for a point on new branch
	pred = predictor(bifpoint, ds; verbose = verbose, ampfactor = Ty(ampfactor))
	if isnothing(pred); return nothing, nothing; end

	verbose && printstyled(color = :green, "\n--> Start branch switching. \n--> Bifurcation type = ", type(bifpoint), "\n----> newp = ", pred.p, ", δp = ", br.bifpoint[ind_bif].param - pred.p, "\n")

	if usedeflation
		verbose && println("\n----> Compute point on the current branch with nonlinear deflation...")
		optn = optionsCont.newtonOptions
		bifpt = br.bifpoint[ind_bif]
		# find the bifurcated branch using nonlinear deflation
		solbif, _, flag, _ = newton(F, dF, bifpt.x, pred.x, set(br.params, br.param_lens, pred.p), optn; kwargs...)[1]
		copyto!(pred.x, solbif)

	end

	# perform continuation
	branch, u, tau =  continuation(F, dF, pred.x, set(br.params, br.param_lens, pred.p), br.bifpoint[ind_bif].x, br.bifpoint[ind_bif].param,br.param_lens, optionsCont; kwargs...)
	return Branch(branch, bifpoint), u, tau
end

continuation(F, dF, d2F, d3F, br::Branch, ind_bif::Int, optionsCont::ContinuationPar ; kwargs...) = continuation(F, dF, d2F, d3F, getContResult(br), ind_bif, optionsCont ; kwargs...)


function multicontinuation(F, dF, d2F, d3F, br::ContResult, ind_bif::Int, optionsCont::ContinuationPar ; Jt = nothing, δ = 1e-8, δp = nothing, ampfactor = 1, nev = optionsCont.nev, issymmetric = false, usedeflation = false, Teigvec = vectortype(br), kwargs...)

	verbose = get(kwargs, :verbosity, 0) > 0 ? true : false

	bpnf = computeNormalForm(F, dF, d2F, d3F, br, ind_bif; Jt = Jt, δ = δ, nev = nev, verbose = verbose, issymmetric = issymmetric, Teigvec = Teigvec)

	# dimension of the kernel
	n = length(bpnf.ζ)

	# compute predictor for point on new branch
	ds = isnothing(δp) ? optionsCont.ds : δp |> abs
	Ty = typeof(ds)

	# find zeros for the normal on each side of the bifurcation point
	function getRootsNf(_ds)
		deflationOp = DeflationOperator(2.0, (x, y) -> dot(x, y), 1.0, [zeros(n)])
		failures = 0
		# we allow for 10 failures of nonlinear deflation
		while failures < 10
			outdef1, _, flag, _ = newton((x, p) -> bpnf(Val(:reducedForm),x, p[1]), rand(n), [_ds], NewtonPar(maxIter = 50), deflationOp)
			flag && push!(deflationOp, outdef1)
			~flag && (failures += 1)
		end
		return deflationOp.roots
	end
	rootsNFp =  getRootsNf(ds)
	rootsNFm =  getRootsNf(-ds)
	println("\n--> BS from Non simple branch point")
	printstyled(color=:green, "--> we find $(length(rootsNFm)) (resp. $(length(rootsNFp))) roots on the left (resp. right) of the bifurcation point (Reduced equation).\n")

	# attempting to convert the guesses from the normal form into true zeros of F
	par = br.params
	defOpp = DeflationOperator(2.0, dot, 1.0, Vector{typeof(bpnf.x0)}())

	optn = optionsCont.newtonOptions
	cbnewton = get(kwargs, :callbackN, (x, f, J, res, iteration, itlinear, optionsN; kwgs...) -> true)
	@show cbnewton
	for xsol in rootsNFp
		@show 1
		solbif, _, flag, _ = newton(F, dF, bpnf(xsol, ds), set(par, br.param_lens, bpnf.p + ds), setproperties(optn; maxIter = 10optn.maxIter, verbose = true), defOpp; callback = cbnewton)
		flag && push!(defOpp, solbif)
	end
	defOpm = DeflationOperator(2.0, dot, 1.0, Vector{typeof(bpnf.x0)}())
	for xsol in rootsNFm
		solbif, _, flag, _ = newton(F, dF, bpnf(xsol, ds), set(par, br.param_lens, bpnf.p - ds), setproperties(optn; maxIter = 15optn.maxIter, verbose = false), defOpm)
		flag && push!(defOpm, solbif)
	end
	printstyled(color=:green, "--> we find $(length(defOpm)) (resp. $(length(defOpp))) roots on the left (resp. right) of the bifurcation point.\nprintstyled(color=:green, ")

	# compute the different branches
	function _continue(_sol, _dp, _ds)
		continuation(F, dF, _sol, set(par, br.param_lens, bpnf.p + _dp), br.param_lens, (@set optionsCont.ds = _ds); kwargs...)
	end

	branches = ContResult[]
	for id in 2:length(defOpm)
		# br, _, _ = _continue(defOpm[id], -ds, ds); push!(branches, br)
		br, _, _ = _continue(defOpm[id], -ds, -ds); push!(branches, br)
	end

	for id in 2:length(defOpp)
		br, _, _ = _continue(defOpp[id], ds, ds); push!(branches, br)
		# br, _, _ = _continue(defOpp[id], ds, -ds); push!(branches, br)
	end

	return branches, (before = defOpm, after = defOpp)
end

multicontinuation(F, dF, d2F, d3F, br::Branch, ind_bif::Int, optionsCont::ContinuationPar ; kwargs...) = multicontinuation(F, dF, d2F, d3F, getContResult(br), ind_bif, optionsCont ; kwargs...)
