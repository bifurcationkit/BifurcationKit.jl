"""
$(SIGNATURES)

This function is the analog of [`continuation`](@ref) when the two first points on the branch are passed (instead of a single one). Hence `x0` is the first point on the branch (with palc `s=0`) with parameter `par0` and `x1` is the second point with parameter `set(par0, lens, p1)`.
"""
function continuation(Fhandle, Jhandle, x0::Tv, par0, x1::Tv, p1::Real, lens::Lens, contParams::ContinuationPar; linearAlgo = BorderingBLS(), kwargs...) where Tv
	# Create a bordered linear solver using the newton linear solver provided by the user
	_linearAlgo = @set linearAlgo.solver = contParams.newtonOptions.linsolver
	# check the sign of ds
	dsfactor = sign(p1 - get(par0, lens))
	# create an iterable
	_contParams = @set contParams.ds = abs(contParams.ds) * dsfactor
	it = ContIterable(Fhandle, Jhandle, x0, par0, lens, _contParams, _linearAlgo; kwargs...)
	return continuation(it, x0, get(par0, lens), x1, p1)
end

function continuation(it::ContIterable, x0, p0::Real, x1, p1::Real)
	## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	# The result type of this method
	# is not known at compile time so we
	# need a function barrier to resolve it
	#############################################

	# we compute the cache for the continuation, i.e. state::ContState
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
- `F, dF, d2F, d3F`: function `(x, p) -> F(x, p)` and its differentials `(x, p, dx) -> d1F(x, p, dx)`, `(x, p, dx1, dx2) -> d2F(x, p, dx1, dx2)`...
- `br` branch result from a call to [`continuation`](@ref)
- `ind_bif` index of the bifurcation point in `br` from which you want to branch from
- `optionsCont` options for the call to [`continuation`](@ref)

# Optional arguments
- `Jt` associated jacobian transpose, it should be implemented in an efficient manner. For matrix-free methods, `transpose` is not readily available and the user must provide a dedicated method. In the case of sparse based jacobian, `Jt` should not be passed as it is computed internally more efficiently, i.e. it avoid recomputing the jacobian as it would be if you pass `Jt = (x, p) -> transpose(dF(x, p))`.
- `δ` used internally to compute derivatives w.r.t the parameter `p`.
- `δp` used to specify a particular guess for the parameter on the bifurcated branch which is otherwise determined by `optionsCont.ds`. This allows to use a step larger than `optionsCont.dsmax`.
- `ampfactor = 1` factor which alter the amplitude of the bifurcated solution. Useful to magnify the bifurcated solution when the bifurcated branch is very steep.
- `nev` number of eigenvalues to be computed to get the right eigenvector
- `issymmetric` whether the Jacobian is Symmetric, avoid computing the left eigenvectors in the computation of the reduced equation.
- `usedeflation = true` whether to use nonlinear deflation (see [Deflated problems](@ref)) to help finding the guess on the bifurcated branch
- `kwargs` optional arguments to be passed to [`continuation`](@ref), the regular `continuation` one.

!!! tip "Advanced use"
    In the case of a very large model and use of special hardware (GPU, cluster), we suggest to discouple the computation of the reduced equation, the predictor and the bifurcated branches. Have a look at `methods(BifurcationKit.multicontinuation)` to see how to call these versions. It has been tested on GPU with very high memory pressure.
"""
function continuation(F, dF, d2F, d3F, br::ContResult, ind_bif::Int, optionsCont::ContinuationPar ; Jt = nothing, δ = 1e-8, δp = nothing, ampfactor = 1, nev = optionsCont.nev, issymmetric = false, usedeflation = false, Teigvec = vectortype(br), scaleζ = norm, verbosedeflation = false, maxIterDeflation = min(50, 15optionsCont.newtonOptions.maxIter), perturb = identity, kwargs...)
	# The usual branch switching algorithm is described in Keller. Numerical solution of bifurcation and nonlinear eigenvalue problems. We do not use this one but compute the Lyapunov-Schmidt decomposition instead and solve the polynomial equation instead.

	if kerneldim(br, ind_bif) > 1
		return multicontinuation(F, dF, d2F, d3F, br, ind_bif, optionsCont; Jt = Jt, δ = δ, δp = δp, nev = nev, issymmetric = issymmetric, usedeflation = usedeflation, scaleζ = scaleζ, verbosedeflation = verbosedeflation, maxIterDeflation = maxIterDeflation, perturb = perturb, kwargs...)
	end

	verbose = get(kwargs, :verbosity, 0) > 0 ? true : false

	@assert br.type == :Equilibrium "Error! This bifurcation type is not handled.\n Branch point from $(br.type)"
	@assert br.bifpoint[ind_bif].type == :bp "Error! This bifurcation type is not handled.\n Branch point from $(br.bifpoint[ind_bif].type)"

	# compute predictor for point on new branch
	ds = isnothing(δp) ? optionsCont.ds : δp
	Ty = typeof(ds)

	# compute the normal form of the branch point
	bifpoint = computeNormalForm1d(F, dF, d2F, d3F, br, ind_bif; Jt = Jt, δ = δ, nev = nev, verbose = verbose, issymmetric = issymmetric, Teigvec = Teigvec, scaleζ = scaleζ)

	# compute predictor for a point on new branch
	pred = predictor(bifpoint, ds; verbose = verbose, ampfactor = Ty(ampfactor))
	if isnothing(pred); return nothing, nothing; end

	verbose && printstyled(color = :green, "\n--> Start branch switching. \n--> Bifurcation type = ", type(bifpoint), "\n----> newp = ", pred.p, ", δp = ", br.bifpoint[ind_bif].param - pred.p, "\n")

	if usedeflation
		verbose && println("\n----> Compute point on the current branch with nonlinear deflation...")
		optn = optionsCont.newtonOptions
		bifpt = br.bifpoint[ind_bif]
		# find the bifurcated branch using nonlinear deflation
		solbif, _, flag, _ = newton(F, dF, bifpt.x, pred.x, set(br.params, br.param_lens, pred.p), setproperties(optn; verbose = verbose = verbosedeflation); kwargs...)[1]
		copyto!(pred.x, solbif)
	end

	# perform continuation
	branch, u, tau =  continuation(F, dF,
			bifpoint.x0, bifpoint.params,	# first point on the branch
			pred.x, pred.p,					# second point on the branch
			br.param_lens, optionsCont; kwargs...)
	return Branch(branch, bifpoint), u, tau
end

# same but for a Branch
continuation(F, dF, d2F, d3F, br::Branch, ind_bif::Int, optionsCont::ContinuationPar ; kwargs...) = continuation(F, dF, d2F, d3F, getContResult(br), ind_bif, optionsCont ; kwargs...)

"""
$(SIGNATURES)

Automatic branch switching at branch points based on a computation of the normal form. More information is provided in [Branch switching](@ref). An example of use is provided in [A generalized Bratu–Gelfand problem in two dimensions](@ref).

# Arguments
- `F, dF, d2F, d3F`: function `(x, p) -> F(x, p)` and its differentials `(x, p, dx) -> d1F(x, p, dx)`, `(x, p, dx1, dx2) -> d2F(x, p, dx1, dx2)`...
- `br` branch result from a call to [`continuation`](@ref)
- `ind_bif` index of the bifurcation point in `br` from which you want to branch from
- `optionsCont` options for the call to [`continuation`](@ref)

# Optional arguments
- `Jt` associated jacobian transpose, it should be implemented in an efficient manner. For matrix-free methods, `transpose` is not readily available and the user must provide a dedicated method. In the case of sparse based jacobian, `Jt` should not be passed as it is computed internally more efficiently, i.e. it avoid recomputing the jacobian as it would be if you pass `Jt = (x, p) -> transpose(dF(x, p))`.
- `δ` used internally to compute derivatives w.r.t the parameter `p`.
- `δp` used to specify a particular guess for the parameter on the bifurcated branch which is otherwise determined by `optionsCont.ds`. This allows to use a step larger than `optionsCont.dsmax`.
- `ampfactor = 1` factor which alter the amplitude of the bifurcated solution. Useful to magnify the bifurcated solution when the bifurcated branch is very steep.
- `nev` number of eigenvalues to be computed to get the right eigenvector
- `issymmetric` whether the Jacobian is Symmetric, avoid computing the left eigenvectors in the computation of the reduced equation.
- `verbosedeflation = true` whether to display the nonlinear deflation iterations (see [Deflated problems](@ref)) to help finding the guess on the bifurcated branch
- `scaleζ` norm used to normalise eigenbasis when computing the reduced equation
- `Teigvec` type of the eigenvector. Useful when `br` was loaded from a file and this information was lost
- `ζs` basis of the kernel
- `kwargs` optional arguments to be passed to [`continuation`](@ref), the regular `continuation` one.

!!! tip "Advanced use"
    In the case of a very large model and use of special hardware (GPU, cluster), we suggest to discouple the computation of the reduced equation, the predictor and the bifurcated branches. Have a look at `methods(BifurcationKit.multicontinuation)` to see how to call these versions. It has been tested on GPU with very high memory pressure.
"""
function multicontinuation(F, dF, d2F, d3F, br::ContResult, ind_bif::Int, optionsCont::ContinuationPar ; Jt = nothing, δ = 1e-8, δp = nothing, nev = optionsCont.nev, issymmetric = false, usedeflation = false, Teigvec = vectortype(br), ζs = nothing, verbosedeflation = false, scaleζ = norm, kwargs...)

	verbose = get(kwargs, :verbosity, 0) > 0 ? true : false

	bpnf = computeNormalForm(F, dF, d2F, d3F, br, ind_bif; Jt = Jt, δ = δ, nev = nev, verbose = verbose, issymmetric = issymmetric, Teigvec = Teigvec, ζs = ζs, scaleζ = scaleζ)

	return multicontinuation(F, dF, br, bpnf, optionsCont; nev = nev, issymmetric = issymmetric, usedeflation = usedeflation, Teigvec = Teigvec, ζs = ζs, δp = δp, verbosedeflation = verbosedeflation, scaleζ = scaleζ, kwargs...)
end

function multicontinuation(F, dF, br::BranchResult, bpnf::NdBranchPoint, optionsCont::ContinuationPar ; δp = nothing, perturb = identity, kwargs...)

	verbose = get(kwargs, :verbosity, 0) > 0 ? true & get(kwargs, :verbosedeflation, true) : false

	# compute predictor for point on new branch
	ds = isnothing(δp) ? optionsCont.ds : δp |> abs

	# get prediction from solving the reduced equation
	rootsNFm, rootsNFp = predictor(bpnf, ds;  verbose = verbose, perturb = perturb)

	return multicontinuation(F, dF, br, bpnf, (before = rootsNFm, after = rootsNFp), optionsCont; δp = δp, kwargs...)
end

function multicontinuation(F, dF, br::BranchResult, bpnf::NdBranchPoint, solfromRE, optionsCont::ContinuationPar ; Jt = nothing, δ = 1e-8, δp = nothing, nev = optionsCont.nev, issymmetric = false, usedeflation = false, Teigvec = vectortype(br), ζs = nothing, verbosedeflation = false, scaleζ = norm, maxIterDeflation = min(50, 15optionsCont.newtonOptions.maxIter), lsdefop = DeflatedLinearSolver(), kwargs...)

	# compute predictor for point on new branch
	ds = isnothing(δp) ? optionsCont.ds : δp |> abs
	dscont = abs(optionsCont.ds)

	rootsNFm = solfromRE.before
	rootsNFp = solfromRE.after

	# attempting now to convert the guesses from the normal form into true zeros of F
	par = bpnf.params
	optn = optionsCont.newtonOptions
	cbnewton = get(kwargs, :callbackN, (x, f, J, res, iteration, itlinear, optionsN; kwgs...) -> true)

	println("--> Looking for solutions after the bifurcation point...")
	defOpp = DeflationOperator(2.0, dot, 1.0, Vector{typeof(bpnf.x0)}())
	for xsol in rootsNFp
		solbif, _, flag, _ = newton(F, dF, bpnf(xsol, ds), set(par, br.param_lens, bpnf.p + ds), setproperties(optn; maxIter = maxIterDeflation, verbose = verbosedeflation), defOpp, lsdefop; callback = cbnewton)
		flag && push!(defOpp, solbif)
	end

	println("--> Looking for solutions before the bifurcation point...")
	defOpm = DeflationOperator(2.0, dot, 1.0, Vector{typeof(bpnf.x0)}())
	for xsol in rootsNFm
		solbif, _, flag, _ = newton(F, dF, bpnf(xsol, ds), set(par, br.param_lens, bpnf.p - ds), setproperties(optn; maxIter = maxIterDeflation, verbose = verbosedeflation), defOpm, lsdefop; callback = cbnewton)
		flag && push!(defOpm, solbif)
	end
	printstyled(color=:green, "--> we find $(length(defOpm)) (resp. $(length(defOpp))) roots on the left (resp. right) of the bifurcation point.\n")

	# compute the different branches
	function _continue(_sol, _dp, _ds)
		# needed to reset the tangent algorithm in case fields are used
		emptypredictor!(get(kwargs, :tangentAlgo, nothing))
		println("#"^50)
		continuation(F, dF,
			bpnf.x0, par,		# first point on the branch
			_sol, bpnf.p + _dp, # second point on the branch
			br.param_lens, (@set optionsCont.ds = _ds); kwargs...)
		# continuation(F, dF, _sol, set(par, br.param_lens, bpnf.p + _dp), br.param_lens, (@set optionsCont.ds = _ds); kwargs...)
	end

	branches = Branch[]
	for id in 2:length(defOpm)
		br, = _continue(defOpm[id], -ds, -dscont); push!(branches, Branch(br, bpnf))
		# br, = _continue(defOpm[id], -ds, dscont); push!(branches, Branch(br, bpnf))
	end

	for id in 2:length(defOpp)
		br, = _continue(defOpp[id], ds, dscont); push!(branches, Branch(br, bpnf))
		# br, = _continue(defOpp[id], ds, -dscont); push!(branches, Branch(br, bpnf))
	end

	return branches, (before = defOpm, after = defOpp), (before = rootsNFm, after = rootsNFp)
end

# same but for a Branch
multicontinuation(F, dF, d2F, d3F, br::Branch, ind_bif::Int, optionsCont::ContinuationPar ; kwargs...) = multicontinuation(F, dF, d2F, d3F, getContResult(br), ind_bif, optionsCont ; kwargs...)
