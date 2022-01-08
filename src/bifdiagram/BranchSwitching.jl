"""
$(SIGNATURES)

This function is the analog of [`continuation`](@ref) when the two first points on the branch are passed (instead of a single one). Hence `x0` is the first point on the branch (with palc `s=0`) with parameter `par0` and `x1` is the second point with parameter `set(par0, lens, p1)`.
"""
function continuation(F, J, x0::Tv, par0, x1::Tv, p1::Real, lens::Lens, contParams::ContinuationPar; linearAlgo = BorderingBLS(), kwargs...) where Tv
	# Create a bordered linear solver using the newton linear solver provided by the user
	# dont modify it if the user passed its own version
	if isnothing(linearAlgo.solver)
	_linearAlgo = @set linearAlgo.solver = contParams.newtonOptions.linsolver
	else
		_linearAlgo = linearAlgo
	end
	# check the sign of ds
	dsfactor = sign(p1 - get(par0, lens))
	# create an iterable
	_contParams = @set contParams.ds = abs(contParams.ds) * dsfactor
	it = ContIterable(F, J, x0, par0, lens, _contParams, _linearAlgo; kwargs...)
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
	state, _ = iterateFromTwoPoints(it, x0, p0, x1, p1)

	# variable to hold the result from continuation, i.e. a branch
	contRes = ContResult(it, state)

	# perform the continuation
	return continuation!(it, state, contRes)
end

"""
$(SIGNATURES)

Automatic branch switching at branch points based on a computation of the normal form. More information is provided in [Branch switching](@ref Branch-switching-page). An example of use is provided in [2d generalized Bratu–Gelfand problem](@ref).

# Arguments
- `F, dF, d2F, d3F`: function `(x, p) -> F(x, p)` and its differentials `(x, p, dx) -> d1F(x, p, dx)`, `(x, p, dx1, dx2) -> d2F(x, p, dx1, dx2)`...
- `br` branch result from a call to [`continuation`](@ref)
- `ind_bif` index of the bifurcation point in `br` from which you want to branch from
- `optionsCont` options for the call to [`continuation`](@ref)

# Optional arguments
- `Jᵗ` associated jacobian transpose, it should be implemented in an efficient manner. For matrix-free methods, `transpose` is not readily available and the user must provide a dedicated method. In the case of sparse based jacobian, `Jᵗ` should not be passed as it is computed internally more efficiently, i.e. it avoid recomputing the jacobian as it would be if you pass `Jᵗ = (x, p) -> transpose(dF(x, p))`.
- `δ` used internally to compute derivatives (with finite differences) w.r.t the parameter `p`.
- `δp` used to specify a particular guess for the parameter on the bifurcated branch which is otherwise determined by `optionsCont.ds`. This allows to use a step larger than `optionsCont.dsmax`.
- `ampfactor = 1` factor which alters the amplitude of the bifurcated solution. Useful to magnify the bifurcated solution when the bifurcated branch is very steep.
- `nev` number of eigenvalues to be computed to get the right eigenvector
- `issymmetric` whether the Jacobian is Symmetric, avoid computing the left eigenvectors in the computation of the reduced equation.
- `usedeflation = true` whether to use nonlinear deflation (see [Deflated problems](@ref Deflated-problems)) to help finding the guess on the bifurcated branch
- `kwargs` optional arguments to be passed to [`continuation`](@ref), the regular `continuation` one.

!!! tip "Advanced use"
    In the case of a very large model and use of special hardware (GPU, cluster), we suggest to discouple the computation of the reduced equation, the predictor and the bifurcated branches. Have a look at `methods(BifurcationKit.multicontinuation)` to see how to call these versions. These methods has been tested on GPU with very high memory pressure.
"""
function continuation(F, dF, d2F, d3F, br::ContResult, ind_bif::Int, optionsCont::ContinuationPar = br.contparams ;
		Jᵗ = nothing,
		δ::Real = 1e-8, δp = nothing, ampfactor::Real = 1,
		nev = optionsCont.nev, issymmetric = false,
		usedeflation::Bool = false,
		Teigvec = getvectortype(br),
		scaleζ = norm,
		verbosedeflation::Bool = false,
		maxIterDeflation::Int = min(50, 15optionsCont.newtonOptions.maxIter),
		perturb = identity,
		kwargs...)
	# The usual branch switching algorithm is described in Keller. Numerical solution of bifurcation and nonlinear eigenvalue problems. We do not use this one but compute the Lyapunov-Schmidt decomposition instead and solve the polynomial equation instead.

	verbose = get(kwargs, :verbosity, 0) > 0 ? true : false
	verbose && println("--> Considering bifurcation point:"); _show(stdout, br.specialpoint[ind_bif], ind_bif)

	if kernelDim(br, ind_bif) > 1
		return multicontinuation(F, dF, d2F, d3F, br, ind_bif, optionsCont; Jᵗ = Jᵗ, δ = δ, δp = δp, ampfactor = ampfactor, nev = nev, issymmetric = issymmetric, scaleζ = scaleζ, verbosedeflation = verbosedeflation, maxIterDeflation = maxIterDeflation, perturb = perturb, Teigvec = Teigvec, kwargs...)
	end

	@assert br.type == :Equilibrium "Error! This bifurcation type is not handled.\n Branch point from $(br.type)"
	@assert br.specialpoint[ind_bif].type == :bp "Error! This bifurcation type is not handled.\n Branch point from $(br.specialpoint[ind_bif].type)"

	# compute predictor for point on new branch
	ds = isnothing(δp) ? optionsCont.ds : δp
	Ty = typeof(ds)

	# compute the normal form of the bifurcation point
	specialpoint = computeNormalForm1d(F, dF, d2F, d3F, br, ind_bif; Jᵗ = Jᵗ, δ = δ, nev = nev, verbose = verbose, issymmetric = issymmetric, Teigvec = Teigvec, scaleζ = scaleζ)

	# compute predictor for a point on new branch
	pred = predictor(specialpoint, ds; verbose = verbose, ampfactor = Ty(ampfactor))
	if isnothing(pred); return nothing, nothing; end

	verbose && printstyled(color = :green, "\n--> Start branch switching. \n--> Bifurcation type = ", type(specialpoint), "\n----> newp = ", pred.p, ", δp = ", br.specialpoint[ind_bif].param - pred.p, "\n")

	if usedeflation
		verbose && println("\n----> Compute point on the current branch with nonlinear deflation...")
		optn = optionsCont.newtonOptions
		bifpt = br.specialpoint[ind_bif]
		# find the bifurcated branch using nonlinear deflation
		solbif, _, flag, _ = newton(F, dF, convert(Teigvec, bifpt.x), pred.x, setParam(br, pred.p), setproperties(optn; verbose = verbose = verbosedeflation); kwargs...)[1]
		copyto!(pred.x, solbif)
	end

	# perform continuation
	branch, u, τ = continuation(F, dF,
			specialpoint.x0, specialpoint.params,	# first point on the branch
			pred.x, pred.p,					# second point on the branch
			br.lens, optionsCont; kwargs...)
	return Branch(branch, specialpoint), u, τ
end

# same but for a Branch
continuation(F, dF, d2F, d3F, br::AbstractBranchResult, ind_bif::Int, optionsCont::ContinuationPar = br.contparams ; kwargs...) = continuation(F, dF, d2F, d3F, getContResult(br), ind_bif, optionsCont ; kwargs...)

"""
$(SIGNATURES)

Automatic branch switching at branch points based on a computation of the normal form. More information is provided in [Branch switching](@ref). An example of use is provided in [2d generalized Bratu–Gelfand problem](@ref).

# Arguments
- `F, dF, d2F, d3F`: function `(x, p) -> F(x, p)` and its differentials `(x, p, dx) -> d1F(x, p, dx)`, `(x, p, dx1, dx2) -> d2F(x, p, dx1, dx2)`...
- `br` branch result from a call to [`continuation`](@ref)
- `ind_bif` index of the bifurcation point in `br` from which you want to branch from
- `optionsCont` options for the call to [`continuation`](@ref)

# Optional arguments
- `Jᵗ` associated jacobian transpose, it should be implemented in an efficient manner. For matrix-free methods, `transpose` is not readily available and the user must provide a dedicated method. In the case of sparse based jacobian, `Jᵗ` should not be passed as it is computed internally more efficiently, i.e. it avoid recomputing the jacobian as it would be if you pass `Jᵗ = (x, p) -> transpose(dF(x, p))`.
- `δ` used internally to compute derivatives w.r.t the parameter `p`.
- `δp` used to specify a particular guess for the parameter on the bifurcated branch which is otherwise determined by `optionsCont.ds`. This allows to use a step larger than `optionsCont.dsmax`.
- `ampfactor = 1` factor which alters the amplitude of the bifurcated solution. Useful to magnify the bifurcated solution when the bifurcated branch is very steep.
- `nev` number of eigenvalues to be computed to get the right eigenvector
- `issymmetric` whether the Jacobian is Symmetric, avoid computing the left eigenvectors in the computation of the reduced equation.
- `verbosedeflation = true` whether to display the nonlinear deflation iterations (see [Deflated problems](@ref Deflated-problems)) to help finding the guess on the bifurcated branch
- `scaleζ` norm used to normalize eigenbasis when computing the reduced equation
- `Teigvec` type of the eigenvector. Useful when `br` was loaded from a file and this information was lost
- `ζs` basis of the kernel
- `perturbGuess = identity` perturb the guess from the predictor just before the deflated-newton correction
- `kwargs` optional arguments to be passed to [`continuation`](@ref), the regular `continuation` one.

!!! tip "Advanced use"
    In the case of a very large model and use of special hardware (GPU, cluster), we suggest to discouple the computation of the reduced equation, the predictor and the bifurcated branches. Have a look at `methods(BifurcationKit.multicontinuation)` to see how to call these versions. These methods has been tested on GPU with very high memory pressure.
"""
function multicontinuation(F, dF, d2F, d3F, br::AbstractBranchResult, ind_bif::Int, optionsCont::ContinuationPar = br.contparams;
		Jᵗ = nothing,
		δ::Real = 1e-8,
		δp = nothing,
		ampfactor::Real = getvectoreltype(br)(1),
		nev::Int = optionsCont.nev,
		issymmetric::Bool = false,
		Teigvec = getvectortype(br),
		ζs = nothing,
		verbosedeflation::Bool = false,
		scaleζ = norm,
		perturbGuess = identity,
		kwargs...)

	verbose = get(kwargs, :verbosity, 0) > 0 ? true : false

	bpnf = computeNormalForm(F, dF, d2F, d3F, br, ind_bif; Jᵗ = Jᵗ, δ = δ, nev = nev, verbose = verbose, issymmetric = issymmetric, Teigvec = Teigvec, ζs = ζs, scaleζ = scaleζ)

	return multicontinuation(F, dF, br, bpnf, optionsCont; Teigvec = Teigvec, δp = δp, ampfactor = ampfactor, verbosedeflation = verbosedeflation, kwargs...)
end

# for AbstractBifurcationPoint (like Hopf, BT, ...), it must return nothing
multicontinuation(F, dF, br::AbstractBranchResult, bpnf::AbstractBifurcationPoint, optionsCont::ContinuationPar; kwargs...) = nothing

# general function for branching from Nd bifurcation points
function multicontinuation(F, dF, br::AbstractBranchResult, bpnf::NdBranchPoint, optionsCont::ContinuationPar = br.contparams; δp = nothing, ampfactor = getvectoreltype(br)(1), perturb = identity, kwargs...)

	verbose = get(kwargs, :verbosity, 0) > 0 ? true & get(kwargs, :verbosedeflation, true) : false

	# compute predictor for point on new branch
	ds = abs(isnothing(δp) ? optionsCont.ds : δp)

	# get prediction from solving the reduced equation
	rootsNFm, rootsNFp = predictor(bpnf, ds;  verbose = verbose, perturb = perturb, ampfactor = ampfactor)

	return multicontinuation(F, dF, br, bpnf, (before = rootsNFm, after = rootsNFp), optionsCont; δp = δp, kwargs...)
end

"""
$(SIGNATURES)

Function to transform predictors `solfromRE` in the normal form coordinates of `bpnf` into solutions. Note that `solfromRE = (before = Vector{vectype}, after = Vector{vectype})`.
"""
function getFirstPointsOnBranch(F, dF, br::AbstractBranchResult,
		bpnf::NdBranchPoint, solfromRE,
		optionsCont::ContinuationPar = br.contparams ;
		δp = nothing,
		Teigvec = getvectortype(br),
		verbosedeflation = false,
		maxIterDeflation = min(50, 15optionsCont.newtonOptions.maxIter),
		lsdefop = DeflatedLinearSolver(),
		perturbGuess = identity,
		kwargs...)
	# compute predictor for point on new branch
	ds = isnothing(δp) ? optionsCont.ds : δp |> abs
	dscont = abs(optionsCont.ds)

	rootsNFm = solfromRE.before
	rootsNFp = solfromRE.after

	# attempting now to convert the guesses from the normal form into true zeros of F
	optn = optionsCont.newtonOptions

	# options for newton
	cbnewton = get(kwargs, :callbackN, cbDefault)
	normn = get(kwargs, :normN, norm)

	printstyled(color = :magenta, "--> Looking for solutions after the bifurcation point...\n")
	defOpp = DeflationOperator(2, 1.0, Vector{typeof(bpnf.x0)}(), _copy(bpnf.x0); autodiff = true)
	for (ind, xsol) in pairs(rootsNFp)
		# print("\n--> attempt to converge zero #$ind")
		solbif, _, flag, _ = newton(F, dF, perturbGuess(bpnf(xsol, ds)), setParam(br, bpnf.p + ds), setproperties(optn; maxIter = maxIterDeflation, verbose = verbosedeflation), defOpp, lsdefop; callback = cbnewton, normN = normn)
		flag && push!(defOpp, solbif)
	end

	printstyled(color = :magenta, "--> Looking for solutions before the bifurcation point...\n")
	defOpm = DeflationOperator(2, 1.0, Vector{typeof(bpnf.x0)}(), _copy(bpnf.x0); autodiff = true)
	for (ind, xsol) in pairs(rootsNFm)
		# print("\n--> attempt to converge zero #$ind")
		solbif, _, flag, _ = newton(F, dF, perturbGuess(bpnf(xsol, ds)), setParam(br, bpnf.p - ds), setproperties(optn; maxIter = maxIterDeflation, verbose = verbosedeflation), defOpm, lsdefop; callback = cbnewton, normN = normn)
		flag && push!(defOpm, solbif)
	end
	printstyled(color=:magenta, "--> we find $(length(defOpm)) (resp. $(length(defOpp))) roots after (resp. before) the bifurcation point.\n")
	return (before = defOpm, after = defOpp, bpm = bpnf.p - ds, bpp = bpnf.p + ds)
end

# In this function, I keep usedeflation although it is not used to simplify the calls
function multicontinuation(F, dF, br::AbstractBranchResult,
		bpnf::NdBranchPoint, solfromRE,
		optionsCont::ContinuationPar = br.contparams ;
		δp = nothing,
		Teigvec = getvectortype(br),
		verbosedeflation = false,
		maxIterDeflation = min(50, 15optionsCont.newtonOptions.maxIter),
		lsdefop = DeflatedLinearSolver(),
		perturbGuess = identity,
		kwargs...)

	defOpm, defOpp, _, _ = getFirstPointsOnBranch(F, dF, br, bpnf, solfromRE, optionsCont; δp = δp, verbosedeflation = verbosedeflation, maxIterDeflation = maxIterDeflation, lsdefop = lsdefop, perturbGuess = perturbGuess, kwargs...)

	multicontinuation(F, dF, br,
			bpnf, defOpm, defOpp, optionsCont;
			δp = δp,
			Teigvec = Teigvec,
			verbosedeflation = verbosedeflation,
			maxIterDeflation = maxIterDeflation,
			lsdefop = lsdefop,
			kwargs...)
end

"""
$(SIGNATURES)

Automatic branch switching at branch points based on a computation of the normal form. More information is provided in [Branch switching](@ref). An example of use is provided in [2d generalized Bratu–Gelfand problem](@ref).

# Arguments
- `F, dF`: function `(x, p) -> F(x, p)` and its differential `(x, p, dx) -> d1F(x, p, dx)`
- `br` branch result from a call to [`continuation`](@ref)
- `bpnf` normal form
- `defOpm::DeflationOperator, defOpp::DeflationOperator` to specify converged points on nonn-trivial branches before/after the bifurcation points.

The rest is as the regular `multicontinuation` function.
"""
function multicontinuation(F, dF, br::AbstractBranchResult,
		bpnf::NdBranchPoint, defOpm::DeflationOperator, defOpp::DeflationOperator,
		optionsCont::ContinuationPar = br.contparams ;
		δp = nothing,
		Teigvec = getvectortype(br),
		verbosedeflation = false,
		maxIterDeflation = min(50, 15optionsCont.newtonOptions.maxIter),
		lsdefop = DeflatedLinearSolver(),
		kwargs...)

	ds = isnothing(δp) ? optionsCont.ds : δp |> abs
	dscont = abs(optionsCont.ds)
	par = bpnf.params

	# compute the different branches
	function _continue(_sol, _dp, _ds)
		# needed to reset the tangent algorithm in case fields are used
		empty!(get(kwargs, :tangentAlgo, nothing))
		println("#"^50)
		continuation(F, dF,
			bpnf.x0, par,		# first point on the branch
			_sol, bpnf.p + _dp, # second point on the branch
			br.lens, (@set optionsCont.ds = _ds); kwargs...)
		# continuation(F, dF, _sol, setParam(br, bpnf.p + _dp), br.lens, (@set optionsCont.ds = _ds); kwargs...)
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

	return branches, (before = defOpm, after = defOpp)
end

# same but for a Branch
multicontinuation(F, dF, d2F, d3F, br::Branch, ind_bif::Int, optionsCont::ContinuationPar = br.contparams; kwargs...) = multicontinuation(F, dF, d2F, d3F, getContResult(br), ind_bif, optionsCont ; kwargs...)
