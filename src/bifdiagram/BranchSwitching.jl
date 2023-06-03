"""
$(SIGNATURES)

[Internal] This function is not meant to be called directly.

This function is the analog of [`continuation`](@ref) when the first two points on the branch are passed (instead of a single one). Hence `x0` is the first point on the branch (with palc `s=0`) with parameter `par0` and `x1` is the second point with parameter `set(par0, lens, p1)`.
"""
function continuation(prob::AbstractBifurcationProblem,
					x0::Tv, par0,		# first point on the branch
					x1::Tv, p1::Real,	# second point on the branch
					alg, lens::Lens,
					contParams::ContinuationPar;
					kwargs...) where Tv
	# update alg linear solver with contParams.newtonOptions.linsolver
	alg = update(alg, contParams, nothing)
	# check the sign of ds
	dsfactor = sign(p1 - get(par0, lens))
	# create an iterable
	_contParams = @set contParams.ds = abs(contParams.ds) * dsfactor
	prob2 = reMake(prob; lens = lens, params = par0)
	it = ContIterable(prob2, alg, _contParams; kwargs...)
	return continuation(it, x0, get(par0, lens), x1, p1)
end

function continuation(it::ContIterable, x0, p0::Real, x1, p1::Real)
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
- `br` branch result from a call to [`continuation`](@ref)
- `ind_bif` index of the bifurcation point in `br` from which you want to branch from
- `optionsCont` options for the call to [`continuation`](@ref)

# Optional arguments
- `alg = br.alg` continuation algorithm to be used, default value: `br.alg`
- `δp` used to specify a particular guess for the parameter on the bifurcated branch which is otherwise determined by `optionsCont.ds`. This allows to use a step larger than `optionsCont.dsmax`.
- `ampfactor = 1` factor which alters the amplitude of the bifurcated solution. Useful to magnify the bifurcated solution when the bifurcated branch is very steep.
- `nev` number of eigenvalues to be computed to get the right eigenvector
- `usedeflation = true` whether to use nonlinear deflation (see [Deflated problems](@ref Deflated-problems)) to help finding the guess on the bifurcated
- `plotSolution` change plot solution method in the problem `br.prob`
- `kwargs` optional arguments to be passed to [`continuation`](@ref), the regular `continuation` one.

!!! tip "Advanced use"
    In the case of a very large model and use of special hardware (GPU, cluster), we suggest to discouple the computation of the reduced equation, the predictor and the bifurcated branches. Have a look at `methods(BifurcationKit.multicontinuation)` to see how to call these versions. These methods has been tested on GPU with very high memory pressure.
"""
function continuation(br::AbstractResult{EquilibriumCont, Tprob}, ind_bif::Int, optionsCont::ContinuationPar = br.contparams ;
		alg = br.alg,
		δp = nothing, ampfactor::Real = 1,
		nev = optionsCont.nev,
		usedeflation::Bool = false,
		Teigvec = getvectortype(br),
		scaleζ = norm,
		verbosedeflation::Bool = false,
		maxIterDeflation::Int = min(50, 15optionsCont.newtonOptions.maxIter),
		perturb = identity,
		plotSolution = plotSolution(br.prob),
		tolFold = 1e-3,
		kwargs...) where Tprob
	# The usual branch switching algorithm is described in Keller. Numerical solution of bifurcation and nonlinear eigenvalue problems. We do not use this one but compute the Lyapunov-Schmidt decomposition instead and solve the polynomial equation instead.

	verbose = get(kwargs, :verbosity, 0) > 0 ? true : false
	verbose && println("──▶ Considering bifurcation point:"); _show(stdout, br.specialpoint[ind_bif], ind_bif)

	if kernelDim(br, ind_bif) > 1
		return multicontinuation(br, ind_bif, optionsCont; δp = δp, ampfactor = ampfactor, nev = nev, scaleζ = scaleζ, verbosedeflation = verbosedeflation, maxIterDeflation = maxIterDeflation, perturb = perturb, Teigvec = Teigvec, alg = alg, plotSolution = plotSolution, kwargs...)
	end

	@assert br.specialpoint[ind_bif].type == :bp "This bifurcation type is not handled.\n Branch point from $(br.specialpoint[ind_bif].type)"

	# compute predictor for point on new branch
	ds = isnothing(δp) ? optionsCont.ds : δp
	Ty = typeof(ds)

	# compute the normal form of the bifurcation point
	specialpoint = getNormalForm1d(br, ind_bif; nev = nev, verbose = verbose, Teigvec = Teigvec, scaleζ = scaleζ, tolFold = tolFold)

	# compute predictor for a point on new branch
	pred = predictor(specialpoint, ds; verbose = verbose, ampfactor = Ty(ampfactor))
	if isnothing(pred); return nothing; end

	verbose && printstyled(color = :green, "\n──▶ Start branch switching. \n──▶ Bifurcation type = ", type(specialpoint), "\n────▶ newp = ", pred.p, ", δp = ", br.specialpoint[ind_bif].param - pred.p, "\n")

	if usedeflation
		verbose && println("\n────▶ Compute point on the current branch with nonlinear deflation...")
		optn = optionsCont.newtonOptions
		bifpt = br.specialpoint[ind_bif]
		# find the bifurcated branch using nonlinear deflation
		solbif = newton(br.prob, convert(Teigvec, bifpt.x), pred.x, setParam(br, pred.p), setproperties(optn; verbose = verbose = verbosedeflation); kwargs...)[1]
		copyto!(pred.x, solbif.u)
	end

	# perform continuation
	branch = continuation(reMake(br.prob, plotSolution=plotSolution),
			specialpoint.x0, specialpoint.params,	# first point on the branch
			pred.x, pred.p,							# second point on the branch
			alg, getLens(br),
			optionsCont; kwargs...)
	return Branch(branch, specialpoint)
end

# same but for a Branch
continuation(br::AbstractBranchResult, ind_bif::Int, optionsCont::ContinuationPar = br.contparams ; kwargs...) = continuation(getContResult(br), ind_bif, optionsCont ; kwargs...)

"""
$(SIGNATURES)

Automatic branch switching at branch points based on a computation of the normal form. More information is provided in [Branch switching](@ref). An example of use is provided in [2d generalized Bratu–Gelfand problem](@ref).

# Arguments
- `br` branch result from a call to [`continuation`](@ref)
- `ind_bif` index of the bifurcation point in `br` from which you want to branch from
- `optionsCont` options for the call to [`continuation`](@ref)

# Optional arguments
- `alg = br.alg` continuation algorithm to be used, default value: `br.alg`
- `δp` used to specify a particular guess for the parameter on the bifurcated branch which is otherwise determined by `optionsCont.ds`. This allows to use a step larger than `optionsCont.dsmax`.
- `ampfactor = 1` factor which alters the amplitude of the bifurcated solution. Useful to magnify the bifurcated solution when the bifurcated branch is very steep.
- `nev` number of eigenvalues to be computed to get the right eigenvector
- `verbosedeflation = true` whether to display the nonlinear deflation iterations (see [Deflated problems](@ref Deflated-problems)) to help finding the guess on the bifurcated branch
- `scaleζ` norm used to normalize eigenbasis when computing the reduced equation
- `Teigvec` type of the eigenvector. Useful when `br` was loaded from a file and this information was lost
- `ζs` basis of the kernel
- `perturbGuess = identity` perturb the guess from the predictor just before the deflated-newton correction
- `kwargs` optional arguments to be passed to [`continuation`](@ref), the regular `continuation` one.

!!! tip "Advanced use"
    In the case of a very large model and use of special hardware (GPU, cluster), we suggest to discouple the computation of the reduced equation, the predictor and the bifurcated branches. Have a look at `methods(BifurcationKit.multicontinuation)` to see how to call these versions. These methods has been tested on GPU with very high memory pressure.
"""
function multicontinuation(br::AbstractBranchResult, ind_bif::Int, optionsCont::ContinuationPar = br.contparams;
		δp = nothing,
		ampfactor::Real = getvectoreltype(br)(1),
		nev::Int = optionsCont.nev,
		Teigvec = getvectortype(br),
		ζs = nothing,
		verbosedeflation::Bool = false,
		scaleζ = norm,
		perturbGuess = identity,
		plotSolution = plotSolution(br.prob),
		kwargs...)

	verbose = get(kwargs, :verbosity, 0) > 0 ? true : false

	bpnf = getNormalForm(br, ind_bif; nev = nev, verbose = verbose, Teigvec = Teigvec, ζs = ζs, scaleζ = scaleζ)

	return multicontinuation(br, bpnf, optionsCont; Teigvec = Teigvec, δp = δp, ampfactor = ampfactor, verbosedeflation = verbosedeflation, plotSolution = plotSolution, kwargs...)
end

# for AbstractBifurcationPoint (like Hopf, BT, ...), it must return nothing
multicontinuation(br::AbstractBranchResult, bpnf::AbstractBifurcationPoint, optionsCont::ContinuationPar; kwargs...) = nothing

# general function for branching from Nd bifurcation points
function multicontinuation(br::AbstractBranchResult,
						bpnf::NdBranchPoint,
						optionsCont::ContinuationPar = br.contparams;
						δp = nothing,
						ampfactor = getvectoreltype(br)(1),
						perturb = identity,
						plotSolution = plotSolution(br.prob),
						kwargs...)

	verbose = get(kwargs, :verbosity, 0) > 0 ? true & get(kwargs, :verbosedeflation, true) : false

	# compute predictor for point on new branch
	ds = abs(isnothing(δp) ? optionsCont.ds : δp)

	# get prediction from solving the reduced equation
	rootsNFm, rootsNFp = predictor(bpnf, ds;  verbose = verbose, perturb = perturb, ampfactor = ampfactor)

	return multicontinuation(br, bpnf, (before = rootsNFm, after = rootsNFp), optionsCont; δp = δp, plotSolution = plotSolution, kwargs...)
end

"""
$(SIGNATURES)

Function to transform predictors `solfromRE` in the normal form coordinates of `bpnf` into solutions. Note that `solfromRE = (before = Vector{vectype}, after = Vector{vectype})`.
"""
function getFirstPointsOnBranch(br::AbstractBranchResult,
		bpnf::NdBranchPoint, solfromRE,
		optionsCont::ContinuationPar = br.contparams ;
		δp = nothing,
		Teigvec = getvectortype(br),
		usedeflation = true,
		verbosedeflation = false,
		maxIterDeflation = min(50, 15optionsCont.newtonOptions.maxIter),
		lsdefop = DeflatedProblemCustomLS(),
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

	printstyled(color = :magenta, "──▶ Looking for solutions after the bifurcation point...\n")
	defOpp = DeflationOperator(2, 1.0, Vector{typeof(bpnf.x0)}(), _copy(bpnf.x0); autodiff = true)
	optnDf = setproperties(optn; maxIter = maxIterDeflation, verbose = verbosedeflation)

	for (ind, xsol) in pairs(rootsNFp)
		probp = reMake(br.prob; u0 = perturbGuess(bpnf(xsol, ds)),
								params = setParam(br, bpnf.p + ds))
		if usedeflation
			solbif = newton(probp, defOpp, optnDf, lsdefop; callback = cbnewton, normN = normn)
		else
			solbif = newton(probp, optnDf; callback = cbnewton, normN = normn)
		end
		converged(solbif) && push!(defOpp, solbif.u)
	end

	printstyled(color = :magenta, "──▶ Looking for solutions before the bifurcation point...\n")
	defOpm = DeflationOperator(2, 1.0, Vector{typeof(bpnf.x0)}(), _copy(bpnf.x0); autodiff = true)
	for (ind, xsol) in pairs(rootsNFm)
		probm = reMake(br.prob; u0 = perturbGuess(bpnf(xsol, ds)),
								params = setParam(br, bpnf.p - ds))
		if usedeflation
			solbif = newton(probm, defOpm, optnDf, lsdefop; callback = cbnewton, normN = normn)
		else
			solbif = newton(probm, optnDf; callback = cbnewton, normN = normn)
		end
		converged(solbif) && push!(defOpm, solbif.u)
	end
	printstyled(color=:magenta, "──▶ we find $(length(defOpp)) (resp. $(length(defOpm))) roots after (resp. before) the bifurcation point.\n")
	return (before = defOpm, after = defOpp, bpm = bpnf.p - ds, bpp = bpnf.p + ds)
end

# In this function, I keep usedeflation although it is not used to simplify the calls
function multicontinuation(br::AbstractBranchResult,
		bpnf::NdBranchPoint, solfromRE,
		optionsCont::ContinuationPar = br.contparams ;
		δp = nothing,
		Teigvec = getvectortype(br),
		verbosedeflation = false,
		maxIterDeflation = min(50, 15optionsCont.newtonOptions.maxIter),
		lsdefop = DeflatedProblemCustomLS(),
		perturbGuess = identity,
		kwargs...)

	defOpm, defOpp, _, _ = getFirstPointsOnBranch(br, bpnf, solfromRE, optionsCont; δp = δp, verbosedeflation = verbosedeflation, maxIterDeflation = maxIterDeflation, lsdefop = lsdefop, perturbGuess = perturbGuess, kwargs...)

	multicontinuation(br,
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
- `br` branch result from a call to [`continuation`](@ref)
- `bpnf` normal form
- `defOpm::DeflationOperator, defOpp::DeflationOperator` to specify converged points on nonn-trivial branches before/after the bifurcation points.

The rest is as the regular `multicontinuation` function.
"""
function multicontinuation(br::AbstractBranchResult,
		bpnf::NdBranchPoint,
		defOpm::DeflationOperator,
		defOpp::DeflationOperator,
		optionsCont::ContinuationPar = br.contparams ;
		alg = br.alg,
		δp = nothing,
		Teigvec = getvectortype(br),
		verbosedeflation = false,
		maxIterDeflation = min(50, 15optionsCont.newtonOptions.maxIter),
		lsdefop = DeflatedProblemCustomLS(),
		plotSolution = plotSolution(br.prob),
		kwargs...)

	ds = isnothing(δp) ? optionsCont.ds : δp |> abs
	dscont = abs(optionsCont.ds)
	par = bpnf.params
	prob = reMake(br.prob; plotSolution = plotSolution)

	# compute the different branches
	function _continue(_sol, _dp, _ds)
		# needed to reset the tangent algorithm in case fields are used
		println("━"^50)
		continuation(prob,
			bpnf.x0, par,		# first point on the branch
			_sol, bpnf.p + _dp, # second point on the branch
			empty(alg), getLens(br),
			(@set optionsCont.ds = _ds); kwargs...)
	end

	branches = Branch[]
	for id in 2:length(defOpm)
		br = _continue(defOpm[id], -ds, -dscont); push!(branches, Branch(br, bpnf))
		# br, = _continue(defOpm[id], -ds, dscont); push!(branches, Branch(br, bpnf))
	end

	for id in 2:length(defOpp)
		br = _continue(defOpp[id], ds, dscont); push!(branches, Branch(br, bpnf))
		# br, = _continue(defOpp[id], ds, -dscont); push!(branches, Branch(br, bpnf))
	end

	return branches
end

# same but for a Branch
multicontinuation(br::Branch, ind_bif::Int, optionsCont::ContinuationPar = br.contparams; kwargs...) = multicontinuation(getContResult(br), ind_bif, optionsCont ; kwargs...)
