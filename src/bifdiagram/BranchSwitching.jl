"""
$(TYPEDSIGNATURES)

[Internal] This function is not meant to be called directly.

This function is the analog of [`continuation`](@ref) when the first two points on the branch are passed (instead of a single one). Hence `x0` is the first point on the branch (with pseudo arc length `s=0`) with parameter `par0` and `x1` is the second point with parameter `set(par0, lens, p1)`.
"""
function continuation(prob::AbstractBifurcationProblem,
                      x0::Tv, par0,     # first point on the branch
                      x1::Tv, p1::Real, # second point on the branch
                      alg, lens::AllOpticTypes,
                      contParams::ContinuationPar;
                      bothside::Bool = false,
                      kwargs...) where Tv
    # update alg linear solver with contParams.newton_options.linsolver
    alg = update(alg, contParams, nothing)
    # check the sign of ds
    dsfactor = sign(p1 - _get(par0, lens))
    # create an iterable
    _contParams = @set contParams.ds = abs(contParams.ds) * dsfactor
    prob2 = re_make(prob; lens, params = par0)
    if ~bothside
        it = ContIterable(prob2, alg, _contParams; kwargs...)
        return continuation(it, x0, _get(par0, lens), x1, p1)
    else
        itfw = ContIterable(prob2, alg, _contParams; kwargs...)
        itbw = deepcopy(itfw)
        resfw = continuation(itfw, x0, _get(par0, lens), x1, p1)
        resbw = continuation(itbw, x1, p1, x0, _get(par0, lens))
        return _merge(resfw, resbw)
    end
end

function continuation(it::ContIterable, x0, p0::Real, x1, p1::Real)
    # we compute the cache for the continuation, i.e. state::ContState
    # In this call, we also compute the initial point on the branch (and its stability) and the initial tangent
    state, _ = iterate_from_two_points(it, x0, p0, x1, p1)

    # variable to hold the result from continuation, i.e. a branch
    contRes = ContResult(it, state)

    # perform the continuation
    return continuation!(it, state, contRes)
end

"""
$(TYPEDSIGNATURES)

Automatic branch switching at branch points based on a computation of the normal form. More information is provided in [Branch switching](@ref Branch-switching-page). An example of use is provided in [2d generalized Bratu–Gelfand problem](@ref).

# Arguments
- `br`: branch result from a call to [`continuation`](@ref) containing the bifurcation point
- `ind_bif`: index of the bifurcation point in `br` from which to branch
- `options_cont`: continuation parameters for the new branch

# Optional arguments
- `alg = br.alg` continuation algorithm to be used, default value: `br.alg`
- `δp` used to specify a specific value for the parameter on the bifurcated branch which is otherwise determined by `options_cont.ds`. This allows to use a step larger than `options_cont.dsmax`.
- `ampfactor = 1` factor to alter the amplitude of the bifurcated solution. Useful to magnify the bifurcated solution when the bifurcated branch is very steep. Can also be used to select the upper/lower branch in Pitchfork bifurcations. See also `use_normal_form` below.
- `use_normal_form = true`. If `use_normal_form = true`, the normal form is computed as well as its predictor and a guess is automatically formed. If `use_normal_form = false`, the parameter value `p = p₀ + δp` and the guess `x = x₀ + ampfactor .* e` (where `e` is a vector of the kernel) are used as initial guess. This is useful in case automatic branch switching does not work.
- `nev` number of eigenvalues to be computed to get the right eigenvector
- `scaleζ = norm` pass a norm to normalize vectors during normal form computation
- `plot_solution` change plot solution method in the problem `br.prob`
- `usedeflation = false` whether to use nonlinear deflation (see [Deflated problems](@ref Deflated-problems)) to help finding the guess on the bifurcated
- `verbosedeflation` print deflated newton iterations
- `max_iter_deflation` number of newton steps in deflated newton
- `perturb = identity` which perturbation function to use during deflated newton
- `Teigvec = _getvectortype(br)` type of the eigenvector. Useful when `br` was loaded from a file and this information was lost
- `kwargs` optional arguments to be passed to [`continuation`](@ref), the regular `continuation` one and to [`get_normal_form`](@ref).

!!! tip "Advanced use"
    In the case of a very large model and use of special hardware (GPU, cluster), we suggest to decouple the computation of the reduced equation, the predictor and the bifurcated branches. Have a look at `methods(BifurcationKit.multicontinuation)` to see how to call these versions. These methods has been tested on GPU with very high memory pressure.
"""
function continuation(br::AbstractResult{EquilibriumCont, Tprob}, 
                      ind_bif::Int, 
                      options_cont::ContinuationPar = br.contparams,
                      Teigvec::Type{𝒯eigvec} = _getvectortype(br);

                      alg = getalg(br),

                      δp = nothing, 
                      ampfactor::Real = 1,
                      use_normal_form = true,
                      bls = MatrixBLS(),
                      bls_block = bls,


                      nev = options_cont.nev,
                      scaleζ = norm,
                      autodiff = true,

                      usedeflation::Bool = false,
                      verbosedeflation::Bool = false,
                      max_iter_deflation::Int = min(50, 15options_cont.newton_options.max_iterations),
                      perturb = identity,

                      plot_solution = plot_solution(br.prob),
                      tol_fold = 1e-3,
                      kwargs_deflated_newton = (),
                      kwargs...) where {Tprob, 𝒯eigvec}
    # The usual branch switching algorithm is described in the work of Keller. 
    # "Numerical solution of bifurcation and nonlinear eigenvalue problems."
    # We do not use this algorithm but instead compute the Lyapunov-Schmidt decomposition and solve the polynomial equation.

    if (br.specialpoint[ind_bif].type in (:bp, :nd)) == false 
        error("You cannot branch from a :$(br.specialpoint[ind_bif].type) point using these arguments.\n")
    end

    verbose = get(kwargs, :verbosity, 0) > 0
    verbose && println("──▶ Considering bifurcation point:")
    verbose && _show(stdout, br.specialpoint[ind_bif], ind_bif)

    if kernel_dimension(br, ind_bif) > 1
        return multicontinuation(br,
                                ind_bif,
                                options_cont;
                                Teigvec,
                                alg,

                                δp,
                                ampfactor,
                                nev,
                                scaleζ,
                                autodiff,
                                bls_block,

                                verbosedeflation,
                                max_iter_deflation,
                                perturb,
                                plot_solution,
                                kwargs...)
    end

    # compute predictor for point on new branch
    ds = isnothing(δp) ? options_cont.ds : δp
    Ty = typeof(ds)

    # compute the normal form of the bifurcation point
    bp = get_normal_form1d(br.prob, br, ind_bif, Teigvec; 
                            nev,
                            verbose,
                            scaleζ,
                            bls,
                            autodiff,
                            tol_fold)

    # compute predictor for a point on new branch
    if ~use_normal_form
        pred = (;x0 = bp.x0, 
                x1 = bp.x0 .+ ampfactor .* real.(bp.ζ), 
                p =  bp.p + ds, 
                amp = ampfactor)
    else
        pred = predictor(bp, ds; verbose, ampfactor = Ty(ampfactor))
    end

    if isnothing(pred)
        @debug "[aBS] The predictor is nothing. Probably a Fold point. See\n $bp"
        return nothing
    end

    verbose && printstyled(color = :green,  "\n──▶ Start branch switching. 
                                             \n──▶ Bifurcation type = ", type(bp), 
                                            "\n────▶ newp      = ", pred.p, ", δp = ", pred.p - br.specialpoint[ind_bif].param, 
                                            "\n────▶ amplitude = ", pred.amp,
                                            "\n")

    if pred.amp > 0.1
        @debug "The guess for the amplitude of the first periodic orbit on the bifurcated branch obtained by the predictor is not small: $(pred.amp). This may lead to convergence failure of the first newton step or select a branch far from the bifurcation point.\nYou can either decrease `ds` or `δp` (which is how far from the bifurcation point you want the branch of equilibria to start). Alternatively, you can specify a multiplicative factor `ampfactor` to be applied to the predictor amplitude."
    end

    if usedeflation
        verbose && println("\n────▶ Compute point on the current branch with nonlinear deflation...")
        optn = options_cont.newton_options
        bifpt = br.specialpoint[ind_bif]
        # find the bifurcated branch using nonlinear deflation
        solbif = newton(br.prob, 
                        convert(Teigvec, pred.x0), 
                        pred.x1, 
                        setparam(br, pred.p), 
                        setproperties(optn; verbose = verbosedeflation); 
                        kwargs...)
        if ~converged(solbif[1])
            @warn "Deflated newton did not converge for the first guess on the bifurcated branch."
        end
        _copyto!(pred.x1, solbif[1].u)
    end

    # perform continuation
    kwargs_cont = _keep_opts_cont(values(kwargs))
    branch = continuation(re_make(br.prob; plot_solution),
                            bp.x0, bp.params, # first point on the branch
                            pred.x1, pred.p,  # second point on the branch
                            alg, getlens(br),
                            options_cont; 
                            kwargs_cont...)
    return Branch(branch, bp)
end

# same but for a Branch
continuation(br::AbstractBranchResult, 
            ind_bif::Int, 
            options_cont::ContinuationPar = br.contparams ; 
            kwargs...) = continuation(get_contresult(br), 
                                      ind_bif, 
                                      options_cont; 
                                      kwargs...)

"""
$(TYPEDSIGNATURES)

Automatic branch switching at branch points based on a computation of the normal form. More information is provided in [Branch switching](@ref). An example of use is provided in [2d generalized Bratu–Gelfand problem](@ref).

# Arguments
- `br` branch result from a call to [`continuation`](@ref)
- `ind_bif` index of the bifurcation point in `br` from which you want to branch from
- `options_cont` options for the call to [`continuation`](@ref)

# Optional arguments
- `alg = br.alg` continuation algorithm to be used, default value: `br.alg`
- `δp` used to specify a particular guess for the parameter on the bifurcated branch which is otherwise determined by `options_cont.ds`. This allows to use a step larger than `options_cont.dsmax`.
- `ampfactor = 1` factor which alters the amplitude of the bifurcated solution. Useful to magnify the bifurcated solution when the bifurcated branch is very steep.
- `nev` number of eigenvalues to be computed to get the right eigenvector
- `verbosedeflation = true` whether to display the nonlinear deflation iterations (see [Deflated problems](@ref Deflated-problems)) to help finding the guess on the bifurcated branch
- `scaleζ` norm used to normalize eigenbasis when computing the reduced equation
- `Teigvec` type of the eigenvector. Useful when `br` was loaded from a file and this information was lost
- `ζs` basis of the kernel
- `perturb_guess = identity` perturb the guess from the predictor just before the deflated-newton correction
- `kwargs` optional arguments to be passed to [`continuation`](@ref), the regular `continuation` one.

!!! tip "Advanced use"
    In the case of a very large model and use of special hardware (GPU, cluster), we suggest to discouple the computation of the reduced equation, the predictor and the bifurcated branches. Have a look at `methods(BifurcationKit.multicontinuation)` to see how to call these versions. These methods has been tested on GPU with very high memory pressure.
"""
function multicontinuation(br::AbstractBranchResult,
                            ind_bif::Int, 
                            options_cont::ContinuationPar = br.contparams,
                            Teigvec::Type{𝒯eigvec} = _getvectortype(br);

                            δp = nothing,
                            ampfactor::Real = 1,

                            nev::Int = options_cont.nev,
                            ζs = nothing,
                            scaleζ = norm,
                            autodiff = true,
                            bls_block = MatrixBLS(),

                            verbosedeflation::Bool = false,
                            perturb_guess = identity,

                            plot_solution = plot_solution(br.prob),
                            kwargs...) where {𝒯eigvec}

    verbose = get(kwargs, :verbosity, 0) > 0 ? true : false

    bpnf = get_normal_form(getprob(br), br, ind_bif, Teigvec; nev, verbose, ζs, scaleζ, autodiff, bls_block)

    return multicontinuation(br,
                            bpnf,
                            options_cont;
                            Teigvec,
                            δp,
                            ampfactor,
                            verbosedeflation,
                            plot_solution,
                            kwargs...)
end

# for AbstractBifurcationPoint (like Hopf, BT, ...), it must return nothing to prevent from calling the function
multicontinuation(br::AbstractBranchResult, bpnf::AbstractBifurcationPoint, options_cont::ContinuationPar; kwargs...) = nothing

# general function for branching from Nd bifurcation points
function multicontinuation(br::AbstractBranchResult,
                           bp::NdBranchPoint,
                           options_cont::ContinuationPar = br.contparams;
                           δp = nothing,
                           ampfactor = 1,
                           perturb = identity,
                           plot_solution = plot_solution(br.prob),
                           kwargs...)

    verbose = get(kwargs, :verbosity, 0) > 0 ? true & get(kwargs, :verbosedeflation, true) : false

    # compute predictor for point on new branch
    ds = abs(isnothing(δp) ? options_cont.ds : δp)

    # get prediction from solving the reduced equation
    rootsNFm, rootsNFp = predictor(bp, ds;  verbose, perturb, ampfactor)

    return multicontinuation(br, bp, (before = rootsNFm, after = rootsNFp), options_cont; δp, plot_solution, kwargs...)
end

"""
$(TYPEDSIGNATURES)

Function to transform predictors `solfromRE` in the normal form coordinates of `bpnf` into solutions. Note that `solfromRE = (before = Vector{vectype}, after = Vector{vectype})`.
"""
function get_first_points_on_branch(br::AbstractBranchResult,
                                    bpnf::NdBranchPoint, solfromRE,
                                    options_cont::ContinuationPar = br.contparams ;
                                    δp = nothing,
                                    Teigvec = _getvectortype(br),
                                    usedeflation = true,
                                    verbosedeflation = false,
                                    max_iter_deflation = min(50, 15options_cont.newton_options.max_iterations),
                                    lsdefop = DeflatedProblemCustomLS(),
                                    perturb_guess = identity,
                                    kwargs...)
    # compute predictor for point on new branch
    ds = isnothing(δp) ? options_cont.ds : δp |> abs
    dscont = abs(options_cont.ds)

    rootsNFm = solfromRE.before
    rootsNFp = solfromRE.after

    𝒯 = VI.scalartype(bpnf.x0)

    # attempting now to convert the guesses from the normal form into true zeros of F
    optn = options_cont.newton_options
    optnDf = setproperties(optn; max_iterations = max_iter_deflation, verbose = verbosedeflation)

    # options for newton
    cbnewton = get(kwargs, :callback_newton, cb_default)
    normn = get(kwargs, :normC, norm)

    printstyled(color = :magenta, "──▶ Looking for solutions after the bifurcation point...\n")
    defOpp = DeflationOperator(2, one(𝒯), Vector{typeof(bpnf.x0)}(), _copy(bpnf.x0); autodiff = true)

    for (ind, xsol) in pairs(rootsNFp)
        probp = re_make(br.prob; u0 = perturb_guess(bpnf(xsol, ds)),
                                params = setparam(br, bpnf.p + ds))
        if usedeflation
            solbif = solve(probp, defOpp, optnDf, lsdefop; callback = cbnewton, normN = normn)
        else
            solbif = solve(probp, Newton(), optnDf; callback = cbnewton, normN = normn)
        end
        converged(solbif) && push!(defOpp, solbif.u)
    end

    printstyled(color = :magenta, "──▶ Looking for solutions before the bifurcation point...\n")
    defOpm = DeflationOperator(2, one(𝒯), Vector{typeof(bpnf.x0)}(), _copy(bpnf.x0); autodiff = true)
    for (ind, xsol) in pairs(rootsNFm)
        probm = re_make(br.prob; u0 = perturb_guess(bpnf(xsol, ds)),
                                params = setparam(br, bpnf.p - ds))
        if usedeflation
            solbif = solve(probm, defOpm, optnDf, lsdefop; callback = cbnewton, normN = normn)
        else
            solbif = solve(probm, Newton(), optnDf; callback = cbnewton, normN = normn)
        end
        converged(solbif) && push!(defOpm, solbif.u)
    end
    printstyled(color=:magenta, "──▶ we find $(length(defOpp)) (resp. $(length(defOpm))) roots after (resp. before) the bifurcation point.\n")
    return (before = defOpm, after = defOpp, bpm = bpnf.p - ds, bpp = bpnf.p + ds)
end

# In this function, I keep usedeflation although it is not used to simplify the calls
function multicontinuation(br::AbstractBranchResult,
                            bpnf::NdBranchPoint, solfromRE,
                            options_cont::ContinuationPar = br.contparams ;
                            δp = nothing,
                            Teigvec = _getvectortype(br),
                            verbosedeflation = false,
                            max_iter_deflation = min(50, 15options_cont.newton_options.max_iterations),
                            lsdefop = DeflatedProblemCustomLS(),
                            perturb_guess = identity,
                            kwargs...)

    defOpm, defOpp, _, _ = get_first_points_on_branch(br,
                                            bpnf,
                                            solfromRE,
                                            options_cont;
                                            δp,
                                            verbosedeflation,
                                            max_iter_deflation,
                                            lsdefop,
                                            perturb_guess,
                                            kwargs...)

    multicontinuation(br,
                    bpnf, defOpm, defOpp, options_cont;
                    δp,
                    Teigvec,
                    verbosedeflation,
                    max_iter_deflation,
                    lsdefop,
                    kwargs...)
end

"""
$(TYPEDSIGNATURES)

Automatic branch switching at branch points based on a computation of the normal form. More information is provided in [Branch switching](@ref). An example of use is provided in [2d generalized Bratu–Gelfand problem](@ref).

# Arguments
- `br` branch result from a call to [`continuation`](@ref)
- `bpnf` normal form
- `defOpm::DeflationOperator, defOpp::DeflationOperator` to specify converged points on non-trivial branches before/after the bifurcation points. The points are located in `defOpm.roots` and `defOpp.roots`. Note that we only continue from the second points in the roots vectors, the first one is meant to be the trivial branch.

The rest is as the regular `multicontinuation` function.
"""
function multicontinuation(br::AbstractBranchResult,
                           bpnf::NdBranchPoint,
                           defOpm::DeflationOperator,
                           defOpp::DeflationOperator,
                           options_cont::ContinuationPar = br.contparams ;
                           alg = getalg(br),
                           δp = nothing,
                           Teigvec = _getvectortype(br),
                           verbosedeflation = false,
                           max_iter_deflation = min(50, 15options_cont.newton_options.max_iterations),
                           lsdefop = DeflatedProblemCustomLS(),
                           plot_solution = plot_solution(getprob(br)),
                           kwargs...)

    ds = isnothing(δp) ? options_cont.ds : δp |> abs
    dscont = abs(options_cont.ds)
    par = bpnf.params
    prob = re_make(br.prob; plot_solution)

    # compute the different branches
    function _continue(_sol, _dp, _ds)
        # needed to reset the tangent algorithm in case fields are used
        println("━"^50)
        continuation(prob,
            bpnf.x0, par,       # first point on the branch
            _sol, bpnf.p + _dp, # second point on the branch
            empty(alg), getlens(br),
            (@set options_cont.ds = _ds); kwargs...)
    end

    branches = Branch[]
    for id in 2:length(defOpm)
        br = _continue(defOpm[id], -ds, -dscont); push!(branches, Branch(br, bpnf))
        # br = _continue(defOpm[id], -ds, dscont); push!(branches, Branch(br, bpnf))
    end

    for id in 2:length(defOpp)
        br = _continue(defOpp[id], ds, dscont); push!(branches, Branch(br, bpnf))
        # br = _continue(defOpp[id], ds, -dscont); push!(branches, Branch(br, bpnf))
    end

    return branches
end

# same but for a Branch
multicontinuation(br::Branch, ind_bif::Int, options_cont::ContinuationPar = br.contparams; kwargs...) = multicontinuation(get_contresult(br), ind_bif, options_cont ; kwargs...)
