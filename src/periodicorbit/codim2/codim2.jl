function d2PO(f, x, dx1, dx2)
   return ForwardDiff.derivative(t2 -> ForwardDiff.derivative( t1 -> f(x .+ t1 .* dx1 .+ t2 .* dx2,), 0.), 0.)
end

struct FloquetWrapperBLS{T} <: AbstractBorderedLinearSolver
    solver::T # use solver as a field is good for BLS
end

(ls::FloquetWrapperBLS)(J, args...; k...) = ls.solver(J, args...; k...)
(ls::FloquetWrapperBLS)(J::FloquetWrapper, args...; k...) = ls.solver(J.jacpb, args...; k...)
Base.transpose(J::FloquetWrapper) = transpose(J.jacpb)

for op in (:NeimarkSackerProblemMinimallyAugmented,
            :PeriodDoublingProblemMinimallyAugmented)
    @eval begin
        """
        $(TYPEDEF)

        Structure to encode functional based on a Minimally Augmented formulation.

        # Fields

        $(FIELDS)
        """
        mutable struct $op{Tprob <: AbstractBifurcationProblem, vectype, T <: Real, S <: AbstractLinearSolver, Sa <: AbstractLinearSolver, Sbd <: AbstractBorderedLinearSolver, Sbda <: AbstractBorderedLinearSolver, Tmass} <: AbstractProblemMinimallyAugmented{Tprob}
            "Functional F(x, p) - vector field - with all derivatives"
            prob_vf::Tprob
            "close to null vector of Jᵗ"
            a::vectype
            "close to null vector of J"
            b::vectype
            "vector zero, to avoid allocating it many times"
            zero::vectype
            "Lyapunov coefficient"
            l1::Complex{T}
            "Cusp test value"
            CP::T
            "Bogdanov-Takens test value"
            FOLDNS::T
            "Generalised period douling test value"
            GPD::T
            "Fold-NS test values"
            FLIPNS::Int
            "linear solver. Used to invert the jacobian of MA functional"
            linsolver::S
            "linear solver for the jacobian adjoint"
            linsolverAdjoint::Sa
            "bordered linear solver"
            linbdsolver::Sbd
            "linear bordered solver for the jacobian adjoint"
            linbdsolverAdjoint::Sbda
            "wether to use the hessian of prob_vf"
            usehessian::Bool
            "wether to use a mass matrix M for studying M⋅∂tu = F(u), default = I"
            massmatrix::Tmass
        end

        @inline getdelta(pb::$op) = getdelta(pb.prob_vf)
        @inline has_hessian(pb::$op) = has_hessian(pb.prob_vf)
        @inline is_symmetric(pb::$op) = is_symmetric(pb.prob_vf)
        @inline has_adjoint(pb::$op) = has_adjoint(pb.prob_vf)
        @inline has_adjoint_MF(pb::$op) = has_adjoint_MF(pb.prob_vf)
        @inline isinplace(pb::$op) = isinplace(pb.prob_vf)
        @inline getlens(pb::$op) = getlens(pb.prob_vf)
        jad(pb::$op, args...) = jad(pb.prob_vf, args...)

        # constructors
        function $op(prob, a, b, linsolve::AbstractLinearSolver, linbdsolver = MatrixBLS(); usehessian = true, massmatrix = LinearAlgebra.I)
            # determine scalar type associated to vectors a and b
            α = norm(a) # this is valid, see https://jutho.github.io/KrylovKit.jl/stable/#Package-features-and-alternatives-1
            Ty = eltype(α)
            return $op(prob, a, b, 0*a,
                        complex(zero(Ty)), # l1
                        real(one(Ty)),     # cp
                        real(one(Ty)),     # fold-ns
                        real(one(Ty)),     # gpd
                        1,                 # flip-ns
                        linsolve, linsolve, linbdsolver, linbdsolver, usehessian, massmatrix)
        end

        # empty constructor, mainly used for dispatch
        function $op(prob ;linsolve = DefaultLS(), linbdsolver = MatrixBLS(), usehessian = true, massmatrix = LinearAlgebra.I)
            a = b = 0.
            α = norm(a) 
            Ty = eltype(α)
            return $op(prob, a, b, 0*a,
                        complex(zero(Ty)), # l1
                        real(one(Ty)),     # cp
                        real(one(Ty)),     # fold-ns
                        real(one(Ty)),     # gpd
                        1,                 # flip-ns
                        linsolve, linsolve, linbdsolver, linbdsolver, usehessian, massmatrix)
        end
    end
end

function correct_bifurcation(contres::ContResult{<: Union{FoldPeriodicOrbitCont, PDPeriodicOrbitCont, NSPeriodicOrbitCont}})
    if contres.prob.prob isa FoldProblemMinimallyAugmented
        conversion = Dict(:bp => :R1, :hopf => :foldNS, :fold => :cusp, :nd => :nd, :pd => :foldpd, :bt => :R1, :zh => :R1)
    elseif contres.prob.prob isa PeriodDoublingProblemMinimallyAugmented
        conversion = Dict(:bp => :foldFlip, :hopf => :pdNS, :pd => :R2)
    elseif contres.prob.prob isa NeimarkSackerProblemMinimallyAugmented
        conversion = Dict(:bp => :foldNS, :hopf => :nsns, :pd => :pdNS, :R1ch => :R1, :fold => :R1)
    else
        throw("Error! this should not occur. Please open an issue on the website of BifurcationKit.jl")
    end
    for (ind, bp) in pairs(contres.specialpoint)
        if bp.type in keys(conversion)
            @set! contres.specialpoint[ind].type = conversion[bp.type]
        end
    end
    return contres
end

_wrap(prob::PeriodicOrbitOCollProblem, args...) = WrapPOColl(prob, args...)
_wrap(prob::ShootingProblem, args...) = WrapPOSh(prob, args...)
_wrap(prob::PeriodicOrbitTrapProblem, args...) = WrapPOTrap(prob, args...)
####################################################################################################
function (finalizer::Finaliser{<: AbstractMABifurcationProblem})(z, tau, step, contResult; bisection = false, kF...)
    updateSectionEveryStep = finalizer.updateSectionEveryStep
    # we first check that the continuation step was successful
    # if not, we do not update the problem with bad information
    success = converged(get(kF, :state, nothing))
    if success && mod_counter(step, updateSectionEveryStep) == 1 && ~bisection
        # we get the MA problem
        wrap_ma = finalizer.prob
        𝐏𝐛 = wrap_ma.prob
        prob_sh = 𝐏𝐛.prob_vf.prob
        # we get the state vector at bifurcation point
        x = getvec(z.u, 𝐏𝐛)
        # we get the parameters at the bifurcation point
        lenses = get_lenses(wrap_ma)
        p1, = getp(z.u, 𝐏𝐛)   # first parameter, ca bug pour Folds si p1,_ = getp(...)
        p2 = z.p              # second parameter
        pars = set(getparams(prob_sh), lenses, (p1, p2))
        @debug "[Periodic orbit] update section"
        updatesection!(prob_sh, x, pars)
    end
    if isnothing(finalizer.finalise_solution)
        return true
    else
        return finalizer.finalise_solution(z, tau, step, contResult; prob = finalizer.prob, kF...)
    end
end

function (finalizer::Finaliser{<: AbstractMABifurcationProblem{ <: AbstractProblemMinimallyAugmented{ <: WrapPOColl}}})(Z, tau, step, contResult; bisection = false, kF...)
    updateSectionEveryStep = finalizer.updateSectionEveryStep
    𝐏𝐛 = finalizer.prob.prob
    coll = 𝐏𝐛.prob_vf.prob
     # we get the state vector at bifurcation point
     x = getvec(Z.u, 𝐏𝐛)
    # we first check that the continuation step was successful
    # if not, we do not update the problem with bad information
    state = get(kF, :state, nothing)
    success = isnothing(state) ? false : converged(state)
    # mesh adaptation
    if success && coll.meshadapt && ~bisection
        @debug "[Collocation] update mesh"
        oldsol = _copy(x) # avoid possible overwrite in compute_error!
        oldmesh = get_times(coll) .* getperiod(coll, oldsol, nothing)
        adapt = compute_error!(coll, oldsol;
                    verbosity = coll.verbose_mesh_adapt,
                    K = coll.K
                    )
        if ~adapt.success # stop continuation if mesh adaptation fails
            return false
        end
    end
    if success && mod_counter(step, updateSectionEveryStep) == 1 && ~bisection
        @debug "[collocation] update section"
        updatesection!(coll, x, nothing) # collocation does not need the parameter for updatesection!
    end
    if isnothing(finalizer.finalise_solution)
        return true
    else
        return finalizer.finalise_solution(Z, tau, step, contResult; prob = coll, kF...)
    end
end
####################################################################################################
function continuation(br::AbstractResult{Tkind, Tprob}, ind_bif::Int,
                    options_cont::ContinuationPar,
                    probPO::AbstractPeriodicOrbitProblem;
                    detect_codim2_bifurcation::Int = 0,
                    autodiff = true,
                    kwargs...) where {Tkind, Tprob <: Union{HopfMAProblem}}
    verbose = get(kwargs, :verbosity, 0) > 1 ? true : false
    verbose && (println("──▶ Considering bifurcation point:"); _show(stdout, br.specialpoint[ind_bif], ind_bif))
    nf = get_normal_form(getprob(br), br, ind_bif; detailed = true, autodiff)
    _contParams = detect_codim2_parameters(detect_codim2_bifurcation, options_cont; kwargs...)
    @set! _contParams.newton_options.eigsolver = getsolver(_contParams.newton_options.eigsolver)
    return _continuation(nf, br, _contParams, probPO; kwargs...)
end

function _continuation(gh::Bautin, br::AbstractResult{Tkind, Tprob},
                        _contParams::ContinuationPar,
                        probPO::AbstractPeriodicOrbitProblem;
                        alg = br.alg,
                        linear_algo = nothing,
                        δp = nothing, ampfactor::Real = 1,
                        nev = _contParams.nev,
                        detect_codim2_bifurcation::Int = 0,
                        Teigvec = getvectortype(br),
                        scaleζ = norm,
                        # start_with_eigen = false,
                        Jᵗ = nothing,
                        bdlinsolver::AbstractBorderedLinearSolver = getprob(br).prob.linbdsolver,
                        kwargs...) where {Tkind, Tprob <: Union{HopfMAProblem}}
    verbose = get(kwargs, :verbosity, 0) > 1 ? true : false
    # compute predictor for point on new branch
    ds = isnothing(δp) ? _contParams.ds : δp |> abs
    𝒯 = typeof(ds)
    pred = predictor(gh, Val(:FoldPeriodicOrbitCont), ds; verbose = verbose, ampfactor = 𝒯(ampfactor))
    pred0 = predictor(gh, Val(:FoldPeriodicOrbitCont), 0; verbose = verbose, ampfactor = 𝒯(ampfactor))

    M = get_mesh_size(probPO)
    ϕ = 0
    orbitguess_a = [pred.orbit(t - ϕ) for t in LinRange(0, 2pi, M + 1)[1:M]]

    # extract the vector field and use it possibly to affect the PO functional
    lens1, lens2 = gh.lens
    newparams = set(gh.params, lens1, pred.params[1])
    newparams = set(newparams, lens2, pred.params[2])

    prob_ma = getprob(br).prob
    prob_vf = re_make(prob_ma.prob_vf, params = newparams)

    # build the variable to hold the functional for computing PO based on finite differences
    probPO, orbitguess = re_make(probPO, prob_vf, gh, gh.ζ, orbitguess_a, abs(2pi/pred.ω); orbit = pred.orbit)

    verbose && printstyled(color = :green, "━"^61*
            "\n┌─ Start branching from Bautin bif. point to folds of periodic orbits.",
            "\n├─── Bautin params = ", pred0.params,
            "\n├─── new params  p = ", pred.params, ", p - p0 = ", pred.params - pred0.params,
            "\n├─── period      T = ", 2pi / pred.ω, " (from T = $(2pi / pred0.ω))",
            "\n├─ Method = \n", probPO, "\n")

    if _contParams.newton_options.linsolver isa GMRESIterativeSolvers
        _contParams = @set _contParams.newton_options.linsolver.N = length(orbitguess)
    end

    # change the user provided functions by passing probPO in its parameters
    if probPO isa PeriodicOrbitOCollProblem
        _finsol = modify_po_finalise(FoldMAProblem(FoldProblemMinimallyAugmented(WrapPOColl(probPO))), kwargs, probPO.update_section_every_step)
    else
        _finsol = modify_po_finalise(FoldMAProblem(FoldProblemMinimallyAugmented(WrapPOSh(probPO))), kwargs, probPO.update_section_every_step)
    end
    _recordsol = modify_po_record(probPO, kwargs, getparams(probPO), getlens(probPO))
    _plotsol = modify_po_plot(probPO, kwargs)

    jac = build_jacobian(probPO, orbitguess, getparams(probPO); δ = getdelta(prob_vf))
    pbwrap = _wrap(probPO, jac, orbitguess, getparams(probPO), getlens(probPO), _plotsol, _recordsol)

    # we have to change the Bordered linearsolver to cope with our type FloquetWrapper
    options = _contParams.newton_options
    _linear_algo = isnothing(linear_algo) ?  MatrixBLS() : linear_algo
    linear_algo = @set _linear_algo.solver = FloquetWrapperLS(_linear_algo.solver)
    alg = update(alg, _contParams, linear_algo)

    contParams = (@set _contParams.newton_options.linsolver = FloquetWrapperLS(options.linsolver));

    # set second derivative
    probsh_fold = BifurcationProblem((x, p) -> residual(pbwrap, x, p), orbitguess, getparams(pbwrap), getlens(pbwrap);
                J = (x, p) -> jacobian(pbwrap, x, p),
                Jᵗ = Jᵗ,
                d2F = (x, p, dx1, dx2) -> d2PO(z -> probPO(z, p), x, dx1, dx2),
                record_from_solution = _recordsol,
                plot_solution = _plotsol,
                )

    # create fold point guess
    foldpointguess = BorderedArray(orbitguess, get(newparams, lens1))
    
    # get the approximate null vectors
    jacpo = jacobian(probsh_fold, orbitguess, getparams(probsh_fold)).jacpb
    ls = DefaultLS()
    nj = length(orbitguess)
    p = rand(nj); q = rand(nj)
    rhs = zero(orbitguess); #rhs[end] = 1
    q, = bdlinsolver(jacpo, p, q, 0, rhs, 1); q ./= norm(q) #≈ ker(J)
    p, = bdlinsolver(jacpo', q, p, 0, rhs, 1); p ./= norm(p)

    q, = bdlinsolver(jacpo, p, q, 0, rhs, 1); q ./= norm(q) #≈ ker(J)
    p, = bdlinsolver(jacpo', q, p, 0, rhs, 1); p ./= norm(p)

    @assert sum(isnan, q) == 0 "Please report this error to the website."

    # perform continuation
    branch = continuation_fold(probsh_fold, alg,
        foldpointguess, getparams(probsh_fold),
        lens1, lens2,
        # p, q,
        q, p,
        contParams;
        kind = FoldPeriodicOrbitCont(),
        kwargs...,
        bdlinsolver = FloquetWrapperBLS(bdlinsolver),
        finalise_solution = _finsol
    )
    return Branch(branch, gh)
end

function _continuation(hh::HopfHopf, br::AbstractResult{Tkind, Tprob},
            _contParams::ContinuationPar,
            probPO::AbstractPeriodicOrbitProblem;
            whichns::Int = 1,
            alg = br.alg,
            linear_algo = nothing,
            δp = nothing, ampfactor::Real = 1,
            nev = _contParams.nev,
            detect_codim2_bifurcation::Int = 0,
            Teigvec = getvectortype(br),
            scaleζ = norm,
            Jᵗ = nothing,
            eigsolver = FloquetQaD(getsolver(_contParams.newton_options.eigsolver)),
            bdlinsolver::AbstractBorderedLinearSolver = getprob(br).prob.linbdsolver,
            record_from_solution = nothing,
            plot_solution = nothing,
            kwargs...) where {Tkind, Tprob <: Union{HopfMAProblem}}
    @assert whichns in (1, 2) "This parameter must belong to {1,2}."
    verbose = get(kwargs, :verbosity, 0) > 1 ? true : false
    
    # compute predictor for point on new branch
    ds = isnothing(δp) ? _contParams.ds : δp |> abs
    𝒯 = typeof(ds)
    pred = predictor(hh, Val(:NS), ds; verbose = verbose, ampfactor = 𝒯(ampfactor))
    pred0 = predictor(hh, Val(:NS), 0; verbose = verbose, ampfactor = 𝒯(ampfactor))

    _orbit = whichns == 1 ? pred.ns1 : pred.ns2
    period = whichns == 1 ? pred.T1 : pred.T2
    period0 = whichns == 1 ? pred0.T1 : pred0.T2

    M = get_mesh_size(probPO)
    ϕ = 0
    orbitguess_a = [_orbit(t - ϕ) for t in LinRange(0, 2pi, M + 1)[1:M]]

    # extract the vector field and use it possibly to affect the PO functional
    lens1, lens2 = hh.lens
    _params = whichns == 1 ? pred.params1 : pred.params2
    newparams = set(hh.params, lens1, _params[1])
    newparams = set(newparams, lens2, _params[2])

    prob_ma = getprob(br).prob
    prob_vf = re_make(prob_ma.prob_vf, params = newparams)

    @assert lens1 == getlens(prob_vf) "Please open an issue on the website of BifurcationKit"

    # build the variable to hold the functional for computing PO based on finite differences
    probPO, orbitguess = re_make(probPO, prob_vf, hh, hh.ζ.q1, orbitguess_a, period; orbit = _orbit)

    verbose && printstyled(color = :green, "━"^61*
        "\n┌─ Start branching from Hopf-Hopf bif. point to curve of Neimark-Sacker bifurcations of periodic orbits.",
        "\n├─── Hopf-Hopf params = ", pred0.params1,
        "\n├─── new params     p = ", _params, ", p - p0 = ", _params - pred0.params1,
        "\n├─── period         T = ", period, " (from T = $(period0))",
        "\n├─ Method = \n", probPO, "\n")

    if _contParams.newton_options.linsolver isa GMRESIterativeSolvers
        _contParams = @set _contParams.newton_options.linsolver.N = length(orbitguess)
    end

    contParams = compute_eigenelements(_contParams) ? (@set _contParams.newton_options.eigsolver = eigsolver) : _contParams

    # this is to remove this part from the arguments passed to continuation
    _kwargs = (record_from_solution = record_from_solution, plot_solution = plot_solution)
    _recordsol = modify_po_record(probPO, _kwargs, getparams(probPO), getlens(probPO))
    _plotsol = modify_po_plot(probPO, _kwargs)

    jac = build_jacobian(probPO, orbitguess, getparams(probPO); δ = getdelta(prob_vf))
    pbwrap = _wrap(probPO, jac, orbitguess, getparams(probPO), getlens(probPO), _plotsol, _recordsol)

    # we have to change the Bordered linearsolver to cope with our type FloquetWrapper
    options = _contParams.newton_options
    _linear_algo = isnothing(linear_algo) ?  MatrixBLS() : linear_algo
    linear_algo = @set _linear_algo.solver = FloquetWrapperLS(_linear_algo.solver)
    alg = update(alg, _contParams, linear_algo)

    contParams = (@set contParams.newton_options.linsolver = FloquetWrapperLS(options.linsolver));

    # create fold point guess
    ωₙₛ = whichns == 1 ? pred.k1 : pred.k2
    nspointguess = BorderedArray(_copy(orbitguess), [get(newparams, lens1), ωₙₛ])
    
    # get the approximate null vectors
    if pbwrap isa WrapPOColl
        @debug "Collocation, get borders"
        jac = jacobian(pbwrap, orbitguess, getparams(pbwrap))
        J = Complex.(copy(jac.jacpb))
        nj = size(J, 1)
        J[end, :] .= rand(nj) #must be close to eigensapce
        J[:, end] .= rand(nj)
        J[end, end] = 0
        # enforce NS boundary condition
        N, m, Ntst = size(probPO)
        J[end-N:end-1, end-N:end-1] .= UniformScaling(cis(ωₙₛ))(N)

        rhs = zeros(nj); rhs[end] = 1
        q = J  \ rhs; q = q[1:end-1]; q ./= norm(q) #≈ ker(J)
        p = J' \ rhs; p = p[1:end-1]; p ./= norm(p)

        @set! contParams.newton_options.eigsolver = FloquetColl()
    else
        @debug "Shooting, get borders"
        J = jacobian_neimark_sacker(pbwrap, orbitguess, getparams(pbwrap), ωₙₛ)
        nj = length(orbitguess)-1
        q, = bdlinsolver(J, Complex.(rand(nj)), Complex.(rand(nj)), 0, Complex.(zeros(nj)), 1)
        q ./= norm(q)
        p = conj(q)
    end

    @assert sum(isnan, q) == 0 "Please report this error to the website."

    # perform continuation
    # the finalizer is handled in NS continuation
    branch = continuation_ns(pbwrap, alg,
            nspointguess, getparams(pbwrap),
            lens1, lens2,
            p, q,
            # q, p,
            contParams;
            kind = NSPeriodicOrbitCont(),
            kwargs...,
            plot_solution = _plotsol,
            bdlinsolver = FloquetWrapperBLS(bdlinsolver),
    )
    return Branch(branch, hh)
end