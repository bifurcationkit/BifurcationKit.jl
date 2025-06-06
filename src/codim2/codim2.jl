abstract type AbstractProblemMinimallyAugmented{Tprob} end
abstract type AbstractCodim2EigenSolver <: AbstractEigenSolver end

getsolver(eig::AbstractCodim2EigenSolver) = eig.eigsolver

# function to get the two lenses associated to 2-param continuation
@inline function get_lenses(_prob::Union{FoldMAProblem,
                                         HopfMAProblem,
                                         PDMAProblem,
                                         NSMAProblem})
    prob_ma = _prob.prob
    return getlens(prob_ma), getlens(_prob)
end

@inline function get_lenses(br::AbstractResult{Tkind}) where Tkind <: TwoParamCont
    get_lenses(br.prob)
end

for op in (:FoldProblemMinimallyAugmented, :HopfProblemMinimallyAugmented)
    @eval begin
    """
    $(TYPEDEF)

    Structure to encode Fold / Hopf functional based on a Minimally Augmented formulation.

    # Fields

    $(FIELDS)
    """
    mutable struct $op{Tprob <: AbstractBifurcationProblem,
                       vectype,
                       T <: Real,
                       S <: AbstractLinearSolver,
                       Sa <: AbstractLinearSolver,
                       Sbd <: AbstractBorderedLinearSolver,
                       Sbda <: AbstractBorderedLinearSolver,
                       Tmass,
                       Tn} <: AbstractProblemMinimallyAugmented{Tprob}
        "Functional F(x, p) - vector field - with all derivatives."
        prob_vf::Tprob
        "close to null vector of Jᵗ."
        a::vectype
        "close to null vector of J."
        b::vectype
        "vector zero, to avoid allocating it many times."
        zero::vectype
        "Lyapunov coefficient."
        l1::Complex{T}
        "Cusp test value."
        CP::T
        "Bogdanov-Takens test value."
        BT::T
        "Bautin test values."
        GH::T
        "Zero-Hopf test values."
        ZH::Int
        "linear solver. Used to invert the jacobian of MA functional."
        linsolver::S
        "linear solver for the jacobian adjoint."
        linsolverAdjoint::Sa
        "bordered linear solver."
        linbdsolver::Sbd
        "linear bordered solver for the jacobian adjoint."
        linbdsolverAdjoint::Sbda
        "whether to use the hessian of prob_vf."
        usehessian::Bool
        "whether to use a mass matrix M for studying M⋅∂ₜu = F(u), default = I."
        massmatrix::Tmass
        "norm to normalize vector in update or test"
        norm::Tn
        "Update the problem every such step"
        update_minaug_every_step::Int
    end

    @inline getdelta(pb::$op) = getdelta(pb.prob_vf)
    @inline Base.eltype(pb::$op{Tprob, vectype, T}) where {Tprob, vectype, T} = T
    @inline has_hessian(pb::$op) = has_hessian(pb.prob_vf)
    @inline is_symmetric(pb::$op) = is_symmetric(pb.prob_vf)
    @inline has_adjoint(pb::$op) = has_adjoint(pb.prob_vf)
    @inline has_adjoint_MF(pb::$op) = has_adjoint_MF(pb.prob_vf)
    @inline isinplace(pb::$op) = isinplace(pb.prob_vf)
    @inline getlens(pb::$op) = getlens(pb.prob_vf)
    jad(pb::$op, args...) = jad(pb.prob_vf, args...)

    # constructor
    function $op(prob, a, b,
                    linsolve::AbstractLinearSolver,
                    linbdsolver = MatrixBLS();
                    linsolve_adjoint = linsolve,
                    usehessian = true,
                    massmatrix = LinearAlgebra.I,
                    linbdsolve_adjoint = linbdsolver,
                    _norm = norm,
                    update_minaug_every_step = 0)
        # determine scalar type associated to vectors a and b
        α = norm(a) # this is valid, see https://jutho.github.io/KrylovKit.jl/stable/#Package-features-and-alternatives-1
        𝒯 = eltype(α)
        return $op(prob, a, b, 0*a,
                    complex(zero(𝒯)), # l1
                    real(one(𝒯)),     # cp
                    real(one(𝒯)),     # bt
                    real(one(𝒯)),     # gh
                    1,                # zh
                    linsolve, linsolve_adjoint, linbdsolver, linbdsolve_adjoint, usehessian, massmatrix,
                    _norm, update_minaug_every_step)
    end

    # empty constructor, mainly used for dispatch
    function $op(prob ;linsolve = DefaultLS(),
                    linbdsolver = MatrixBLS(),
                    usehessian = true,
                    massmatrix = LinearAlgebra.I,
                    _norm = norm,
                    update_minaug_every_step = 0)
        a = b = 0.
        α = norm(a) 
        𝒯 = eltype(α)
        return $op(prob, a, b, 0*a,
                    complex(zero(𝒯)), # l1
                    real(one(𝒯)),     # cp
                    real(one(𝒯)),     # bt
                    real(one(𝒯)),     # gh
                    1,                # zh
                    linsolve, linsolve, linbdsolver, linbdsolver, usehessian, massmatrix, _norm, update_minaug_every_step)
    end
    end
end
@inline getvec(x, ::FoldProblemMinimallyAugmented) = get_vec_bls(x)
@inline getvec(x, ::HopfProblemMinimallyAugmented) = get_vec_bls(x, 2)
@inline getp(x, ::FoldProblemMinimallyAugmented) = get_par_bls(x)
@inline getp(x, ::HopfProblemMinimallyAugmented) = get_par_bls(x, 2)

update!(::FoldMAProblem, args...; k...) = update_default(args...; k...)
update!(::HopfMAProblem, args...; k...) = update_default(args...; k...)
################################################################################
function detect_codim2_parameters(detect_codim2_bifurcation, options_cont; 
                                    update_minaug_every_step = 1, 
                                    kwargs...)
    if detect_codim2_bifurcation > 0
        if update_minaug_every_step == 0
            @warn "You asked for detection of codim 2 bifurcations but passed the option `update_minaug_every_step = 0`.\n The bifurcation detection algorithm may not work faithfully.\n Please use `update_minaug_every_step > 0`."
        end
        return ContinuationPar(options_cont; 
                    detect_bifurcation = 0,
                    detect_event = detect_codim2_bifurcation,
                    detect_fold = false)
    else
        return options_cont
    end
end

function Base.show(io::IO, prob::AbstractMABifurcationProblem; prefix = "")
    print(io, prefix * "┌─ Problem for bif. points continuation with uType ")
    printstyled(io, _getvectortype(prob), color = :cyan, bold = true)
    print(io, "\n" * prefix * "├─ Inplace:  ")
    printstyled(io, isinplace(prob), color = :cyan, bold = true)
    print(io, "\n" * prefix * "├─ Dimension:  ")
    printstyled(io, length(getu0(prob)), color = :cyan, bold = true)
    print(io, "\n" * prefix * "├─ Jacobian: ")
    printstyled(io, prob.jacobian, color = :cyan, bold = true)
    print(io, "\n" * prefix * "└─ Parameter: ")
    printstyled(io, get_lens_symbol(getlens(prob)), color = :cyan, bold = true)
end

function Base.show(io::IO, prob::AbstractProblemMinimallyAugmented{Tprob}; prefix = "") where {Tprob}
    print(io, prefix * "┌─ Minimally Augmented Problem continuation")
    print(io, "\n" * prefix * "├─ use hessian:  ")
    printstyled(io, prob.usehessian, color = :cyan, bold = true)
    print(io, "\n" * prefix * "├─ linear solver:  ")
    printstyled(io, prob.linsolver, color = :cyan, bold = true)
    print(io, "\n" * prefix * "├─ linear solver for adjoint:  ")
    printstyled(io, prob.linsolverAdjoint, color = :cyan, bold = true)
    print(io, "\n" * prefix * "├─ linear solver for adjoint:  ")
    printstyled(io, prob.linsolverAdjoint, color = :cyan, bold = true)
    print(io, "\n" * prefix * "├─ linear bordered solver for the jacobian:  ")
    printstyled(io, prob.linbdsolver, color = :cyan, bold = true)
    print(io, "\n" * prefix * "├─ linear bordered solver for the jacobian adjoint:  ")
    printstyled(io, prob.linbdsolverAdjoint, color = :cyan, bold = true)
    # print(io, "\n" * prefix * "├─ Dimension:  ")
    # printstyled(io, length(getu0(prob)), color = :cyan, bold = true)
    # print(io, "\n" * prefix * "├─ Jacobian: ")
    # printstyled(io, prob.jacobian, color = :cyan, bold = true)
    # print(io, "\n" * prefix * "└─ Parameter: ")
    # printstyled(io, get_lens_symbol(getlens(prob)), color = :cyan, bold = true)
end
################################################################################
function get_bif_point_codim2(br::AbstractResult{Tkind, Tprob}, ind::Int) where {Tkind, Tprob <: Union{FoldMAProblem, HopfMAProblem, PDMAProblem, NSMAProblem}}
    prob_ma = getprob(br).prob
    𝒯 = _getvectortype(br)

    bifpt = br.specialpoint[ind]
    # get the BT point. We perform a conversion in case GPU is used
    if 𝒯 <: BorderedArray
        x0 = convert(𝒯.parameters[1], getvec(bifpt.x, prob_ma))
    else
        x0 = convert(𝒯, getvec(bifpt.x , prob_ma))
    end

    # parameters for vector field
    p1 = getp(bifpt.x , prob_ma)[1] # get(bifpt.printsol, lens1)
    p2 = bifpt.param
    lenses = get_lenses(br)
    parbif = _set(getparams(br), lenses, (p1, p2))
    return (x = x0, params = parbif)
end

function getparams(br::AbstractResult{Tkind}, ind::Int) where Tkind <: TwoParamCont
    prob_ma = getprob(br).prob
    lenses = get_lenses(br)
    p1 = getp(br.sol[ind].x , prob_ma)[1]
    p2 = br.sol[ind].p
    lenses = get_lenses(br)
    parbif = _set(getparams(br), lenses, (p1, p2))
end
################################################################################
"""
$(SIGNATURES)

This function turns an initial guess for a Fold / Hopf point into a solution to the Fold / Hopf problem based on a Minimally Augmented formulation.

## Arguments
- `br` results returned after a call to [continuation](@ref Library-Continuation)
- `ind_bif` bifurcation index in `br`

# Optional arguments:
- `options::NewtonPar`, default value `br.contparams.newton_options`
- `normN = norm`
- `options` You can pass newton parameters different from the ones stored in `br` by using this argument `options`.
- `bdlinsolver` bordered linear solver for the constraint equation
- `start_with_eigen = false` whether to start the Minimally Augmented problem with information from eigen elements.
- `kwargs` keywords arguments to be passed to the regular Newton-Krylov solver

!!! tip "ODE problems"
    For ODE problems, it is more efficient to use the Matrix based Bordered Linear Solver passing the option `bdlinsolver = MatrixBLS()`

!!! tip "start_with_eigen"
    It is recommended that you use the option `start_with_eigen=true`
"""
function newton(br::AbstractBranchResult,
                ind_bif::Int64; 
                normN = norm,
                options = br.contparams.newton_options,
                start_with_eigen = false,
                lens2::AllOpticTypes = (@optic _),
                kwargs...)
    if isempty(br.specialpoint)
        error("The branch does not contain bifurcation points")
    end
    if br.specialpoint[ind_bif].type == :hopf
        return newton_hopf(br, ind_bif; normN, options, start_with_eigen, kwargs...)
    elseif br.specialpoint[ind_bif].type == :bt
        return newton_bt(br, ind_bif; lens2, normN, options, start_with_eigen, kwargs...)
    else
        return newton_fold(br, ind_bif; normN, options, start_with_eigen, kwargs...)
    end
end
################################################################################
"""
$(SIGNATURES)

Codimension 2 continuation of Fold / Hopf points. This function turns an initial guess for a Fold / Hopf point into a curve of Fold / Hopf points based on a Minimally Augmented formulation. The arguments are as follows
- `br` results returned after a call to [continuation](@ref Library-Continuation)
- `ind_bif` bifurcation index in `br`
- `lens2` second parameter used for the continuation, the first one is the one used to compute `br`, e.g. `getlens(br)`
- `options_cont = br.contparams` arguments to be passed to the regular [continuation](@ref Library-Continuation)

# Optional arguments:
- `linsolve_adjoint` solver for (J+iω)˟ ⋅sol = rhs or Jᵗ ⋅sol = rhs
- `bdlinsolver` bordered linear solver for the constraint equation
- `bdlinsolver_adjoint` bordered linear solver for the constraint equation with top-left block (J-iω)˟ or Jᵗ. Required in the linear solver for the Minimally Augmented Fold/Hopf functional. This option can be used to pass a dedicated linear solver for example with specific preconditioner.
- `update_minaug_every_step` update vectors `a, b` in Minimally Formulation every `update_minaug_every_step` steps
- `detect_codim2_bifurcation ∈ {0,1,2}` whether to detect Bogdanov-Takens, Bautin and Cusp. If equals `1` non precise detection is used. If equals `2`, a bisection method is used to locate the bifurcations. Default value = 2.
- `start_with_eigen = false` whether to start the Minimally Augmented problem with information from eigen elements. If `start_with_eigen = false`, then:

    - `a::Nothing` estimate of null vector of J (resp. J-iω) for Fold (resp. Hopf). If nothing is passed, a random vector is used. In case you do not rely on `AbstractArray`, you should probably pass this.
    - `b::Nothing` estimate of null vector of Jᵗ (resp. (J-iω)˟) for Fold (resp. Hopf). If nothing is passed, a random vector is used. In case you do not rely on `AbstractArray`, you should probably pass this.

- `kwargs` keywords arguments to be passed to the regular [continuation](@ref Library-Continuation)

where the parameters are as above except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of Hopf point in `br` you want to refine.

!!! tip "ODE problems"
    For ODE problems, it is more efficient to use the Matrix based Bordered Linear Solver passing the option `bdlinsolver = MatrixBLS()`

!!! tip "start_with_eigen"
    It is recommended that you use the option `start_with_eigen = true`
"""
function continuation(br::AbstractBranchResult,
            ind_bif,
            lens2::AllOpticTypes,
            options_cont::ContinuationPar = br.contparams ;
            prob = br.prob,
            start_with_eigen = false,
            detect_codim2_bifurcation = 2,
            update_minaug_every_step = 1,
            kwargs...)
    if isempty(br.specialpoint)
        error("The branch does not contain bifurcation points")
    end
    # options to detect codim2 bifurcations
    compute_eigen_elements = options_cont.detect_bifurcation > 0
    _options_cont = detect_codim2_parameters(detect_codim2_bifurcation, options_cont; update_minaug_every_step, kwargs...)

    if br.specialpoint[ind_bif].type == :hopf
        return continuation_hopf(prob, br, ind_bif, lens2, _options_cont;
            start_with_eigen,
            compute_eigen_elements,
            update_minaug_every_step,
            kwargs...)
    else
        return continuation_fold(prob, br, ind_bif, lens2, _options_cont;
            start_with_eigen,
            compute_eigen_elements,
            update_minaug_every_step,
            kwargs...)
    end
end
####################################################################################################
# branch switching at BT / ZH / HH bifurcation point
function continuation(br::AbstractResult{Tkind, Tprob}, ind_bif::Int,
                        options_cont::ContinuationPar = br.contparams;
                        alg = br.alg,
                        δp = nothing, 
                        ampfactor::Real = 1,
                        nev = options_cont.nev,
                        detect_codim2_bifurcation::Int = 0,
                        Teigvec = _getvectortype(br),
                        scaleζ = norm,
                        start_with_eigen = false,
                        autodiff_nf = false,
                        detailed = true,
                        ζs = nothing,
                        ζs_ad = nothing,
                        bdlinsolver::AbstractBorderedLinearSolver = getprob(br).prob.linbdsolver,
                        bdlinsolver_adjoint = bdlinsolver,
                        bdlinsolver_block = bdlinsolver,
                        kwargs...) where {Tkind <: TwoParamCont, Tprob <: Union{FoldMAProblem, HopfMAProblem}}

    verbose = get(kwargs, :verbosity, 0) > 0 ? true : false
    verbose && println("--> Considering bifurcation point:"); _show(stdout, br.specialpoint[ind_bif], ind_bif)

    bif_type = br.specialpoint[ind_bif].type
    @assert bif_type in (:bt, :zh, :hh) "Only branching from Bogdanov-Takens, Zero-Hopf and Hopf-Hopf is allowed."

    if bif_type == :hh
        @assert Tkind <: HopfCont
    end

    # functional
    prob_ma = br.prob.prob
    prob_vf = prob_ma.prob_vf

    # continuation parameters
    compute_eigen_elements = options_cont.detect_bifurcation > 0
    optionsCont = detect_codim2_parameters(detect_codim2_bifurcation, options_cont; kwargs...)

    # scalar type
    Ty = eltype(Teigvec)

    # compute the normal form of the bifurcation point
    nf = get_normal_form(br, ind_bif; 
                            nev,
                            verbose,
                            Teigvec,
                            scaleζ,
                            autodiff = autodiff_nf,
                            detailed,
                            bls = bdlinsolver,
                            bls_adjoint = bdlinsolver_adjoint,
                            bls_block = bdlinsolver_block,
                            ζs,
                            ζs_ad)

    # compute predictor for point on new branch
    ds = isnothing(δp) ? optionsCont.ds : δp

    if prob_ma isa FoldProblemMinimallyAugmented || bif_type == :hh
        # define guess for the first Hopf point on the branch
        pred = predictor(nf, Val(:HopfCurve), ds)

        # new continuation parameters
        parcont = pred.hopf(ds)

        # new full parameters
        params = set(set(nf.params, nf.lens[2], parcont[2]), nf.lens[1], parcont[1])

        # guess for the Hopf point
        hopfpt = BorderedArray(nf.x0 .+ pred.x0(ds), [parcont[1], pred.ω(ds)])

        # estimates for eigenvectors for ±iω
        ζ = pred.EigenVec(ds)
        ζstar = pred.EigenVecAd(ds)

        # put back original options
        @reset optionsCont.newton_options.eigsolver =
                getsolver(optionsCont.newton_options.eigsolver)
        @reset optionsCont.newton_options.linsolver = prob_ma.linsolver

        branch = continuation_hopf(prob_vf, alg,
            hopfpt, params,
            nf.lens...,
            ζ, ζstar,
            optionsCont;
            bdlinsolver = prob_ma.linbdsolver,
            compute_eigen_elements,
            kwargs...
            )
        return Branch(branch, nf)

    else
        @assert prob_ma isa HopfProblemMinimallyAugmented
        pred = predictor(nf, Val(:FoldCurve), 0.)

        # new continuation parameters
        parcont = pred.fold(ds)

        # new full parameters
        params = set(set(nf.params, nf.lens[2], parcont[2]), nf.lens[1], parcont[1])

        # guess for the fold point
        foldpt = BorderedArray(nf.x0 .+ 0 .* pred.x0(ds), parcont[1])

        # estimates for null eigenvectors
        ζ = pred.EigenVec(ds) |> real
        ζstar = pred.EigenVecAd(ds) |> real

        # put back original options
        @reset optionsCont.newton_options.eigsolver =
                getsolver(optionsCont.newton_options.eigsolver)
        @reset optionsCont.newton_options.linsolver = prob_ma.linsolver

        branch = continuation_fold(prob_vf, alg,
            foldpt, params,
            nf.lens...,
            ζ, ζstar,
            optionsCont;
            bdlinsolver = prob_ma.linbdsolver,
            compute_eigen_elements,
            kwargs...
            )
        return Branch(branch, nf)
    end
end
################################################################################
"""
$(SIGNATURES)

This function uses information in the branch to detect codim 2 bifurcations like BT, ZH and Cusp.
"""
function correct_bifurcation(contres::ContResult)
    if contres.prob.prob isa AbstractProblemMinimallyAugmented == false
        return contres
    end
    if contres.prob.prob isa FoldProblemMinimallyAugmented
        conversion = Dict(:bp => :bt, :hopf => :zh, :fold => :cusp, :nd => :nd, :btbp => :bt)
    elseif contres.prob.prob isa HopfProblemMinimallyAugmented
        conversion = Dict(:bp => :zh, :hopf => :hh, :fold => :nd, :nd => :nd, :ghbt => :bt, :btgh => :bt, :btbp => :bt)
    else
        throw("Error! this should not occur. Please open an issue on the website of BifurcationKit.jl")
    end
    for (ind, bp) in pairs(contres.specialpoint)
        if bp.type in keys(conversion)
            @reset contres.specialpoint[ind].type = conversion[bp.type]
        end
    end
    return contres
end
