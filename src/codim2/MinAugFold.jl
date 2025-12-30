"""
$(TYPEDSIGNATURES)

For an initial guess from the index of a Fold bifurcation point located in `ContResult.specialpoint`, returns a point which can can refined using `newtonFold`.
"""
function fold_point(br::AbstractBranchResult, index::Int)
    bptype = br.specialpoint[index].type
    if ~(bptype in (:bp, :nd, :fold))
        error("This should be a Fold / BP point.\nYou passed a $bptype point.")
    end
    specialpoint = br.specialpoint[index]
    return BorderedArray(_copy(specialpoint.x), specialpoint.param)
end
####################################################################################################
function (𝐅::FoldProblemMinimallyAugmented)(x, p::𝒯, params) where 𝒯
    # These are the equations of the minimally augmented (MA) formulation of the Fold bifurcation point
    # input:
    # - x guess for the point at which the jacobian is singular
    # - p guess for the parameter value `<: Real` at which the jacobian is singular
    # The jacobian of the MA problem is solved with a BLS method
    # ┌      ┐┌  ┐   ┌ ┐
    # │ J  a ││v │ = │0│
    # │ b  0 ││σ │   │1│
    # └      ┘└  ┘   └ ┘
    # In the notations of Govaerts 2000, a = w, b = v
    # Thus, b should be a null vector of J
    #       a should be a null vector of J'
    # we solve Jv + a σ1 = 0 with <b, v> = 1
    # the solution is v = -σ1 J\a with σ1 = -1/<b, J^{-1}a>
    a = 𝐅.a
    b = 𝐅.b
    # update parameter
    par = set(params, getlens(𝐅), p)
    J = jacobian(𝐅.prob_vf, x, par)
    _, σ, cv, = 𝐅.linbdsolver(J, a, b, zero(𝒯), 𝐅.zero, one(𝒯))
    ~cv && @debug "[Fold residual] Linear solver for J did not converge."
    return residual(𝐅.prob_vf, x, par), σ
end
###################################################################################################
"""
$(TYPEDSIGNATURES)

Compute the solution of 

```
┌              ┐┌  ┐   ┌   ┐
│ J     𝐅.a    ││v │ = │ 0 │
│ 𝐅.b'   0     ││σ │   │ 1 │
└              ┘└  ┘   └   ┘
```

and the same for the adjoint system.
"""
function _compute_bordered_vectors(𝐅::FoldProblemMinimallyAugmented, J_at_xp, JAd_at_xp)
    a = 𝐅.a
    b = 𝐅.b
    𝒯 = eltype(𝐅)

    # we solve Jv + a σ1 = 0 with <b, v> = 1
    # the solution is v = -σ1 J\a with σ1 = -1/<b, J\a>
    v, _, cv, itv = 𝐅.linbdsolver(J_at_xp, a, b, zero(𝒯), 𝐅.zero, one(𝒯))
    ~cv && @debug "Bordered linear solver for J did not converge. it = $(itv)"

    # we solve J'w + b σ2 = 0 with <a, w> = 1
    # the solution is w = -σ2 J'\b with σ2 = -1/<a, J'\b>
    w, _, cv, itw = 𝐅.linbdsolverAdjoint(JAd_at_xp, b, a, zero(𝒯), 𝐅.zero, one(𝒯))
    ~cv && @debug "Bordered linear solver for J' did not converge."

    return (; v, w, itv, itw, JAd_at_xp)
end

function _get_bordered_terms(𝐅::FoldProblemMinimallyAugmented, x, p::𝒯, par) where 𝒯
    # update parameter
    lens = getlens(𝐅)
    par0 = set(par, lens, p)

    # The jacobian is used at least 3 times below. This avoids doing 3 times the 
    # (possibly) costly building of J(x, p)
    J_at_xp = jacobian(𝐅.prob_vf, x, par0)
    # Avoid computing J_at_xp twice in case 𝐅.Jadjoint is not provided
    if is_symmetric(𝐅.prob_vf)
        JAd_at_xp = J_at_xp
    else
        JAd_at_xp = has_adjoint(𝐅) ? jacobian_adjoint(𝐅.prob_vf, x, par0) : transpose(J_at_xp)
    end


    (;v, w, itv, itw, JAd_at_xp) = _compute_bordered_vectors(𝐅, J_at_xp, JAd_at_xp)

    δ = getdelta(𝐅.prob_vf)
    ϵₚ = ϵₓ = ϵⱼ = 𝒯(δ)
    ################### computation of σx σp ####################
    dₚF = minus(residual(𝐅.prob_vf, x, set(par, lens, p + ϵₚ)),
                residual(𝐅.prob_vf, x, set(par, lens, p - ϵₚ)))
    VI.scale!(dₚF, 𝒯(1 / (2ϵₚ)))
    dJvdp = minus(apply(jacobian(𝐅.prob_vf, x, set(par, lens, p + ϵⱼ)), v),
                  apply(jacobian(𝐅.prob_vf, x, set(par, lens, p - ϵⱼ)), v));
    VI.scale!(dJvdp, 𝒯(1/(2ϵⱼ)))
    σₚ = -VI.inner(w, dJvdp)

    return (;J_at_xp, JAd_at_xp, dₚF, σₚ, δ, ϵₓ, v, w, par0, dJvdp, itv, itw)
end
###################################################################################################
function jacobian(pdpb::FoldMAProblem{Tprob, MinAugMatrixBased}, X::AbstractVector, par) where {Tprob}
    𝐅 = pdpb.prob
    x = @view X[begin:end-1]
    p = X[end]

    (;J_at_xp, JAd_at_xp, dₚF, σₚ, ϵₓ, v, w, par0) = _get_bordered_terms(𝐅, x, p, par)

    u1 = apply_jacobian(𝐅.prob_vf, x + ϵₓ * v, par0, w, true)
    u2 = apply(JAd_at_xp, w) # TODO we know u2!!
    σₓ = minus(u2, u1); VI.scale!(σₓ, 1 / ϵₓ)

    [J_at_xp dₚF ; σₓ' σₚ]
end
###################################################################################################
# Struct to invert the jacobian of the fold MA problem.
struct FoldLinearSolverMinAug <: AbstractLinearSolver; end

function foldMALinearSolver(x, p::𝒯, 𝐅::FoldProblemMinimallyAugmented, par,
                            rhsu, rhsp) where 𝒯
    # Recall that the functional we want to solve is [F(x,p), σ(x,p)] where σ(x,p) is computed in the 
    # function above. The Jacobian Jfold of the vector field is expressed at (x, p).
    # We solve Jfold⋅res = rhs := [rhsu, rhsp]
    # The Jacobian expression of the Fold problem is
    #           ┌         ┐
    #  Jfold =  │  J  dpF │
    #           │ σx   σp │
    #           └         ┘
    # where σx := ∂_xσ and σp := ∂_pσ
    # We recall the expression of
    #  σx = -< w, d2F(x,p)[v, x2]>
    # where (w, σ2) is solution of J'w + b σ2 = 0 with <a, w> = 1
    ################### inversion of Jfold ####################
    (;J_at_xp, JAd_at_xp, dₚF, σₚ, δ, ϵₓ, v, w, par0, itv, itw) = _get_bordered_terms(𝐅, x, p, par)

    if 𝐅.usehessian == false || has_hessian(𝐅) == false
        # We invert the jacobian of the Fold problem when the Hessian of x -> F(x, p) is not known analytically.
        # apply Jacobian adjoint
        u1 = apply_jacobian(𝐅.prob_vf, x + ϵₓ * v, par0, w, true)
        u2 = apply(JAd_at_xp, w) # TODO ON CONNAIT u2!!
        σₓ = minus(u2, u1); VI.scale!(σₓ, 1 / ϵₓ)
        ########## Resolution of the bordered linear system ########
        # we invert Jfold
        dX, dσ, cv, it = 𝐅.linbdsolver(J_at_xp, dₚF, σₓ, σₚ, rhsu, rhsp)
        ~cv && @debug "Bordered linear solver for J did not converge."
    else
        # We invert the jacobian of the Fold problem when the Hessian of x -> F(x, p) is known analytically.
        # we solve it here instead of calling linearBorderedSolver because this removes the need to pass the linear form associated to σₓ
        # !!! Careful, this method makes the linear system singular
        x1, x2, cv, it = 𝐅.linsolver(J_at_xp, rhsu, dₚF)
        ~cv && @debug "Linear solver for J did not converge."

        d2Fv = d2F(𝐅.prob_vf, x, par0, x1, v)
        σx1 = -VI.inner(w, d2Fv )

        _copyto!(d2Fv, d2F(𝐅.prob_vf, x, par0, x2, v))
        σx2 = -VI.inner(w, d2Fv )

        dσ = (rhsp - σx1) / (σₚ - σx2)

        # dX = @. x1 - dσ * x2
        dX = _copy(x1)
        VI.add!(dX, x2, -dσ)
    end
    return dX, dσ, true, sum(it) + sum(itv) + sum(itw)
end

function (foldl::FoldLinearSolverMinAug)(Jfold, du::BorderedArray{vectype, 𝒯}; kwargs...) where {vectype, 𝒯}
    # kwargs is used by AbstractLinearSolver
    out = foldMALinearSolver((Jfold.x).u,
                 (Jfold.x).p,
                 Jfold.pbma,
                 Jfold.params,
                 du.u, du.p)
    return BorderedArray{vectype, 𝒯}(out[1], out[2]), out[3], out[4]
end
###################################################################################################
@inline has_adjoint(foldpb::FoldMAProblem) = has_adjoint(foldpb.prob)
@inline is_symmetric(foldpb::FoldMAProblem) = is_symmetric(foldpb.prob)
jacobian_adjoint(foldpb::FoldMAProblem, args...) = jacobian_adjoint(foldpb.prob, args...)
###################################################################################################
"""
$(TYPEDSIGNATURES)

This function turns an initial guess for a Fold point into a solution to the Fold problem based on a Minimally Augmented formulation. The arguments are as follows
- `prob::AbstractBifurcationFunction`
- `foldpointguess` initial guess (x_0, p_0) for the Fold point. It should be a `BorderedArray` as returned by the function `foldpoint`
- `par` parameters used for the vector field
- `eigenvec` guess for the right null vector
- `eigenvec_ad` guess for the left null vector
- `options::NewtonPar` options for the Newton-Krylov algorithm, see [`NewtonPar`](@ref).

# Optional arguments:
- `normN = norm`
- `bdlinsolver` bordered linear solver for the constraint equation
- `kwargs` keywords arguments to be passed to the regular Newton-Krylov solver

# Simplified call
Simplified call to refine an initial guess for a Fold point. More precisely, the call is as follows

    newtonFold(br::AbstractBranchResult, ind_fold::Int; options = br.contparams.newton_options, kwargs...)

The parameters / options are as usual except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of bifurcation point in `br` you want to refine. You can pass newton parameters different from the ones stored in `br` by using the argument `options`.

!!! tip "Jacobian transpose"
    The adjoint of the jacobian `J` is computed internally when `Jᵗ = nothing` by using `transpose(J)` which works fine when `J` is an `AbstractArray`. In this case, do not pass the jacobian adjoint like `Jᵗ = (x, p) -> transpose(d_xF(x, p))` otherwise the jacobian will be computed twice!

!!! tip "ODE problems"
    For ODE problems, it is more efficient to pass the Bordered Linear Solver using the option `bdlinsolver = MatrixBLS()`
"""
function newton_fold(prob::AbstractBifurcationProblem,
                foldpointguess, par,
                eigenvec, eigenvec_ad,
                options::NewtonPar;
                normN = norm,
                bdlinsolver::AbstractBorderedLinearSolver = MatrixBLS(),
                usehessian = true,
                kwargs...)

    𝐅 = FoldProblemMinimallyAugmented(
        prob,
        _copy(eigenvec),
        _copy(eigenvec_ad),
        options.linsolver,
        # do not change linear solver if the user provides it
        @set bdlinsolver.solver = (isnothing(bdlinsolver.solver) ? options.linsolver : bdlinsolver.solver);
        usehessian = usehessian)

    prob_f = FoldMAProblem(𝐅, nothing, foldpointguess, par, nothing, prob.plotSolution, prob.recordFromSolution)

    # options for the Newton Solver
    opt_fold = @set options.linsolver = FoldLinearSolverMinAug()

    # solve the Fold equations
    return solve(prob_f, Newton(), opt_fold; normN, kwargs...)
end

function newton_fold(br::AbstractBranchResult, ind_fold::Int;
                prob = br.prob,
                normN = norm,
                options = br.contparams.newton_options,
                nev = br.contparams.nev,
                start_with_eigen = false,
                bdlinsolver::AbstractBorderedLinearSolver = MatrixBLS(),
                kwargs...)
    foldpointguess = fold_point(br, ind_fold)
    bifpt = br.specialpoint[ind_fold]
    eigenvec = bifpt.τ.u; VI.scale!(eigenvec, 1 / normN(eigenvec))
    eigenvec_ad = _copy(eigenvec)

    if start_with_eigen
        λ = zero(_getvectoreltype(br))
        p = bifpt.param
        parbif = setparam(br, p)

        # jacobian at bifurcation point
        L = jacobian(prob, bifpt.x, parbif)

        # computation of zero eigenvector
        ζstar, = get_adjoint_basis(L, λ, br.contparams.newton_options.eigsolver; nev, verbose = false)
        eigenvec .= real.(ζstar)

        # computation of adjoint eigenvector
        _Jt = ~has_adjoint(prob) ? adjoint(L) : jacobian_adjoint(prob, bifpt.x, parbif)
        ζstar, = get_adjoint_basis(_Jt, λ, br.contparams.newton_options.eigsolver; nev, verbose = false)
        eigenvec_ad .= real.(ζstar)
        VI.scale!(eigenvec_ad, 1 / normN(eigenvec_ad))
    end

    # solve the Fold equations
    return newton_fold(prob,
                        foldpointguess,
                        getparams(br),
                        eigenvec,
                        eigenvec_ad,
                        options; 
                        normN = normN,
                        bdlinsolver = bdlinsolver,
                        kwargs...)
end

function update!(probma::FoldMAProblem, iter, state)
    # this function is used as an update of 𝐅
    # it is called to update the Minimally Augmented problem
    # by updating the vectors a, b
    # we first check that the continuation step was successful
    # if not, we do not update the problem with bad information!
    # if we are in a bisection, we still update the MA problem, this does not work well otherwise
    𝐅 = probma.prob
    𝒯 = eltype(𝐅)
    success = state.converged
    step = state.step
    if (~mod_counter(step, 𝐅.update_minaug_every_step) || success == false)
        return true
    end

    @debug "[Fold] Update vectors a and b"
    z = getsolution(state)
    x = getvec(z.u) # fold point
    p1 = getp(z.u)  # first parameter
    p2 = z.p        # second parameter

    lens1, lens2 = get_lenses(probma)
    newpar = set(getparams(probma), lens1, p1)
    newpar = set(newpar, lens2, p2)

    a = 𝐅.a
    b = 𝐅.b

    # expression of the jacobian
    J_at_xp = jacobian(𝐅.prob_vf, x, newpar)

    # compute new a, close to left null vector
    if is_symmetric(𝐅)
        JAd_at_xp = J_at_xp
    else
        JAd_at_xp = has_adjoint(𝐅) ? jad(𝐅.prob_vf, x, newpar) : transpose(J_at_xp)
    end

    bd_vec = _compute_bordered_vectors(𝐅, J_at_xp, JAd_at_xp)

    _copyto!(𝐅.a, bd_vec.w); VI.scale!(𝐅.a, 1 / 𝐅.norm(bd_vec.w))
    # do not normalize with dot(newb, 𝐅.a), it prevents from BT detection
    _copyto!(𝐅.b, bd_vec.v); VI.scale!(𝐅.b, 1 / 𝐅.norm(bd_vec.v))

    return true
end

"""
$(TYPEDSIGNATURES)

Codim 2 continuation of Fold points. This function turns an initial guess for a Fold point into a curve of Fold points based on a Minimally Augmented formulation. The arguments are as follows
- `prob::AbstractBifurcationFunction`
- `foldpointguess` initial guess `(x_0, p1_0)` for the Fold point. It should be a `BorderedArray` as returned by the function `foldpoint`
- `par` set of parameters
- `lens1` parameter axis for parameter 1
- `lens2` parameter axis for parameter 2
- `eigenvec` guess for the right null vector
- `eigenvec_ad` guess for the left null vector
- `options_cont` arguments to be passed to the regular [`continuation`](@ref)

# Optional arguments:
- `jacobian_ma = AutoDiff()`, how the linear system of the Fold problem is solved. Can be `AutoDiff(), FiniteDifferencesMF(), FiniteDifferences(), MinAug(), MinAugMatrixBased`.
- `bdlinsolver` bordered linear solver for the constraint equation with top-left block J. Required in the linear solver for the Minimally Augmented Fold functional. This option can be used to pass a dedicated linear solver for example with specific preconditioner.
- `bdlinsolver_adjoint` bordered linear solver for the constraint equation with top-left block J^*. Required in the linear solver for the Minimally Augmented Fold functional. This option can be used to pass a dedicated linear solver for example with specific preconditioner.
- `update_minaug_every_step` update vectors `a, b` in Minimally Formulation every `update_minaug_every_step` steps
- `compute_eigen_elements = false` whether to compute eigenelements. If `options_cont.detect_event>0`, it allows the detection of ZH points.
- `kwargs` keywords arguments to be passed to the regular [`continuation`](@ref)

# Simplified call

    continuation_fold(br::AbstractBranchResult, ind_fold::Int64, lens2::AllOpticTypes, options_cont::ContinuationPar ; kwargs...)

where the parameters are as above except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of Fold point in `br` that you want to continue.

!!! tip "Jacobian transpose"
    The adjoint of the jacobian `J` is computed internally when `Jᵗ = nothing` by using `transpose(J)` which works fine when `J` is an `AbstractArray`. In this case, do not pass the jacobian adjoint like `Jᵗ = (x, p) -> transpose(d_xF(x, p))` otherwise the jacobian would be computed twice!

!!! tip "ODE problems"
    For ODE problems, it is more efficient to use the Matrix based Bordered Linear Solver passing the option `bdlinsolver = MatrixBLS()`. This is the default setting.

!!! tip "Detection of Bogdanov-Takens and Cusp bifurcations"
    In order to trigger the detection, pass `detect_event = 1 or 2` in `options_cont`.
"""
function continuation_fold(prob, alg::AbstractContinuationAlgorithm,
                           foldpointguess::BorderedArray{vectype, 𝒯}, 
                           par,
                           lens1::AllOpticTypes, lens2::AllOpticTypes,
                           eigenvec, eigenvec_ad,
                           options_cont::ContinuationPar ;
                           update_minaug_every_step = 1,
                           normC = norm,

                           bdlinsolver::AbstractBorderedLinearSolver = MatrixBLS(),
                           bdlinsolver_adjoint::AbstractBorderedLinearSolver = bdlinsolver,

                           jacobian_ma::AbstractJacobianType = AutoDiff(),
                           compute_eigen_elements = false,
                           usehessian = true,
                           kind = FoldCont(),
                           record_from_solution = nothing,
                           kwargs...) where {𝒯, vectype}
    @assert lens1 != lens2 "Please choose 2 different parameters. You only passed $lens1"
    @assert lens1 == getlens(prob)

    if alg isa PALC && alg.tangent isa Bordered
        @warn "You selected the PALC continuation algorithm with Bordered predictor.\nThe jacobian being singular on Fold points, this could lead to bad prediction and convergence.\nIf you have issues, try a different tangent predictor like Secant for example, you can pass it using `alg = PALC()`."
    end

    # options for the Newton solver inherited from the ones provided by the user
    options_newton = options_cont.newton_options

    𝐅 = FoldProblemMinimallyAugmented(
            prob,
            _copy(eigenvec_ad), # carefull a =  left null vector
            _copy(eigenvec),    # carefull b = right null vector
            options_newton.linsolver,
            # do not change linear solver if user provides it
            @set bdlinsolver.solver = (isnothing(bdlinsolver.solver) ? options_newton.linsolver : bdlinsolver.solver);
            linbdsolve_adjoint = bdlinsolver_adjoint,
            usehessian,
            _norm = normC,
            update_minaug_every_step)

    # Jacobian for the Fold problem
    if jacobian_ma == AutoDiff()
        foldpointguess = vcat(foldpointguess.u, foldpointguess.p)
        prob_fold = FoldMAProblem(𝐅, AutoDiff(), foldpointguess, par, lens2, prob.plotSolution, prob.recordFromSolution)
        opt_fold_cont = @set options_cont.newton_options.linsolver = DefaultLS()
    elseif jacobian_ma == FiniteDifferencesMF()
        foldpointguess = vcat(foldpointguess.u, foldpointguess.p)
        prob_fold = FoldMAProblem(𝐅, FiniteDifferencesMF(), foldpointguess, par, lens2, prob.plotSolution, prob.recordFromSolution)
        opt_fold_cont = @set options_cont.newton_options.linsolver = options_cont.newton_options.linsolver
    elseif jacobian_ma == FiniteDifferences()
        foldpointguess = vcat(foldpointguess.u, foldpointguess.p)
        prob_fold = FoldMAProblem(𝐅, FiniteDifferences(), foldpointguess, par, lens2, prob.plotSolution, prob.recordFromSolution)
        opt_fold_cont = @set options_cont.newton_options.linsolver = options_cont.newton_options.linsolver
    elseif jacobian_ma == MinAugMatrixBased()
        foldpointguess = vcat(foldpointguess.u, foldpointguess.p)
        prob_fold = FoldMAProblem(𝐅, MinAugMatrixBased(), foldpointguess, par, lens2, prob.plotSolution, prob.recordFromSolution)
        opt_fold_cont = @set options_cont.newton_options.linsolver = options_cont.newton_options.linsolver
    else
        prob_fold = FoldMAProblem(𝐅, nothing, foldpointguess, par, lens2, prob.plotSolution, prob.recordFromSolution)
        opt_fold_cont = @set options_cont.newton_options.linsolver = FoldLinearSolverMinAug()
    end

    # Allow to tackle the case where the two parameters have the same name
    lenses = get_lens_symbol(lens1, lens2)

    # variables to save call back
    𝐅.BT = one(𝒯)
    𝐅.CP = one(𝒯)
    𝐅.ZH = 1

    # the following allows to append information specific to the codim 2 continuation to the user data
    _printsol = record_from_solution
    _printsol2 = isnothing(_printsol) ?
        (u, p; kw...) -> begin 
                (; zip(lenses, (getp(u, 𝐅), p))..., 
                        BT = 𝐅.BT,
                        CP = 𝐅.CP,
                        ZH = 𝐅.ZH,
                        _namedrecordfromsol(BifurcationKit.record_from_solution(prob)(getvec(u), p; kw...))...) 
            end :
        (u, p; kw...) -> begin 
                (; zip(lenses, (getp(u, 𝐅), p))..., 
                    BT = 𝐅.BT, 
                    CP = 𝐅.CP, 
                    ZH = 𝐅.ZH,
                    _namedrecordfromsol(_printsol(getvec(u), p; kw...))...) 
            end

    prob_fold = re_make(prob_fold, record_from_solution = _printsol2)

    # eigen solver
    eigsolver = FoldEig(getsolver(opt_fold_cont.newton_options.eigsolver), prob_fold)

    # define event for detecting codim 2 bifurcations
    # couple it with user passed events
    event_user = get(kwargs, :event, nothing)
    event_bif = ContinuousEvent(2, test_bt_cusp, compute_eigen_elements, ("bt", "cusp"), 0)
    if isnothing(event_user)
        event = PairOfEvents(event_bif, DiscreteEvent(1, test_zh, false, ("zh",)))
    else
        event = SetOfEvents(event_bif, DiscreteEvent(1, test_zh, false, ("zh",)), event_user)
    end

    if prob isa AbstractWrapperFDProblem
        _plotsol = modify_po_plot(prob_fold, getparams(prob_fold), getlens(prob_fold); plot_solution = prob.plotSolution)
        prob_fold = re_make(prob_fold, plot_solution = _plotsol)
    end

    # solve the Fold equations
    br = continuation(
        prob_fold, alg,
        (@set opt_fold_cont.newton_options.eigsolver = eigsolver);
        linear_algo = BorderingBLS(solver = opt_fold_cont.newton_options.linsolver, check_precision = false),
        kwargs...,
        kind,
        normC,
        finalise_solution = get(kwargs, :finalise_solution, finalise_default),
        event
        )
    @assert ~isnothing(br) "Empty branch!"
    return correct_bifurcation(br)
end

function continuation_fold(prob,
                br::AbstractBranchResult, ind_fold::Int,
                lens2::AllOpticTypes,
                options_cont::ContinuationPar = br.contparams ;
                alg = br.alg,
                normC = norm,
                nev = br.contparams.nev,
                start_with_eigen = false,
                bdlinsolver::AbstractBorderedLinearSolver = MatrixBLS(),
                bdlinsolver_adjoint = bdlinsolver,
                a = nothing,
                b = nothing,
                kwargs...)
    foldpointguess = fold_point(br, ind_fold)
    bifpt = br.specialpoint[ind_fold]
    ζ = bifpt.τ.u; VI.scale!(ζ, 1 / norm(ζ))
    ζad = _copy(ζ)

    p = bifpt.param
    parbif = setparam(br, p)

    if start_with_eigen
        # jacobian at bifurcation point
        L = jacobian(prob, bifpt.x, parbif)

        # computation of zero eigenvector
        if bifpt.ind_ev > 0 && haseigenvector(br)
            ζ .= real.( geteigenvector(br.contparams.newton_options.eigsolver, br.eig[bifpt.idx].eigenvecs, bifpt.ind_ev))
            VI.scale!(ζ, 1/normC(ζ))
        else
            error("No index for the eigenvalue has been saved.\nPlease open an issue on the website of BifurcationKit.")
        end

        # jacobian adjoint at bifurcation point
        L★ = has_adjoint(prob) ? jacobian_adjoint(prob, bifpt.x, parbif) : transpose(L)

        # computation of zero adjoint eigenvector
        ζ★, λ★ = get_adjoint_basis(L★, 0, br.contparams.newton_options.eigsolver; nev = nev, verbose = options_cont.newton_options.verbose)
        ζad = real.(ζ★)
        VI.scale!(ζad, 1 / real(VI.inner(ζ, ζ★))) # it can be useful to enforce real(), like for DDE
    else
        # we use a minimally augmented formulation to set the initial vectors
        a = isnothing(a) ? _randn(ζ) : a; VI.scale!(a, 1 / normC(a))
        b = isnothing(b) ? _randn(ζ) : b; VI.scale!(b, 1 / normC(b))

        𝒯 = typeof(p)
        L = jacobian(prob, foldpointguess.u, parbif)
        newb, _, cv, it = bdlinsolver(L, a, b, zero(𝒯), VI.zerovector(a), one(𝒯))
        ~cv && @debug "Bordered linear solver for J did not converge."

        @debug "RIGHT EIGENVECTORS" cv it norminf(residual(prob, bifpt.x, parbif)) norminf(apply(L, newb))

        L★ = has_adjoint(prob) ? jacobian_adjoint(prob, bifpt.x, parbif) : transpose(L)
        newa, _, cv, it = bdlinsolver_adjoint(L★, b, a, zero(𝒯), VI.zerovector(a), one(𝒯))
        ~cv && @debug "Bordered linear solver for J' did not converge."

        @debug "LEFT  EIGENVECTORS" cv it norminf(residual(prob, bifpt.x, parbif)) norminf(apply(L★, newa))

        ζad = newa; VI.scale!(ζad, 1 / normC(ζad))
        ζ   = newb; VI.scale!(ζ,   1 / normC(ζ))
        VI.scale!(ζad, 1 / VI.inner(ζ, ζad))
    end

    return continuation_fold(prob, alg,
            foldpointguess, parbif,
            getlens(br), lens2,
            ζ, ζad,
            options_cont ;
            normC,
            bdlinsolver,
            bdlinsolver_adjoint,
            kwargs...)
end

# Zero-Hopf test function for the Fold functional
function test_zh(iter, state)
    if isnothing(state.eigvals)
        iter.prob.prob.ZH = 1
    else
        ϵ = iter.contparams.tol_stability
        ρ = minimum(abs ∘ real, state.eigvals)
        iter.prob.prob.ZH = mapreduce(x -> ((real(x) > ρ) & (imag(x) > ϵ)), +, state.eigvals)
    end
    return iter.prob.prob.ZH
end

# Bogdanov-Takens / Cusp test function for the Fold functional
function test_bt_cusp(iter, state)
    probma = getprob(iter)
    lens1, lens2 = get_lenses(probma)

    z = getx(state)
    x = getvec(z)    # fold point
    p1 = getp(z)     # first parameter
    p2 = getp(state) # second parameter
    par = getparams(probma)
    newpar = set(par, lens1, p1)
    newpar = set(newpar, lens2, p2)

    𝐅 = probma.prob
    𝒯 = eltype(𝐅)

    # expression of the jacobian
    J_at_xp = jacobian(𝐅.prob_vf, x, newpar)
    JAd_at_xp = has_adjoint(𝐅) ? jacobian_adjoint(𝐅, x, newpar) : transpose(J_at_xp)

    bd_vec = _compute_bordered_vectors(𝐅, J_at_xp, JAd_at_xp)

    # compute new b
    ζ = bd_vec.v
    VI.scale!(ζ, 1 / 𝐅.norm(ζ))

    # compute new a
    ζstar = bd_vec.w
    VI.scale!(ζstar, 1 / 𝐅.norm(ζstar))

    𝐅.BT = VI.inner(ζstar, ζ)
    𝐅.CP = getp(state.τ)

    return 𝐅.BT, 𝐅.CP
end

# structure to compute eigen-elements along branch of Fold points
struct FoldEig{P, S} <: AbstractCodim2EigenSolver
    eigsolver::S
    prob::P
end
FoldEig(solver) = FoldEig(solver, nothing)

function (eig::FoldEig)(Jma, nev; kwargs...)
    # il ne faut pas mettre a jour les deux params?
    n = min(nev, length(getvec(Jma.x)))
    J = jacobian(Jma.pbma.prob_vf, getvec(Jma.x), set(Jma.params, getlens(Jma.pbma), getp(Jma.x)))
    return eig.eigsolver(J, n; kwargs...)
end

@views function (eig::FoldEig)(Jma::AbstractMatrix, nev; kwargs...)
    return eig.eigsolver(Jma[begin:end-1, begin:end-1], nev; kwargs...)
end

geteigenvector(eig::FoldEig, vectors, i::Int) = geteigenvector(eig.eigsolver, vectors, i)

get_bifurcation_type(it::ContIterable, state, status::Symbol, interval::Tuple{T, T}, eig::FoldEig) where T = get_bifurcation_type(it, state, status, interval, eig.eigsolver)
