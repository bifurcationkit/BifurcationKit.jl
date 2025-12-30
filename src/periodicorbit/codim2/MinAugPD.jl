"""
$(SIGNATURES)

For an initial guess from the index of a PD bifurcation point located in `ContResult.specialpoint`, returns a point which can be refined using `newton_fold`.
"""
function pd_point(br::AbstractBranchResult, index::Int)
    bptype = br.specialpoint[index].type
    if bptype != :pd 
        error("This should be a PD point.\nYou passed a $bptype point.")
    end
    specialpoint = br.specialpoint[index]
    return BorderedArray(_copy(specialpoint.x), specialpoint.param)
end

function apply_jacobian_period_doubling(pb, x, par, dx, _transpose = false)
    if _transpose == false
        # THIS CASE IS NOT REALLY USED
        # if hasJvp(pb)
        #  return jvp(pb, x, par, dx)
        # else
        #  return apply(jacobian_period_doubling(pb, x, par), dx)
        # end
        @error("Please report to the website of BifurcationKit")
    else
        # if matrix-free:
        if has_adjoint(pb)
            return jacobian_adjoint_period_doubling_matrix_free(pb, x, par, dx)
        else
            return apply(transpose(jacobian_period_doubling(pb, x, par)), dx)
        end
    end
end
####################################################################################################
@inline getvec(x, ::PeriodDoublingMinimallyAugmentedFormulation) = get_vec_bls(x)
@inline   getp(x, ::PeriodDoublingMinimallyAugmentedFormulation) = get_par_bls(x)

pdtest(JacPD, v, w, J22, _zero, n, lsbd = MatrixBLS()) = lsbd(JacPD, v, w, J22, _zero, n)

# this function encodes the functional
function (𝐏𝐝::PeriodDoublingMinimallyAugmentedFormulation)(x, p::𝒯, params) where 𝒯
    # These are the equations of the minimally augmented (MA) formulation of the Period-Doubling bifurcation point
    # input:
    # - x guess for the point at which the jacobian is singular
    # - p guess for the parameter value `<: Real` at which the jacobian is singular
    # The jacobian of the MA problem is solved with a BLS method
    a = 𝐏𝐝.a
    b = 𝐏𝐝.b
    # update parameter
    par = set(params, getlens(𝐏𝐝), p)
    # ┌        ┐┌  ┐   ┌ ┐
    # │ J+I  a ││v │ = │0│
    # │ b    0 ││σ │   │1│
    # └        ┘└  ┘   └ ┘
    # In the notations of Govaerts 2000, a = w, b = v
    # Thus, b should be a null vector of J +I
    #       a should be a null vector of J'+I
    # we solve Jv + v + a σ1 = 0 with <b, v> = 1
    # the solution is v = -σ1 (J+I)\a with σ1 = -1/<b, (J+I)⁻¹a>.
    # In the case of collocation, the matrix J is simply Jpo without the phase condition and with PD boundary condition.
    J = jacobian_period_doubling(𝐏𝐝.prob_vf, x, par)
    _, σ, cv, = pdtest(J, a, b, zero(𝒯), 𝐏𝐝.zero, one(𝒯), 𝐏𝐝.linbdsolver)
    ~cv && @debug "[PD residual] Linear solver for J+I did not converge."
    return residual(𝐏𝐝.prob_vf, x, par), σ
end
###################################################################################################
"""
$(TYPEDSIGNATURES)

Compute the solution of 

```
┌              ┐┌  ┐   ┌   ┐
│ J+I   𝐅.a    ││v │ = │ 0 │
│ 𝐅.b'   0     ││σ │   │ 1 │
└              ┘└  ┘   └   ┘
```

and the same for the adjoint system.
"""
function _compute_bordered_vectors(𝐏𝐝::PeriodDoublingMinimallyAugmentedFormulation, JPD, JPD★)
    a = 𝐏𝐝.a
    b = 𝐏𝐝.b
    𝒯 = eltype(𝐏𝐝)

    # we solve N[v, σ1] = [0, 1]
    v, σ1, cv, itv = pdtest(JPD, a, b, zero(𝒯), 𝐏𝐝.zero, one(𝒯), 𝐏𝐝.linbdsolver)
    ~cv && @debug "Linear solver for N did not converge."
 
    # we solve Nᵗ[w, σ2] = [0, 1]
    w, σ2, cv, itw = pdtest(JPD★, b, a, zero(𝒯), 𝐏𝐝.zero, one(𝒯), 𝐏𝐝.linbdsolverAdjoint)
    ~cv && @debug "Linear solver for Nᵗ did not converge."
    return (; v, itv, w, itw)
end

function _get_bordered_terms(𝐏𝐝::PeriodDoublingMinimallyAugmentedFormulation, x, p::𝒯, par) where 𝒯
    # get the PO functional, ie a WrapPOSh, WrapPOTrap, WrapPOColl
    POWrap = 𝐏𝐝.prob_vf

    # update parameter
    lens = getlens(𝐏𝐝)
    par0 = set(par, lens, p)
 
    # Avoid doing 3 times the (possibly) costly building of J(x, p)
    JPD = jacobian_period_doubling(POWrap, x, par0) # jacobian with period doubling boundary condition but without phase condition
    # Avoid computing the jacobian twice in case 𝐏𝐝.Jadjoint is not provided
    JPD★ = has_adjoint(𝐏𝐝) ? jacobian_adjoint_period_doubling(POWrap, x, par0) : transpose(JPD)

    (;v, w, itv, itw) = _compute_bordered_vectors(𝐏𝐝, JPD, JPD★)
 
    δ = getdelta(POWrap)
    ϵₚ = ϵₓ = ϵⱼ = ϵₜ = 𝒯(δ)
 
    dₚF = minus(residual(POWrap, x, set(par, lens, p + ϵₚ)),
                residual(POWrap, x, set(par, lens, p - ϵₚ)))
    LA.rmul!(dₚF, 𝒯(1 / (2ϵₚ)))
    dJvdp = minus(apply(jacobian_period_doubling(POWrap, x, set(par, lens, p + ϵⱼ)), v),
                  apply(jacobian_period_doubling(POWrap, x, set(par, lens, p - ϵⱼ)), v));
    LA.rmul!(dJvdp, 𝒯(1/(2ϵⱼ)))
    σₚ = -LA.dot(w, dJvdp)

    return (;JPD, JPD★, dₚF, σₚ, δ, ϵₜ, ϵₓ, v, w, par0, dJvdp, itv, itw)
end
###################################################################################################
function jacobian(pdpb::PDMAProblem{Tprob, MinAugMatrixBased}, X, par) where {Tprob}
    p = X[end]
    x = @view X[begin:end-1]

    𝐏𝐝 = pdpb.prob
    𝒯 = eltype(p)

    POWrap = 𝐏𝐝.prob_vf

    (;dₚF, σₚ, ϵₜ, ϵₓ, v, w, par0) = _get_bordered_terms(𝐏𝐝, x, p, par)

    # TODO!! This is only finite differences
    u1 = apply_jacobian_period_doubling(POWrap, x .+ ϵₓ .* vcat(v, 0), par0, w, true)
    u2 = apply_jacobian_period_doubling(POWrap, x .- ϵₓ .* vcat(v, 0), par0, w, true)
    σₓ = minus(u2, u1); LA.rmul!(σₓ, 1 / (2ϵₓ))

    # TODO!! a bit of a hack
    xtmp = copy(x); xtmp[end] += ϵₜ
    σₜ = (𝐏𝐝(xtmp, p, par0)[end] - 𝐏𝐝(x, p, par0)[end]) / (ϵₜ)

    _Jpo = jacobian(POWrap, x, par0)

    return [_Jpo dₚF ; vcat(σₓ, σₜ)' σₚ]
end
###################################################################################################
# Struct to invert the jacobian of the pd MA problem.
struct PDLinearSolverMinAug <: AbstractLinearSolver; end

function PDMALinearSolver(x, p::𝒯, 𝐏𝐝::PeriodDoublingMinimallyAugmentedFormulation, par,
                            rhsu, rhsp) where 𝒯
    ################################################################################################
    # Recall that the functional we want to solve is [F(x,p), σ(x,p)]
    # where σ(x,p) is computed in the above functions and F is the periodic orbit
    # functional. We recall that N⋅[v, σ] ≡ [0, 1]
    # The Jacobian Jpd of the functional is expressed at (x, p)
    # We solve here Jpd⋅res = rhs := [rhsu, rhsp]
    # The Jacobian expression of the PD problem is
    #           ┌          ┐
    #    Jpd =  │ dxF  dpF │
    #           │ σx   σp  │
    #           └          ┘
    # where σx := ∂ₓσ and σp := ∂ₚσ
    # We recall the expression of
    #            σx = -< w, d2F(x,p)[v, x2]>
    # where (w, σ2) is solution of J'w + b σ2 = 0 with <a, w> = n
    ########################## Extraction of function names ########################################
    # get the PO functional, ie a WrapPOSh, WrapPOTrap, WrapPOColl
    POWrap = 𝐏𝐝.prob_vf

    (; dₚF, σₚ, ϵₜ, ϵₓ, v, w, par0, itv, itw) = _get_bordered_terms(𝐏𝐝, x, p, par)

    if has_hessian(𝐏𝐝) == false || 𝐏𝐝.usehessian == false
        # We invert the jacobian of the PD problem when the Hessian of x -> F(x, p) is not known analytically.
        # apply Jacobian adjoint
        u1 = apply_jacobian_period_doubling(POWrap, x .+ ϵₓ .* vcat(v,0), par0, w, true)
        u2 = apply_jacobian_period_doubling(POWrap, x .- ϵₓ .* vcat(v,0), par0, w, true)
        σₓ = minus(u2, u1); LA.rmul!(σₓ, 1 / (2ϵₓ))

        # a bit of a hack
        xtmp = copy(x); xtmp[end] += ϵₜ
        σₜ = (𝐏𝐝(xtmp, p, par0)[end] - 𝐏𝐝(x, p, par0)[end]) / (ϵₜ)
        ########## Resolution of the bordered linear system ########
        # we invert Jpd
        _Jpo = jacobian(POWrap, x, par0)
        dX, dsig, flag, it = 𝐏𝐝.linbdsolver(_Jpo, dₚF, vcat(σₓ, σₜ), σₚ, rhsu, rhsp)
        ~flag && @debug "Linear solver for J did not converge."
    else
        error("WIP. Please select another jacobian method like `AutoDiff()` or `FiniteDifferences()`. You can also pass the option usehessian = false.")
    end

    return dX, dsig, true, sum(it) + sum(itv) + sum(itw)
end

function (pdls::PDLinearSolverMinAug)(Jpd, rhs::BorderedArray{vectype, 𝒯}; kwargs...) where {vectype, 𝒯}
    # kwargs is used by AbstractLinearSolver
    out = PDMALinearSolver((Jpd.x).u,
                 (Jpd.x).p,
                 Jpd.pbma,
                 Jpd.params,
                 rhs.u, rhs.p)
    # this type annotation enforces type stability
    return BorderedArray{vectype, 𝒯}(out[1], out[2]), out[3], out[4]
end
###################################################################################################
get_wrap_po(pb::PDMAProblem) = get_wrap_po(pb.prob)
@inline has_adjoint(pdpb::PDMAProblem) = has_adjoint(pdpb.prob)
@inline is_symmetric(pdpb::PDMAProblem) = is_symmetric(pdpb.prob)
################################################################################################### Newton / Continuation functions
"""
$(SIGNATURES)

This function turns an initial guess for a PD point into a solution to the PD problem based on a Minimally Augmented formulation. The arguments are as follows
- `prob::AbstractBifurcationFunction`
- `pdpointguess` initial guess (x_0, p_0) for the PD point. It should be a `BorderedArray` as returned by the function `PDPoint`
- `par` parameters used for the vector field
- `eigenvec` guess for the 0 eigenvector
- `eigenvec_ad` guess for the 0 adjoint eigenvector
- `options::NewtonPar` options for the Newton-Krylov algorithm, see [`NewtonPar`](@ref).

# Optional arguments:
- `normN = norm`
- `bdlinsolver` bordered linear solver for the constraint equation
- `kwargs` keywords arguments to be passed to the regular Newton-Krylov solver

# Simplified call
Simplified call to refine an initial guess for a PD point. More precisely, the call is as follows

    newton_pd(br::AbstractBranchResult, ind_pd::Int; options = br.contparams.newton_options, kwargs...)

The parameters / options are as usual except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of bifurcation point in `br` you want to refine. You can pass newton parameters different from the ones stored in `br` by using the argument `options`.

!!! tip "Jacobian transpose"
    The adjoint of the jacobian `J` is computed internally when `Jᵗ = nothing` by using `transpose(J)` which works fine when `J` is an `AbstractArray`. In this case, do not pass the jacobian adjoint like `Jᵗ = (x, p) -> transpose(d_xF(x, p))` otherwise the jacobian will be computed twice!

!!! tip "ODE problems"
    For ODE problems, it is more efficient to pass the Bordered Linear Solver using the option `bdlinsolver = MatrixBLS()`
"""
function newton_pd(prob::AbstractBifurcationProblem,
                pdpointguess, par,
                eigenvec, eigenvec_ad,
                options::NewtonPar;
                normN = norm,
                bdlinsolver::AbstractBorderedLinearSolver = MatrixBLS(),
                usehessian = true,
                kwargs...)

    pdproblem = PeriodDoublingMinimallyAugmentedFormulation(
        prob,
        _copy(eigenvec),
        _copy(eigenvec_ad),
        options.linsolver,
        # do not change linear solver if user provides it
        @set bdlinsolver.solver = (isnothing(bdlinsolver.solver) ? options.linsolver : bdlinsolver.solver);
        usehessian = usehessian)

    pdpointguess = vcat(pdpointguess.u, pdpointguess.p)
    prob_f = PDMAProblem(pdproblem, FiniteDifferences(), pdpointguess, par, nothing, prob.plotSolution, prob.recordFromSolution)

    # options for the Newton Solver
    opt_pd = deepcopy(options)

    # solve the PD equations
    return newton(prob_f, opt_pd; normN, kwargs...)
end
###################################################################################################
function update!(probma::PDMAProblem, iter, state)
    # it is called to update the Minimally Augmented problem
    # by updating the vectors a, b
    # we first check that the continuation step was successful
    # if not, we do not update the problem with bad information!
    𝐏𝐝 = probma.prob
    𝒯 = eltype(𝐏𝐝)
    success = state.converged
    if (~mod_counter(step, 𝐏𝐝.update_minaug_every_step) || success == false)
        # we call the user update
        return update!(𝐏𝐝, iter, state)
    end
    @debug "[codim2 PD] Update a / b in PD"

    z = getsolution(state)
    x = getvec(z.u) # PD point
    p1 = getp(z.u)  # first parameter
    p2 = z.p        # second parameter

    lens1, lens2 = get_lenses(probma)
    newpar = set(getparams(probma), lens1, p1)
    newpar = set(newpar, lens2, p2)

    POWrap = 𝐏𝐝.prob_vf
    JPD = jacobian_period_doubling(POWrap, x, newpar) # jacobian with period doubling boundary condition
    # we do the following in order to avoid computing JPO_at_xp twice in case 𝐏𝐝.Jadjoint is not provided
    JPD★ = has_adjoint(𝐏𝐝) ? jacobian_adjoint_period_doubling(POWrap, x, newpar) : transpose(JPD)

    # normalization
    (;v, w) = _compute_bordered_vectors(𝐏𝐝, JPD, JPD★)
    _copyto!(𝐏𝐝.a, w); LA.rmul!(𝐏𝐝.a, 1/𝐏𝐝.norm(w))
    # do not normalize with dot(newb, 𝐏𝐝.a), it prevents from BT detection
    _copyto!(𝐏𝐝.b, v); LA.rmul!(𝐏𝐝.b, 1/𝐏𝐝.norm(v))

    # call the user-passed update
    return update!(𝐏𝐝, iter, state)
end

function continuation_pd(prob, alg::AbstractContinuationAlgorithm,
                pdpointguess::BorderedArray{vectype, 𝒯}, par,
                lens1::AllOpticTypes, lens2::AllOpticTypes,
                eigenvec, eigenvec_ad,
                options_cont::ContinuationPar ;
                normC = norm,

                update_minaug_every_step = 1,
                bdlinsolver::AbstractBorderedLinearSolver = MatrixBLS(),
                bdlinsolver_adjoint::AbstractBorderedLinearSolver = bdlinsolver,

                jacobian_ma::AbstractJacobianType = AutoDiff(),
                compute_eigen_elements = false,
                plot_solution = BifurcationKit.plot_solution(prob),
                prm::Bool = prob isa WrapPOSh,
                usehessian = false,
                kind = PDCont(),
                kwargs...) where {𝒯, vectype}
    @assert lens1 != lens2 "Please choose 2 different parameters. You only passed $lens1"
    @assert lens1 == getlens(prob)

    # options for the Newton solver inheritated from the ones the user provided
    newton_options = options_cont.newton_options

    𝐏𝐝 = PeriodDoublingMinimallyAugmentedFormulation(
            prob,
            _copy(eigenvec),
            _copy(eigenvec_ad),
            newton_options.linsolver,
            # do not change linear solver if user provides it
            @set bdlinsolver.solver = (isnothing(bdlinsolver.solver) ? newton_options.linsolver : bdlinsolver.solver);
            linbdsolve_adjoint = bdlinsolver_adjoint,
            usehessian,
            _norm = normC,
            newton_options,
            prm,
            update_minaug_every_step)

    # this is to remove this part from the arguments passed to continuation
    _kwargs = (;record_from_solution, plot_solution)

    # Jacobian for the PD problem
    if jacobian_ma == AutoDiff()
        pdpointguess = vcat(pdpointguess.u, pdpointguess.p)
        prob_pd = PDMAProblem(𝐏𝐝, AutoDiff(), pdpointguess, par, lens2, plot_solution, prob.recordFromSolution)
        opt_pd_cont = @set options_cont.newton_options.linsolver = DefaultLS()
    elseif jacobian_ma == FiniteDifferences()
        pdpointguess = vcat(pdpointguess.u, pdpointguess.p...)
        prob_pd = PDMAProblem(𝐏𝐝, FiniteDifferences(), pdpointguess, par, lens2, plot_solution, prob.recordFromSolution)
        opt_pd_cont = @set options_cont.newton_options.linsolver = options_cont.newton_options.linsolver
    elseif jacobian_ma == FiniteDifferencesMF()
        pdpointguess = vcat(pdpointguess.u, pdpointguess.p)
        prob_pd = PDMAProblem(𝐏𝐝, FiniteDifferencesMF(), pdpointguess, par, lens2, plot_solution, prob.recordFromSolution)
        opt_pd_cont = @set options_cont.newton_options.linsolver = options_cont.newton_options.linsolver
    elseif jacobian_ma == MinAugMatrixBased()
        pdpointguess = vcat(pdpointguess.u, pdpointguess.p)
        prob_pd = PDMAProblem(𝐏𝐝, MinAugMatrixBased(), pdpointguess, par, lens2, plot_solution, prob.recordFromSolution)
        opt_pd_cont = @set options_cont.newton_options.linsolver = options_cont.newton_options.linsolver
    else
        prob_pd = PDMAProblem(𝐏𝐝, nothing, pdpointguess, par, lens2, plot_solution, prob.recordFromSolution)
        opt_pd_cont = @set options_cont.newton_options.linsolver = PDLinearSolverMinAug()
    end

    # this functions allows to tackle the case where the two parameters have the same name
    lenses = get_lens_symbol(lens1, lens2)

    # global variables to save call back
    𝐏𝐝.CP  = one(𝒯)
    𝐏𝐝.GPD = one(𝒯)
    𝐏𝐝.R2  = one(𝒯)

    # the following allows to append information specific to the codim 2 continuation to the user data
    _recordsol = get(kwargs, :record_from_solution, nothing)
    _recordsol2 = isnothing(_recordsol) ?
        (u, p; kw...) -> (; zip(lenses, (getp(u, 𝐏𝐝)[1], p))...,
                    period = getperiod(prob, getvec(u, 𝐏𝐝), nothing), # do not work for PoincareShootingProblem
                    CP  = 𝐏𝐝.CP,
                    GPD = 𝐏𝐝.GPD,
                    R₂  = 𝐏𝐝.R2,
                    _namedrecordfromsol(record_from_solution(prob)(getvec(u, 𝐏𝐝), p; kw...))...) :
        (u, p; kw...) -> (; _namedrecordfromsol(_recordsol(getvec(u, 𝐏𝐝), p; kw...))..., zip(lenses, (getp(u, 𝐏𝐝), p))..., 
                            CP  = 𝐏𝐝.CP, 
                            GPD = 𝐏𝐝.GPD,
                            R₂  = 𝐏𝐝.R2,
                            )

    # eigen solver
    eigsolver = FoldEig(getsolver(opt_pd_cont.newton_options.eigsolver), prob_pd)

    # change the plotter
    _kwargs = (record_from_solution = record_from_solution(prob), plot_solution = plot_solution)
    _plotsol = modify_po_plot(prob_pd, getparams(prob_pd), getlens(prob_pd); _kwargs...)
    prob_pd = re_make(prob_pd, record_from_solution = _recordsol2, plot_solution = _plotsol)

    # define event for detecting codim 2 bifurcations.
    # couple it with user passed events
    event_user = get(kwargs, :event, nothing)
    event_bif = ContinuousEvent(3, test_for_pd_gpd_cp, compute_eigen_elements, ("gpd", "cusp", "R2"), opt_pd_cont.tol_stability)
    event = isnothing(event_user) ? event_bif : PairOfEvents(event_bif, event_user)

    # solve the PD equations
    br_pd_po = continuation(
        prob_pd, alg,
        (@set opt_pd_cont.newton_options.eigsolver = eigsolver);
        linear_algo = BorderingBLS(solver = opt_pd_cont.newton_options.linsolver, check_precision = false),
        kwargs...,
        kind,
        normC,
        event,
        )
    correct_bifurcation(br_pd_po)
end

function test_for_pd_gpd_cp(iter, state)
    probma = getprob(iter)
    lens1, lens2 = get_lenses(probma)

    z = getx(state)
    x = getvec(z)    # pd point
    p1 = getp(z)     # first parameter
    p2 = getp(state) # second parameter
    par = getparams(probma)
    newpar = set(par, lens1, p1)
    newpar = set(newpar, lens2, p2)

    𝐏𝐝 = probma.prob
    𝒯 = eltype(𝐏𝐝)
    pbwrap = 𝐏𝐝.prob_vf

    a = 𝐏𝐝.a
    b = 𝐏𝐝.b

    # expression of the jacobian
    JPD = jacobian_period_doubling(pbwrap, x, newpar) # jacobian with period doubling boundary condition

    # we do the following in order to avoid computing JPO_at_xp twice in case 𝐏𝐝.Jadjoint is not provided
    JPD★ = has_adjoint(𝐏𝐝) ? jacobian_adjoint(pbwrap, x, newpar) : transpose(JPD)

    # compute new b
    ζ, _, cv, it = pdtest(JPD, a, b, zero(𝒯), 𝐏𝐝.zero, one(𝒯))
    ~cv && @debug "Linear solver for Pd did not converge."
    ζ ./= norm(ζ)

    # compute new a
    ζ★, _, cv, it = pdtest(JPD★, b, a, zero(𝒯), 𝐏𝐝.zero, one(𝒯), 𝐏𝐝.linbdsolverAdjoint)
    ~cv && @debug "Linear solver for Pdᵗ did not converge."
    ζ★ ./= norm(ζ★)
    𝐏𝐝.R2 = LA.dot(ζ★, ζ)

    pd0 = PeriodDoubling(copy(x), nothing, p1, newpar, lens1, nothing, nothing, nothing, :none)
    if pbwrap.prob isa ShootingProblem
        pd = period_doubling_normal_form(pbwrap, pd0, (1, 1), NewtonPar(𝐏𝐝.newton_options, verbose = false); verbose = false)
        𝐏𝐝.GPD = pd.nf.nf.b3
    end
    if pbwrap.prob isa PeriodicOrbitOCollProblem
        if 𝐏𝐝.prm
            pd = period_doubling_normal_form_prm(pbwrap, pd0; verbose = false)
        else
            pd = period_doubling_normal_form_iooss(pbwrap, pd0; verbose = false)
            𝐏𝐝.GPD = pd.nf.nf.b3
        end
    end
    return 𝐏𝐝.GPD, 𝐏𝐝.CP, 𝐏𝐝.R2
end

function compute_eigenvalues(eig::FoldEig,
                            iter::ContIterable{PDPeriodicOrbitCont},
                            state,
                            u0,
                            par,
                            nev = iter.contparams.nev; k...)
    probma = getprob(iter)
    lens1, lens2 = get_lenses(probma)
    x = getvec(u0)
    p1 = getp(u0)      # first parameter
    p2 = getp(state.z) # second parameter
    par = getparams(probma)
    newpar = _set(par, (lens1, lens2), (p1, p2))
    compute_eigenvalues(eig.eigsolver, iter, state, x, newpar, nev; k...)
end