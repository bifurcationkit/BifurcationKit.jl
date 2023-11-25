
"""
$(TYPEDEF)

Structure to encode Bogdanov-Takens functional based on a Minimally Augmented formulation.

# Fields

$(FIELDS)
"""
mutable struct BTProblemMinimallyAugmented{Tprob <: AbstractBifurcationProblem, vectype, S <: AbstractLinearSolver, Sa <: AbstractLinearSolver, Sbd <: AbstractBorderedLinearSolver, Sbda <: AbstractBorderedLinearSolver, Tlens <: Lens} <: AbstractProblemMinimallyAugmented
    "Functional F(x, p) - vector field - with all derivatives"
    prob_vf::Tprob
    "close to null vector of Jᵗ"
    a::vectype
    "close to null vector of J"
    b::vectype
    "vector zero, to avoid allocating it many times"
    zero::vectype
    "linear solver. Used to invert the jacobian of MA functional"
    linsolver::S
    "linear solver for the jacobian adjoint"
    linsolverAdjoint::Sa
    "bordered linear solver"
    linbdsolver::Sbd
    "linear bordered solver for the jacobian adjoint"
    linbdsolverAdjoint::Sbda
    "second parameter axis"
    lens2::Tlens
    "whether to use the hessian of prob_vf"
    usehessian::Bool
end

@inline has_hessian(pb::BTProblemMinimallyAugmented) = has_hessian(pb.prob_vf)
@inline is_symmetric(pb::BTProblemMinimallyAugmented) = is_symmetric(pb.prob_vf)
@inline has_adjoint(pb::BTProblemMinimallyAugmented) = has_adjoint(pb.prob_vf)
@inline has_adjoint_MF(pb::BTProblemMinimallyAugmented) = has_adjoint_MF(pb.prob_vf)
@inline isinplace(pb::BTProblemMinimallyAugmented) = isinplace(pb.prob_vf)
@inline getlens(pb::BTProblemMinimallyAugmented) = getlens(pb.prob_vf)
@inline getlenses(pb::BTProblemMinimallyAugmented) = (getlens(pb.prob_vf), pb.lens2)
jad(pb::BTProblemMinimallyAugmented, args...) = jad(pb.prob_vf, args...)

# constructor
function BTProblemMinimallyAugmented(prob, a, b,
                            linsolve::AbstractLinearSolver,
                            lens2::Lens;
                            linbdsolver = MatrixBLS(),
                            usehessian = true)
    return BTProblemMinimallyAugmented(prob, a, b, 0*a,
                linsolve, linsolve, linbdsolver, linbdsolver, lens2, usehessian)
end

"""
For an initial guess from the index of a BT bifurcation point located in ContResult.specialpoint, returns a point which will be refined using `newtonBT`.
"""
function bt_point(br::AbstractResult{FoldCont, Tprob}, index::Int) where {Tprob}
    bptype = br.specialpoint[index].type
    @assert bptype == :bt "This should be a BT point"
    specialpoint = br.specialpoint[index]
    prob_ma = br.prob.prob
    return BorderedArray(_copy(getvec(specialpoint.x, prob_ma)), [getp(specialpoint.x, prob_ma), specialpoint.param])
end

function bt_point(br::AbstractResult{HopfCont, Tprob}, index::Int) where {Tprob}
    bptype = br.specialpoint[index].type
    @assert bptype == :bt "This should be a BT point"
    specialpoint = br.specialpoint[index]
    prob_ma = br.prob.prob
    return BorderedArray(_copy(getvec(specialpoint.x, prob_ma)), [getp(specialpoint.x, prob_ma)[1], specialpoint.param])
end
################################################################################
getvec(x, ::BTProblemMinimallyAugmented) = getvec(x)
getp(x, ::BTProblemMinimallyAugmented) = getp(x)

function (𝐁𝐓::BTProblemMinimallyAugmented)(x, p1::T, p2::T, params) where T
    # These are the equations of the minimally augmented (MA) formulation of the bt bifurcation point
    # input:
    # - x guess for the point at which the jacobian is singular
    # - p guess for the parameter value `<: Real` at which the jacobian is singular
    # The jacobian of the MA problem is solved with a BLS method
    a = 𝐁𝐓.a
    b = 𝐁𝐓.b
    # update parameter
    par = set(params, getlens(𝐁𝐓.prob_vf), p1)
    par = set(par, 𝐁𝐓.lens2, p2)
    # ┌      ┐┌  ┐   ┌ ┐
    # │ J  a ││v1│ = │0│
    # │ b  0 ││σ1│   │1│
    # └      ┘└  ┘   └ ┘
    # In the notations of Govaerts 2000, a = w, b = v
    # Thus, b should be a null vector of J
    #       a should be a null vector of J'
    # we solve Jv + a σ1 = 0 with <b, v> = n
    # the solution is v = -σ1 J\a with σ1 = -n/<b, J^{-1}a>
    n = T(1)
    J = jacobian(𝐁𝐓.prob_vf, x, par)
    v1, σ1, cv, it = 𝐁𝐓.linbdsolver(J, a, b, T(0), 𝐁𝐓.zero, n)
    ~cv && @debug "Linear solver for J did not converge."
    # ┌      ┐┌  ┐   ┌   ┐
    # │ J  a ││v2│ = │ v1│
    # │ b  0 ││σ2│   │ 0 │
    # └      ┘└  ┘   └   ┘
    # this could be greatly improved by saving the factorization
    _, σ2, cv, _ = 𝐁𝐓.linbdsolver(J, a, b, T(0), v1, zero(T))
    ~cv && @debug "Linear solver for J did not converge."
    return residual(𝐁𝐓.prob_vf, x, par), σ1, σ2
end

# this function encodes the functional
function (𝐁𝐓::BTProblemMinimallyAugmented)(x::BorderedArray, params)
    res = 𝐁𝐓(x.u, x.p[1], x.p[2], params)
    return BorderedArray(res[1], [res[2], res[3]])
end

@views function (𝐁𝐓::BTProblemMinimallyAugmented)(x::AbstractVector, params)
    res = 𝐁𝐓(x[1:end-2], x[end-1], x[end], params)
    return vcat(res[1], res[2], res[3])
end
################################################################################
# Struct to invert the jacobian of the BT MA problem.
struct BTLinearSolverMinAug <: AbstractLinearSolver; end

function btMALinearSolver(x, p::Vector{T}, 𝐁𝐓::BTProblemMinimallyAugmented, par,
                            rhsu, rhsp) where T
    # Recall that the functional we want to solve is
    #                    [F(x,p1,p2), σ1(x,p1,p2), σ2(x,p1,p2)]
    # where σi(x,p1,p2) is computed in the function above.
    # The jacobian has to be passed as a tuple as Jac_bt_MA(u0, 𝐁𝐓::BTProblemMinimallyAugmented) = (return (u0, 𝐁𝐓, d2F::Bool))
    # The Jacobian J of the vector field is expressed at (x, p)
    # We solve here Jbt⋅res = rhs := [rhsu, rhsp]
    # The Jacobian expression Jbt of the BT problem is
    #           ┌           ┐
    #   Jbt  =  │  J    dpF │
    #           │ σ1x   σ1p │
    #           │ σ2x   σ2p │
    #           └           ┘
    # where σx := ∂_xσ and σp := ∂_pσ
    # We recall the expression of σ1x = -< w1, ∂J v1> where (w, _) is solution of J'w + b σ2 = 0 with <a, w> = n and
    #                             σ2x = -< w2, ∂J v1> - < w1, ∂J v2>
    ################### Extraction of function names ###########################
    a = 𝐁𝐓.a
    b = 𝐁𝐓.b

    p1, p2 = p

    # parameter axis
    lens = getlens(𝐁𝐓)
    # update parameter
    par0 = set(par, getlens(𝐁𝐓.prob_vf), p1)
    par0 = set(par0, 𝐁𝐓.lens2, p2)

    # par0 = set(par, lens, p)

    # we define the following jacobian. It is used at least 3 times below. This avoids doing 3 times the (possibly) costly building of J(x, p)
    J_at_xp = jacobian(𝐁𝐓.prob_vf, x, par0)

    # we do the following in order to avoid computing J_at_xp twice in case 𝐁𝐓.Jadjoint is not provided
    if is_symmetric(𝐁𝐓.prob_vf)
        JAd_at_xp = J_at_xp
    else
        JAd_at_xp = has_adjoint(𝐁𝐓) ? jad(𝐁𝐓.prob_vf, x, par0) : transpose(J_at_xp)
    end

    # normalization
    n = T(1)

    # we solve Jv + a σ1 = 0 with <b, v> = n
    # the solution is v = -σ1 J\a with σ1 = -n/<b, J\a>
    v1, σ1, cv, itv1 = 𝐁𝐓.linbdsolver(J_at_xp, a, b, T(0), 𝐁𝐓.zero, n)
    ~cv && @debug "Linear solver for J did not converge."

    v2, σ2, cv, itv2 = 𝐁𝐓.linbdsolver(J_at_xp, a, b, T(0), v1, zero(T))
    ~cv && @debug "Linear solver for J did not converge."

    # we solve J'w + b σ2 = 0 with <a, w> = n
    # the solution is w = -σ2 J'\b with σ2 = -n/<a, J'\b>
    w1, _, cv, itw1 = 𝐁𝐓.linbdsolver(JAd_at_xp, b, a, T(0), 𝐁𝐓.zero, n)
    ~cv && @debug "Linear solver for J' did not converge."

    w2, _, cv, itw2 = 𝐁𝐓.linbdsolver(JAd_at_xp, b, a, T(0), w1, zero(T))
    ~cv && @debug "Linear solver for J' did not converge."

    δ = getdelta(𝐁𝐓.prob_vf)
    ϵ1, ϵ2, ϵ3 = T(δ), T(δ), T(δ)
    ################### computation of σx σp ####################
    ################### and inversion of Jbt ####################
    lens1, lens2 = getlenses(𝐁𝐓)
    dp1F = minus(residual(𝐁𝐓.prob_vf, x, set(par, lens1, p1 + ϵ1)),
                 residual(𝐁𝐓.prob_vf, x, set(par, lens1, p1 - ϵ1))); rmul!(dp1F, T(1/(2ϵ1)))
    dp2F = minus(residual(𝐁𝐓.prob_vf, x, set(par, lens2, p2 + ϵ1)),
                 residual(𝐁𝐓.prob_vf, x, set(par, lens2, p2 - ϵ1))); rmul!(dp2F, T(1/(2ϵ1)))

    dJvdp1 = minus(apply(jacobian(𝐁𝐓.prob_vf, x, set(par, lens1, p1 + ϵ3)), v1),
                   apply(jacobian(𝐁𝐓.prob_vf, x, set(par, lens1, p1 - ϵ3)), v1)); rmul!(dJvdp1, T(1/(2ϵ3)))
    σ1p1 = -dot(w1, dJvdp1) / n

    dJvdp2 = minus(apply(jacobian(𝐁𝐓.prob_vf, x, set(par, lens2, p2 + ϵ3)), v1),
                   apply(jacobian(𝐁𝐓.prob_vf, x, set(par, lens2, p2 - ϵ3)), v1)); rmul!(dJvdp2, T(1/(2ϵ3)))
    σ1p2 = -dot(w1, dJvdp2) / n

    dJv1dp1 = minus(apply(jacobian(𝐁𝐓.prob_vf, x, set(par, lens1, p1 + ϵ3)), v1),
                    apply(jacobian(𝐁𝐓.prob_vf, x, set(par, lens1, p1 - ϵ3)), v1)); rmul!(dJv1dp1, T(1/(2ϵ3)))
    dJv2dp1 = minus(apply(jacobian(𝐁𝐓.prob_vf, x, set(par, lens1, p1 + ϵ3)), v2),
                    apply(jacobian(𝐁𝐓.prob_vf, x, set(par, lens1, p1 - ϵ3)), v2)); rmul!(dJv2dp1, T(1/(2ϵ3)))
    σ2p1 = -dot(w2, dJv1dp1) / n - dot(w1, dJv2dp1) / n


    dJv1dp2 = minus(apply(jacobian(𝐁𝐓.prob_vf, x, set(par, lens2, p2 + ϵ3)), v1),
                    apply(jacobian(𝐁𝐓.prob_vf, x, set(par, lens2, p2 - ϵ3)), v1)); rmul!(dJv1dp2, T(1/(2ϵ3)))
    dJv2dp2 = minus(apply(jacobian(𝐁𝐓.prob_vf, x, set(par, lens2, p2 + ϵ3)), v2),
                    apply(jacobian(𝐁𝐓.prob_vf, x, set(par, lens2, p2 - ϵ3)), v2)); rmul!(dJv2dp2, T(1/(2ϵ3)))
    σ2p2 = -dot(w2, dJv1dp2) / n - dot(w1, dJv2dp2) / n
    σp = [σ1p1 σ1p2; σ2p1 σ2p2]

    if 1==1
        # We invert the jacobian of the bt problem when the Hessian of x -> F(x, p) is not known analytically.
        # apply Jacobian adjoint
        u11 = apply_jacobian(𝐁𝐓.prob_vf, x + ϵ2 * v1, par0, w1, true)
        u12 = apply(JAd_at_xp, w1)
        σ1x = minus(u12, u11); rmul!(σ1x, 1 / ϵ2)

        u21 = apply_jacobian(𝐁𝐓.prob_vf, x + ϵ2 * v1, par0, w2, true)
        u22 = apply(JAd_at_xp, w2)
        σ2x1 = minus(u22, u21); rmul!(σ2x1, 1 / ϵ2)

        u21 = apply_jacobian(𝐁𝐓.prob_vf, x + ϵ2 * v2, par0, w1, true)
        u22 = apply(JAd_at_xp, w1)
        σ2x2 = minus(u22, u21); rmul!(σ2x2, 1 / ϵ2)
        σ2x = σ2x1 + σ2x2
        ########## Resolution of the bordered linear system ########
        # we invert Jbt
        dX, dsig, flag, it = 𝐁𝐓.linbdsolver(Val(:Block), J_at_xp, (dp1F, dp2F), (σ1x, σ2x), σp, rhsu, rhsp)
        ~flag && @debug "Linear solver for J did not converge."
    end

    return dX, dsig, true, sum(it) + sum(itv1) + sum(itw1) + sum(itv2) + sum(itw2)
end

function (btls::BTLinearSolverMinAug)(Jbt, du::BorderedArray{vectype, T}; debugArray = nothing, kwargs...) where {vectype, T}
    # kwargs is used by AbstractLinearSolver
    out =  btMALinearSolver((Jbt.x).u,
                 (Jbt.x).p,
                 Jbt.fldpb,
                 Jbt.params,
                 du.u, du.p)
    # this type annotation enforces type stability
    return BorderedArray{vectype, T}(out[1], out[2]), out[3], out[4]
end
###################################################################################################
@inline has_adjoint(BTpb::BTMAProblem) = has_adjoint(BTpb.prob)
@inline is_symmetric(BTpb::BTMAProblem) = is_symmetric(BTpb.prob)
residual(BTpb::BTMAProblem, x, p) = BTpb.prob(x, p)
jacobian(BTpb::BTMAProblem, x, p) = (x = x, params = p, fldpb = BTpb.prob)
jad(BTpb::BTMAProblem, args...) = jad(BTpb.prob, args...)
################################################################################################### Newton functions
"""
$(SIGNATURES)

This function turns an initial guess for a BT point into a solution to the BT problem based on a Minimally Augmented formulation. The arguments are as follows
- `prob::AbstractBifurcationFunction`
- `btpointguess` initial guess (x_0, p_0) for the BT point. It should be a `BorderedArray` as returned by the function `BTPoint`
- `par` parameters used for the vector field
- `eigenvec` guess for the 0 eigenvector
- `eigenvec_ad` guess for the 0 adjoint eigenvector
- `options::NewtonPar` options for the Newton-Krylov algorithm, see [`NewtonPar`](@ref).

# Optional arguments:
- `normN = norm`
- `bdlinsolver` bordered linear solver for the constraint equation
- `jacobian_ma::Symbol = true` specify the way the (newton) linear system is solved. Can be (:autodiff, :finitedifferences, :minaug)
- `kwargs` keywords arguments to be passed to the regular Newton-Krylov solver

# Simplified call
Simplified call to refine an initial guess for a BT point. More precisely, the call is as follows

    newton(br::AbstractBranchResult, ind_bt::Int; options = br.contparams.newton_options, kwargs...)

The parameters / options are as usual except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of bifurcation point in `br` you want to refine. You can pass newton parameters different from the ones stored in `br` by using the argument `options`.

!!! tip "Jacobian transpose"
    The adjoint of the jacobian `J` is computed internally when `Jᵗ = nothing` by using `transpose(J)` which works fine when `J` is an `AbstractArray`. In this case, do not pass the jacobian adjoint like `Jᵗ = (x, p) -> transpose(d_xF(x, p))` otherwise the jacobian will be computed twice!

!!! tip "ODE problems"
    For ODE problems, it is more efficient to pass the option `jacobian_ma = :autodiff`
"""
function newton_bt(prob::AbstractBifurcationProblem,
                btpointguess, par,
                lens2::Lens,
                eigenvec, eigenvec_ad,
                options::NewtonPar;
                normN = norm,
                jacobian_ma::Symbol = :autodiff,
                usehessian = false,
                bdlinsolver::AbstractBorderedLinearSolver = MatrixBLS(),
                kwargs...)

    @assert jacobian_ma in (:autodiff, :finitedifferences, :minaug)

    𝐁𝐓 = BTProblemMinimallyAugmented(
        prob,
        _copy(eigenvec_ad), # a
        _copy(eigenvec),    # b
        options.linsolver,
        lens2;
        # do not change linear solver if user provides it
        linbdsolver = (@set bdlinsolver.solver = isnothing(bdlinsolver.solver) ? options.linsolver : bdlinsolver.solver),
        usehessian = usehessian)

    Ty = eltype(btpointguess)

    if jacobian_ma == :autodiff
        prob_f = BifurcationProblem(𝐁𝐓, btpointguess, par)
        optn_bt = @set options.linsolver = DefaultLS()
    elseif jacobian_ma == :finitedifferences
        prob_f = BifurcationProblem(𝐁𝐓, btpointguess, par;
            J = (x, p) -> finite_differences(z -> 𝐁𝐓(z, p), x))
        optn_bt = @set options.linsolver = DefaultLS()
    else
        prob_f = BTMAProblem(𝐁𝐓, jacobian_ma, BorderedArray(btpointguess[1:end-2], btpointguess[end-1:end]), par, nothing, prob.plotSolution, prob.recordFromSolution)
        # options for the Newton Solver
        optn_bt = @set options.linsolver = BTLinearSolverMinAug()
    end

    # solve the BT equations
    sol = newton(prob_f, optn_bt; normN = normN, kwargs...)

    # save the solution in BogdanovTakens
    pbt = extractParBLS(sol.u, 2)
    parbt = set(par, getlens(prob), pbt[1])
    parbt = set(parbt, lens2, pbt[2])
    bt = BogdanovTakens(x0 = get_vec_bls(sol.u, 2), params = parbt, lens = (getlens(prob), lens2), ζ = nothing, ζ★ = nothing, type = :none, nf = (a = missing, b = missing ),
    nfsupp = (K2 = zero(Ty),))
    @set sol.u = bt
end

"""
$(SIGNATURES)

This function turns an initial guess for a Bogdanov-Takens point into a solution to the Bogdanov-Takens problem based on a Minimally Augmented formulation.

## Arguments
- `br` results returned after a call to [continuation](@ref Library-Continuation)
- `ind_bif` bifurcation index in `br`

# Optional arguments:
- `options::NewtonPar`, default value `br.contparams.newton_options`
- `normN = norm`
- `options` You can pass newton parameters different from the ones stored in `br` by using this argument `options`.
- `jacobian_ma::Symbol = true` specify the way the (newton) linear system is solved. Can be (:autodiff, :finitedifferences, :minaug)
- `bdlinsolver` bordered linear solver for the constraint equation
- `start_with_eigen = false` whether to start the Minimally Augmented problem with information from eigen elements.
- `kwargs` keywords arguments to be passed to the regular Newton-Krylov solver

!!! tip "ODE problems"
    For ODE problems, it is more efficient to pass the option `jacobian = :autodiff`

!!! tip "start_with_eigen"
    For ODE problems, it is more efficient to pass the option `start_with_eigen = true`
"""
function newton_bt(br::AbstractResult{Tkind, Tprob}, ind_bt::Int;
                probvf = br.prob.prob.prob_vf,
                normN = norm,
                options = br.contparams.newton_options,
                nev = br.contparams.nev,
                start_with_eigen = false,
                bdlinsolver::AbstractBorderedLinearSolver = MatrixBLS(),
                kwargs...) where {Tkind, Tprob <: Union{FoldMAProblem, HopfMAProblem}}

    prob_ma = br.prob.prob

    btpointguess = bt_point(br, ind_bt)

    # we look for a solution which is a Vector so we can use ForwardDiff
    btpointguess = vcat(getvec(btpointguess, prob_ma), getp(btpointguess, prob_ma))

    bifpt = br.specialpoint[ind_bt]
    eigenvec = getvec(bifpt.τ.u, prob_ma); rmul!(eigenvec, 1/normN(eigenvec))
    # in the case of Fold continuation, this could be ill-defined.
    if ~isnothing(findfirst(isnan, eigenvec)) && ~start_with_eigen
        @warn "Eigenvector ill defined (has NaN). Use the option start_with_eigen = true"
    end
    eigenvec_ad = _copy(eigenvec)

    if start_with_eigen
        x0, parbif = get_bif_point_codim2(br, ind_bt)

        # jacobian at bifurcation point
        L = jacobian(prob_ma.prob_vf, x0, parbif)

        # computation of zero eigenvector
        λ = zero(getvectoreltype(br))
        ζ, = get_adjoint_basis(L, λ, br.contparams.newton_options.eigsolver.eigsolver; nev = nev, verbose = false)
        eigenvec .= real.(ζ)
        rmul!(eigenvec, 1/normN(eigenvec))

        # computation of adjoint eigenvector
        Lt = has_adjoint(prob_ma.prob_vf) ? jad(prob_ma.prob_vf, x0, parbif) : adjoint(L)
        ζstar, = get_adjoint_basis(Lt, λ, br.contparams.newton_options.eigsolver.eigsolver; nev = nev, verbose = false)
        eigenvec_ad .= real.(ζstar)
        rmul!(eigenvec_ad, 1/normN(eigenvec_ad))
    end

    # solve the BT equations
    return newton_bt(prob_ma.prob_vf, btpointguess, getparams(br), getlens(br), eigenvec, eigenvec_ad, options; normN = normN, bdlinsolver = bdlinsolver, kwargs...)
end
