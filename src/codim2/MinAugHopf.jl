"""
$(TYPEDSIGNATURES)

For an initial guess from the index of a Hopf bifurcation point located in `ContResult.specialpoint`, returns a point which can be refined using `newton_hopf`.
"""
function hopf_point(br::AbstractBranchResult, index::Int)
    if br.specialpoint[index].type != :hopf 
        error("The provided index does not refer to a Hopf point")
    end
    specialpoint = br.specialpoint[index] # Hopf point
    p = specialpoint.param                # parameter value at the Hopf point
    ω = imag(br.eig[specialpoint.idx].eigenvals[specialpoint.ind_ev]) # frequency at the Hopf point
    return BorderedArray(_copy(saved_solution(specialpoint.x)), [p, ω] )
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# this function encodes the functional in the case where the Mass matrix is I, passed as ::Nothing
hopf_ma_test(𝐇, M, J, a, b, J22, _zero, n, ω::𝒯) where {𝒯} = _hopf_ma_test(𝐇.linbdsolver, M, J, a, b, J22, _zero, n, Complex{𝒯}(0, -ω))
_hopf_ma_test(linbdsolver, ::IdentityOperator, J, a, b, J22, _zero, n, a₀ ) = linbdsolver(ShiftedOperator(;J, a₀), a, b, J22, _zero, n)

function (𝐇::HopfMinimallyAugmentedFormulation)(x, p::𝒯, ω::𝒯, params) where 𝒯
    # These are the equations of the minimally augmented (MA) formulation of the 
    # Hopf bifurcation point
    # input:
    # - x guess for the point at which the jacobian has a purely imaginary eigenvalue
    # - p guess for the parameter for which the jacobian has a purely imaginary eigenvalue
    # The jacobian of the MA problem is solved with a BLS method
    # ┌             ┐┌  ┐   ┌ ┐
    # │ J-iω⋅M  M⋅a ││v │ = │0│
    # │  M⋅b      0 ││σ1│   │1│
    # └             ┘└  ┘   └ ┘
    # In the notations of Govaerts 2000, a = w, b = v
    # Thus, b should be a null vector of J - iω⋅M
    #       a should be a null vector of J'+ iω⋅M'
    a = 𝐇.a
    b = 𝐇.b
    # update parameter
    par = set(params, getlens(𝐇), p)
    # we solve (J - iω)⋅v + M⋅a σ1 = 0 with <M⋅b, v> = 1
    # note that the shift argument only affect J in this call:
    J = jacobian(𝐇.prob_vf, x, par)
    M = getmassmatrix(𝐇.prob_vf, x, par)
    _, σ1, cv, = hopf_ma_test(𝐇, M, J, a, b, zero(𝒯), 𝐇.zero, one(𝒯), ω)
    ~cv && @debug "[Hopf residual] Linear solver for (J-iω) did not converge."
    return residual(𝐇.prob_vf, x, par), real(σ1), imag(σ1)
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
$(TYPEDSIGNATURES)

Compute the solution (v, σ) of 

```
┌                    ┐ ┌  ┐   ┌   ┐
│ J - iω⋅M    M⋅𝐇.a  │ │v │ = │ 0 │
│  (M⋅𝐇.b)'     0    │ │σ │   │ 1 │
└                    ┘ └  ┘   └   ┘
```

and the same for the adjoint system with solution (w, τ).
"""
function _compute_bordered_vectors(𝐇::HopfMinimallyAugmentedFormulation, M, M★, J, J★, ω)
    return __compute_bordered_vectors_hopf(𝐇.linbdsolver,
                                      𝐇.linbdsolverAdjoint,
                                      M,
                                      M★,
                                      J,
                                      J★,
                                      ω,
                                      𝐇.a,
                                      𝐇.b,
                                      𝐇.zero)
end

function __compute_bordered_vectors_hopf(linbdsolver, linbdsolver_adjoint, ::IdentityOperator, M★, J, J★, ω::𝒯, a, b, _zero_vector) where {𝒯}
    v, σ1, cv, itv = linbdsolver(ShiftedOperator(J = J, a₀ = Complex{𝒯}(0, -ω) ), a, b, zero(𝒯), _zero_vector, one(𝒯))
    # we solve (J-iωI)v + a σ1 = 0 with <b, v> = 1
    ~cv && @debug "Bordered linear solver for (J-iωI) did not converge."

    # we solve (J+iωI)'w + b σ2 = 0 with <a, w> = 1
    w, σ2, cv, itw = linbdsolver_adjoint(ShiftedOperator(J = J★, a₀ = Complex{𝒯}(0, ω) ), b, a, zero(𝒯), _zero_vector, one(𝒯))
    ~cv && @debug "Bordered linear solver for (J+iωI)' did not converge."
    # we should have σ1 ≈ conj(σ2)
    return (; v, w, itv, itw, σ1, σ2)
end

function _get_bordered_terms(𝐇::HopfMinimallyAugmentedFormulation, x, p::𝒯, ω::𝒯, par) where 𝒯
    # update parameter
    lens = getlens(𝐇)
    par0 = set(par, lens, p)

    # This avoids doing 3 times the possibly costly building of J(x, p)
    J = jacobian(𝐇.prob_vf, x, par0)
    M = getmassmatrix(𝐇.prob_vf, x, par0)
    # Avoid computing J twice in case 𝐇.Jadjoint is not provided
    J★ = has_adjoint(𝐇) ? jacobian_adjoint(𝐇.prob_vf, x, par0) : transpose(J)
    M★ = has_massmatrix_adjoint(𝐇.prob_vf) ? getmassmatrix_adjoint(𝐇.prob_vf, x, par0) : adjoint(M)

    (; v, w, itv, itw, σ1, σ2) = _compute_bordered_vectors(𝐇, M, M★, J, J★, ω)

    δ = getdelta(𝐇.prob_vf)
    ϵ2 = ϵₚ = 𝒯(δ)
    ################### computation of σω σp ####################
    dₚF  = R01(𝐇.prob_vf, x, set(par, lens, p))
    dₚJv = R11(𝐇.prob_vf, x, set(par, lens, p), v)
    σₚ = -VI.inner(w, dₚJv)
    if is_mass_matrix_constant(𝐇.prob_vf) == false
        σₚ += _dₚσ_mass(𝐇.prob_vf, x, par0, 𝐇.a, 𝐇.b, v, w, σ1, σ2, ω)
    end
    σω = Complex{𝒯}(0, 1) * dot_with_mass(w, M, v)
    return (;J_at_xp = J, JAd_at_xp = J★, dₚF, σₚ, δ, ϵ2, v, w, par0, itv, itw, σω, M, σ1, σ2)
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function jacobian(pdpb::HopfMAProblem{Tprob, MinAugMatrixBased}, X::AbstractVector{𝒯}, par) where {Tprob, 𝒯}
    𝐇 = get_formulation(pdpb)
    x = @view X[begin:end-2]
    p = X[end-1]
    ω = X[end]

    (;J_at_xp, JAd_at_xp, dₚF, σₚ, ϵ2, v, w, par0, σω, σ1, σ2) = _get_bordered_terms(𝐇, x, p, ω, par)

    cw = conj(w)
    vr = real(v); vi = imag(v)
    # this is not R20, there is a transpose
    u1r = apply_jacobian(𝐇.prob_vf, x + ϵ2 * vr, par0, cw, true)
    u1i = apply_jacobian(𝐇.prob_vf, x + ϵ2 * vi, par0, cw, true)
    u2 = apply(JAd_at_xp,  cw)
    σxv2r = @. -(u1r - u2) / ϵ2
    σxv2i = @. -(u1i - u2) / ϵ2
    σₓ = @. σxv2r + Complex{𝒯}(0, 1) * σxv2i

    if is_mass_matrix_constant(𝐇.prob_vf) == false
        # nonconstant mass matrix contribution ∇_x σ1 to the state row of σₓ
        σₓ .+= _dₓσ_mass(𝐇.prob_vf, x, par0, 𝐇.a, 𝐇.b, v, w, σ1, σ2, ω)
    end

    Jhopf = hcat(J_at_xp, dₚF, VI.zerovector(dₚF))
    Jhopf = vcat(Jhopf, vcat(real(σₓ), real(σₚ), real(σω))')
    Jhopf = vcat(Jhopf, vcat(imag(σₓ), imag(σₚ), imag(σω))')
    return Jhopf
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# contribution of a nonconstant mass matrix M(x, p) to ∂σ1. The border scalar solves
#     (J  - iωM )v + M a σ1 = 0,   <M b, v> = 1
#     (J' + iωM')w + M b σ2 = 0,   <M a, w> = 1
# Differentiating σ1 with respect to the state gives
#     ∇_x σ1 = iω ∇_x⟨w,Mv⟩ - σ1 ∇_x⟨w,Ma⟩ - conj(σ2) conj.(∇_x⟨v,Mb⟩)
# the δJ term -<w, δJ v> being handled separately. The derivatives of
# `⟨w, M(x, p) v⟩` are provided by `∇_x_mass_matrix` / `R01_mass_matrix` (the
# former can be user provided, see `MassFunction`).

# Returns the vector ∇_x σ1 restricted to the mass contribution.
function _dₓσ_mass(prob_vf, x, par0, a, b, v, w, σ1, σ2, ω)
    return Complex{typeof(ω)}(0, 1) * ω * ∇_x_mass_matrix(prob_vf, x, par0, v, w) -
                                     σ1 * ∇_x_mass_matrix(prob_vf, x, par0, a, w) -
                         conj(σ2) * conj.(∇_x_mass_matrix(prob_vf, x, par0, b, v))
end

# parameter direction: σₚ stores the plain ∂_pσ1 (no conjugation).
function _dₚσ_mass(prob_vf, x, par0, a, b, v, w, σ1, σ2, ω)
    return Complex{typeof(ω)}(0, 1) * ω * R01_mass_matrix(prob_vf, x, par0, v, w) -
                                     σ1 * R01_mass_matrix(prob_vf, x, par0, a, w) -
                          conj(σ2) * conj(R01_mass_matrix(prob_vf, x, par0, b, v))
end

# Struct to invert the jacobian of the Hopf MA problem.
struct HopfLinearSolverMinAug <: AbstractLinearSolver; end

"""
This function solves the linear problem associated with a linearization of the minimally augmented formulation of the Hopf bifurcation point.
"""
function _hopf_MA_linear_solver(x, p::𝒯, ω::𝒯, 𝐇::HopfMinimallyAugmentedFormulation, par,
                              duu, dup, duω) where 𝒯
    # N = length(du) - 2
    # The Jacobian J of the vector field is expressed at (x, p)
    # the jacobian expression Jhopf of the hopf problem is
    #           ┌             ┐
    #  Jhopf =  │  J  dₚF   0 │
    #           │ σx   σp  σω │
    #           └             ┘
    ########## Resolution of the bordered linear system ########
    # solve for x1, x2: J⋅x1 = du and J⋅x2 = dₚF
    # J⋅dX      + dₚF⋅dp           = du => dX = x1 - dp⋅x2
    # The second equation
    #    <σx, dX> +  σp⋅dp + σω⋅dω = du[end-1:end]
    # thus becomes
    #   (σp - <σx, x2>)⋅dp + σω⋅dω = du[end-1:end] - <σx, x1>
    # This 2 x 2 system is then solved to get (dp, dω)
    ################### inversion of Jhopf ####################
    (;J_at_xp, JAd_at_xp, dₚF, σₚ, ϵ2, v, w, par0, itv, itw, σω, σ1, σ2) = _get_bordered_terms(𝐇, x, p, ω, par)
    # we should have σ1 ≈ conj(σ2)

    # we solve J⋅x1 = duu and J⋅x2 = dₚF
    x1, x2, cv, (it1, it2) = 𝐇.linsolver(J_at_xp, duu, dₚF)
    ~cv && @debug "Linear solver for J did not converge"

    # the case of ∂ₓσ is a bit more involved
    # we first need to compute the value of ∂ₓσ written σx
    σx = similar(x, Complex{𝒯})
    # σ_i := -⟨w, ∂_i(dF·v)⟩ = -⟨w, d²F[v, e_i]⟩.
    # σ = -(d²F[v, ·])^* w
    # ⟨σ, ξ⟩ = -⟨w, d²F[v, ξ]⟩.

    if 𝐇.usehessian == false || has_hessian(𝐇) == false
        # finite differences version
        # (J(x+εv)ᵀ w - J(x)ᵀ w)/ε  =  (d²F[v,·])^* w  =  -σx
        cw = conj(w); vr = real(v); vi = imag(v)
        # apply jacobian adjoint
        u1r = apply_jacobian(𝐇.prob_vf, x + ϵ2 * vr, par0, cw, true)
        u1i = apply_jacobian(𝐇.prob_vf, x + ϵ2 * vi, par0, cw, true)
        u2 = apply(JAd_at_xp,  cw)
        σxv2r = @. -(u1r - u2) / ϵ2
        σxv2i = @. -(u1i - u2) / ϵ2

        σx = @. σxv2r + Complex{𝒯}(0, 1) * σxv2i

        σxx1 = VI.inner(σx, x1)
        σxx2 = VI.inner(σx, x2)
    else
        # ∂_ξσ1 = -⟨w, d²F[ξ,v]⟩ + iωM⟨w,(∂_ξM)v⟩ - σ1⟨w,(∂_ξM)a⟩ - σ2⟨(∂_ξM)b, v⟩
        # σ_i = -⟨w, ∂_i(dF·v)⟩ = -⟨w, d²F[v, e_i]⟩
        # hence ⟨σx, ξ⟩ = -⟨w, d²F[v, ξ]⟩ for ξ = x1, x2
        d2Fv = d2F(𝐇.prob_vf, x, par0, v, x1)
        σxx1 = -conj(VI.inner(w, d2Fv))
        d2Fv = d2F(𝐇.prob_vf, x, par0, v, x2)
        σxx2 = -conj(VI.inner(w, d2Fv))
    end
    if is_mass_matrix_constant(𝐇.prob_vf) == false
        σx_mass = _dₓσ_mass(𝐇.prob_vf, x, par0, 𝐇.a, 𝐇.b, v, w, σ1, σ2, ω)
        σxx1 += VI.inner(σx_mass, x1)
        σxx2 += VI.inner(σx_mass, x2)
    end
    # We need to be careful here because the dot produces conjugates. 
    # Hence the + dot(σx, x2) and + imag(dot(σx, x1) and not the opposite
    LS = Matrix{𝒯}(undef, 2, 2);
    rhs = Vector{𝒯}(undef, 2);
    LS[1, 1] = real(σₚ - σxx2); LS[1, 2] = real(σω)
    LS[2, 1] = imag(σₚ + σxx2); LS[2, 2] = imag(σω)
    rhs[1] = dup - real(σxx1); rhs[2] =  duω + imag(σxx1)
    dp, dω = LS \ rhs
    return x1 .- dp .* x2, dp, dω, true, it1 + it2 + sum(itv) + sum(itw)
end

function (::HopfLinearSolverMinAug)(Jhopf, du::BorderedArray{vectype, 𝒯}; kwargs...)  where {vectype, 𝒯}
    # kwargs is used by AbstractLinearSolver
    out = _hopf_MA_linear_solver((Jhopf.x).u, #!! TODO !! This seems TU
                (Jhopf.x).p[1],
                (Jhopf.x).p[2],
                Jhopf.pbma,
                Jhopf.params,
                du.u, du.p[1], du.p[2])
    return BorderedArray{vectype, 𝒯}(out[1], [out[2], out[3]]), out[4], out[5]
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@inline has_adjoint(pb::HopfMAProblem) = has_adjoint(get_formulation(pb))
@inline is_symmetric(pb::HopfMAProblem) = is_symmetric(get_formulation(pb))

function finalise_solution(iter::ContIterable{HopfCont},
                            state::AbstractContinuationState, 
                            contres)
    isbt = isnothing(contres) ? true : isnothing(findfirst(x -> x.type in (:bt, :ghbt, :btgh), contres.specialpoint))
    fin_user = iter.finalise_solution(getsolution(state),
                                  state.τ,
                                  state.step,
                                  contres; 
                                  state,
                                  iter)
    return isbt && fin_user
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
$(TYPEDSIGNATURES)

This function turns an initial guess for a Hopf point into a solution to the Hopf problem based on a Minimally Augmented formulation. The arguments are as follows
- `prob::AbstractBifurcationProblem` where `p` is a set of parameters.
- `hopfpointguess` initial guess (x_0, p_0) for the Hopf point. It should a `BorderedArray` as returned by the function `hopf_point`.
- `par` parameters used for the vector field
- `eigenvec` guess for the  iωM eigenvector
- `eigenvec_ad` guess for the -iωM eigenvector
- `options::NewtonPar` options for the Newton-Krylov algorithm, see [`NewtonPar`](@ref).

# Optional arguments:
- `normN = norm`
- `bdlinsolver` bordered linear solver for the constraint equation
- `kwargs` keywords arguments to be passed to the regular Newton-Krylov solver

# Simplified call:
Simplified call to refine an initial guess for a Hopf point. More precisely, the call is as follows

    newton_hopf(br::AbstractBranchResult, ind_hopf::Int; normN = norm, options = br.contparams.newton_options, kwargs...)

The parameters / options are as usual except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of bifurcation point in `br` you want to refine. You can pass newton parameters different from the ones stored in `br` by using the argument `options`.

!!! tip "Jacobian transpose"
    The adjoint of the jacobian `J` is computed internally when `Jᵗ = nothing` by using `transpose(J)` which works fine when `J` is an `AbstractArray`. In this case, do not pass the jacobian adjoint like `Jᵗ = (x, p) -> transpose(dₓF(x, p))` otherwise the jacobian will be computed twice!

!!! tip "ODE problems"
    For ODE problems, it is more efficient to use the Matrix based Bordered Linear Solver passing the option `bdlinsolver = MatrixBLS()`
"""
function newton_hopf(prob,
                    guessₕ::BorderedArray,
                    par,
                    ζ, ζ★,
                    options::NewtonPar;
                    normN = norm,
                    bdlinsolver::AbstractBorderedLinearSolver = MatrixBLS(),
                    usehessian = true,
                    jacobian_ma::AbstractJacobianType = AutoDiff(),
                    kwargs...)
    𝐇 = HopfMinimallyAugmentedFormulation(
            re_make(prob; params = par),
            _copy(ζ★), # this is pb.a ≈ null space of (J - iω M)^*
            _copy(ζ),    # this is pb.b ≈ null space of  J - iω M
            options.linsolver,
            # do not change linear solver if user provides it
            @set bdlinsolver.solver = (isnothing(bdlinsolver.solver) ? options.linsolver : bdlinsolver.solver);
            usehessian)
    if jacobian_ma in (AutoDiff(), FiniteDifferencesMF(), FiniteDifferences(), MinAugMatrixBased())
        guessₕ = vcat(guessₕ.u, guessₕ.p)
        prob_h = HopfMAProblem(𝐇, jacobian_ma, guessₕ, nothing, plot_solution(prob), record_from_solution(prob))
        opt_hopf = options
    else
        prob_h = HopfMAProblem(𝐇, nothing, guessₕ, nothing, plot_solution(prob), record_from_solution(prob))
        opt_hopf = @set options.linsolver = HopfLinearSolverMinAug()
    end
    return solve(prob_h, Newton(), opt_hopf; normN, kwargs...)
end

# this version extracts the border vectors
function newton_hopf(prob,
                    guessₕ::BorderedArray,
                    par,
                    options::NewtonPar;
                    nev = 10,
                    start_with_eigen = false,
                    bdlinsolver::AbstractBorderedLinearSolver = MatrixBLS(),
                    bdlinsolver_adjoint = bdlinsolver,
                    a = nothing,
                    b = nothing,
                    ζ = nothing,
                    normN = norm,
                    kwargs...)
    xₕ = guessₕ.u
    ω = guessₕ.p[2]
    if start_with_eigen
        ζ ./= normN(ζ)
        ζ★ = conj.(ζ)
        # computation of adjoint eigenvalue. Recall that b should be a null vector of J-iωM
        λ = Complex(0, ω)

        # jacobian at bifurcation point
        L = jacobian(prob, xₕ, par)
        Mass = getmassmatrix(prob, xₕ, par)

        # computation of adjoint eigenvector
        L★ = ~has_adjoint(prob) ? adjoint(L) : jacobian_adjoint(prob, xₕ, par)
        ζ★, _ = _get_target_eigenvector_from_eigensolver(L★, conj(λ), options.eigsolver; nev)
        ζ★ ./= dot_with_mass(ζ★, Mass, ζ)
    else
        (; ζ, ζad) = _init_hopf_vectors_minaug(prob, xₕ, par, ω, bdlinsolver, bdlinsolver_adjoint, a, b, normN)
        ζ★ = ζad
    end
    return newton_hopf(prob, guessₕ, par, ζ, ζ★, options; normN, bdlinsolver, kwargs...)
end

function newton_hopf(br::AbstractBranchResult, ind_hopf::Int;
            prob = getprob(br),
            options = br.contparams.newton_options,
            kw...)
    guessₕ = hopf_point(br, ind_hopf)
    bifpt = br.specialpoint[ind_hopf]
    options.verbose && println("--> Newton Hopf, the eigenvalue considered here is ", br.eig[bifpt.idx].eigenvals[bifpt.ind_ev])
    @assert bifpt.idx == bifpt.step + 1 "Error, the bifurcation index does not refer to the correct step."
    @assert ~isempty(br.eig[bifpt.idx].eigenvecs) "You must save the eigenvectors for this to work."
    ζ = geteigenvector(options.eigsolver, br.eig[bifpt.idx].eigenvecs, bifpt.ind_ev)
    return newton_hopf(prob, guessₕ, getparams(br), options; nev = br.contparams.nev, ζ, kw...)
end

function update!(𝐏𝐛::HopfMAProblem, iter, state::ContState)
    # it is called to update the Minimally Augmented problem
    # by updating the vectors a, b
    # we first check that the continuation step was successful
    # if not, we do not update the problem with bad information!
    # if we are in a bisection, we still update the MA problem, this does not work well otherwise
    𝐇 = get_formulation(𝐏𝐛)
    success = converged(state)
    step = state.step
    if (~mod_counter(step, 𝐇.update_minaug_every_step) || success == false) || in_bisection(state)
        # update vector field
        return update!(𝐇, iter, state)
    end

    @debug "[Hopf] Update vectors a and b"
    zu = getx(state)
    ω = get_frequency(zu, 𝐇)

    # expression of the jacobian
    x = getvec(zu, 𝐇) # fold point
    newpar = getparams(iter, state)
    J = jacobian(𝐇.prob_vf, x, newpar)
    J★ = has_adjoint(𝐇) ? jacobian_adjoint(𝐇.prob_vf, x, newpar) : adjoint(J)
    M = getmassmatrix(𝐇.prob_vf, x, newpar)
    M★ = has_massmatrix_adjoint(𝐇.prob_vf) ? getmassmatrix_adjoint(𝐇.prob_vf, x, newpar) : adjoint(M)

    bd_vec = _compute_bordered_vectors(𝐇, M, M★, J, J★, ω)

    𝐇.a .= bd_vec.w ./ 𝐇.norm(bd_vec.w)
    # do not normalize with dot(newb, 𝐇.a), it prevents from BT detection
    𝐇.b .= bd_vec.v ./ 𝐇.norm(bd_vec.v)

    # we stop continuation at Bogdanov-Takens points
    threshBT = 100 * iter.contparams.newton_options.tol
    # if the frequency is null, this is not a Hopf point, we halt the process
    isbt = abs(ω) < threshBT

    if isbt
        p1 = get_parameter(zu, 𝐇)
        p2 = getp(state)
        @warn "[Codim 2 Hopf - update!]\nThe Hopf curve seems to be close to a BT point: ω ≈ $ω.\nStopping computations at ($p1, $p2) .\nIf the BT point is not detected, try lowering Newton tolerance or dsmax."
    end

    # call the user-passed update
    update_result = update!(𝐇, iter, state)

    return ((abs(ω) >= threshBT) || in_bisection(state) == false) && (~isbt) && update_result
end

function record_from_solution(iter::ContIterable{Tkind, <: HopfMAProblem},
                              state::AbstractContinuationState) where {Tkind <: AbstractTwoParamCont}
    𝐏𝐛 = getprob(iter)
    𝐇 = get_formulation(𝐏𝐛)
    lens1, lens2 = get_lenses(𝐏𝐛)
    lenses = get_lens_symbol(lens1, lens2)
    u = getx(state)
    p = getp(state)

    return (; zip(lenses, (getp(u, 𝐇)[1], p))..., 
                        ωₕ = getp(u, 𝐇)[2],
                        l1 = 𝐇.l1,
                        BT = 𝐇.BT,
                        GH = 𝐇.GH,
                        _namedrecordfromsol(𝐏𝐛.recordFromSolution(getvec(u, 𝐇), p; iter, state))...
                        ) 
end

"""
$(TYPEDSIGNATURES)

codim 2 continuation of Hopf points. This function turns an initial guess for a Hopf point into a curve of Hopf points based on a Minimally Augmented formulation. The arguments are as follows
- `prob::AbstractBifurcationProblem`
- `hopfpointguess` initial guess (x_0, p1_0) for the Hopf point. It should be a `Vector` or a `BorderedArray`
- `par` set of parameters
- `lens1` parameter axis for parameter 1
- `lens2` parameter axis for parameter 2
- `eigenvec` guess for the iω eigenvector at p1_0
- `eigenvec_ad` guess for the -iω eigenvector at p1_0
- `options_cont` keywords arguments to be passed to the regular [`continuation`](@ref)

# Optional arguments:
- `jacobian_ma = AutoDiff()`, how the linear system of the Hopf problem is solved. Can be `AutoDiff(), FiniteDifferencesMF(), FiniteDifferences(), MinAug(), MinAugMatrixBased`.
- `linsolve_adjoint` solver for (J+iωM)^* ⋅sol = rhs
- `bdlinsolver` bordered linear solver for the constraint equation with top-left block (J-iωM). Required in the linear solver for the Minimally Augmented Hopf functional. This option can be used to pass a dedicated linear solver for example with specific preconditioner.
- `bdlinsolver_adjoint` bordered linear solver for the constraint equation with top-left block (J-iωM)^*. Required in the linear solver for the Minimally Augmented Hopf functional. This option can be used to pass a dedicated linear solver for example with specific preconditioner.
- `update_minaug_every_step` update vectors `a,b` in Minimally Formulation every `update_minaug_every_step` steps
- `compute_eigen_elements = false` whether to compute eigenelements. If `options_cont.detect_event > 0`, it allows the detection of ZH, HH points.
- `kwargs` keywords arguments to be passed to the regular [`continuation`](@ref)

# Simplified call:

    continuation_hopf(br::AbstractBranchResult, ind_hopf::Int, lens2::AllOpticTypes, options_cont::ContinuationPar ;  kwargs...)

where the parameters are as above except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of Hopf point in `br` that you want to refine.

!!! tip "ODE problems"
    For ODE problems, it is more efficient to use the Matrix based Bordered Linear Solver passing the option `bdlinsolver = MatrixBLS()`. This is the default setting.

!!! tip "Jacobian transpose"
    The adjoint of the jacobian `J` is computed internally when `Jᵗ = nothing` by using `transpose(J)` which works fine when `J` is an `AbstractArray`. In this case, do not pass the jacobian adjoint like `Jᵗ = (x, p) -> transpose(d_xF(x, p))` otherwise the jacobian would be computed twice!

!!! tip "Detection of Bogdanov-Takens and Bautin bifurcations"
    In order to trigger the detection, pass `detect_event = 1,2` in `options_cont`. Note that you need to provide `d3F` in `prob`.
"""
function continuation_hopf(prob_vf, alg::AbstractContinuationAlgorithm,
                hopfpointguess::BorderedArray{vectype, Tb}, par,
                lens1::AllOpticTypes, lens2::AllOpticTypes,
                eigenvec, eigenvec_ad,
                options_cont::ContinuationPar ;
                update_minaug_every_step = 1,
                normC = norm,

                linsolve_adjoint = options_cont.newton_options.linsolver,
                bdlinsolver::AbstractBorderedLinearSolver = MatrixBLS(),
                bdlinsolver_adjoint::AbstractBorderedLinearSolver = bdlinsolver,

                jacobian_ma::AbstractJacobianType = AutoDiff(),
                compute_eigen_elements = false,
                usehessian = true,
                kind = HopfCont(),
                record_from_solution = nothing,
                kwargs...) where {Tb, vectype}
    lens1 == lens2 && error("Please choose 2 different parameters. You only passed $lens1")
    lens1 != getlens(prob_vf) && error("lens1 must be the continuation parameter. You passed $lens1")

    # options for the Newton solver inherited from the ones provided by the user
    options_newton = options_cont.newton_options
    # tolerance for detecting BT bifurcation and stopping continuation
    threshBT = 100options_newton.tol

    𝐇 = HopfMinimallyAugmentedFormulation(
        re_make(prob_vf; params = par),
        _copy(eigenvec_ad), # this is a ≈ null space of (J - iω M)^*
        _copy(eigenvec),    # this is b ≈ null space of  J - iω M
        options_newton.linsolver,
        # do not change linear solver if user provides it
        @set bdlinsolver.solver = (isnothing(bdlinsolver.solver) ? options_newton.linsolver : bdlinsolver.solver);
        linsolve_adjoint = linsolve_adjoint,
        linbdsolve_adjoint = bdlinsolver_adjoint,
        usehessian,
        _norm = normC,
        update_minaug_every_step
        )

    # jacobians for the Hopf problem
    record_hopf = RecordForHopf(record_from_solution, BifurcationKit.record_from_solution(prob_vf))
    if jacobian_ma in (AutoDiff(), FiniteDifferencesMF(), FiniteDifferences(), MinAugMatrixBased())
        hopfpointguess = vcat(hopfpointguess.u, hopfpointguess.p)
        prob_hopf = HopfMAProblem(𝐇, jacobian_ma, hopfpointguess, lens2, plot_solution(prob_vf), record_hopf)
        opt_hopf_cont = deepcopy(options_cont)
    else
        prob_hopf = HopfMAProblem(𝐇, nothing, hopfpointguess, lens2, plot_solution(prob_vf), record_hopf)
        opt_hopf_cont = @set options_cont.newton_options.linsolver = HopfLinearSolverMinAug()
    end

    # current lyapunov coefficient
    eTb = eltype(Tb)
    𝐇.l1 = Complex{eTb}(0, 0)
    𝐇.BT = one(eTb)
    𝐇.GH = one(eTb)

    # eigen solver
    eigsolver = HopfEig(getsolver(opt_hopf_cont.newton_options.eigsolver), prob_hopf)

    # define event for detecting codim 2 bifurcations
    # couple it with user passed events
    event_user = get(kwargs, :event, nothing)
    event_bif = ContinuousEvent(2, test_bt_gh, compute_eigen_elements, ("bt", "gh"), threshBT)

    if compute_eigen_elements #|| event_user == BifDetectEvent
        if isnothing(event_user)
            event = PairOfEvents(event_bif, BifDetectEvent)
        else
            event = SetOfEvents(event_bif, BifDetectEvent, event_user)
        end
        # careful here, we need to adjust the tolerance for stability to avoid
        # spurious ZH or HH bifurcations
        @reset opt_hopf_cont.tol_stability = max(10opt_hopf_cont.newton_options.tol, opt_hopf_cont.tol_stability)
    else
        if isnothing(event_user)
            event = event_bif
        else
            event = PairOfEvents(event_bif, event_user)
        end
    end

    # solve the hopf equations
    br = continuation(
                prob_hopf, alg,
                (@set opt_hopf_cont.newton_options.eigsolver = eigsolver);
                kwargs...,
                kind,
                linear_algo = BorderingBLS(solver = opt_hopf_cont.newton_options.linsolver, check_precision = false),
                normC,
                finalise_solution = get(kwargs, :finalise_solution, finalise_default),
                event
            )
    @assert ~isnothing(br) "Empty branch!"
    return _correct_event_labels(br)
end

function continuation_hopf(prob,
                        br::AbstractBranchResult, ind_hopf::Int64,
                        lens2::AllOpticTypes,
                        options_cont::ContinuationPar = br.contparams;
                        alg = getalg(br),
                        normC = norm,
                        nev = br.contparams.nev,
                        start_with_eigen = false,
                        bdlinsolver::AbstractBorderedLinearSolver = MatrixBLS(),
                        bdlinsolver_adjoint = bdlinsolver,
                        a = nothing,
                        b = nothing,
                        kwargs...)
    hopfpointguess = hopf_point(br, ind_hopf)
    ω = hopfpointguess.p[2]
    bifpt = br.specialpoint[ind_hopf]

    p = bifpt.param
    parbif = setparam(br, p)
    # we put the problem back to the state it was
    restore_problem!(prob, bifpt.x, parbif)

    if start_with_eigen
        if ~haseigenvector(br)
            error("The branch contains no eigenvectors for the Hopf point.\nPlease provide one.")
        end
        ζ = geteigenvector(br.contparams.newton_options.eigsolver, br.eig[bifpt.idx].eigenvecs, bifpt.ind_ev)
        VI.scale!(ζ, 1 / normC(ζ))

        # computation of adjoint eigenvalue
        λ = br.eig[bifpt.idx].eigenvals[bifpt.ind_ev]
        L = jacobian(prob, bifpt.x, parbif)
        Mass = getmassmatrix(prob, bifpt.x, parbif)
        L★ = ~has_adjoint(prob) ? adjoint(L) : jacobian_adjoint(prob, bifpt.x, parbif)

        ζ★, = _get_target_eigenvector_from_eigensolver(L★, conj(λ), br.contparams.newton_options.eigsolver; nev, verbose = options_cont.newton_options.verbose)
        ζad = VI.scale(ζ★, 1 / dot_with_mass(ζ★, Mass, ζ))
    else
        (; ζ, ζad) = _init_hopf_vectors_minaug(prob, bifpt.x, parbif, ω, bdlinsolver, bdlinsolver_adjoint, a, b, normC)
    end

    return continuation_hopf(getprob(br), alg,
                    hopfpointguess, parbif,
                    getlens(br), lens2,
                    ζ, ζad,
                    options_cont ;
                    normC,
                    bdlinsolver,
                    bdlinsolver_adjoint,
                    kwargs...)
end

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
Compute the initial (right / left) eigenvectors `(ζ, ζad)` of the Hopf point
using the minimally augmented formulation.
"""
function _init_hopf_vectors_minaug(prob, xₕ, parₕ, ω, bdlinsolver, bdlinsolver_adjoint, a, b, normC)
    # we use a minimally augmented formulation to set the initial vectors
    # we start with a vector similar to an eigenvector, we must ensure that
    # it is complex valued
    ζ = VI.scale(_copy(xₕ), one(Complex{VI.scalartype(xₕ)}))
    a = isnothing(a) ? _randn(ζ) : a; VI.scale!(a, 1 / normC(a))
    b = isnothing(b) ? _randn(ζ) : b; VI.scale!(b, 1 / normC(b))

    L = jacobian(prob, xₕ, parₕ)
    L★ = ~has_adjoint(prob) ? adjoint(L) : jacobian_adjoint(prob, xₕ, parₕ)
    M = getmassmatrix(prob, xₕ, parₕ)
    M★ = has_massmatrix_adjoint(prob) ? getmassmatrix_adjoint(prob, xₕ, parₕ) : adjoint(M)

    (; v, w, itv, itw) = __compute_bordered_vectors_hopf(bdlinsolver, bdlinsolver_adjoint, M, M★, L, L★, ω, a, b, VI.zerovector(a))

    @debug "RIGHT EIGENVECTORS" ω itv norminf(residual(prob, xₕ, parₕ)) norminf(apply(L,  v) - complex(0,ω)*apply(M, v)) norminf(apply(L, v) + complex(0,ω)*apply(M, v))
    @debug "LEFT  EIGENVECTORS" ω itw norminf(residual(prob, xₕ, parₕ)) norminf(apply(L★, w) - complex(0,ω)*apply(M★,w)) norminf(apply(L★,w) + complex(0,ω)*apply(M★,w))

    ζad = VI.scale(w,  1 / normC(w))
    ζ   = VI.scale(v,  1 / normC(v))
    return (; ζ, ζad)
end

function test_bt_gh(iter, state)
    𝐏𝐛 = getprob(iter)
    𝐇 = get_formulation(𝐏𝐛)
    𝒯 = eltype(𝐇)

    zu = getx(state)
    ω = get_frequency(zu, 𝐇)

    # expression of the jacobian
    x = getvec(zu, 𝐇) # fold point
    newpar = getparams(iter, state)
    L = jacobian(𝐇.prob_vf, x, newpar)
    L★ = has_adjoint(𝐇) ? jacobian_adjoint(𝐇.prob_vf, x, newpar) : transpose(L)
    Mass = getmassmatrix(𝐇.prob_vf, x, newpar)
    Mass★ = has_massmatrix_adjoint(𝐇.prob_vf) ? getmassmatrix_adjoint(𝐇.prob_vf, x, newpar) : adjoint(Mass)

    bd_vec = _compute_bordered_vectors(𝐇, Mass, Mass★, L, L★, ω)

    # compute new b
    ζ = bd_vec.v
    ζ ./= 𝐇.norm(ζ)

    # compute new a
    ζ★ = bd_vec.w

    # test function for Bogdanov-Takens
    𝐇.BT = ω
    ζ★ ./= dot_with_mass(ζ★, Mass, ζ)
    @debug "Hopf normal form computation"
    hp0 = Hopf(x, nothing, get_parameter(zu, 𝐇), ω, newpar, get_lenses(𝐏𝐛)[1], ζ, ζ★, (a = zero(Complex{𝒯}), b = zero(Complex{𝒯})), :hopf)
    hp = __hopf_normal_form(𝐇.prob_vf, hp0, 𝐇.linsolver; L, Mass)
    𝐇.l1 = hp.nf.b
    # test for Bautin bifurcation.
    # If GH is too large, we take the previous value to avoid spurious detection
    # GH will be large close to BR points
    𝐇.GH = abs(real(hp.nf.b)) < 1e5 ? real(hp.nf.b) : state.eventValue[2][2]
    return 𝐇.BT, 𝐇.GH
end

# structure to compute the eigenvalues along the Hopf branch
struct HopfEig{P, S} <: AbstractCodim2EigenSolver
    eigsolver::S
    prob::P
end

function (eig::HopfEig)(Jma, nev; k...)
    n = min(nev, length(getvec(Jma.x)))
    x = Jma.x.u     # hopf point
    p1, _ = Jma.x.p # first parameter
    newpar = set(Jma.params, getlens(Jma.pbma), p1)
    J = jacobian(Jma.pbma.prob_vf, x, newpar)
    eigenelts = eig.eigsolver(J, n; k...)
    return eigenelts
end

@views function (eig::HopfEig)(Jma::AbstractMatrix, nev; k...)
    return eig.eigsolver(Jma[begin:end-2, begin:end-2], nev; k...)
end

geteigenvector(eig::HopfEig, vectors, i::Int) = geteigenvector(eig.eigsolver, vectors, i)
