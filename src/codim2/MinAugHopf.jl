"""
For an initial guess from the index of a Hopf bifurcation point located in ContResult.specialpoint, returns a point which can be refined using `newton_hopf`.
"""
function hopf_point(br::AbstractBranchResult, index::Int)
    if br.specialpoint[index].type != :hopf 
        error("The provided index does not refer to a Hopf point")
    end
    specialpoint = br.specialpoint[index] # Hopf point
    p = specialpoint.param                # parameter value at the Hopf point
    ω = imag(br.eig[specialpoint.idx].eigenvals[specialpoint.ind_ev]) # frequency at the Hopf point
    return BorderedArray(specialpoint.x, [p, ω] )
end
###################################################################################################
# this function encodes the functional
function (𝐇::HopfProblemMinimallyAugmented)(x, p::𝒯, ω::𝒯, params) where 𝒯
    # These are the equations of the minimally augmented (MA) formulation of the 
    # Hopf bifurcation point
    # input:
    # - x guess for the point at which the jacobian has a purely imaginary eigenvalue
    # - p guess for the parameter for which the jacobian has a purely imaginary eigenvalue
    # The jacobian of the MA problem is solved with a BLS method
    # ┌         ┐┌  ┐   ┌ ┐
    # │ J-iω  a ││v │ = │0│
    # │  b    0 ││σ1│   │1│
    # └         ┘└  ┘   └ ┘
    # In the notations of Govaerts 2000, a = w, b = v
    # Thus, b should be a null vector of J - iω
    #       a should be a null vector of J'+ iω
    a = 𝐇.a
    b = 𝐇.b
    # update parameter
    par = set(params, getlens(𝐇), p)
    # we solve (J - iω)⋅v + a σ1 = 0 with <b, v> = 1
    # note that the shift argument only affect J in this call:
    _, σ1, cv, = 𝐇.linbdsolver(jacobian(𝐇.prob_vf, x, par), a, b, zero(𝒯), 𝐇.zero, one(𝒯); shift = Complex{𝒯}(0, -ω))
    ~cv && @debug "[Hopf residual] Linear solver for (J-iω) did not converge."
    return residual(𝐇.prob_vf, x, par), real(σ1), imag(σ1)
end

# this function encodes the functional
function (𝐇::HopfProblemMinimallyAugmented)(x::BorderedArray, params)
    res = 𝐇(x.u, x.p[1], x.p[2], params)
    return BorderedArray(res[1], [res[2], res[3]])
end

@views function (𝐇::HopfProblemMinimallyAugmented)(x::AbstractVector, params)
    res = 𝐇(x[begin:end-2], x[end-1], x[end], params)
    return vcat(res[1], res[2], res[3])
end
###################################################################################################
function _compute_bordered_vectors(𝐇::HopfProblemMinimallyAugmented, J_at_xp, JAd_at_xp, ω)
    a = 𝐇.a
    b = 𝐇.b
    𝒯 = eltype(𝐇)

     # we solve (J-iω)v + a σ1 = 0 with <b, v> = 1
    v, _, cv, itv = 𝐇.linbdsolver(J_at_xp, a, b, zero(𝒯), 𝐇.zero, one(𝒯); shift = Complex{𝒯}(0, -ω))
    ~cv && @debug "Bordered linear solver for (J-iω) did not converge."

    # we solve (J+iω)'w + b σ1 = 0 with <a, w> = 1
    w, _, cv, itw = 𝐇.linbdsolverAdjoint(JAd_at_xp, b, a, zero(𝒯), 𝐇.zero, one(𝒯); shift = Complex{𝒯}(0, ω))
    ~cv && @debug "Bordered linear solver for (J+iω)' did not converge."

    return (; v, w, itv, itw)
end

function _get_bordered_terms(𝐇::HopfProblemMinimallyAugmented, x, p::𝒯, ω::𝒯, par) where 𝒯
    # update parameter
    lens = getlens(𝐇)
    par0 = set(par, lens, p)

    # This avoids doing 3 times the possibly costly building of J(x, p)
    J_at_xp = jacobian(𝐇.prob_vf, x, par0)
    # Avoid computing J_at_xp twice in case 𝐇.Jadjoint is not provided
    JAd_at_xp = has_adjoint(𝐇) ? jad(𝐇.prob_vf, x, par0) : transpose(J_at_xp)

    (; v, w, itv, itw) = @time "--> bd_vec" _compute_bordered_vectors(𝐇, J_at_xp, JAd_at_xp, ω)

    δ = getdelta(𝐇.prob_vf)
    ϵ1, ϵ2, ϵ3 = 𝒯(δ), 𝒯(δ), 𝒯(δ)
    ################### computation of σx σp ####################
    # TODO!! This is only finite differences
    dₚF   = (residual(𝐇.prob_vf, x, set(par, lens, p + ϵ1)) -
             residual(𝐇.prob_vf, x, set(par, lens, p - ϵ1))) / 𝒯(2ϵ1)
    dₚJv = (apply(jacobian(𝐇.prob_vf, x, set(par, lens, p + ϵ3)), v) -
             apply(jacobian(𝐇.prob_vf, x, set(par, lens, p - ϵ3)), v)) / 𝒯(2ϵ3)
    σₚ = -dot(w, dₚJv)

    # case of sigma_omega
    # σω = dot(w, Complex{T}(0, 1) * v)
    σω = Complex{𝒯}(0, 1) * dot(w, v)

    return (;J_at_xp, JAd_at_xp, dₚF, σₚ, δ, ϵ2, v, w, par0, itv, itw, σω)
end
###################################################################################################
# since this is matrix based, it requires X to ba an AbstractVector
function jacobian(pdpb::HopfMAProblem{Tprob, MinAugMatrixBased}, X::AbstractVector{𝒯}, par) where {Tprob, 𝒯}
    𝐇 = pdpb.prob
    x = @view X[begin:end-2]
    p = X[end-1]
    ω = X[end]

    (;J_at_xp, JAd_at_xp, dₚF, σₚ, ϵ2, v, w, par0, σω) = _get_bordered_terms(𝐇, x, p, ω, par)

    cw = conj(w)
    vr = real(v); vi = imag(v)
    u1r = apply_jacobian(𝐇.prob_vf, x + ϵ2 * vr, par0, cw, true)
    u1i = apply_jacobian(𝐇.prob_vf, x + ϵ2 * vi, par0, cw, true)
    u2 = apply(JAd_at_xp,  cw)
    σxv2r = @. -(u1r - u2) / ϵ2
    σxv2i = @. -(u1i - u2) / ϵ2
    σₓ = @. σxv2r + Complex{𝒯}(0, 1) * σxv2i

    Jhopf = hcat(_get_matrix(J_at_xp), dₚF, zero(dₚF))
    Jhopf = vcat(Jhopf, vcat(real(σₓ), real(σₚ), real(σω))')
    Jhopf = vcat(Jhopf, vcat(imag(σₓ), imag(σₚ), imag(σω))')
end
################################################################################
# Struct to invert the jacobian of the Hopf MA problem.
struct HopfLinearSolverMinAug <: AbstractLinearSolver; end

"""
This function solves the linear problem associated with a linearization of the minimally augmented formulation of the Hopf bifurcation point. The keyword `debugArray` is used to debug the routine by returning several key quantities.
"""
function _hopf_MA_linear_solver(x, p::𝒯, ω::𝒯, 𝐇::HopfProblemMinimallyAugmented, par,
                            duu, dup, duω) where 𝒯
    ################################################################################################
    # N = length(du) - 2
    # The Jacobian J of the vector field is expressed at (x, p)
    # the jacobian expression Jhopf of the hopf problem is
    #           ┌             ┐
    #  Jhopf =  │  J  dpF   0 │
    #           │ σx   σp  σω │
    #           └             ┘
    ########## Resolution of the bordered linear system ########
    # J * dX      + dpF * dp           = du => dX = x1 - dp * x2
    # The second equation
    #    <σx, dX> +  σp * dp + σω * dω = du[end-1:end]
    # thus becomes
    #   (σp - <σx, x2>) * dp + σω * dω = du[end-1:end] - <σx, x1>
    # This 2 x 2 system is then solved to get (dp, dω)
    ################### inversion of Jhopf ####################

    (;J_at_xp, JAd_at_xp, dₚF, σₚ, δ, ϵ2, v, w, par0, itv, itw, σω) = _get_bordered_terms(𝐇, x, p, ω, par)

    # we solve J⋅x1 = duu and J⋅x2 = dₚF
    x1, x2, cv, (it1, it2) = 𝐇.linsolver(J_at_xp, duu, dₚF)
    ~cv && @debug "Linear solver for J did not converge"

    # the case of ∂ₓσ is a bit more involved
    # we first need to compute the value of ∂ₓσ written σx
    σx = similar(x, Complex{𝒯})

    if 𝐇.usehessian == false || has_hessian(𝐇) == false
        cw = conj(w)
        vr = real(v); vi = imag(v)
        u1r = apply_jacobian(𝐇.prob_vf, x + ϵ2 * vr, par0, cw, true)
        u1i = apply_jacobian(𝐇.prob_vf, x + ϵ2 * vi, par0, cw, true)
        u2 = apply(JAd_at_xp,  cw)
        σxv2r = @. -(u1r - u2) / ϵ2
        σxv2i = @. -(u1i - u2) / ϵ2
        σx = @. σxv2r + Complex{𝒯}(0, 1) * σxv2i

        σxx1 = dot(σx, x1)
        σxx2 = dot(σx, x2)
    else
        d2Fv = d2F(𝐇.prob_vf, x, par0, v, x1)
        σxx1 = -conj(dot(w, d2Fv))
        d2Fv = d2F(𝐇.prob_vf, x, par0, v, x2)
        σxx2 = -conj(dot(w, d2Fv))
    end
    # We need to be careful here because the dot produces conjugates. 
    # Hence the + dot(σx, x2) and + imag(dot(σx, x1) and not the opposite
    LS = Matrix{𝒯}(undef, 2, 2);
    rhs = Vector{𝒯}(undef, 2);
    LS[1,1] = real(σₚ - σxx2); LS[1,2] = real(σω)
    LS[2,1] = imag(σₚ + σxx2); LS[2,2] = imag(σω)
    rhs[1] = dup - real(σxx1); rhs[2] =  duω + imag(σxx1)
    dp, dω = LS \ rhs
    return x1 .- dp .* x2, dp, dω, true, it1 + it2 + sum(itv) + sum(itw)
end

function (hopfl::HopfLinearSolverMinAug)(Jhopf, du::BorderedArray{vectype, 𝒯}; kwargs...)  where {vectype, 𝒯}
    # kwargs is used by AbstractLinearSolver
    out = _hopf_MA_linear_solver((Jhopf.x).u, #!! TODO !! This seems TU
                (Jhopf.x).p[1],
                (Jhopf.x).p[2],
                Jhopf.hopfpb,
                Jhopf.params,
                du.u, du.p[1], du.p[2])
    return BorderedArray{vectype, 𝒯}(out[1], [out[2], out[3]]), out[4], out[5]
end
###################################################################################################
# define a problem <: AbstractBifurcationProblem
@inline has_adjoint(hopfpb::HopfMAProblem) = has_adjoint(hopfpb.prob)
@inline is_symmetric(hopfpb::HopfMAProblem) = is_symmetric(hopfpb.prob)
@inline getdelta(hopfpb::HopfMAProblem) = getdelta(hopfpb.prob)
residual(hopfpb::HopfMAProblem, x, p) = hopfpb.prob(x, p)
residual!(hopfpb::HopfMAProblem, out, x, p) = (copyto!(out, hopfpb.prob(x, p)); out)
save_solution(::HopfMAProblem, x ,p) = x

# jacobian(hopfpb::HopfMAProblem, x, p) = hopfpb.jacobian(x, p)
jacobian(hopfpb::HopfMAProblem{Tprob, Nothing}, x, p) where {Tprob} = (x = x, params = p, hopfpb = hopfpb.prob)

jacobian(hopfpb::HopfMAProblem{Tprob, AutoDiff}, x, p) where {Tprob} = ForwardDiff.jacobian(z -> hopfpb.prob(z, p), x)

jacobian(hopfpb::HopfMAProblem{Tprob, FiniteDifferences}, x, p) where {Tprob} = finite_differences( z -> hopfpb.prob(z, p), x; δ = 1e-8)

jacobian(hopfpb::HopfMAProblem{Tprob, FiniteDifferencesMF}, x, p) where {Tprob} = dx -> (hopfpb.prob(x .+ 1e-8 .* dx, p) .- hopfpb.prob(x .- 1e-8 .* dx, p)) / (2e-8)
###################################################################################################
"""
$(SIGNATURES)

This function turns an initial guess for a Hopf point into a solution to the Hopf problem based on a Minimally Augmented formulation. The arguments are as follows
- `prob::AbstractBifurcationProblem` where `p` is a set of parameters.
- `hopfpointguess` initial guess (x_0, p_0) for the Hopf point. It should a `BorderedArray` as returned by the function `HopfPoint`.
- `par` parameters used for the vector field
- `eigenvec` guess for the  iω eigenvector
- `eigenvec_ad` guess for the -iω eigenvector
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
    The adjoint of the jacobian `J` is computed internally when `Jᵗ = nothing` by using `transpose(J)` which works fine when `J` is an `AbstractArray`. In this case, do not pass the jacobian adjoint like `Jᵗ = (x, p) -> transpose(d_xF(x, p))` otherwise the jacobian will be computed twice!

!!! tip "ODE problems"
    For ODE problems, it is more efficient to use the Matrix based Bordered Linear Solver passing the option `bdlinsolver = MatrixBLS()`
"""
function newton_hopf(prob,
            hopfpointguess::BorderedArray,
            par,
            eigenvec, eigenvec_ad,
            options::NewtonPar;
            normN = norm,
            bdlinsolver::AbstractBorderedLinearSolver = MatrixBLS(),
            usehessian = true,
            kwargs...)
    # we first need to update d2F and d3F for them to accept complex arguments

    hopfproblem = HopfProblemMinimallyAugmented(
        prob,
        _copy(eigenvec_ad), # this is pb.a ≈ null space of (J - iω I)^*
        _copy(eigenvec),    # this is pb.b ≈ null space of  J - iω I
        options.linsolver,
        # do not change linear solver if user provides it
        @set bdlinsolver.solver = (isnothing(bdlinsolver.solver) ? options.linsolver : bdlinsolver.solver);
        usehessian = usehessian)

    prob_h = HopfMAProblem(hopfproblem, nothing, hopfpointguess, par, nothing, prob.plotSolution, prob.recordFromSolution)

    # options for the Newton Solver
    opt_hopf = @set options.linsolver = HopfLinearSolverMinAug()

    # solve the hopf equations
    return solve(prob_h, Newton(), opt_hopf; normN, kwargs...)
end

function newton_hopf(br::AbstractBranchResult, ind_hopf::Int;
            prob = br.prob,
            normN = norm,
            options = br.contparams.newton_options,
            verbose = true,
            nev = br.contparams.nev,
            start_with_eigen = false,
            kwargs...)
    hopfpointguess = hopf_point(br, ind_hopf)
    ω = hopfpointguess.p[2]
    bifpt = br.specialpoint[ind_hopf]
    options.verbose && println("--> Newton Hopf, the eigenvalue considered here is ", br.eig[bifpt.idx].eigenvals[bifpt.ind_ev])
    @assert bifpt.idx == bifpt.step + 1 "Error, the bifurcation index does not refer to the correct step."
    @assert ~isempty(br.eig[bifpt.idx].eigenvecs) "You must save the eigenvectors for this to work."
    ζ = geteigenvector(options.eigsolver, br.eig[bifpt.idx].eigenvecs, bifpt.ind_ev)
    ζ ./= normN(ζ)
    ζad = LinearAlgebra.conj.(ζ)

    if start_with_eigen
        # computation of adjoint eigenvalue. Recall that b should be a null vector of J-iω
        λ = Complex(0, ω)
        p = bifpt.param
        parbif = setparam(br, p)

        # jacobian at bifurcation point
        L = jacobian(prob, bifpt.x, parbif)

        # computation of adjoint eigenvector
        _Jt = ~has_adjoint(prob) ? adjoint(L) : jad(prob, bifpt.x, parbif)

        ζstar, λstar = get_adjoint_basis(_Jt, conj(λ), options.eigsolver; nev = nev, verbose = false)
        ζad .= ζstar ./ dot(ζstar, ζ)
    end

    # solve the hopf equations
    return newton_hopf(prob, hopfpointguess, getparams(br), ζ, ζad, options; normN, kwargs...)
end

"""
$(SIGNATURES)

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
- `jacobian_ma::Symbol = :autodiff`, how the linear system of the Fold problem is solved. Can be `:autodiff, :finiteDifferencesMF, :finiteDifferences, :minaug`
- `linsolve_adjoint` solver for (J+iω)^* ⋅sol = rhs
- `bdlinsolver` bordered linear solver for the constraint equation with top-left block (J-iω). Required in the linear solver for the Minimally Augmented Hopf functional. This option can be used to pass a dedicated linear solver for example with specific preconditioner.
- `bdlinsolver_adjoint` bordered linear solver for the constraint equation with top-left block (J-iω)^*. Required in the linear solver for the Minimally Augmented Hopf functional. This option can be used to pass a dedicated linear solver for example with specific preconditioner.
- `update_minaug_every_step` update vectors `a,b` in Minimally Formulation every `update_minaug_every_step` steps
- `compute_eigen_elements = false` whether to compute eigenelements. If `options_cont.detect_event>0`, it allows the detection of ZH, HH points.
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
                massmatrix = LinearAlgebra.I,
                record_from_solution = nothing,
                kwargs...) where {Tb, vectype}
    @assert lens1 != lens2 "Please choose 2 different parameters. You only passed $lens1"
    @assert lens1 == getlens(prob_vf)

    # options for the Newton Solver inherited from the ones the user provided
    options_newton = options_cont.newton_options
    # tolerance for detecting BT bifurcation and stopping continuation
    threshBT = 100options_newton.tol

    𝐇 = HopfProblemMinimallyAugmented(
        prob_vf,
        _copy(eigenvec_ad), # this is a ≈ null space of (J - iω I)^*
        _copy(eigenvec),    # this is b ≈ null space of  J - iω I
        options_newton.linsolver,
        # do not change linear solver if user provides it
        @set bdlinsolver.solver = (isnothing(bdlinsolver.solver) ? options_newton.linsolver : bdlinsolver.solver);
        linsolve_adjoint = linsolve_adjoint,
        linbdsolve_adjoint = bdlinsolver_adjoint,
        usehessian,
        massmatrix,
        _norm = normC,
        update_minaug_every_step
        )

    # Jacobian for the Hopf problem
    if jacobian_ma == AutoDiff()
        hopfpointguess = vcat(hopfpointguess.u, hopfpointguess.p)
        prob_hopf = HopfMAProblem(𝐇, AutoDiff(), hopfpointguess, par, lens2, prob_vf.plotSolution, prob_vf.recordFromSolution)
        opt_hopf_cont = @set options_cont.newton_options.linsolver = DefaultLS()
    elseif jacobian_ma == FiniteDifferencesMF()
        hopfpointguess = vcat(hopfpointguess.u, hopfpointguess.p)
        prob_hopf = HopfMAProblem(𝐇, FiniteDifferencesMF(), hopfpointguess, par, lens2, prob_vf.plotSolution, prob_vf.recordFromSolution)
        opt_hopf_cont = @set options_cont.newton_options.linsolver = options_cont.newton_options.linsolver
    elseif jacobian_ma == FiniteDifferences()
        hopfpointguess = vcat(hopfpointguess.u, hopfpointguess.p)
        prob_hopf = HopfMAProblem(𝐇, FiniteDifferences(), hopfpointguess, par, lens2, prob_vf.plotSolution, prob_vf.recordFromSolution)
        opt_hopf_cont = @set options_cont.newton_options.linsolver = options_cont.newton_options.linsolver
    elseif jacobian_ma == MinAugMatrixBased()
        hopfpointguess = vcat(hopfpointguess.u, hopfpointguess.p)
        prob_hopf = HopfMAProblem(𝐇, MinAugMatrixBased(), hopfpointguess, par, lens2, prob_vf.plotSolution, prob_vf.recordFromSolution)
        opt_hopf_cont = @set options_cont.newton_options.linsolver = options_cont.newton_options.linsolver
    else
        prob_hopf = HopfMAProblem(𝐇, nothing, hopfpointguess, par, lens2, prob_vf.plotSolution, prob_vf.recordFromSolution)
        opt_hopf_cont = @set options_cont.newton_options.linsolver = HopfLinearSolverMinAug()
    end

    # this functions allows to tackle the case where the two parameters have the same name
    lenses = get_lens_symbol(lens1, lens2)

    # current lyapunov coefficient
    eTb = eltype(Tb)
    𝐇.l1 = Complex{eTb}(0, 0)
    𝐇.BT = one(eTb)
    𝐇.GH = one(eTb)

    # this function is used as a Finalizer
    # it is called to update the Minimally Augmented problem
    # by updating the vectors a, b
    function update_minaug_hopf(z, tau, step, contResult; kUP...)
        # user-passed finalizer
        finaliseUser = get(kwargs, :finalise_solution, nothing)

        # we first check that the continuation step was successful
        # if not, we do not update the problem with bad information!
        # if we are in a bisection, we still update the MA problem, this does not work well otherwise
        success = get(kUP, :state, nothing).converged
        if (~mod_counter(step, update_minaug_every_step) || success == false)
            # we call the user finalizer
            return isnothing(finaliseUser) ? true : finaliseUser(z, tau, step, contResult; prob = 𝐇, kUP...)
        end

        @debug "[Hopf] Update vectors a and b"
        x = getvec(z.u, 𝐇)   # hopf point
        p1, ω = getp(z.u, 𝐇) # first parameter
        p2 = z.p              # second parameter
        newpar = set(par, lens1, p1)
        newpar = set(newpar, lens2, p2)

        # expression of the jacobian
        J_at_xp = jacobian(𝐇.prob_vf, x, newpar)
        JAd_at_xp = has_adjoint(𝐇) ? jad(𝐇.prob_vf, x, newpar) : adjoint(J_at_xp)

        bd_vec = _compute_bordered_vectors(𝐇, J_at_xp, JAd_at_xp, ω)

        𝐇.a .= bd_vec.w ./ normC(bd_vec.w)
        # do not normalize with dot(newb, 𝐇.a), it prevents from BT detection
        𝐇.b .= bd_vec.v ./ normC(bd_vec.v)

        # we stop continuation at Bogdanov-Takens points
        # CA NE DEVRAIT PAS ETRE ISSNOT?
        isbt = isnothing(contResult) ? true : isnothing(findfirst(x -> x.type in (:bt, :ghbt, :btgh), contResult.specialpoint))

        # if the frequency is null, this is not a Hopf point, we halt the process
        if abs(ω) < threshBT
            @warn "[Codim 2 Hopf - Finalizer] The Hopf curve seems to be close to a BT point: ω ≈ $ω. Stopping computations at ($p1, $p2). If the BT point is not detected, try lowering Newton tolerance or dsmax."
        end

        # call the user-passed finalizer
        final_result = isnothing(finaliseUser) ? true : finaliseUser(z, tau, step, contResult; prob = 𝐇, kUP...)

        return abs(ω) >= threshBT && isbt && final_result
    end

    # the following allows to append information specific to the codim 2 continuation to the user data
    _printsol = record_from_solution
    _printsol2 = isnothing(_printsol) ?
        (u, p; kw...)  -> begin
                 (; zip(lenses, (getp(u, 𝐇)[1], p))..., 
                            ωₕ = getp(u, 𝐇)[2],
                            l1 = 𝐇.l1,
                            BT = 𝐇.BT,
                            GH = 𝐇.GH,
                            _namedrecordfromsol(BifurcationKit.record_from_solution(prob_vf)(getvec(u, 𝐇), p; kw...))...
                            )
            end :
        (u, p; kw...) -> begin
            (; zip(lenses, (getp(u, 𝐇)[1], p))..., 
                        ωₕ = getp(u, 𝐇)[2],
                        l1 = 𝐇.l1,
                        BT = 𝐇.BT,
                        GH = 𝐇.GH,
                        _namedrecordfromsol(_printsol(getvec(u, 𝐇), p; kw...))...
                        )
        end

    prob_hopf = re_make(prob_hopf, record_from_solution = _printsol2)

    # eigen solver
    eigsolver = HopfEig(getsolver(opt_hopf_cont.newton_options.eigsolver), prob_hopf)

    # Define event for detecting codim 2 bifurcations.
    # Couple it with user passed events
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

    prob_hopf = re_make(prob_hopf, record_from_solution = _printsol2)

    # solve the hopf equations
    br = continuation(
                prob_hopf, alg,
                (@set opt_hopf_cont.newton_options.eigsolver = eigsolver);
                kwargs...,
                kind = kind,
                linear_algo = BorderingBLS(solver = opt_hopf_cont.newton_options.linsolver, check_precision = false),
                normC = normC,
                finalise_solution = update_minaug_every_step == 0 ? get(kwargs, :finalise_solution, finalise_default) : update_minaug_hopf,
                event = event
            )
    @assert ~isnothing(br) "Empty branch!"
    return correct_bifurcation(br)
end

function continuation_hopf(prob,
                        br::AbstractBranchResult, ind_hopf::Int64,
                        lens2::AllOpticTypes,
                        options_cont::ContinuationPar = br.contparams;
                        alg = br.alg,
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

    if isnothing(br.eig) 
        error("The branch contains no eigen elements. This is strange because a Hopf point was detected. Please open an issue on the website.")
    end

    p = bifpt.param
    parbif = setparam(br, p)

    if start_with_eigen
        if ~haseigenvector(br)
            error("The branch contains no eigenvectors for the Hopf point.\nPlease provide one.")
        end
        ζ = geteigenvector(br.contparams.newton_options.eigsolver, br.eig[bifpt.idx].eigenvecs, bifpt.ind_ev)
        rmul!(ζ, 1 / normC(ζ))
        ζad = conj.(ζ)

        # computation of adjoint eigenvalue
        λ = br.eig[bifpt.idx].eigenvals[bifpt.ind_ev]
        # jacobian at bifurcation point
        L = jacobian(prob, bifpt.x, parbif)

        # jacobian adjoint at bifurcation point
        L★ = ~has_adjoint(prob) ? adjoint(L) : jad(prob, bifpt.x, parbif)

        ζ★, λ★ = get_adjoint_basis(L★, conj(λ), br.contparams.newton_options.eigsolver; nev = nev, verbose = options_cont.newton_options.verbose)
        axpby!(1 / dot(ζ★, ζ), ζ★, 0, ζad)
    else
        # we use a minimally augmented formulation to set the initial vectors
        # we start with a vector similar to an eigenvector
        ζ = _copy(getu0(br.prob))
        a = isnothing(a) ? _randn(ζ) : a
        b = isnothing(b) ? _randn(ζ) : b

        𝒯 = typeof(ω)
        L = jacobian(prob, bifpt.x, parbif)
        newb, _, cv, it = bdlinsolver(L, a, b, zero(𝒯), zero(a), one(𝒯); shift = Complex{𝒯}(0, -ω))
        ~cv && @debug "Bordered linear solver for (J-iω) did not converge."

        @debug "EIGENVECTORS" ω cv it norminf(residual(prob, bifpt.x, parbif)) norminf(apply(L,newb) - complex(0,ω)*newb) norminf(apply(L,newb) + complex(0,ω)*newb)

        L★ = ~has_adjoint(prob) ? adjoint(L) : jad(prob, bifpt.x, parbif)
        newa, _, cv, it = bdlinsolver_adjoint(L★, b, a, zero(𝒯), zero(a), one(𝒯); shift = Complex{𝒯}(0, ω))
        ~cv && @debug "Bordered linear solver for (J+iω)' did not converge."

        @debug "EIGENVECTORS" ω cv it norminf(residual(prob, bifpt.x, parbif)) norminf(apply(L★,newa) - complex(0,ω)*newa) norminf(apply(L★,newa) + complex(0,ω)*newa)

        ζad = newa ./ normC(newa)
        ζ   = newb ./ normC(newb)
    end

    return continuation_hopf(br.prob, alg,
                    hopfpointguess, parbif,
                    getlens(br), lens2,
                    ζ, ζad,
                    options_cont ;
                    normC,
                    bdlinsolver,
                    bdlinsolver_adjoint,
                    kwargs...)
end

function test_bt_gh(iter, state)
    probma = getprob(iter)
    𝐇 = probma.prob
    𝒯 = eltype(𝐇) 
    lens1, lens2 = get_lenses(probma)

    z = getx(state)
    x = getvec(z, 𝐇)   # hopf point
    p1, ω = getp(z, 𝐇) # first parameter
    p2 = getp(state)   # second parameter
    par = getparams(probma)
    newpar = set(par, lens1, p1)
    newpar = set(newpar, lens2, p2)

    probhopf = iter.prob.prob

    a = probhopf.a
    b = probhopf.b

    # expression of the jacobian
    J_at_xp = jacobian(probhopf.prob_vf, x, newpar)
    JAd_at_xp = has_adjoint(probhopf) ? jad(probhopf.prob_vf, x, newpar) : transpose(J_at_xp)

    bd_vec = _compute_bordered_vectors(𝐇, J_at_xp, JAd_at_xp, ω)

    # compute new b
    ζ = bd_vec.v
    ζ ./= 𝐇.norm(ζ)

    # compute new a
    ζ★ = bd_vec.w

    # test function for Bogdanov-Takens
    probhopf.BT = ω
    BT2 = real( dot(ζ★ ./ 𝐇.norm(ζ★), ζ) )
    ζ★ ./= dot(ζ, ζ★)
    @debug "Hopf normal form computation"
    hp0 = Hopf(x, nothing, p1, ω, newpar, lens1, ζ, ζ★, (a = zero(Complex{𝒯}), b = zero(Complex{𝒯})), :hopf)
    hp = hopf_normal_form(𝐇.prob_vf, hp0, 𝐇.linsolver; verbose = false, autodiff = false) # TODO! WE NEED A KWARGS here
    # lyapunov coefficient
    probhopf.l1 = hp.nf.b
    # test for Bautin bifurcation.
    # If GH is too large, we take the previous value to avoid spurious detection
    # GH will be large close to BR points
    probhopf.GH = abs(real(hp.nf.b)) < 1e5 ? real(hp.nf.b) : state.eventValue[2][2]
    return probhopf.BT, probhopf.GH
end

# structure to compute the eigenvalues along the Hopf branch
struct HopfEig{P, S} <: AbstractCodim2EigenSolver
    eigsolver::S
    prob::P
end

function (eig::HopfEig)(Jma, nev; k...)
    n = min(nev, length(Jma.x.u))
    x = Jma.x.u     # hopf point
    p1, ω = Jma.x.p # first parameter
    newpar = set(Jma.params, getlens(Jma.hopfpb), p1)
    J = jacobian(Jma.hopfpb.prob_vf, x, newpar)
    eigenelts = eig.eigsolver(J, n; k...)
    return eigenelts
end

@views function (eig::HopfEig)(Jma::AbstractMatrix, nev; k...)
    eigenelts = eig.eigsolver(Jma[1:end-2, 1:end-2], nev; k...)
end

geteigenvector(eig::HopfEig, vectors, i::Int) = geteigenvector(eig.eigsolver, vectors, i)
