@enum MoorePenroseLS direct=1 pInv=2 iterative=3
"""
    Moore-Penrose predictor / corrector

Moore-Penrose continuation algorithm.

Additional information is available on the [website](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/MooreSpence/).

# Constructors

`alg = MoorePenrose()`

`alg = MoorePenrose(tangent = PALC())`

# Fields

$(TYPEDFIELDS)
"""
struct MoorePenrose{T, Tls <: AbstractLinearSolver} <: AbstractContinuationAlgorithm
    "Tangent predictor, for example `PALC()`"
    tangent::T
    "Moore Penrose linear solver. Can be BifurcationKit.direct, BifurcationKit.pInv or BifurcationKit.iterative"
    method::MoorePenroseLS
    "(Bordered) linear solver"
    ls::Tls
end
# important for bisection algorithm, switch on / off internal adaptive behavior
internal_adaptation!(alg::MoorePenrose, swch::Bool) = internal_adaptation!(alg.tangent, swch)
@inline getdot(alg::MoorePenrose) = getdot(alg.tangent)
@inline getθ(alg::MoorePenrose) = getθ(alg.tangent)

"""
$(SIGNATURES)
"""
function MoorePenrose(;tangent = PALC(),
                        method = direct,
                        ls = nothing)
    if ~(method == iterative)
        ls = DefaultLS()
    else
        if isnothing(ls)
            if tangent isa PALC
                ls = tangent.bls
            else
                ls = MatrixBLS()
            end
        end
    end
    return MoorePenrose(tangent, method, ls)
end

getpredictor(alg::MoorePenrose) = getpredictor(alg.tangent)
getlinsolver(alg::MoorePenrose) = getlinsolver(alg.tangent)

function Base.empty!(alg::MoorePenrose)
    empty!(alg.tangent)
    alg
end

function update(alg0::MoorePenrose,
                contParams::ContinuationPar,
                linearAlgo)
    tgt = update(alg0.tangent, contParams, linearAlgo)
    alg = @set alg0.tangent = tgt

    # for direct methods, we need a direct solver
    if alg.method != iterative || alg.ls isa AbstractBorderedLinearSolver
        @reset alg.ls = DefaultLS()
    end

    if isnothing(linearAlgo) && alg.method != iterative
        if hasproperty(alg.ls, :solver) && isnothing(alg.ls.solver)
            return @set alg.ls.solver = contParams.newton_options.linsolver
        end
    else
        return @set alg.ls = isnothing(linearAlgo) ? MatrixBLS() : linearAlgo
    end
    alg
end

initialize!(state::AbstractContinuationState,
                        iter::AbstractContinuationIterable,
                        alg::MoorePenrose, nrm = false) = initialize!(state, iter, alg.tangent, nrm)

function getpredictor!(state::AbstractContinuationState,
                        iter::AbstractContinuationIterable,
                        alg::MoorePenrose, nrm = false)
    (iter.verbosity > 0) && println("Predictor:  MoorePenrose")
    # we just compute the tangent
    getpredictor!(state, iter, alg.tangent, nrm)
end

update_predictor!(state::AbstractContinuationState,
                        iter::AbstractContinuationIterable,
                        alg::MoorePenrose,
                        nrm = false) = update_predictor!(state, iter, alg.tangent, nrm)

# corrector based on natural formulation
function corrector!(state::AbstractContinuationState,
                it::AbstractContinuationIterable,
                algo::MoorePenrose;
                kwargs...)
    if state.z_pred.p <= it.contparams.p_min || state.z_pred.p >= it.contparams.p_max
        state.z_pred.p = clamp_predp(state.z_pred.p, it)
        return corrector!(state, it, Natural(); kwargs...)
    end
    sol = newton_moore_penrose(it, state, getdot(algo); normN = it.normC, callback = it.callback_newton, kwargs...)

    # update fields
    _update_field_but_not_sol!(state, sol)

    # update solution
    if converged(sol)
        copyto!(state.z, sol.u)
    end

    return true
end

function newton_moore_penrose(iter::AbstractContinuationIterable,
                              state::AbstractContinuationState, dotθ;
                              normN = norm,
                              callback = cb_default, kwargs...)
    prob = iter.prob
    par = getparams(prob)
    ϵ = getdelta(prob)
    paramlens = getlens(iter)
    contparams = getcontparams(iter)
    𝒯 = eltype(iter)

    @unpack method = iter.alg

    z0 = getsolution(state)
    τ0 = state.τ
    z_pred = state.z_pred
    ds = state.ds

    @unpack tol, max_iterations, verbose = contparams.newton_options
    @unpack p_min, p_max = contparams
    linsolver = iter.alg.ls

    # initialise variables
    x = _copy(z_pred.u)
    p = z_pred.p
    x_pred = _copy(x)
    res_f = residual(prob, x, set(par, paramlens, p))

    dX = _copy(res_f) # copy(res_f)
    # dFdp = (F(x, p + ϵ) - res_f) / ϵ
    dFdp = _copy(residual(prob, x, set(par, paramlens, p + ϵ)))
    minus!(dFdp, res_f); rmul!(dFdp, one(𝒯) / ϵ)

    res = normN(res_f)
    residuals = [res]

    # step count
    step = 0

    # total number of linear iterations
    itlinear = 0
    itlineartot = 0

    verbose && print_nonlinear_step(step, res)
    line_step = true

    compute = callback((;x, res_f, residual = res, step, contparams, p, residuals, z0); fromNewton = false, kwargs...)

    X = BorderedArray(x, p)
    if linsolver isa AbstractIterativeLinearSolver || (method == iterative)
        ϕ = _copy(τ0)
        rmul!(ϕ,  one(𝒯) / norm(ϕ))
    end

    while (step < max_iterations) && (res > tol) && line_step && compute
        step += 1
        # dFdp = (F(x, p + ϵ) - F(x, p)) / ϵ)
        copyto!(dFdp, residual(prob, x, set(par, paramlens, p + ϵ)))
        minus!(dFdp, res_f); rmul!(dFdp, one(𝒯) / ϵ)

        # compute jacobian
        J = jacobian(prob, x, set(par, paramlens, p))
        if method == direct || method == pInv
            Jb = hcat(J, dFdp)

            if method == direct
                dx, flag, itlinear = linsolver(Jb, res_f)
                ~flag && @debug "[MoorePenrose] Linear solver did not converge."
            else
                # dx = pinv(Array(Jb)) * res_f #seems to work better than the following
                dx = LinearAlgebra.pinv(Array(Jb)) * res_f
                flag = true;
                itlinear = 1
            end
            x .-= @view dx[begin:end-1]
            p -= dx[end]
        else
            @debug "Moore-Penrose Iterative"
            # A = hcat(J, dFdp); A = vcat(A, ϕ')
            # X .= X .- A \ vcat(res_f, 0)
            # x .= X[begin:end-1]; p = X[end]
            du, dup, flag, itlinear1 = linsolver(J, dFdp, ϕ.u, ϕ.p, res_f, zero(𝒯), one(𝒯), one(𝒯))
            ~flag && @debug "[MoorePenrose] Bordered linear solver did not converge."
            minus!(x, du)
            p -= dup
            verbose && print_nonlinear_step(step, nothing, itlinear1)
        end

        p = clamp(p, p_min, p_max)
        res_f .= residual(prob, x, set(par, paramlens, p))
        res = normN(res_f)

        if method == iterative
            # compute jacobian
            J = jacobian(prob, x, set(par, paramlens, p))
            copyto!(dFdp, residual(prob, x, set(par, paramlens, p + ϵ)))
            minus!(dFdp, res_f); rmul!(dFdp, one(𝒯) / ϵ)
            # A = hcat(J, dFdp); A = vcat(A, ϕ')
            # ϕ .= A \ vcat(zero(x),1)
            u, up, flag, itlinear2 = linsolver(J, dFdp, ϕ.u, ϕ.p, zero(x), one(𝒯), one(𝒯), one(𝒯))
            ~flag && @debug "[MoorePenrose] Linear solver did not converge."
            ϕ.u .= u; ϕ.p = up
            # rmul!(ϕ,  one(𝒯) / norm(ϕ))
            itlinear = (itlinear1 .+ itlinear2)
        end
        push!(residuals, res)

        verbose && print_nonlinear_step(step, res, itlinear)

        # break the while-loop?
        compute = callback((;x, res_f, J, residual=res, step, itlinear, contparams, p, residuals, z0); fromNewton = false, kwargs...)
    end
    verbose && print_nonlinear_step(step, res, 0, true) # display last line of the table
    flag = (residuals[end] < tol) & callback((;x, res_f, nothing, residual=res, step, contparams, p, residuals, z0); fromNewton = false, kwargs...)

    return NonLinearSolution(BorderedArray(x, p),
                            prob,
                            residuals,
                            flag,
                            step,
                            itlineartot)
end
