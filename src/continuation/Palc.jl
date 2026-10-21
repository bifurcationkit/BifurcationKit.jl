struct NormalisedDot{Tdot}
    dot::Tdot
end
(dt::NormalisedDot)(x, y) = dt.dot(x, y) / length(x)

# 1 / length(x) in the scalar type of x: `1 / length(x)` is a Float64 and promotes Float32 states
_inverse_length(x) = inv(convert(VI.scalartype(x), length(x)))

__scaling_function_dot_palc(x) = VI.scale!(x, _inverse_length(x))

"""
$(TYPEDEF)

# Internal fields
$(TYPEDFIELDS)

This parametric type allows to define a new dot product from the one saved in `dt::dot`. More precisely:

    dt(u1, u2, p1::T, p2::T, theta::T) where {T <: Real}

computes, the weighted dot product ``\\langle (u_1,p_1), (u_2,p_2)\\rangle_\\theta = \\theta \\Re \\langle u_1,u_2\\rangle  +(1-\\theta)p_1p_2`` where ``u_i\\in\\mathbb R^N``. The ``\\Re`` factor is put to ensure a real valued result despite possible complex valued arguments.

!!! info "Info"
    This is used in the pseudo-arclength constraint with the dot product ``\\frac{1}{N} \\langle u_1, u_2\\rangle,\\quad u_i\\in\\mathbb R^N``
"""
struct DotTheta{Tdot, Ta}
    "dot product used in pseudo-arclength constraint"
    dot::Tdot
    "Linear operator associated with dot product, i.e. dot(x, y) = <x, Ay>, where <,> is the standard dot product on R^N. You must provide an inplace function which evaluates A. For example `x -> rmul!(x, 1/length(x))`."
    apply!::Ta
end

DotTheta() = DotTheta( NormalisedDot(VI.inner), __scaling_function_dot_palc)
DotTheta(dt) = DotTheta(dt, nothing)
_get_apply_dot(dt::DotTheta) = dt.apply!

# we restrict the type of the parameters because for complex problems, we still want the parameter to be real
(dt::DotTheta)(u1, u2, p1::T, p2, θ, θₚ = one(T) - θ) where {T <: Real} = real(dt.dot(u1, u2) * θ + p1 * p2 * θₚ)

# implementation of the norm associated to DotTheta
(dt::DotTheta)(u, p::T, θ::T) where T = sqrt(dt(u, u, p, p, θ))

(dt::DotTheta)(a::BorderedArray{vec, T}, b::BorderedArray{vec, T}, θ::T) where {vec, T} = dt(a.u, b.u, a.p, b.p, θ)
(dt::DotTheta)(a::BorderedArray{vec, T}, θ::T) where {vec, T} = dt(a.u, a.p, θ)

"""
$(TYPEDSIGNATURES)

Cosine, in the norm of `dt`, of the angle between the tangent `τ` at `z` and the chord from `z` to `znew`. An empty chord has cosine 1: it does not turn.
"""
function chord_cosine(dt::DotTheta, τ::BorderedArray, z::BorderedArray, znew::BorderedArray, θ::Real)
    Δ = _copy(znew)
    Δ = VI.add!!(Δ, z, -one(θ))
    length_chord = dt(Δ.u, Δ.p, θ)
    if iszero(length_chord)
        return one(length_chord)
    end
    return dt(τ.u, Δ.u, τ.p, Δ.p, θ) / (dt(τ.u, τ.p, θ) * length_chord)
end

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# equation of the arc length constraint
arc_length_eq(dt::DotTheta, u, p, du, dp, θ, ds) = dt(u, du, p, dp, θ) - ds

"""
Compute
    θ⋅dot(u1 - u2, τ0.u) / n + (1 - θ)⋅(p - z0.p)⋅τ0.p - ds
"""
function arc_length_eq(dt::DotTheta, u1, u2, p, du, dp, θ, ds)
    # θ⋅dot(x - z0.u, τ0.u) / n + (1 - θ)⋅(p - z0.p)⋅τ0.p - ds
    #  arc_length_eq(dotθ, minus(u, z0.u), _p - z0.p, τ0.u, τ0.p, θ, ds)
    return arc_length_eq(dt, u1, p, du, dp, θ, ds) - 
           arc_length_eq(dt, u2, p, du, 0, θ, 0)

end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
$(TYPEDEF)

Pseudo-arclength continuation algorithm.

Additional information is available on the [website](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/PALC/).

# Internal fields

$(TYPEDFIELDS)

"""
@with_kw struct PALC{Ttang <: AbstractTangentComputation, Tbls <: AbstractLinearSolver, T, Tdot, Tctl} <: AbstractContinuationAlgorithm
    "Tangent (predictor), must be a subtype of `AbstractTangentComputation`. For example `Secant()` or `Bordered()`, etc."
    tangent::Ttang = Secant()
    "`θ` is a parameter in the arclength constraint. It is very **important** to tune it. It should be tuned for the continuation to work properly especially in the case of large problems where the < x - x_0, dx_0 > component in the constraint equation might be favoured too much. Also, large thetas favour p as the corresponding term in N involves the term 1-theta."
    θ::T = 0.5
    "[internal], not yet used."
    _bothside::Bool = false
    "Bordered linear solver used to invert the jacobian of the bordered problem during newton iterations. It is also used to compute the tangent for the predictor `Bordered()`."
    bls::Tbls = MatrixBLS()
    "`dotθ = DotTheta()`, this sets up a dot product `(x, y) -> dot(x, y) / length(x)` used to define the weighted dot product (resp. norm) ``\\|(x, p)\\|^2_\\theta`` in the constraint ``N(x, p)`` (see online docs on [PALC](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/PALC/)). This argument can be used to remove the factor `1/length(x)` for example in problems where the dimension of the state space changes (mesh adaptation, ...) or when a specific (FEM) dot product is provided."
    dotθ::Tdot = DotTheta()
    "Step control by the quality of the corrector, a [`CorrectorQuality`](@ref) or `nothing` (the default). With `nothing`, `ds` is controlled by the number of Newton iterations, see the parameter `a` of [`ContinuationPar`](@ref)."
    step_control::Tctl = nothing
    "Orientation test: reject a step whose new tangent has a negative dot product with the previous one (they point more than `π / 2` apart), as when the corrector converged onto a nearby part of the curve that runs the other way. A `Secant` tangent is the chord to the new point, so the test costs one dot product. A `Bordered` tangent is oriented as the curve is, by the sign of the determinant of the bordered matrix `[J dFdl; τ']` (Allgower and Georg, *Introduction to Numerical Continuation Methods*, definition (2.1.7)), with the sign fixed to the direction of travel at the start. That determinant needs a `MatrixBLS` solver and a dense LU factorisation of size `n + 1` per step, `O(n^3)`. Off by default."
    orientation_check::Bool = false

    @assert ~(tangent isa Constant) "You cannot use a constant predictor with PALC"
    @assert 0 <= θ <= 1 "θ must belong to [0, 1]"
    @assert ~orientation_check || tangent isa Union{Secant, Bordered} "orientation_check needs a Secant or Bordered tangent"
end
get_bordered_linsolver(alg::PALC) = alg.bls
getdot(alg::PALC) = alg.dotθ
getθ(alg::PALC) = alg.θ
# we also extend this for ContIterable
getdot(it::ContIterable) = getdot(it.alg)
getθ(it::ContIterable) = oftype(it.contparams.ds, getθ(it.alg))

# important for bisection algorithm, switch on / off internal adaptive behavior
internal_adaptation!(alg::PALC, on_or_off::Bool) = internal_adaptation!(alg.tangent, on_or_off)

function Base.empty!(alg::PALC)
    empty!(alg.tangent)
    alg
end

function update(alg::PALC, contParams::ContinuationPar, linear_algo::Tla) where {Tla}
    if Tla == Nothing
        if isnothing(alg.bls.solver)
            bls = alg.bls
            return @set alg.bls = update_bls(bls, contParams.newton_options.linsolver)
        end
    else
        return @set alg.bls = linear_algo
    end
    alg
end

function initialize!(state::AbstractContinuationState,
                     iter::AbstractContinuationIterable,
                     alg::PALC,
                     nrm = false)
    # for the initialization step, we do not use a Bordered predictor which 
    # fails at bifurcation points. Instead, we start with a Secant predictor
    gettangent!(state, iter, Secant(), getdot(alg))
    # we want to start at (u0, p0), not at (u1, p1)
    _copyto!(state.z, state.z_old)
    if alg.orientation_check
        init_orientation_check!(state, iter, alg)
    end
    # then update the predictor state.z_pred
    update_predictor!(state, iter, alg, nrm)
end

function getpredictor!(state::AbstractContinuationState,
                       iter::AbstractContinuationIterable,
                       alg::PALC,
                       nrm = false)
    _getpredictor_palc!(state, iter, alg, nrm)
end

# this function can also be called by Natural
function _getpredictor_palc!(state::AbstractContinuationState,
                       iter::AbstractContinuationIterable,
                       alg::PALC,
                       nrm = false)
    # we compute the tangent
    # if the state has not converged, we dot not update the tangent
    # state.z has been updated only if converged(state) == true
    if converged(state)
        @debug "Update tangent"
        if isnothing(state.next_tangent)
            gettangent!(state, iter, alg.tangent, getdot(alg))
        else
            _copyto!(state.τ, state.next_tangent)
            state.next_tangent = nothing
        end
    end
    # then update the predictor state.z_pred
    update_predictor!(state, iter, alg, nrm)
end

update_predictor!(state::AbstractContinuationState,
                  ::AbstractContinuationIterable,
                  ::PALC,
                  nrm = false) = addtangent!(state, nrm)

"""
$(TYPEDEF)

Data of [`PALC`](@ref) carried across continuation steps in `ContState.cache`, used by `step_control` and `orientation_check`.

$(TYPEDFIELDS)
"""
struct PALCCache{T, Tq, Tv}
    "factor by which step size control multiplies `ds` after the current step, or `nothing` to use the Newton iteration count."
    step_factor::Union{Nothing, T}
    "sign (±1) relating the determinant orientation of a `Bordered` tangent to the direction of travel, set at the start."
    orientation::Int
    "quality of the current corrector, `nothing` without `step_control`."
    quality::Tq
    "whether `step_control` rejected the current corrector."
    rejected::Bool
    "whether the Newton callback has seen the first iterate of the current corrector."
    started::Bool
    "last Newton iterate of the current corrector, `nothing` without `step_control`."
    previous::Tv
end

function _update(cache::PALCCache;
                 step_factor = cache.step_factor,
                 orientation::Int = cache.orientation,
                 quality = cache.quality,
                 rejected::Bool = cache.rejected,
                 started::Bool = cache.started)
    return typeof(cache)(step_factor, orientation, quality, rejected, started, cache.previous)
end

_continuation_cache(::AbstractContinuationAlgorithm, z, ::Type) = nothing

function _continuation_cache(alg::PALC, z::BorderedArray, ::Type{T}) where {T}
    if isnothing(alg.step_control)
        return PALCCache{T, Nothing, Nothing}(nothing, 1, nothing, false, false, nothing)
    end
    quality = StepQuality(CorrectorQuality{T}(alg.step_control))
    return PALCCache{T, typeof(quality), typeof(z)}(nothing, 1, quality, false, false, _copy(z))
end

_step_factor(::Nothing) = nothing
_step_factor(cache::PALCCache) = cache.step_factor

function step_size_control!(state::AbstractContinuationState,
                            iter::AbstractContinuationIterable,
                            alg::PALC)
    factor = _step_factor(state.cache)
    if isnothing(factor) || ~converged(state) || state.stopcontinuation || ~stepsizecontrol(state)
        return @invoke step_size_control!(state::AbstractContinuationState, iter::AbstractContinuationIterable, alg::AbstractContinuationAlgorithm)
    end
    state.ds = clamp_ds(state.ds * factor, getcontparams(iter))
    return
end

"""
$(TYPEDEF)

Newton callback of [`PALC`](@ref) with `step_control` or `orientation_check`, wrapping the Newton callback `callback`.
After each Newton step it applies the tests of `control` to the distance between consecutive iterates, in the norm of the arc length constraint.
After the last step it applies the orientation test to the converged point. A rejection returns `false`, otherwise the value of `callback`.
"""
struct PALCStepCheck{Tstate, Titer, Talg, Tc, T, Tcb}
    state::Tstate
    iter::Titer
    alg::Talg
    control::Tc
    tol::T
    callback::Tcb
end

_control(::Nothing, ::Type) = nothing
_control(control::CorrectorQuality, ::Type{T}) where {T} = CorrectorQuality{T}(control)
_quality(::Nothing) = nothing
_quality(control::CorrectorQuality) = StepQuality(control)

_record_iterate!(::Nothing, x, p) = nothing
function _record_iterate!(previous::BorderedArray, x, p)
    _copyto!(previous.u, x)
    previous.p = p
    return nothing
end

function (check::PALCStepCheck)(newton_state; kwargs...)
    state = check.state
    cache = state.cache
    p = get(newton_state, :p, state.z_pred.p)
    if ~cache.started
        # the call before the first Newton step
        _record_iterate!(cache.previous, newton_state.x, p)
        state.cache = _update(cache; started = true)
        return check.callback(newton_state; kwargs...)
    end
    if haskey(newton_state, :itlinear)
        # the call after a Newton step
        if _rejects_newton_step!(check, check.control, newton_state.x, p)
            return false
        end
        return check.callback(newton_state; kwargs...)
    end
    # the call after the last Newton step
    if cache.rejected
        return false
    end
    verdict = check.callback(newton_state; kwargs...)
    if verdict === false || (verdict === true && newton_state.residual >= check.tol)
        return verdict
    end
    if ~_turns_forward!(check, BorderedArray(newton_state.x, p))
        return false
    end
    return verdict
end

_rejects_newton_step!(::PALCStepCheck, ::Nothing, x, p) = false

function _rejects_newton_step!(check::PALCStepCheck, control::CorrectorQuality, x, p)
    state = check.state
    cache = state.cache
    previous = cache.previous
    𝒯 = eltype(check.iter)
    # previous.u becomes the Newton step x - previous.u, then the iterate x
    previous.u = VI.add!!(previous.u, x, one(𝒯), -one(𝒯))
    Δp = p - previous.p
    distance = convert(𝒯, sqrt(getdot(check.alg)(previous.u, previous.u, Δp, Δp, getθ(check.iter))))
    _record_iterate!(previous, x, p)
    if rejects_step(control, cache.quality, distance, check.tol)
        state.cache = _update(cache; rejected = true)
        if check.iter.contparams.newton_options.verbose
            printstyled("Step rejected by the corrector quality tests at Newton step $(cache.quality.steps + 1)\n", color = :red)
        end
        return true
    end
    state.cache = _update(cache; quality = observe(control, cache.quality, distance, check.tol))
    return false
end

function _turns_forward!(check::PALCStepCheck, znew::BorderedArray)
    if ~check.alg.orientation_check
        return true
    end
    state = check.state
    cosine, next_tangent = turn_cosine(check.alg.tangent, state, check.iter, znew, getdot(check.alg))
    if cosine < 0
        if check.iter.verbosity > 0
            printstyled("Step rejected: the new tangent points away from the previous one (dot product $cosine)\n", color = :red)
        end
        return false
    end
    state.next_tangent = next_tangent
    return true
end

function corrector!(state::AbstractContinuationState,
                    it::AbstractContinuationIterable,
                    alg::PALC;
                    kwargs...)
    state.next_tangent = nothing
    callback = get(kwargs, :callback, it.callback_newton)
    # the bisection that locates special points prescribes `ds` (`stepsizecontrol(state) == false`) and accepts every converged step
    if stepsizecontrol(state) && (~isnothing(alg.step_control) || alg.orientation_check)
        if ~(state.cache isa PALCCache)
            throw(ArgumentError("step_control and orientation_check of PALC need PALC as the continuation algorithm of the iterator"))
        end
        control = _control(alg.step_control, eltype(it))
        state.cache = _update(state.cache; step_factor = nothing, quality = _quality(control), rejected = false, started = false)
        _corrector!(state, it, alg, PALCStepCheck(state, it, alg, control, it.contparams.newton_options.tol, callback); kwargs...)
        if converged(state) && ~isnothing(control)
            state.cache = _update(state.cache; step_factor = growth_factor(control, state.cache.quality))
            @debug "step growth" _group = :step_control step_factor = state.cache.step_factor capped = ~isnothing(control.max_growth) && growth_capped(control, state.cache.quality)
        end
        return true
    end
    return _corrector!(state, it, alg, callback; kwargs...)
end

function _corrector!(state::AbstractContinuationState,
                     it::AbstractContinuationIterable,
                     alg::PALC,
                     callback;
                     kwargs...)
    if state.z_pred.p <= it.contparams.p_min || state.z_pred.p >= it.contparams.p_max
        state.z_pred.p = clamp_predp(state.z_pred.p, it)
        return corrector!(state, it, Natural(); kwargs..., callback)
    end
    sol = newton_palc(it, state, getdot(alg);
                      linearbdalgo = alg.bls,
                      normN = it.normC,
                      kwargs...,
                      callback)

    # update fields, in particular the `converged` one
    _update_field_but_not_solution!(state, sol)

    # update solution
    if converged(sol)
        _copyto!(state.z, sol.u)
    end
    return true
end

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
$(TYPEDSIGNATURES)

This is the classical Newton-Krylov solver for `F(x, p) = 0` together
with the scalar condition `n(x, p) ≡ θ ⋅ <x - x0, τx> + (1 - θ) ⋅ (p - p0) * τp - n0 = 0`. This makes a problem of dimension N + 1.

The initial guess for the newton method is located in `state.z_pred`. An iterate outside `[p_min, p_max]` stops the corrector, not converged, before `F` is evaluated there.
"""
function newton_palc(iter::AbstractContinuationIterable,
                    state::AbstractContinuationState,
                    dotθ = getdot(iter);
                    normN = norm,
                    callback = cb_default,
                    kwargs...)
    prob = iter.prob
    par = getparams(prob)
    ϵ = getdelta(prob)
    paramlens = getlens(iter)
    contparams = getcontparams(iter)
    𝒯 = eltype(iter)
    θ = getθ(iter)

    z0 = getsolution(state)
    τ0 = state.τ
    (;z_pred, ds) = state

    (;tol, max_iterations, verbose, α, αmin, linesearch) = contparams.newton_options
    (;p_min, p_max) = contparams
    linsolver = get_bordered_linsolver(iter)

    # record the damping parameter
    α0 = α

    N(u, _p) = arc_length_eq(dotθ, u, z0.u, _p - z0.p, τ0.u, τ0.p, θ, ds)
    normAC(resf, resn) = max(normN(resf), abs(resn))

    # initialise variables
    x = _copy(z_pred.u)
    p = z_pred.p
    x_pred = _copy(x)

    res_f = residual(prob, x, set(par, paramlens, p));  res_n = N(x, p)

    # dFdp = (F(x, p + ϵ) - res_f) / ϵ
    dFdp = _copy(residual(prob, x, set(par, paramlens, p + ϵ)))
    dFdp = minus!!(dFdp, res_f) # dFdp = dFdp - res_f
    dFdp = VI.scale!(dFdp, one(𝒯) / ϵ)

    res       = normAC(res_f, res_n)
    residuals = [res]
    step = 0
    itlineartot = 0
    rejected = false

    verbose && print_nonlinear_step(step, res)
    line_step = true

    compute = callback((;x, res_f, residual = res, step, contparams, z0, p, residuals, options = (;linsolver)); fromNewton = false, kwargs...)

    while (step < max_iterations) && (res > tol) && line_step && compute
        # dFdp = (F(x, p + ϵ) - F(x, p)) / ϵ)
        _copyto!(dFdp, residual(prob, x, set(par, paramlens, p + ϵ)))
        dFdp = minus!!(dFdp, res_f); dFdp = VI.scale!(dFdp, one(𝒯) / ϵ)

        # compute jacobian
        J = jacobian(prob, x, set(par, paramlens, p))
        
        # solve linear system
        # ┌            ┐┌  ┐   ┌     ┐
        # │ J     dFdp ││u │ = │res_f│
        # │ τ0.u  τ0.p ││up│   │res_n│
        # └            ┘└  ┘   └     ┘
        u, up, flag, itlinear = solve_bls_palc(linsolver, iter, state, J, dFdp, res_f, res_n)
        ~flag && @debug "[newton_palc] Linear solver for J did not converge."
        itlineartot += sum(itlinear)

        if linesearch
            line_step = false
            while !line_step && (α > αmin)
                # x_pred = x - α * u
                _copyto!(x_pred, x); x_pred = VI.add!!(x_pred, u, -α)

                p_pred = p - α * up
                # F may be undefined outside [p_min, p_max]: an iterate there stops the corrector, not converged
                if ~(p_min <= p_pred <= p_max)
                    rejected = true
                    break
                end
                _copyto!(res_f, residual(prob, x_pred, set(par, paramlens, p_pred)))

                res_n  = N(x_pred, p_pred)
                res = normAC(res_f, res_n)

                if res < residuals[end]
                    if (res < residuals[end] / 4) && (α < 1)
                        α *= 2
                    end
                    line_step = true
                    _copyto!(x, x_pred)

                    p = p_pred
                else
                    α /= 2
                end
            end
            if rejected
                break
            end
            # we put back the initial value
            α = α0
        else
            # F may be undefined outside [p_min, p_max]: an iterate there stops the corrector, not converged
            if ~(p_min <= p - up <= p_max)
                rejected = true
                break
            end
            x = minus!!(x, u)
            p = p - up
            _copyto!(res_f, residual(prob, x, set(par, paramlens, p)))
            res_n  = N(x, p); res = normAC(res_f, res_n)
        end

        push!(residuals, res)
        step += 1

        verbose && print_nonlinear_step(step, res, itlinear)

        # shall we break the loop?
        compute = callback((;x, res_f, J, residual=res, step, itlinear, contparams, z0, p, residuals, options = (;linsolver)); fromNewton = false, kwargs...)
    end
    verbose && print_nonlinear_step(step, res, 0, true) # display last line of the table
    flag = (residuals[end] < tol) & callback((;x, res_f, residual = res, step, contparams, p, residuals, options = (;linsolver)); fromNewton = false, kwargs...) & ~rejected

    return NonLinearSolution(BorderedArray(x, p),
                            prob,
                            residuals,
                            flag,
                            step,
                            itlineartot)
end
