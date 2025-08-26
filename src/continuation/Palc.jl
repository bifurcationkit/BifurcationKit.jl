"""
$(TYPEDEF)

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

DotTheta() = DotTheta( (x, y) -> dot(x, y) / length(x), x -> rmul!(x, 1/length(x))   )
DotTheta(dt) = DotTheta(dt, nothing)

# we restrict the type of the parameters because for complex problems, we still want the parameter to be real
(dt::DotTheta)(u1, u2, p1::T, p2, θ, θₚ = one(T) - θ) where {T <: Real} = real(dt.dot(u1, u2) * θ + p1 * p2 * θₚ)

# implementation of the norm associated to DotTheta
(dt::DotTheta)(u, p::T, θ::T) where T = sqrt(dt(u, u, p, p, θ))

(dt::DotTheta)(a::BorderedArray{vec, T}, b::BorderedArray{vec, T}, θ::T) where {vec, T} = dt(a.u, b.u, a.p, b.p, θ)
(dt::DotTheta)(a::BorderedArray{vec, T}, θ::T) where {vec, T} = dt(a.u, a.p, θ)
####################################################################################################
# equation of the arc length constraint
arc_length_eq(dt::DotTheta, u, p, du, dp, θ, ds) = dt(u, du, p, dp, θ) - ds

"""
Compute
    θ⋅dot(u1 - u2, τ0.u) / n + (1 - θ)⋅(p - z0.p)⋅τ0.p - ds
"""
function arc_length_eq(dt::DotTheta, u1, u2, p, du, dp, θ, ds)
    # θ⋅dot(x - z0.u, τ0.u) / n + (1 - θ)⋅(p - z0.p)⋅τ0.p - ds
    #  arc_length_eq(dotθ, minus(u, z0.u), _p - z0.p, τ0.u, τ0.p, θ, ds)
    out = arc_length_eq(dt, u1, p, du, dp, θ, ds) - 
          arc_length_eq(dt, u2, p, du, 0, θ, 0)

end
####################################################################################################
"""
$(TYPEDEF)

Pseudo-arclength continuation algorithm.

Additional information is available on the [website](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/PALC/).

# Fields

$(TYPEDFIELDS)

"""
@with_kw struct PALC{Ttang <: AbstractTangentComputation, Tbls <: AbstractLinearSolver, T, Tdot} <: AbstractContinuationAlgorithm
    "Tangent predictor, must be a subtype of `AbstractTangentComputation`. For example `Secant()` or `Bordered()`, "
    tangent::Ttang = Secant()
    "`θ` is a parameter in the arclength constraint. It is very **important** to tune it. It should be tuned for the continuation to work properly especially in the case of large problems where the < x - x_0, dx_0 > component in the constraint equation might be favoured too much. Also, large thetas favour p as the corresponding term in N involves the term 1-theta."
    θ::T = 0.5
    "[internal], "
    _bothside::Bool = false
    "Bordered linear solver used to invert the jacobian of the bordered problem during newton iterations. It is also used to compute the tangent for the predictor `Bordered()`, "
    bls::Tbls = MatrixBLS()
    "`dotθ = DotTheta()`, this sets up a dot product `(x, y) -> dot(x, y) / length(x)` used to define the weighted dot product (resp. norm) ``\\|(x, p)\\|^2_\\theta`` in the constraint ``N(x, p)`` (see online docs on [PALC](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/PALC/)). This argument can be used to remove the factor `1/length(x)` for example in problems where the dimension of the state space changes (mesh adaptation, ...) or when a specific (FEM) dot product is provided."
    dotθ::Tdot = DotTheta()

    @assert ~(predictor isa ConstantPredictor) "You cannot use a constant predictor with PALC"
    @assert 0 <= θ <= 1 "θ must belong to [0, 1]"
end
getlinsolver(alg::PALC) = alg.bls
getdot(alg::PALC) = alg.dotθ
getθ(alg::PALC) = alg.θ
# we also extend this for ContIterable
getdot(it::ContIterable) = getdot(it.alg)
getθ(it::ContIterable) = getθ(it.alg)

# important for bisection algorithm, switch on / off internal adaptive behavior
internal_adaptation!(alg::PALC, on_or_off::Bool) = internal_adaptation!(alg.tangent, on_or_off)

function Base.empty!(alg::PALC)
    empty!(alg.tangent)
    alg
end

function update(alg::PALC, contParams::ContinuationPar, linear_algo)
    if isnothing(linear_algo)
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
    copyto!(state.z, state.z_old)
    # then update the predictor state.z_pred
    addtangent!(state, nrm)
end

function getpredictor!(state::AbstractContinuationState,
                       iter::AbstractContinuationIterable,
                       alg::PALC,
                       nrm = false)
    # we compute the tangent
    # if the state has not converged, we dot not update the tangent
    # state.z has been updated only if converged(state) == true
    if converged(state)
        @debug "Update tangent"
        gettangent!(state, iter, alg.tangent, getdot(alg))
    end
    # then update the predictor state.z_pred
    addtangent!(state, nrm)
end

# this function only mutates z_pred
# the nrm argument allows to just the increment z_pred.p by ds
function addtangent!(state::AbstractContinuationState, nrm = false)
    # we perform z_pred = z + ds * τ
    # note that state.z contains the last converged state
    copyto!(state.z_pred, state.z)
    ds = state.ds
    ρ = nrm ? ds / state.τ.p : ds
    axpy!(ρ, state.τ, state.z_pred)
end

update_predictor!(state::AbstractContinuationState,
                  iter::AbstractContinuationIterable,
                  alg::PALC,
                  nrm = false) = addtangent!(state, nrm)

function corrector!(state::AbstractContinuationState,
                    it::AbstractContinuationIterable,
                    alg::PALC;
                    kwargs...)
    if state.z_pred.p <= it.contparams.p_min || state.z_pred.p >= it.contparams.p_max
        state.z_pred.p = clamp_predp(state.z_pred.p, it)
        return corrector!(state, it, Natural(); kwargs...)
    end
    sol = newton_palc(it, state, getdot(alg); 
                      linearbdalgo = alg.bls, 
                      normN = it.normC, 
                      callback = it.callback_newton, 
                      kwargs...)

    # update fields, in particular the `converged` one
    _update_field_but_not_sol!(state, sol)

    # update solution
    if converged(sol)
        copyto!(state.z, sol.u)
    end

    return true
end

###############################################
"""
    Secant Tangent predictor
"""
struct Secant <: AbstractTangentComputation end

# This function is used for initialization in iterate_from_two_points
function _secant_tangent!(τ::M, 
                          z₁::M, 
                          z₀::M, 
                          it::AbstractContinuationIterable, 
                          ds, 
                          θ, 
                          verbosity, 
                          dotθ) where {T, vectype, M <: BorderedArray{vectype, T}}
    (verbosity > 0) && println("Predictor:  Secant")
    # secant predictor: τ = z₁ - z₀; tau *= sign(ds) / normtheta(tau)
    copyto!(τ, z₁)
    minus!(τ, z₀)
    α = sign(ds) / dotθ(τ, θ)
    rmul!(τ, α)
end
# important for bisection algorithm, switch on / off internal adaptive behavior
internal_adaptation!(::Secant, ::Bool) = nothing

gettangent!(state::AbstractContinuationState,
            iter::AbstractContinuationIterable,
            algo::Secant,
            dotθ) = _secant_tangent!(state.τ, 
                                     state.z, 
                                     state.z_old, 
                                     iter, 
                                     state.ds, 
                                     getθ(iter), 
                                     iter.verbosity, 
                                     dotθ)
###############################################
"""
    Bordered Tangent predictor
"""
struct Bordered <: AbstractTangentComputation end
# important for bisection algorithm, switch on / off internal adaptive behavior
internal_adaptation!(::Bordered, ::Bool) = nothing

# tangent computation using Bordered system
# τ is the tangent prediction found by solving
# ┌                           ┐┌  ┐   ┌   ┐
# │      J            dFdl    ││τu│ = │ 0 │
# │  θ/N ⋅ τ.u     (1-θ)⋅τ.p  ││τp│   │ 1 │
# └                           ┘└  ┘   └   ┘
# it is updated inplace
function gettangent!(state::AbstractContinuationState,
                    it::AbstractContinuationIterable,
                    tgt_algo::Bordered, 
                    dotθ)
    (it.verbosity > 0) && println("Predictor: Bordered")
    ϵ = getdelta(it.prob)
    τ = state.τ
    θ = getθ(it)
    T = eltype(it)

    # dFdl = (F(z.u, z.p + ϵ) - F(z.u, z.p)) / ϵ
    dFdl = residual(it.prob, state.z.u, setparam(it, state.z.p + ϵ))
    minus!(dFdl, residual(it.prob, state.z.u, setparam(it, state.z.p)))
    rmul!(dFdl, 1/ϵ)

    # compute jacobian at the current solution
    J = jacobian(it.prob, state.z.u, setparam(it, state.z.p))

    # extract tangent as solution of the above bordered linear system
    τu, τp, flag, itl = solve_bls_palc(getlinsolver(it),
                                        it, state,
                                        J, dFdl,
                                        0*state.z.u, one(T)) # Right-hand side
    ~flag && @warn "Linear solver failed to converge in tangent computation with type ::Bordered"

    # we scale τ in order to have ||τ||_θ = 1 and sign <τ, τold> = 1
    α = one(T) / sqrt(dotθ(τu, τu, τp, τp, θ))
    α *= sign(dotθ(τ.u, τu, τ.p, τp, θ))

    copyto!(τ.u, τu)
    τ.p = τp
    rmul!(τ, α)
end
####################################################################################################
"""
    Polynomial Tangent predictor

$(TYPEDFIELDS)

# Constructor(s)

    Polynomial(pred, n, k, v0)

    Polynomial(n, k, v0)

- `n` order of the polynomial
- `k` length of the last solutions vector used for the polynomial fit
- `v0` example of solution to be stored. It is only used to get the `eltype` of the tangent.

Can be used like

    PALC(tangent = Polynomial(Bordered(), 2, 6, rand(1)))
"""
mutable struct Polynomial{T <: Real, Tvec, Ttg <: AbstractTangentComputation} <: AbstractTangentComputation
    "Order of the polynomial"
    n::Int64

    "Length of the last solutions vector used for the polynomial fit"
    k::Int64

    "Matrix for the interpolation"
    A::Matrix{T}

    "Algo for tangent when polynomial predictor is not possible"
    tangent::Ttg

    "Vector of solutions"
    solutions::DataStructures.CircularBuffer{Tvec}

    "Vector of parameters"
    parameters::DataStructures.CircularBuffer{T}

    "Vector of arclengths"
    arclengths::DataStructures.CircularBuffer{T}

    "Coefficients for the polynomials for the solution"
    coeffsSol::Vector{Tvec}

    "Coefficients for the polynomials for the parameter"
    coeffsPar::Vector{T}

    "Update the predictor by adding the last point (x, p)? This can be disabled in order to just use the polynomial prediction. It is useful when the predictor is called mutiple times during bifurcation detection using bisection."
    update::Bool
end
# important for bisection algorithm, switch on / off internal adaptive behavior
internal_adaptation!(alg::Polynomial, swch::Bool) = alg.update = swch

function Polynomial(pred, n, k, v0)
    @assert n<k "k must be larger than the degree of the polynomial"
    Polynomial(n, k, zeros(eltype(v0), k, n+1), pred,
        DataStructures.CircularBuffer{typeof(v0)}(k),  # solutions
        DataStructures.CircularBuffer{eltype(v0)}(k),  # parameters
        DataStructures.CircularBuffer{eltype(v0)}(k),  # arclengths
        Vector{typeof(v0)}(undef, n+1), # coeffsSol
        Vector{eltype(v0)}(undef, n+1), # coeffsPar
        true)
end
Polynomial(n, k, v0) = Polynomial(Secant(), n, k, v0)

isready(ppd::Polynomial) = length(ppd.solutions) >= ppd.k

function Base.empty!(ppd::Polynomial)
    empty!(ppd.solutions); empty!(ppd.parameters); empty!(ppd.arclengths);
    ppd
end

function getstats(polypred::Polynomial)
    Sbar = sum(polypred.arclengths) / length(polypred.arclengths)
    σ = sqrt(sum(x->(x-Sbar)^2, polypred.arclengths ) / length(polypred.arclengths))
    return Sbar, σ
end

function (polypred::Polynomial)(ds::T) where T
    sbar, σ = getstats(polypred)
    s = polypred.arclengths[end] + ds
    snorm = (s-sbar)/σ
    # vector of powers of snorm
    S = Vector{T}(undef, polypred.n+1); S[1] = T(1)
    for jj = 1:polypred.n; S[jj+1] = S[jj] * snorm; end
    p = sum(S .* polypred.coeffsPar)
    x = sum(S .* polypred.coeffsSol)
    return x, p
end

function update_pred!(polypred::Polynomial)
    Sbar, σ = getstats(polypred)
    # re-scale the previous arclengths so that the Vandermond matrix is well conditioned
    Ss = (polypred.arclengths .- Sbar) ./ σ
    # construction of the Vandermond Matrix
    polypred.A[:, 1] .= 1
    for jj in 1:polypred.n; polypred.A[:, jj+1] .= polypred.A[:, jj] .* Ss; end
    # invert linear system for least square fitting
    B = (polypred.A' * polypred.A) \ polypred.A'
    mul!(polypred.coeffsSol, B, polypred.solutions)
    mul!(polypred.coeffsPar, B, polypred.parameters)
    return true
end

function gettangent!(state::AbstractContinuationState,
                    it::AbstractContinuationIterable,
                    polypred::Polynomial, dotθ)
    (it.verbosity > 0) && println("Predictor: Polynomial")
    ds = state.ds
    # do we update the predictor with last converged point?
    if polypred.update
        if length(polypred.arclengths) == 0
            push!(polypred.arclengths, ds)
        else
            push!(polypred.arclengths, polypred.arclengths[end] + ds)
        end
        push!(polypred.solutions, state.z.u)
        push!(polypred.parameters, state.z.p)
    end

    if ~isready(polypred) || ~polypred.update
        return gettangent!(state, it, polypred.tangent, dotθ)
    else
        return polypred.update ? update_pred!(polypred) : true
    end
end


####################################################################################################
"""
This is the classical Newton-Krylov solver for `F(x, p) = 0` together
with the scalar condition `n(x, p) ≡ θ ⋅ <x - x0, τx> + (1-θ) ⋅ (p - p0) * τp - n0 = 0`. This makes a problem of dimension N + 1.

The initial guess for the newton method is located in `state.z_pred`
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
    linsolver = getlinsolver(iter)

    # record the damping parameter
    α0 = α

    N(u, _p) = arc_length_eq(dotθ, u, z0.u, _p - z0.p, τ0.u, τ0.p, θ, ds)
    normAC(resf, resn) = max(normN(resf), abs(resn))

    # initialise variables
    x = _copy(z_pred.u)
    p = z_pred.p
    x_pred = _copy(x)

    res_f = residual(prob, x, set(par, paramlens, p));  res_n = N(x, p)
    dp = zero(𝒯)
    up = zero(𝒯)

    # dFdp = (F(x, p + ϵ) - res_f) / ϵ
    dFdp = _copy(residual(prob, x, set(par, paramlens, p + ϵ)))
    minus!(dFdp, res_f) # dFdp = dFdp - res_f
    rmul!(dFdp, one(𝒯) / ϵ)

    res       = normAC(res_f, res_n)
    residuals = [res]
    step = 0
    itlineartot = 0

    verbose && print_nonlinear_step(step, res)
    line_step = true

    compute = callback((;x, res_f, residual = res, step, contparams, z0, p, residuals, options = (;linsolver)); fromNewton = false, kwargs...)

    while (step < max_iterations) && (res > tol) && line_step && compute
        # dFdp = (F(x, p + ϵ) - F(x, p)) / ϵ)
        copyto!(dFdp, residual(prob, x, set(par, paramlens, p + ϵ)))
        minus!(dFdp, res_f); rmul!(dFdp, one(𝒯) / ϵ)

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
                copyto!(x_pred, x); axpy!(-α, u, x_pred)

                p_pred = p - α * up
                copyto!(res_f, residual(prob, x_pred, set(par, paramlens, p_pred)))

                res_n  = N(x_pred, p_pred)
                res = normAC(res_f, res_n)

                if res < residuals[end]
                    if (res < residuals[end] / 4) && (α < 1)
                        α *= 2
                    end
                    line_step = true
                    copyto!(x, x_pred)

                    # p = p_pred
                    p  = clamp(p_pred, p_min, p_max)
                else
                    α /= 2
                end
            end
            # we put back the initial value
            α = α0
        else
            minus!(x, u)
            p = clamp(p - up, p_min, p_max)
            copyto!(res_f, residual(prob, x, set(par, paramlens, p)))
            res_n  = N(x, p); res = normAC(res_f, res_n)
        end

        push!(residuals, res)
        step += 1

        verbose && print_nonlinear_step(step, res, itlinear)

        # shall we break the loop?
        compute = callback((;x, res_f, J, residual=res, step, itlinear, contparams, z0, p, residuals, options = (;linsolver)); fromNewton = false, kwargs...)
    end
    verbose && print_nonlinear_step(step, res, 0, true) # display last line of the table
    flag = (residuals[end] < tol) & callback((;x, res_f, residual = res, step, contparams, p, residuals, options = (;linsolver)); fromNewton = false, kwargs...)

    return NonLinearSolution(BorderedArray(x, p),
                            prob,
                            residuals,
                            flag,
                            step,
                            itlineartot)
end
