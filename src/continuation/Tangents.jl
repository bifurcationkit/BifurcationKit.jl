struct Constant <: AbstractTangentComputation end
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    Secant Tangent predictor
"""
struct Secant <: AbstractTangentComputation end
# important for bisection algorithm, switch on / off internal adaptive behavior
internal_adaptation!(::Secant, ::Bool) = nothing
_shortname(::PALC{Secant}) = "PALC [Secant]"

# This function is used for initialization in iterate_from_two_points
function _secant_tangent!(τ::M, 
                          z₁::M, 
                          z₀::M, 
                          ::AbstractContinuationIterable, 
                          ds, 
                          θ, 
                          verbosity, 
                          dotθ) where {T, vectype, M <: BorderedArray{vectype, T}}
    (verbosity > 0) && println("Predictor:  Secant")
    # secant predictor: τ = z₁ - z₀; tau *= sign(ds) / normtheta(tau)
    _copyto!(τ, z₁)
    minus!!(τ, z₀)
    α = sign(ds) / dotθ(τ, θ)
    VI.scale!(τ, α)
end

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
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    Bordered Tangent predictor
"""
struct Bordered <: AbstractTangentComputation end
# important for bisection algorithm, switch on / off internal adaptive behavior
internal_adaptation!(::Bordered, ::Bool) = nothing
_shortname(::PALC{Bordered}) = "PALC [Bordered]"

# tangent computation using Bordered system
# τ is the tangent prediction found by solving
# ┌                           ┐┌  ┐   ┌   ┐
# │      J            dFdl    ││τu│ = │ 0 │
# │  θ/N ⋅ τ.u     (1-θ)⋅τ.p  ││τp│   │ 1 │
# └                           ┘└  ┘   └   ┘
# it is updated inplace
function gettangent!(state::AbstractContinuationState,
                    iter::AbstractContinuationIterable,
                    ::Bordered, 
                    dotθ)
    (iter.verbosity > 0) && println("Predictor: Bordered")
    ϵ = getdelta(iter.prob)
    τ = state.τ
    θ = getθ(iter)
    T = eltype(iter)

    # dFdl = (F(z.u, z.p + ϵ) - F(z.u, z.p)) / ϵ
    dFdl = residual(iter.prob, state.z.u, setparam(iter, state.z.p + ϵ))
    dFdl = minus!!(dFdl, residual(iter.prob, state.z.u, setparam(iter, state.z.p)))
    dFdl = VI.scale!!(dFdl, 1/ϵ)

    # compute jacobian at the current solution
    J = jacobian(iter.prob, state.z.u, setparam(iter, state.z.p))

    # extract tangent as solution of the above bordered linear system
    τu, τp, flag, iterl = solve_bls_palc(get_bordered_linsolver(iter),
                                        iter, state,
                                        J, dFdl,
                                        VI.zerovector(state.z.u), 
                                        one(T)) # Right-hand side
    ~flag && @warn "Linear solver failed to converge in tangent computation with type ::Bordered"

    # we scale τ in order to have ||τ||_θ = 1 and sign <τ, τold> = 1
    α = one(T) / sqrt(dotθ(τu, τu, τp, τp, θ))
    α *= sign(dotθ(τ.u, τu, τ.p, τp, θ))

    _copyto!(τ.u, τu)
    τ.p = τp
    VI.scale!(τ, α)
end
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    Polynomial Tangent predictor

# Internal fields
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
_shortname(::PALC{Polynomial}) = "PALC [Polynomial]"

function Polynomial(pred, n, k, v0)
    @assert n<k "k must be larger than the degree of the polynomial"
    𝒯 = VI.scalartype(v0)
    Polynomial(n, k, zeros(𝒯, k, n+1), pred,
        DataStructures.CircularBuffer{typeof(v0)}(k),  # solutions
        DataStructures.CircularBuffer{𝒯}(k),  # parameters
        DataStructures.CircularBuffer{𝒯}(k),  # arclengths
        Vector{typeof(v0)}(undef, n+1), # coeffsSol
        Vector{𝒯}(undef, n+1), # coeffsPar
        true)
end
Polynomial(n, k, v0) = Polynomial(Secant(), n, k, v0)

isready(ppd::Polynomial) = length(ppd.solutions) >= ppd.k

function Base.empty!(ppd::Polynomial)
    empty!(ppd.solutions); empty!(ppd.parameters); empty!(ppd.arclengths);
    ppd
end

function _getstats(polypred::Polynomial)
    Sbar = sum(polypred.arclengths) / length(polypred.arclengths)
    σ = sqrt(sum(x->(x-Sbar)^2, polypred.arclengths ) / length(polypred.arclengths))
    return Sbar, σ
end

function (polypred::Polynomial)(ds::T) where T
    sbar, σ = _getstats(polypred)
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
    Sbar, σ = _getstats(polypred)
    # re-scale the previous arclengths so that the Vandermond matrix is well conditioned
    Ss = (polypred.arclengths .- Sbar) ./ σ
    # construction of the Vandermond Matrix
    polypred.A[:, 1] .= 1
    for jj in 1:polypred.n; polypred.A[:, jj+1] .= polypred.A[:, jj] .* Ss; end
    # invert linear system for least square fitting
    B = (polypred.A' * polypred.A) \ polypred.A'
    LA.mul!(polypred.coeffsSol, B, polypred.solutions)
    LA.mul!(polypred.coeffsPar, B, polypred.parameters)
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