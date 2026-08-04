using SciMLBase: remake, ODEProblem, EnsembleProblem, EnsembleThreads, DAEProblem, isinplace as isinplace_sciml
import SciMLBase

@with_kw_noshow struct FlowDE{Tprob, Talg, Tjac, TprobMono, TalgMono, Tkwde, Tcb, Tvjp, TR01, TR11, TR20, TR30, Tδ} <: AbstractFlow
    "Store the ODEProblem associated to the flow of the Cauchy problem"
    odeprob::Tprob

    "ODE time stepper passed to DifferentialEquations.solve"
    alg::Talg

    "Store the ODEProblem associated to the flow of the variational problem"
    odeprob_mono::TprobMono = nothing

    "ODE time stepper passed to DifferentialEquations.solve"
    alg_mono::TalgMono = nothing

    "Keyword arguments passed to DifferentialEquations.solve"
    kwargsDE::Tkwde

    "Store possible callback"
    callback::Tcb

    "How the monodromy (matrix) is computed"
    jacobian::Tjac = nothing

    "adjoint of the monodromy (matrix-free)."
    vjp::Tvjp = nothing

    "[Optional] Derivatives of the flow with respect to the parameter `lens`.
    `R01(x, pars, t, lens, p)` returns `∂ₚφ(x, p, t)`, the derivative of the flow map with respect to the parameter `lens` evaluated at `p`, as a vector of the size of `x`.
    It is used by the Poincaré return map and the normal forms."
    R01::TR01 = nothing

    "[Optional] Derivatives of the flow with respect to the parameter `lens`.
    `R11(x, pars, dx, t, lens, p)` returns `∂ₚ[dφ(x, p, t)⋅dx]`, the mixed derivative of the JVP with respect to the parameter, as a vector of the size of `x`.
    It is used by the Poincaré return map and the normal forms."
    R11::TR11 = nothing

    "[Optional] Higher-order differentials of the flow with respect to `x`.
    `R20(x, pars, h1, h2, t)` returns `d²φ(x, p, t)(h1, h2)`, the second differential of the flow map applied to `h1`, `h2`. Used by the normal forms."
    R20::TR20 = nothing

    "[Optional] Higher-order differentials of the flow with respect to `x`.
    `R30(x, pars, h1, h2, h3, t)` returns `d³φ(x, p, t)(h1, h2, h3)`, the third differential of the flow map applied to `h1`, `h2`, `h3`.
    Used by the normal forms."
    R30::TR30 = nothing

    "delta used in finite differences w.r.t. to parameter. Used for example in PALC."
    delta::Tδ
end

has_monodromy_DE(::FlowDE{Tprob, Talg, Tjac, TprobMono}) where {Tprob, Talg, Tjac, TprobMono} = ~(TprobMono == Nothing)
@inline getdelta(fl::FlowDE) = fl.delta
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# constructors
"""
$(TYPEDSIGNATURES)

Creates a `::FlowDE <: AbstractFlow` variable based on a `prob::ODEProblem` and ODE solver `alg`. Also, the derivative of the flow is estimated with finite differences.
"""
function Flow(odeprob::Union{ODEProblem, EnsembleProblem, DAEProblem}, alg; kwargsDE...)
    return FlowDE(;odeprob, alg, kwargsDE, callback = get(kwargsDE, :callback, nothing), delta = 1e-8)
end

function Flow(odeprob::Union{ODEProblem, EnsembleProblem}, 
              alg, 
              odeprob_mono::Union{ODEProblem, EnsembleProblem}, 
              alg_mono; 
              kwargsDE...)
    return FlowDE(;odeprob, alg, odeprob_mono, alg_mono, kwargsDE, callback = get(kwargsDE, :callback, nothing), delta = 1e-8)
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_apply_vector_field(prob::ODEProblem, o, x, p) = prob.f(o, x, p, prob.tspan[1])
_apply_vector_field(prob::ODEProblem, x, p) = prob.f(x, p, prob.tspan[1])
_apply_vector_field(prob::EnsembleProblem, x, p) = _apply_vector_field(prob.prob, x, p)
_apply_vector_field(prob::EnsembleProblem, o, x, p) = _apply_vector_field(prob.prob, o, x, p)

@inline _isinplace(pb::ODEProblem) = isinplace_sciml(pb)
@inline _isinplace(pb::EnsembleProblem) = isinplace_sciml(pb.prob)

function vector_field(fl::FlowDE, x, pars)
    if _isinplace(fl.odeprob)
        out = similar(x)
        _apply_vector_field(fl.odeprob, out, x, pars)
        return out
    else
        return _apply_vector_field(fl.odeprob, x, pars)
    end
end

function _flow(x, pars, tm, pb::ODEProblem, alg; kwargs...)
    _prob = remake(pb; u0 = x, tspan = (zero(tm), tm), p = pars)
    # the use of concrete_solve makes it compatible with Zygote
    sol = SciMLBase.solve(_prob, alg; save_everystep = false, kwargs...)
    return (t = sol.t[end], u = sol.u[end])
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# methods for the flow
# this function takes into account a parameter passed to the vector field
# Putting the options `save_start = false` seems to give bugs with Sundials
function evolve(fl::FlowDE{T1}, x::AbstractArray, pars, tm; kw...) where {T1 <: ODEProblem}
    return _flow(x, pars, tm, fl.odeprob, fl.alg; fl.kwargsDE..., kw...)
end

function evolve(fl::FlowDE{T1}, x::AbstractArray, pars, tm; kw...) where {T1 <: EnsembleProblem}
    # modify the function which assigns new initial conditions
    # see docs at https://docs.sciml.ai/dev/features/ensemble/#Performing-an-Ensemble-Simulation-1
    # Compat: accept both SciMLBase v1/v2 (prob, i, repeat) and v3 (prob, ctx) prob_func signatures.
    _prob_func = (prob, ctx_or_i, _rest...) -> begin
        ii = ctx_or_i isa Integer ? ctx_or_i : ctx_or_i.sim_id
        remake(prob, u0 = x[:, ii], tspan = (zero(eltype(tm[ii])), tm[ii]), p = pars)
    end
    _epb = setproperties(fl.odeprob, output_func = (sol, _ctx_or_i) -> ((t = sol.t[end], u = sol.u[end]), false), prob_func = _prob_func)
    sol = SciMLBase.solve(_epb, fl.alg, EnsembleThreads(); trajectories = size(x, 2), save_everystep = false, fl.kwargsDE..., kw...)
    # sol.u contains a vector of tuples (sol_i.t[end], sol_i[end])
    return sol.u
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# monodromy matrix
function monodromy_matrix!(J, fl::FlowDE, x, pars, tm)
    ForwardDiff.jacobian!(J, z -> evolve(fl, Val(:SerialTimeSol), z, pars, tm).u, x)
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Differential of the flow, aka JVP
function _dflow_jvp_serial(x::AbstractVector, pars, dx, tm, pb_monodromy_jvp::ODEProblem, alg; k...)
    n = length(x)
    _prob = remake(pb_monodromy_jvp; u0 = vcat(x, dx), tspan = (zero(tm), tm), p = pars)
    # the use of concrete_solve makes it compatible with Zygote
    sol = SciMLBase.solve(_prob, alg; save_everystep = false, k...).u[end]
    return (t = tm, u = sol[1:n], du = sol[n+1:end])
end

function _dflow_finitediff_serial(x, pars, dx, tm, ode_prob::ODEProblem, alg; δ = convert(VI.scalartype(x), 1e-9), kwargs...)
    sol1 = _flow(x .+ δ .* dx, pars, tm, ode_prob, alg; kwargs...).u
    sol2 = _flow(x           , pars, tm, ode_prob, alg; kwargs...).u
    return (t = tm, u = sol2, du = (sol1 .- sol2) ./ δ)
end

# function used to compute the derivative of the flow, so pb encodes the variational equation
# differential of the flow when a problem is passed for the Monodromy
# default behavior (the FD case is handled by dispatch)
function jvp(fl::FlowDE{T1}, x::AbstractArray, pars, dx, tm;  kw...) where {T1 <: ODEProblem}
    _dflow_jvp_serial(x, pars, dx, tm, fl.odeprob_mono, fl.alg_mono; fl.kwargsDE..., kw...)
end

function vjp(fl::FlowDE{T1}, x::AbstractArray, pars, dx, tm;  kw...) where {T1 <: ODEProblem}
    fl.vjp(x, pars, dx, tm)
end

# differential of the flow when a problem is passed for the Monodromy
function jvp(fl::FlowDE{T1}, x::AbstractArray, pars, dx, tm;  kw...) where {T1 <: EnsembleProblem}
    N = size(x, 1)
    # Compat: accept both SciMLBase v1/v2 (prob, i, repeat) and v3 (prob, ctx) prob_func signatures.
    _prob_func = (prob, ctx_or_i, _rest...) -> begin
        ii = ctx_or_i isa Integer ? ctx_or_i : ctx_or_i.sim_id
        remake(prob, u0 = vcat(x[:, ii], dx[:, ii]), tspan = (zero(tm[ii]), tm[ii]), p = pars)
    end
    _epb = setproperties(fl.odeprob_mono, output_func = (sol, _ctx_or_i) -> ((t = sol.t[end], u = sol.u[end][1:N], du = sol.u[end][N+1:end]), false), prob_func = _prob_func)
    sol = SciMLBase.solve(_epb, fl.alg_mono, EnsembleThreads(); trajectories = size(x, 2), save_everystep = false, kw...)
    return sol.u
end

# when no ODEProblem is passed for the monodromy, we use finite differences
function jvp(fl::FlowDE{T1, Talg, Tjac, Nothing}, x::AbstractArray, pars, dx, tm;  δ = convert(VI.scalartype(x), getdelta(fl)), kw...) where {T1 <: Union{ODEProblem, EnsembleProblem},Talg, Tjac}
    if T1 <: ODEProblem
        return _dflow_finitediff_serial(x, pars, dx, tm, fl.odeprob, fl.alg; δ = δ, fl.kwargsDE..., kw...)
    else
        sol1 = evolve(fl, x .+ δ .* dx, pars, tm; kw...)
        sol2 = evolve(fl, x           , pars, tm; kw...)
        return [(t = sol1[ii][1], u = sol2[ii][2], du = (sol1[ii][2] .- sol2[ii][2]) ./ δ) for ii = 1:size(x,2) ]
    end
end
######### Optional methods
# this gives access to the full solution
# this function takes into account a parameter passed to the vector field and returns the full solution from the ODE solver. This is useful in Poincare Shooting to extract the period.
function evolve(fl::FlowDE{T1}, ::Val{:Full}, x::AbstractArray, pars, tm; kw...) where {T1 <: ODEProblem}
    _prob = remake(fl.odeprob; u0 = x, tspan = (zero(tm), tm), p = pars)
    return SciMLBase.solve(_prob, fl.alg; fl.kwargsDE..., kw...)
end

function evolve(fl::FlowDE{T1}, ::Val{:Full}, x::AbstractMatrix, pars, tm; kw...) where {T1 <: EnsembleProblem}
    # Compat: accept both SciMLBase v1/v2 (prob, i, repeat) and v3 (prob, ctx) prob_func signatures.
    _prob_func = (prob, ctx_or_i, _rest...) -> begin
        ii = ctx_or_i isa Integer ? ctx_or_i : ctx_or_i.sim_id
        remake(prob, u0 = x[:, ii], tspan = (zero(eltype(tm[ii])), tm[ii]), p = pars)
    end
    _epb = setproperties(fl.odeprob, prob_func = _prob_func)
    return SciMLBase.solve(_epb, fl.alg, EnsembleThreads(); trajectories = size(x, 2), fl.kwargsDE..., kw...)
end

function evolve(fl::FlowDE{T1}, ::Val{:SerialTimeSol}, x::AbstractArray, pars, δt; k...) where {T1 <: ODEProblem}
    return evolve(fl, x, pars, δt; k...)
end

function evolve(fl::FlowDE{T1}, ::Val{:SerialTimeSol}, x::AbstractArray, pars, tm; kw...) where {T1 <: EnsembleProblem}
    _flow(x, pars, tm, fl.odeprob.prob, fl.alg; fl.kwargsDE..., kw...)
end

function evolve(fl::FlowDE{T1,T2,Tjac,T3}, ::Val{:SerialdFlow}, x::AbstractArray, pars, dx, tm; δ = convert(eltype(x), getdelta(fl)), kw...) where {T1 <: ODEProblem, T2, Tjac, T3}
    if T3 === Nothing
        return _dflow_finitediff_serial(x, pars, dx, tm, fl.odeprob, fl.alg; δ = δ, fl.kwargsDE..., kw...)
    else # monodromy based on stacked system [vf, jvp(vf)]
        return _dflow_jvp_serial(x, pars, dx, tm, fl.odeprob_mono, fl.alg_mono; fl.kwargsDE..., kw...)
    end
end

function evolve(fl::FlowDE{T1}, ::Val{:SerialdFlow}, x::AbstractArray, pars, dx, tm; kw...) where {T1 <: EnsembleProblem}
    _dflow_jvp_serial(x, pars, dx, tm, fl.odeprob_mono.prob, fl.alg_mono; fl.kwargsDE..., kw...)
end

function evolve(fl::FlowDE{T1,T2,Tjac,Nothing,T4,T5,T6}, ::Val{:SerialdFlow}, x::AbstractArray, pars, dx, tm; δ = convert(eltype(x), getdelta(fl)), kw...) where {T1 <: EnsembleProblem,T2,T4,T5,T6, Tjac}
    _dflow_finitediff_serial(x, pars, dx, tm, fl.odeprob.prob, fl.alg; δ = δ, fl.kwargsDE..., kw...)
end

function R01(fl::FlowDE, x, pars, tΣ, lens, p₀)
    ForwardDiff.derivative(p -> evolve(fl, Val(:SerialTimeSol), x, set(pars, lens, p), tΣ).u, p₀)
end

function R11(fl::FlowDE, x, pars, dx, tΣ, lens, p₀::𝒯) where {𝒯}
    # If we were to use ForwardDiff, it would return a section R11
    δ = convert(𝒯, 1e-4)
    ∂²ϕ_∂x∂p_h₁ = ( evolve(fl, Val(:SerialdFlow), x, set(pars, lens, p₀ + δ), dx, tΣ).du .- 
                    evolve(fl, Val(:SerialdFlow), x, set(pars, lens, p₀ - δ), dx, tΣ).du) ./ (2δ)
    return ∂²ϕ_∂x∂p_h₁
end

R20(fl::FlowDE, x, pars, h1, h2, t) = fl.R20(x, pars, h1, h2, t)
R30(fl::FlowDE, x, pars, h1, h2, h3, t) = fl.R30(x, pars, h1, h2, h3, t)

function R20(fl::FlowDE{Tprob, Talg, Tjac, TprobMono, TalgMono, Tkwde, Tcb, Tvjp, TR01, TR11, Nothing}, x, pars, h1, h2, t)where {Tprob, Talg, Tjac, TprobMono, TalgMono, Tkwde, Tcb, Tvjp, TR01, TR11}
    ForwardDiff.derivative(ϵ -> evolve(fl, Val(:SerialdFlow), x .+ ϵ .* h2, pars, h1, t).du, 0)
end

function R30(fl::FlowDE{Tprob, Talg, Tjac, TprobMono, TalgMono, Tkwde, Tcb, Tvjp, TR01, TR11, TR20, Nothing}, x, pars, h1, h2, h3, t) where {Tprob, Talg, Tjac, TprobMono, TalgMono, Tkwde, Tcb, Tvjp, TR01, TR11, TR20}
    ForwardDiff.derivative(ϵ -> R20(fl, x .+ ϵ .* h3, pars, h1, h2, t), 0)
end