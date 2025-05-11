using SciMLBase: remake, ODEProblem, EnsembleProblem, EnsembleThreads, DAEProblem, isinplace as isinplace_sciml
import SciMLBase

struct FlowDE{Tprob, Talg, Tjac, TprobMono, TalgMono, Tkwde, Tcb, Tvjp, Tδ} <: AbstractFlow
    "Store the ODEProblem associated to the flow of the Cauchy problem"
    odeprob::Tprob

    "ODE time stepper passed to DifferentialEquations.solve"
    alg::Talg

    "Store the ODEProblem associated to the flow of the variational problem"
    odeprob_mono::TprobMono
    alg_mono::TalgMono

    "Keyword arguments passed to DifferentialEquations.solve"
    kwargsDE::Tkwde

    "Store possible callback"
    callback::Tcb

    # SHOULD BE USED!!!
    "How the monodromy is computed"
    jacobian::Tjac

    "adjoint of the monodromy (Matrix-Free)."
    vjp::Tvjp

    "delta used in finite differences wrt to parameter. Used for example in PALC."
    delta::Tδ
end

has_monodromy_DE(::FlowDE{Tprob, Talg, Tjac, TprobMono}) where {Tprob, Talg, Tjac, TprobMono} = ~(TprobMono == Nothing)
getdelta(fl::FlowDE) = fl.delta
####################################################################################################
# constructors
"""
Creates a `Flow` variable based on a `prob::ODEProblem` and ODE solver `alg`. The vector field `F` has to be passed, this will be resolved in the future as it can be recovered from `prob`. Also, the derivative of the flow is estimated with finite differences.
"""
function Flow(prob::Union{ODEProblem, EnsembleProblem, DAEProblem}, alg; kwargs...)
    # this constructor takes into account a parameter passed to the vector field
    return FlowDE(prob, alg, nothing, nothing, kwargs, get(kwargs, :callback, nothing), nothing, nothing, 1e-8)
end

function Flow(prob1::Union{ODEProblem, EnsembleProblem}, 
              alg1, 
              prob2::Union{ODEProblem, EnsembleProblem}, 
              alg2; 
              kwargs...)
    return FlowDE(prob1, alg1, prob2, alg2, kwargs, get(kwargs, :callback, nothing), nothing, nothing, 1e-8)
end
####################################################################################################
_apply_vector_field(prob::ODEProblem, o, x, p) = prob.f(o, x, p, prob.tspan[1])
_apply_vector_field(prob::ODEProblem, x, p) = prob.f(x, p, prob.tspan[1])
_apply_vector_field(prob::EnsembleProblem, x, p) = _apply_vector_field(prob.prob, x, p)
_apply_vector_field(prob::EnsembleProblem, o, x, p) = _apply_vector_field(prob.prob, o, x, p)

@inline _isinplace(pb::ODEProblem) = isinplace_sciml(pb)
@inline _isinplace(pb::EnsembleProblem) = isinplace_sciml(pb.prob)

function vf(fl::FlowDE, x, pars)
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
####################################################################################################
######### methods for the flow
# this function takes into account a parameter passed to the vector field
# Putting the options `save_start = false` seems to give bugs with Sundials
function evolve(fl::FlowDE{T1}, x::AbstractArray, pars, tm; kw...) where {T1 <: ODEProblem}
    return _flow(x, pars, tm, fl.odeprob, fl.alg; fl.kwargsDE..., kw...)
end

function evolve(fl::FlowDE{T1}, x::AbstractArray, pars, tm; kw...) where {T1 <: EnsembleProblem}
    # modify the function which assigns new initial conditions
    # see docs at https://docs.sciml.ai/dev/features/ensemble/#Performing-an-Ensemble-Simulation-1
    _prob_func = (prob, ii, repeat) -> prob = remake(prob, u0 = x[:, ii], tspan = (zero(eltype(tm[ii])), tm[ii]), p = pars)
    _epb = setproperties(fl.prob, output_func = (sol, i) -> ((t = sol.t[end], u = sol.u[end]), false), prob_func = _prob_func)
    sol = SciMLBase.solve(_epb, fl.alg, EnsembleThreads(); trajectories = size(x, 2), save_everystep = false, fl.kwargsDE..., kw...)
    # sol.u contains a vector of tuples (sol_i.t[end], sol_i[end])
    return sol.u
end
####################################################################################################
######### Differential of the flow
function dflowMonoSerial(x::AbstractVector, pars, dx, tm, pb::ODEProblem, alg; k...)
    n = length(x)
    _prob = remake(pb; u0 = vcat(x, dx), tspan = (zero(tm), tm), p = pars)
    # the use of concrete_solve makes it compatible with Zygote
    sol = SciMLBase.solve(_prob, alg; save_everystep = false, k...)[end]
    return (t = tm, u = sol[1:n], du = sol[n+1:end])
end

function dflow_fdSerial(x, pars, dx, tm, pb::ODEProblem, alg; δ = convert(eltype(x), 1e-9), kwargs...)
    sol1 = _flow(x .+ δ .* dx, pars, tm, pb, alg; kwargs...).u
    sol2 = _flow(x           , pars, tm, pb, alg; kwargs...).u
    return (t = tm, u = sol2, du = (sol1 .- sol2) ./ δ)
end

# function used to compute the derivative of the flow, so pb encodes the variational equation
# differential of the flow when a problem is passed for the Monodromy
# default behavior (the FD case is handled by dispatch)
function jvp(fl::FlowDE{T1}, x::AbstractArray, pars, dx, tm;  kw...) where {T1 <: ODEProblem}
    dflowMonoSerial(x, pars, dx, tm, fl.odeprob_mono, fl.alg_mono; fl.kwargsDE..., kw...)
end

function vjp(fl::FlowDE{T1}, x::AbstractArray, pars, dx, tm;  kw...) where {T1 <: ODEProblem}
    fl.vjp(x, pars, dx, tm)
end

# differential of the flow when a problem is passed for the Monodromy
function jvp(fl::FlowDE{T1}, x::AbstractArray, pars, dx, tm;  kw...) where {T1 <: EnsembleProblem}
    N = size(x, 1)
    _prob_func = (prob, ii, repeat) -> prob = remake(prob, u0 = vcat(x[:, ii], dx[:, ii]), tspan = (zero(tm[ii]), tm[ii]), p = pars)
    _epb = setproperties(fl.odeprob_mono, output_func = (sol,i) -> ((t = sol.t[end], u = sol[end][1:N], du = sol[end][N+1:end]), false), prob_func = _prob_func)
    sol = SciMLBase.solve(_epb, fl.alg_mono, EnsembleThreads(); trajectories = size(x, 2), save_everystep = false, kw...)
    return sol.u
end

# when no ODEProblem is passed for the monodromy, we use finite differences
function jvp(fl::FlowDE{T1, Talg, Tjac, Nothing}, x::AbstractArray, pars, dx, tm;  δ = convert(eltype(x), 1e-9), kw...) where {T1 <: Union{ODEProblem, EnsembleProblem},Talg, Tjac}
    if T1 <: ODEProblem
        return dflow_fdSerial(x, pars, dx, tm, fl.prob, fl.alg; δ = δ, fl.kwargsDE..., kw...)
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
    sol = SciMLBase.solve(_prob, fl.alg; fl.kwargsDE..., kw...)
end

function evolve(fl::FlowDE{T1}, ::Val{:Full}, x::AbstractArray, pars, tm; kw...) where {T1 <: EnsembleProblem}
    _prob_func = (prob, ii, repeat) -> prob = remake(prob, u0 = x[:, ii], tspan = (zero(eltype(tm[ii])), tm[ii]), p = pars)
    _epb = setproperties(fl.odeprob, prob_func = _prob_func)
    sol = SciMLBase.solve(_epb, fl.alg, EnsembleThreads(); trajectories = size(x, 2), fl.kwargsDE..., kw...)
end

function evolve(fl::FlowDE{T1}, ::Val{:SerialTimeSol}, x::AbstractArray, pars, δt; k...) where {T1 <: ODEProblem}
    return evolve(fl, x, pars, δt; k...)
end

function evolve(fl::FlowDE{T1}, ::Val{:SerialTimeSol}, x::AbstractArray, pars, tm; kw...) where {T1 <: EnsembleProblem}
    _flow(x, pars, tm, fl.odeprob.prob, fl.alg; fl.kwargsDE..., kw...)
end

function evolve(fl::FlowDE{T1,T2,Tjac,T3}, ::Val{:SerialdFlow}, x::AbstractArray, pars, dx, tm; δ = convert(eltype(x), 1e-9), kw...) where {T1 <: ODEProblem, T2, Tjac, T3}
    if T3 === Nothing
        return dflow_fdSerial(x, pars, dx, tm, fl.odeprob, fl.alg; δ = δ, fl.kwargsDE..., kw...)
    else
        return dflowMonoSerial(x, pars, dx, tm, fl.odeprob_mono, fl.alg_mono; fl.kwargsDE..., kw...)
    end
end

function evolve(fl::FlowDE{T1}, ::Val{:SerialdFlow}, x::AbstractArray, pars, dx, tm; kw...) where {T1 <: EnsembleProblem}
    dflowMonoSerial(x, pars, dx, tm, fl.odeprob_mono.prob, fl.alg_mono; fl.kwargsDE..., kw...)
end

function evolve(fl::FlowDE{T1,T2,Tjac,Nothing,T4,T5,T6}, ::Val{:SerialdFlow}, x::AbstractArray, pars, dx, tm; δ = convert(eltype(x), 1e-9), kw...) where {T1 <: EnsembleProblem,T2,T4,T5,T6, Tjac}
    dflow_fdSerial(x, pars, dx, tm, fl.odeprob.prob, fl.alg; δ = δ, fl.kwargsDE..., kw...)
end
