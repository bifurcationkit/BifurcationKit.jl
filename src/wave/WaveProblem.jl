abstract type AbstractModulatedWaveFD <: AbstractPOFDProblem end
abstract type AbstractModulatedWaveShooting <: AbstractShootingProblem end

"""
$(TYPEDEF)

This composite type implements a functional for freezing symmetries in order, for example, to compute traveling waves (TW). Note that you can freeze many symmetries, not just one, by passing many Lie generators. When you call `pb(x, par)`, it computes:

                    ┌                   ┐
                    │ f(x, par) - s⋅∂⋅x │
                    │   <x - u₀, ∂⋅u₀>  │
                    └                   ┘

## Arguments
- `prob` bifurcation problem with continuous symmetries
- `∂::Tuple = (T1, T2, ⋯)` tuple of Lie generators. In effect, each of these is an (differential) operator which can be specified as a (sparse) matrix or as an operator implementing `LinearAlgebra.mul!`.
- `u₀` reference solution

## Additional Constructor(s)

    pb = TWProblem(prob, ∂, u₀; kw...)

This simplified call handles the case where a single symmetry needs to be frozen.

## Useful function

- `updatesection!(pb::TWProblem, u0)` updates the reference solution of the problem using `u0`.
- `nb_constraints(::TWProblem)` number of constraints (or Lie generators)

## Fields
$(TYPEDFIELDS)

"""
@with_kw_noshow struct TWProblem{Tprob, Tu0, TDu0, TD, Tj} <: AbstractBifurcationProblem
    "vector field, must be `AbstractBifurcationProblem`."
    prob_vf::Tprob
    "Infinitesimal generator of symmetries, differential operator."
    ∂::TD
    "reference solution, we only need one!"
    u₀::Tu0
    ∂u₀::TDu0 = (∂ * u₀,)
    DAE::Int = 0
    "[Internal] number of constraints"
    nc::Int = 1
    jacobian::Tj = AutoDiff()
    @assert 0 <= DAE <= 1
    @assert 0 < nc
    @assert jacobian in [MatrixFree(), AutoDiffMF(), FullLU(), FiniteDifferences(), AutoDiff()] "This jacobian is not defined. Please chose another one."
end
getparams(tw::TWProblem) = getparams(tw.prob_vf)

function TWProblem(prob, ∂::Tuple, u₀; DAE = 0, jacobian = AutoDiff())
    # ∂u₀ = Tuple( apply(_D, u₀) for _D in ∂)
    ∂u₀ = Tuple( LA.mul!(zero(u₀), _D, u₀, 1, 0) for _D in ∂)
    return TWProblem(prob_vf = prob, ∂ = ∂,
        u₀ = u₀,
        ∂u₀ = ∂u₀,
        # u₀∂u₀ = Tuple( dot(u₀, u) for u in ∂u₀),
        DAE = DAE,
        nc = length(∂),
        jacobian = jacobian )
end

# constructor
TWProblem(prob, ∂, u₀; kw...) = TWProblem(prob, (∂,), u₀; kw...)

@inline nb_constraints(pb::TWProblem) = pb.nc

function Base.show(io::IO, tw::TWProblem)
    println(io, "┌─ Travelling wave functional")
    println(io, "├─ type          : Vector{", eltype(tw.u₀), "}")
    println(io, "├─ # constraints : ", tw.nc)
    println(io, "├─ lens          : ", get_lens_symbol(getlens(tw.prob_vf)))
    println(io, "├─ jacobian      : ", tw.jacobian)
    println(io, "└─ DAE           : ", tw.DAE)
end

# we put type information to ensure the user pass a correct u0
function updatesection!(pb::TWProblem{Tprob, Tu0, TDu0, TD}, u₀::Tu0) where {Tprob, Tu0, TDu0, TD}
    _copyto!(pb.u₀, u₀)
    for (∂, ∂u₀) in zip(pb.∂, pb.∂u₀)
        # pb.u₀∂u₀ = Tuple( dot(u₀, u) for u in ∂u₀)
        _copyto!(∂u₀, ∂ * u₀)
    end
end

"""
$(TYPEDSIGNATURES)

- `ss` tuple of speeds
- `D` tuple of Lie generators
"""
function applyD(pb::TWProblem, out, ss, u)
    for (D, s) in zip(pb.∂, ss)
        # out .-=  s .* (D * u)
        LA.mul!(out, D, u, -s, 1)
    end
    out
end
applyD(pb::TWProblem, u) = applyD(pb, zero(u), 1, u)

# s is the speed.
# Return F(u, p) - s * D * u
@views function VF_plus_D(pb::TWProblem, u::AbstractVector, s::Tuple, pars)
    # apply the vector field
    out = residual(pb.prob_vf, u, pars)
    # we add the freezing, it can be done now since out is filled by the previous call!!
    applyD(pb, out, s, u)
    return out
end

# function (u, p) -> F(u, p) - s * D * u to be used with shooting or Trapezoid
VFtw(pb::TWProblem, u::AbstractVector, parsFreez) = VF_plus_D(pb, u, parsFreez.s, parsFreez.user)

# vector field of the TW problem
@views function residual!(pb::TWProblem, out, x::AbstractVector, pars)
    # number of constraints
    nc = pb.nc
    # number of unknowns
    N = length(x) - nc
    u = x[1:N]
    outu = out[1:N]
    # get the speed
    s = Tuple(x[end-nc+1:end])
    # apply the vector field
    outu .= VF_plus_D(pb, u, s, pars)
    # we put the constraints
    for ii in 0:nc-1
        out[end-ii] = LA.dot(u, pb.∂u₀[ii+1])
        if pb.DAE == 0
            out[end-ii] -= LA.dot(pb.u₀, pb.∂u₀[ii+1])
        end
    end
    return out
end

residual(pb::TWProblem, x::AbstractVector, pars) = residual!(pb, similar(x), x, pars)

# jacobian-free function
@views function (pb::TWProblem)(x::AbstractVector, pars, dx::AbstractVector)
    # number of constraints
    nc = pb.nc
    # number of unknowns
    N = length(x) - nc
    # array containing the result
    out = similar(x)
    u = x[1:N]
    du = dx[1:N]
    outu = out[1:N]
    # get the speed
    s = Tuple(x[end-nc+1:end])
    ds = Tuple(dx[end-nc+1:end])
    # get the jacobian
    J = jacobian(pb.prob_vf, u, pars)
    outu .= apply(J, du)
    applyD(pb, outu, s, du)
    applyD(pb, outu, ds, u)
    # we put the constraints
    for ii in 0:nc-1
        out[end-ii] = LA.dot(du, pb.∂u₀[ii+1])
    end
    return out
end

# build the sparse jacobian of the freezed problem
function (pb::TWProblem)(::Val{:JacFullSparse}, ufreez::AbstractVector, par; δ = 1e-9)
    # number of constraints
    nc = nb_constraints(pb)
    # number of unknowns
    N = length(ufreez) - nc
    # get the speed
    s = Tuple(ufreez[end-nc+1:end])
    # get the state space vector
    u = ufreez[1:N]
    # the jacobian of the
    J1 = jacobian(pb.prob_vf, u, par)
    # we add the Lie algebra generators
    rightpart = zeros(N, nc)
    for ii in 1:nc
        J1 = J1 - s[ii] * pb.∂[ii]
        LA.mul!(view(rightpart, :, ii), pb.∂[ii], u, -1, 0)
    end
    J2 = hcat(J1, rightpart)
    for ii in 1:nc
        J2 = vcat(J2, vcat(pb.∂u₀[ii], zeros(nc))')
    end
    return J2
end
################################################################################
function modify_tw_record(probTW, kwargs, par, lens)
    _recordsol0 = get(kwargs, :record_from_solution, nothing)
    if isnothing(_recordsol0) == false
        _recordsol0 = get(kwargs, :record_from_solution, nothing)
        return _recordsol = (x, p; k...) -> _recordsol0(x, (prob = probTW, p = p); k...)
    else
        return _recordsol = (x, p; k...) -> (s = x[end],)
    end
end
################################################################################
jacobian(tw::WrapTW, x, p) = jacobian(tw, tw.jacobian, x, p)
residual(tw::WrapTW, x, p) = residual(tw.prob, x, p)
@inline save_solution(::WrapTW, x, p) = x
@inline is_symmetric(::WrapTW) = false
@inline has_adjoint(::WrapTW) = false
@inline getdelta(::WrapTW) = 1e-8
dF(tw::WrapTW, x, p, dx1) = ForwardDiff.derivative(t -> residual(tw.prob .+ t .* dx1, p), 0)
d2F(tw::WrapTW, x, p, dx1, dx2) = ForwardDiff.derivative(t -> dF(tw, x .+ t .* dx2, p, dx1), 0)
d3F(tw::WrapTW, x, p, dx1, dx2, dx3) = ForwardDiff.derivative(t -> d2F(tw, x .+ t .* dx3, p, dx1, dx2), 0)
@inline update!(::WrapTW, args...; k...) = update_default(args...; k...)

_generate_jacobian(probPO::TWProblem, J::Union{MatrixFree, AutoDiffMF, FullLU, FiniteDifferences, AutoDiff}, o, pars; k...) = J
jacobian(prob::WrapTW, ::AutoDiff, x, p) = ForwardDiff.jacobian(z -> residual(prob, z, p), x)
jacobian(prob::WrapTW, ::FullLU, x, p) = prob.prob(Val(:JacFullSparse), x, p)
jacobian(prob::WrapTW, ::MatrixFree, x, p) = (dx ->  prob.prob(x, p, dx))

function newton(prob::TWProblem, 
                orbitguess, 
                optn::NewtonPar; 
                δ = convert(VI.scalartype(orbitguess), 1e-8),
                kwargs...)
    jacobianTW = prob.jacobian
    jac = _generate_jacobian(prob, jacobianTW, orbitguess, getparams(prob); δ)
    probwp = WrapTW(prob, jac, orbitguess, getparams(prob.prob_vf), getlens(prob.prob_vf), record_from_solution(prob.prob_vf), plot_solution(prob.prob_vf))
    return solve(probwp, Newton(), optn; kwargs...,)
end

function continuation(prob::TWProblem,
                    orbitguess, 
                    alg::AbstractContinuationAlgorithm, 
                    contParams::ContinuationPar;
                    record_from_solution = nothing,
                    plot_solution = plot_solution(prob.prob_vf),
                    δ = convert(VI.scalartype(orbitguess), 1e-8),
                    kwargs...)
    jacobianTW = prob.jacobian
    @assert jacobianTW in (MatrixFree(), AutoDiffMF(), AutoDiff(), FullLU(), FiniteDifferences())
    # define the mass matrix for the eigensolver
    N = length(orbitguess)
    B = spdiagm(vcat(ones(N-1), 0))
    # convert eigsolver to generalised one
    old_eigsolver = contParams.newton_options.eigsolver
    contParamsWave = @set contParams.newton_options.eigsolver = convertToGEV(old_eigsolver, B)

    # update record function
    # this is to remove this part from the arguments passed to continuation
    _kwargs = (;record_from_solution, plot_solution)
    _recordsol = modify_tw_record(prob, _kwargs, getparams(prob.prob_vf), getlens(prob.prob_vf))
    jac = _generate_jacobian(prob, jacobianTW, orbitguess, getparams(prob); δ)
    probwp = WrapTW(prob, jac, orbitguess, getparams(prob.prob_vf), getlens(prob.prob_vf), plot_solution, _recordsol)

    # call continuation
    branch = continuation(probwp, alg, contParamsWave; kind = TravellingWaveCont(), kwargs...,)
    return branch
end
