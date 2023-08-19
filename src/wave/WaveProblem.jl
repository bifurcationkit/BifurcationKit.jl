abstract type abstractModulatedWaveFD <: AbstractPOFDProblem end
abstract type abstractModulatedWaveShooting <: AbstractShootingProblem end

"""
TWProblem(prob, ∂::Tuple, u₀; DAE = 0, jacobian::Symbol = :AutoDiff)

This composite type implements a functional for freezing symmetries in order, for example, to compute travelling waves (TW). Note that you can freeze many symmetries, not just one, by passing many Lie generators. When you call `pb(x, par)`, it computes:

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
- `nbConstraints(::TWProblem)` number of constraints (or Lie generators)

"""
@with_kw_noshow struct TWProblem{Tprob, Tu0, TDu0, TD} <: AbstractBifurcationProblem
    "vector field, must be `AbstractBifurcationProblem`"
    prob_vf::Tprob
    "Infinitesimal generator of symmetries, differential operator"
    ∂::TD
    "reference solution, we only need one!"
    u₀::Tu0
    ∂u₀::TDu0 = (∂ * u₀,)
    DAE::Int = 0
    nc::Int = 1  # number of constraints
    jacobian::Symbol = :AutoDiff
    @assert 0 <= DAE <= 1
    @assert 0 < nc
end

function TWProblem(prob, ∂::Tuple, u₀; DAE = 0, jacobian::Symbol = :AutoDiff)
    # ∂u₀ = Tuple( apply(_D, u₀) for _D in ∂)
    ∂u₀ = Tuple( mul!(zero(u₀), _D, u₀, 1, 0) for _D in ∂)
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

@inline nbConstraints(pb::TWProblem) = pb.nc

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
    copyto!(pb.u₀, u₀)
    for (∂, ∂u₀) in zip(pb.∂, pb.∂u₀)
        # pb.u₀∂u₀ = Tuple( dot(u₀, u) for u in ∂u₀)
        copyto!(∂u₀, ∂ * u₀)
    end
end

"""
- `ss` tuple of speeds
- `D` tuple of Lie generators
"""
function applyD(pb::TWProblem, out, ss, u)
    for (D, s) in zip(pb.∂, ss)
        # out .-=  s .* (D * u)
        mul!(out, D, u, -s, 1)
    end
    out
end
applyD(pb::TWProblem, u) = applyD(pb, zero(u), 1, u)

# s is the speed.
# Return F(u, p) - s * D * u
@views function VFplusD(pb::TWProblem, u::AbstractVector, s::Tuple, pars)
    # apply the vector field
    out = residual(pb.prob_vf, u, pars)
    # we add the freezing, it can be done now since out is filled by the previous call!!
    applyD(pb, out, s, u)
    return out
end

# function (u, p) -> F(u, p) - s * D * u to be used with shooting or Trapezoid
VFtw(pb::TWProblem, u::AbstractVector, parsFreez) = VFplusD(pb, u, parsFreez.s, parsFreez.user)

# vector field of the TW problem
@views function (pb::TWProblem)(x::AbstractVector, pars)
    # number of constraints
    nc = pb.nc
    # number of unknowns
    N = length(x) - nc
    # array containing the result
    out = similar(x)
    u = x[1:N]
    outu = out[1:N]
    # get the speed
    s = Tuple(x[end-nc+1:end])
    # apply the vector field
    outu .= VFplusD(pb, u, s, pars)
    # we put the constraints
    for ii in 0:nc-1
        out[end-ii] = dot(u, pb.∂u₀[ii+1])
        if pb.DAE == 0
            out[end-ii] -= dot(pb.u₀, pb.∂u₀[ii+1])
        end
    end
    return out
end

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
        out[end-ii] = dot(du, pb.∂u₀[ii+1])
    end
    return out
end

# build the sparse jacobian of the freezed problem
function (pb::TWProblem)(::Val{:JacFullSparse}, ufreez::AbstractVector, par; δ = 1e-9)
    # number of constraints
    nc = nbConstraints(pb)
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
        mul!(view(rightpart, :, ii), pb.∂[ii], u, -1, 0)
    end
    J2 = hcat(J1, rightpart)
    for ii in 1:nc
        J2 = vcat(J2, vcat(pb.∂u₀[ii], zeros(nc))')
    end
    return J2
end
################################################################################
function modify_tw_record(probTW, kwargs, par, lens)
    _recordsol0 = get(kwargs, :recordFromSolution, nothing)
    if isnothing(_recordsol0) == false
        _recordsol0 = get(kwargs, :recordFromSolution, nothing)
        return _recordsol = (x, p; k...) -> _recordsol0(x, (prob = probTW, p = p); k...)
    else
        return _recordsol = (x, p; k...) -> (s = x[end],)
    end
end
################################################################################
residual(tw::WrapTW, x, p) = tw.prob(x, p)
jacobian(tw::WrapTW, x, p) = tw.jacobian(x, p)
@inline is_symmetric(::WrapTW) = false
@inline has_adjoint(::WrapTW) = false
@inline getdelta(::WrapTW) = 1e-8
dF(tw::WrapTW, x, p, dx1) = ForwardDiff.derivative(t -> tw.prob(x .+ t .* dx1, p), 0.)
d2F(tw::WrapTW, x, p, dx1, dx2) = ForwardDiff.derivative(t -> dF(tw, x .+ t .* dx2, p, dx1), 0.)
d3F(tw::WrapTW, x, p, dx1, dx2, dx3) = ForwardDiff.derivative(t -> d2F(tw, x .+ t .* dx3, p, dx1, dx2), 0.)

function newton(prob::TWProblem, orbitguess, optn::NewtonPar; kwargs...)
    jacobian = prob.jacobian
    @assert jacobian in (:MatrixFree, :MatrixFreeAD, :AutoDiff, :FullLU, :FiniteDifferences)
    if jacobian == :AutoDiff
        jac = (x, p) -> sparse(ForwardDiff.jacobian(z -> prob(z, p), x))
    elseif jacobian == :MatrixFreeAD
        jac = (x, p) -> (dx -> ForwardDiff.derivative(t -> prob(x .+ t .* dx, p), 0))
    elseif jacobian == :FullLU
        jac = (x, p) -> prob(Val(:JacFullSparse), x, p)
    elseif jacobian == :FiniteDifferences
        jac = (x, p) -> finiteDifferences(z -> prob(z, p), x)
    elseif jacobian == :MatrixFree
        jac = (x, p) -> (dx ->  prob(x, p, dx))
    end
    probwp = WrapTW(prob, jac, orbitguess, getparams(prob.prob_vf), getlens(prob.prob_vf), record_from_solution(prob.prob_vf), plot_solution(prob.prob_vf))
    return newton(probwp, optn; kwargs...,)
end

function continuation(prob::TWProblem,
        orbitguess, alg::AbstractContinuationAlgorithm, contParams::ContinuationPar;
        kwargs...)
    jacobian = prob.jacobian
    @assert jacobian in (:MatrixFree, :MatrixFreeAD, :AutoDiff, :FullLU, :FiniteDifferences)

    if jacobian == :AutoDiff
        jac = (x, p) -> sparse(ForwardDiff.jacobian(z -> prob(z, p), x))
    elseif jacobian == :MatrixFreeAD
        jac = (x, p) -> (dx -> ForwardDiff.derivative(t -> prob(x .+ t .* dx, p), 0))
    elseif jacobian == :FullLU
        jac = (x, p) -> prob(Val(:JacFullSparse), x, p)
    elseif jacobian == :FiniteDifferences
        jac = (x, p) -> finiteDifferences(z -> prob(z, p), x)
    elseif jacobian == :MatrixFree
        jac = (x, p) -> (dx ->  prob(x, p, dx))
    end
    # define the mass matrix for the eigensolver
    N = length(orbitguess)
    B = spdiagm(vcat(ones(N-1),0))
    # convert eigsolver to generalised one
    old_eigsolver = contParams.newtonOptions.eigsolver
    contParamsWave = @set contParams.newtonOptions.eigsolver = convertToGEV(old_eigsolver, B)

    # update record function
    _recordsol = modify_tw_record(prob, kwargs, getparams(prob.prob_vf), getlens(prob.prob_vf))

    probwp = WrapTW(prob, jac, orbitguess, getparams(prob.prob_vf), getlens(prob.prob_vf), plot_solution(prob.prob_vf), _recordsol)

    # call continuation
    branch = continuation(probwp, alg, contParamsWave; kind = TravellingWaveCont(), kwargs...,)
    return branch
end
