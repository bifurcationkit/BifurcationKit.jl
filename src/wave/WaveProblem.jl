abstract type AbstractTravelingWaveDiscretization end
"""
$(TYPEDEF)

Composite type implementing the freezing of continuous symmetries to compute,
for example, traveling waves (TW). Several symmetries can be frozen at once by
passing many Lie generators. `TWModel` is a discretization: the residual of the
frozen system is obtained by wrapping `pb` into a `TravellingWave` functional,
`residual(TravellingWave(pb), x, par)`, which computes:

                    ┌                                      ┐
                    │ f(x, par) - ∑ᵢ sᵢ ⋅ ∂ᵢ ⋅ x            │
                    │   ⟨x - u₀, ∂ᵢ ⋅ u₀⟩,  i = 1, …, N_g   │
                    └                                      ┘

The unknowns are `(u, s₁, …, s_{N_g})`, *i.e.* the state `u` with the speeds `sᵢ` appended at the end. The reference solution `u₀` (and the vectors `∂ᵢ ⋅ u₀`) is updated during continuation, see [`updatesection!`](@ref).

# Arguments
- `prob_vf` bifurcation problem with continuous symmetries, must be an `AbstractBifurcationProblem`
- `∂::Tuple` tuple of Lie generators. Each of these is a (differential) operator, *e.g.* a (sparse) matrix or an operator implementing `LinearAlgebra.mul!`.
- `u₀` reference solution

# Keyword arguments
- `DAE = fill(true, length(∂))` vector of flags, one per symmetry. If `DAE[i] = true`, the i-th phase condition is `⟨u - u₀, ∂ᵢ ⋅ u₀⟩ = 0` (phase fixed relative to the reference solution); if `false`, it reduces to `⟨u, ∂ᵢ ⋅ u₀⟩ = 0`.
- `jacobian = AutoDiff()` type of jacobian used in the Newton iterations, one of:
    - `AutoDiff()`: dense jacobian via ForwardDiff
    - `FiniteDifferences()`: dense finite differences
    - `FullLU()`: (sparse) assembly of the frozen jacobian using the jacobian of the underlying problem `prob_vf`
    - `MatrixFree()`: matrix-free evaluation of the jacobian-vector product
    - `AutoDiffMF()`: matrix-free jacobian-vector product via ForwardDiff
- `update_section_every_step = 1`: the reference solution `u₀` is updated every `update_section_every_step` steps during continuation.

# Useful functions
- `updatesection!(pb::TWModel, U₀)` updates the reference solution using `U₀` (state with speeds appended).
- `nb_constraints(::TWModel)` number of constraints (or Lie generators).
- `newton(pb::TWModel, orbitguess, options)` finds a frozen solution.
- `continuation(pb::TWModel, orbitguess, alg, contParams)` continues the wave.

# Internal fields
$(TYPEDFIELDS)
"""
@with_kw_noshow struct TWModel{Tprob, Tu0, TDu0, TD, Tj} <: AbstractTravelingWaveDiscretization
    "vector field, must be `AbstractBifurcationProblem`."
    prob_vf::Tprob
    "Infinitesimal generator of symmetries, differential operator."
    ∂::TD
    "reference solution, we only need one!"
    u₀::Tu0
    "generator ⋅ u₀"
    ∂u₀::TDu0 = (∂ * u₀,)
    "Vector of flags, one per symmetry. When `true`, the corresponding phase condition is `⟨u - u₀, ∂⋅u₀⟩ = 0` (phase fixed relative to the reference solution `u₀`); when `false`, it reduces to `⟨u, ∂⋅u₀⟩ = 0`."
    DAE::Vector{Bool} = fill(true, nc)
    "[Internal] number of constraints."
    nc::Int = 1
    "Type of jacobian for the frozen problem, one of `AutoDiff()`, `FiniteDifferences()`, `FullLU()`, `MatrixFree()` or `AutoDiffMF()`."
    jacobian::Tj = AutoDiff()
    "Update the section every `update_section_every_step` step during continuation."
    update_section_every_step::UInt = 1
    @assert 0 <= all(x-> 0<=x<=1, DAE)
    @assert 0 < nc
    @assert jacobian in (MatrixFree(), AutoDiffMF(), FullLU(), FiniteDifferences(), AutoDiff()) "This jacobian is not defined. Please chose another one."
end
@inline getparams(tw::TWModel) = getparams(tw.prob_vf)
@inline getlens(tw::TWModel) = getlens(tw.prob_vf)
@inline getdelta(tw::TWModel) = getdelta(tw.prob_vf)

function TWModel(prob, ∂::Tuple, u₀; DAE = [true for _ in ∂], jacobian = AutoDiff(), k...)
    # ∂u₀ = Tuple( apply(_D, u₀) for _D in ∂)
    ∂u₀ = Tuple( LA.mul!(zero(u₀), _D, u₀, 1, 0) for _D in ∂)
    return TWModel(;prob_vf = prob, 
        ∂,
        u₀,
        ∂u₀,
        # u₀∂u₀ = Tuple( dot(u₀, u) for u in ∂u₀),
        DAE,
        nc = length(∂),
        jacobian,
        k... )
end

TWModel(prob, ∂, u₀; kw...) = TWModel(prob, (∂,), u₀; kw...)

function re_make(tw::TWModel; params = getparams(tw))
    new_prob = re_make(tw.prob_vf; params)
    return (@set tw.prob_vf = new_prob)
end

@inline nb_constraints(pb::TWModel) = pb.nc

function Base.show(io::IO, tw::TWModel)
    println(io, "┌─ Travelling wave functional")
    println(io, "├─ type           : Vector{", VI.scalartype(tw.u₀), "}")
    println(io, "├─ # constraints  : ", tw.nc)
    println(io, "├─ lens           : ", get_lens_symbol(getlens(tw.prob_vf)))
    println(io, "├─ update section : ", tw.update_section_every_step)
    println(io, "├─ jacobian       : ", tw.jacobian)
    println(io, "└─ DAE            : ", tw.DAE)
end

# we put type information to ensure the user pass a correct u0
function updatesection!(pb::TWModel{Tprob, Tu0, TDu0, TD}, U₀::Tu0) where {Tprob, Tu0, TDu0, TD}
    u₀ = @view U₀[1:end-pb.nc]
    _copyto!(pb.u₀, u₀)
    for (∂, ∂u₀) in zip(pb.∂, pb.∂u₀)
        _copyto!(∂u₀, ∂ * u₀)
    end
end

"""
$(TYPEDSIGNATURES)

- `ss` tuple of speeds
- `D` tuple of Lie generators
"""
function applyD!(pb::TWModel, out, ss, u)
    for (D, s) in zip(pb.∂, ss)
        # out .-=  s .* (D * u)
        LA.mul!(out, D, u, -s, 1)
    end
    out
end
applyD(pb::TWModel, u) = applyD!(pb, zero(u), 1, u)

"""
Return `F(u, p) - s * D * u` where `s` is the speed.
"""
@views function _VF_plus_D(pb::TWModel, u::AbstractVector, s::Tuple, pars)
    # apply the vector field
    out = residual(pb.prob_vf, u, pars)
    # we add the freezing, it can be done now since out is filled by the previous call!!
    applyD!(pb, out, s, u)
    return out
end

# function (u, p) -> F(u, p) - s * D * u to be used with shooting or Trapeze
VFtw(pb::TWModel, u::AbstractVector, parsFreez) = _VF_plus_D(pb, u, parsFreez.s, parsFreez.user)

"""
Return `dF(u, p)⋅du - s * D * du - ds * D * u` where `s` is the speed.
"""
function _jvp_VF_plus_D!(pb,
                        out::AbstractVector,
                        u::AbstractVector,
                        du::AbstractVector,
                        s::Tuple,
                        ds::Tuple,
                        pars,
                        ::Val{add_ds} = Val(true)) where {add_ds}
    out .= dF(pb.prob_vf, u, pars, du)
    applyD!(pb, out, s, du)
    if add_ds
        applyD!(pb, out, ds, u)
    end
    return out
end

# vector field of the TW problem
@views function residual_tw!(pb::TWModel, out, x::AbstractVector, pars)
    # number of constraints
    nc = pb.nc
    # number of unknowns
    N = length(x) - nc
    u = x[1:N]
    outu = out[1:N]
    # get the speed
    s = Tuple(x[end-nc+1:end])
    # apply the vector field
    outu .= _VF_plus_D(pb, u, s, pars)
    # we put the constraints
    for ii in 0:nc-1
        out[end-ii] = LA.dot(u, pb.∂u₀[ii+1])
        if pb.DAE[ii+1]
            out[end-ii] -= LA.dot(pb.u₀, pb.∂u₀[ii+1])
        end
    end
    return out
end

residual!(pb::TravellingWave, out, x::AbstractVector, pars) = residual_tw!(get_discretization(pb), out, x, pars)
residual(pb::TravellingWave, x::AbstractVector, pars) = residual_tw!(get_discretization(pb), similar(x), x, pars)

# jacobian-vector-product function
@views function (pb::TWModel)(x::AbstractVector, pars, dx::AbstractVector)
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
    _jvp_VF_plus_D!(pb, outu, u, du, s, ds, pars)
    # we put the constraints
    for ii in 0:nc-1
        out[end-ii] = LA.dot(du, pb.∂u₀[ii+1])
    end
    return out
end

# build the sparse jacobian of the freezed problem
function (pb::TWModel)(::Val{:JacFullSparse}, ufreez::AbstractVector, par; δ = getdelta(pb))
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
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
jacobian(tw::WrapTW, x, p) = _jacobian_tw(tw, tw.jacobian, x, p)
isinplace(tw::TWModel) = false
@inline save_solution(::WrapTW, x, p) = x
@inline is_symmetric(::WrapTW) = false
@inline has_adjoint(::WrapTW) = false
R01(tw::WrapTW, x, p) = R01(FiniteDifferences(), tw, x, p)
R02(tw::WrapTW, x, p) = R02(FiniteDifferences(), tw, x, p)
R11(tw::WrapTW, x, p, dx) = R11(FiniteDifferences(), tw, x, p, dx)
dF(tw::WrapTW, x, p, dx1) = get_discretization(tw)(x, p, dx1)
d2F(tw::WrapTW, x, p, dx1, dx2) = ForwardDiff.derivative(t -> dF(tw, x .+ t .* dx2, p, dx1), 0)
d3F(tw::WrapTW, x, p, dx1, dx2, dx3) = ForwardDiff.derivative(t -> d2F(tw, x .+ t .* dx3, p, dx1, dx2), 0)

function update!(wrap::WrapTW, iter, state::ContState)
    prob = get_discretization(wrap)
    success = converged(state)
    bisection = in_bisection(state)
    update_section_every_step = prob.update_section_every_step
    step = state.step
    z = getsolution(state)
    if success && mod_counter(step, update_section_every_step) && bisection == false
        @debug "[Wave problem] update section"
        # Trapeze and Shooting need the parameters for section update:
        updatesection!(prob, z.u)
    end
    return true
end

_generate_jacobian(probPO::TWModel, J::Union{MatrixFree, AutoDiffMF, FullLU, FiniteDifferences, AutoDiff}, o, pars; k...) = J

_jacobian_tw(prob::WrapTW, ::AutoDiff, x, p) = ForwardDiff.jacobian(z -> residual(prob, z, p), x)
_jacobian_tw(prob::WrapTW, ::FullLU, x, p) = get_discretization(prob)(Val(:JacFullSparse), x, p)
_jacobian_tw(prob::WrapTW, ::MatrixFree, x, p) = (dx ->  get_discretization(prob)(x, p, dx))

function _jacobian_tw(prob::WrapTW, ::AutoDiffMF, x, p)
    return dx -> ForwardDiff.derivative(z -> residual(prob, x .+ z .* dx, p), 0)
end

function _jacobian_tw(prob::WrapTW, ::FiniteDifferences, x, p)
    return finite_differences(z -> residual(prob, z, p), x)
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
apply_mass_matrix(wrap::WrapTW, x, p, dx) = apply(getmassmatrix(wrap, x, p), dx)

function getmassmatrix(wrap::WrapTW, x::AbstractVector, p)
    twprob = get_discretization(wrap)
    # @error "getmassmatrix(::WrapTW"
    Mass = getmassmatrix(twprob.prob_vf, x, p)
    if Mass isa IdentityOperator
        N = length(x)
        return SPA.spdiagm(vcat(ones(N - twprob.nc), zeros(twprob.nc)))
    else
        return SPA.blockdiag(Mass, SPA.sparse(LA.I, twprob.nc, twprob.nc))
    end
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function newton(tw::TWModel, 
                orbitguess, 
                optn::NewtonPar; 
                δ = convert(VI.scalartype(orbitguess), getdelta(tw)),
                kwargs...)
    jacobianTW = tw.jacobian
    jac = _generate_jacobian(tw, jacobianTW, orbitguess, getparams(tw); δ)
    wrap = WrapTW(tw, jac, orbitguess, BifurcationKit.record_from_solution(tw.prob_vf), plot_solution(tw.prob_vf))
    return solve(wrap, Newton(), optn; kwargs...,)
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function re_make(prob::Union{AbstractWaveProblem, AbstractWrapperPeriodicOrbitProblem};
                u0 = prob.u0,
                lens = getlens(prob),
                params = getparams(prob),
                record_from_solution = prob.recordFromSolution,
                plot_solution = plot_solution(prob))
    disc = re_make(get_discretization(prob); params)
    setproperties(prob; disc, u0, plotSolution = plot_solution, recordFromSolution = record_from_solution)
end

function record_from_solution(iter::ContIterable{TravellingWaveCont},
                              state::AbstractContinuationState)
    probTW = getprob(iter)
    if probTW.recordFromSolution isa Nothing
        return (s = getx(state)[end],)
    else
        return probTW.recordFromSolution(getx(state), (prob = get_discretization(probTW), p = getp(state)); iter, state)
    end
end

"""
$(TYPEDSIGNATURES)

Continuation of waves (travelling or rotating waves) computed with the freezing
method of [`TWModel`](@ref). The problem `prob` is wrapped in a `WrapTW` and
continued with the standard [`continuation`](@ref) machinery (with
`kind = TravellingWaveCont()`).

# Arguments
- `prob::TWModel`: frozen wave problem built with [`TWModel`](@ref).
- `orbitguess`: initial guess `vcat(u₀, s₀)`: the state followed by the
  speed(s) `s₀` (one per frozen symmetry). A guess is provided, *e.g.*, by
  `newton(prob, vcat(u₀, s₀), options)`.
- `alg::AbstractContinuationAlgorithm`: continuation algorithm (*e.g.* `PALC()`).
- `contParams::ContinuationPar`: continuation options.

# Keyword arguments
- `eigsolver = GEigenWave()`: eigensolver used to assess the stability of the
  wave. Two wave-specific eigensolvers are available:
  * `GEigenWave()` (default): the stability is obtained from the generalized
    eigenvalue problem of the full frozen jacobian with the mass matrix
    `blockdiag(M, I_nc)` (identity on the speed components), see
    [Wave stability](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/intro_wave/#Wave-stability).
    The mass matrix `M` is the one of the underlying vector field (or the
    identity) and the mass `diag(1, …, 1, 0)` (zero weight on the speed
    components) is used to convert the eigensolver stored in
    `contParams.newton_options.eigsolver` into a generalized one.
  * `EigenWave(eigsolver, matrix_free)`: the stability is obtained from the
    eigenvalues of `J + η⋅∂` (jacobian of the frozen system `F - s⋅∂` with the
    speed(s) `η` fixed); the phase constraints are removed from `J`. `eigsolver`
    is the underlying eigensolver (defaulting to the one in
    `contParams.newton_options.eigsolver`) and `matrix_free` selects a
    matrix-free evaluation of the jacobian-vector products.
- `record_from_solution = nothing`: by default, records the speed(s) `s`.
- `plot_solution = plot_solution(prob.prob_vf)`: plotting callback.
- `δ = getdelta(prob)`: step used when the jacobian of `prob` requires finite
  differences.
- additional keyword arguments are forwarded to [`continuation`](@ref).

# See also
- [`TWModel`](@ref), [`GEigenWave`](@ref), [`EigenWave`](@ref)
"""
function continuation(prob::TWModel,
                    orbitguess, 
                    alg::AbstractContinuationAlgorithm, 
                    contParams::ContinuationPar;
                    eigsolver = GEigenWave(),
                    record_from_solution = nothing,
                    plot_solution = plot_solution(prob.prob_vf),
                    δ = convert(VI.scalartype(orbitguess), getdelta(prob)),
                    kwargs...)
    # define the mass matrix for the eigensolver
    N = length(orbitguess)
    B = SPA.spdiagm(vcat(ones(N - prob.nc), zeros(prob.nc)))
    # convert eigsolver to generalised one
    old_eigsolver = contParams.newton_options.eigsolver
    contParamsWave = @set contParams.newton_options.eigsolver = convert_to_wave_eigen_solver(eigsolver, old_eigsolver, B)
    # this is to remove this part from the arguments passed to continuation
    jac = _generate_jacobian(prob, prob.jacobian, orbitguess, getparams(prob); δ)
    probwp = WrapTW(prob, jac, orbitguess, plot_solution, record_from_solution)
    return continuation(probwp, alg, contParamsWave; kind = TravellingWaveCont(), kwargs...,)
end
