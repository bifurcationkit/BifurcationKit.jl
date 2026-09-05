_vector_field_prm(sh::Shooting, x, p) = vector_field(sh.flow, x, p)
_vector_field_prm(coll::Collocation, x, p) = residual(coll.prob_vf, x, p)

"""
$(TYPEDEF)

Construct a Poincaré return map `Π` to an hyperplane `Σ` from an `AbstractBoundaryValueDiscretization`.
If the state space is of size `Nₓ x N𝕪`, then we can evaluate the map as `Π(xₛ, par)` where `xₛ ∈ Σ` is of size `Nₓ x N𝕪`.

# Internal fields
$(TYPEDFIELDS)
"""
struct PoincaréMap{Tp, Tpo, Ts <: AbstractSection, To}
    "periodic orbit problem."
    probpo::Tp
    "Periodic orbit."
    po::Tpo
    "section."
    Σ::Ts
    "Newton options."
    options::To
end

@inline get_mesh_size(Π::PoincaréMap{ <: PeriodicOrbitFunctionalSh}) = get_mesh_size(get_discretization(Π.probpo)) - 1

@views function get_time_slices(Π::PoincaréMap{ <: PeriodicOrbitFunctionalSh}, x::AbstractVector)
    M = get_mesh_size(Π)
    if M == 0
        return x
    end
    N = div(length(x) - 1, M)
    return reshape(x[begin:end-1], N, M)
end

function (Π::PoincaréMap)(xₛ, par)
    solΠ = _solve(Π, xₛ, par)
    _extend(Π, solΠ, par, xₛ)
end

"""
$(TYPEDSIGNATURES)

Constructor for the Poincaré return map from a shooting-based periodic orbit problem. The Poincaré section `Σ` is initialized by deep-copying the section from the discretization, then updated so that its center is the first point of the orbit `po` and its normal is the vector field at that point (subsequently normalized). Returns a `PoincaréMap`.
"""
function PoincareMap(wrap::PeriodicOrbitFunctionalSh, po, par, optn)
    sh = get_discretization(wrap)
    Π = PoincaréMap(wrap, _copy(po), deepcopy(sh.section), optn)
    po_m = get_time_slices(sh, Π.po)
    @views update!(Π.Σ, _vector_field_prm(sh, _copy(po_m[:, begin]), par), _copy(po_m[:, begin]))
    Π.Σ.normal ./= norm(sh.section.normal)
    return Π
end

"""
$(TYPEDSIGNATURES)

Constructor for the Poincaré return map. Return a `PoincaréMap`.
"""
function PoincareMap(wrap::PeriodicOrbitFunctionalColl, po, par, optn)
    coll = get_discretization(wrap)
    N, _, _ = size(coll)
    Σ = SectionSS(rand(N), rand(N))
    update!(Σ, residual(coll.prob_vf, po[1:N], par), po[1:N]) # do not put @views to prevent shadowing
    Σ.normal ./= norm(Σ.normal)
    return PoincaréMap(wrap, _copy(po), Σ, optn)
end

"""
$(TYPEDSIGNATURES)

Evaluate the Poincaré return map functional for a shooting-based periodic orbit discretization.
Returns the residual vector: the differences between consecutive time slices after flow integration, and the section condition at the final point.
"""
function poincaré_functional(Π::PoincaréMap{ <: PeriodicOrbitFunctionalSh }, x, par, x₁)
    sh = get_discretization(Π.probpo)

    M = get_mesh_size(Π)
    N = div(length(Π.po) - 1, M+1)
    T⁰ = getperiod(sh, Π.po)  # period of the reference periodic orbit
    tₘ = _extract_period(x)   # estimate of the last bit for the return time

    # extract the orbit guess and reshape it into a matrix as it is more convenient to handle
    po_m = get_time_slices(sh, Π.po)
    # unknowns are po₁, po₂, ..., poₘ, period
    @assert size(po_m) == (N, M+1)

    xm = get_time_slices(Π, x)
    # unknowns are x₂, ..., xₘ, tΣ

    # variable to hold the computed result
    out = similar(x, typeof(x[1] * x₁[1] * _get(par, getlens(sh))))
    outm = get_time_slices(Π, out)

    if M == 0
        𝒯 = typeof(x[1] * x₁[1])
        # this type promotion is for ForwardDiff
        out[1] = Π.Σ(_evolve_flow_prm(Π, 𝒯.(x₁), par, tₘ * T⁰).u, T⁰)
        return out
    end

    if ~isparallel(sh)
        outm[:, 1] .= _evolve_flow_prm(Π, x₁, par, sh.ds[1] * T⁰).u .- xm[:, 1]
        for ii in 1:M-1
            outm[:, ii+1] .= _evolve_flow_prm(Π, xm[:, ii], par, sh.ds[ii] * T⁰).u .- xm[:, ii+1]
        end
        out[end] = Π.Σ(_evolve_flow_prm(Π, xm[:, M], par, tₘ * T⁰).u, T⁰)
    else
        # call jacobian of the flow
        solOde = _evolve_flow_prm(Π, hcat(x₁, xm), par, sh.ds .* T⁰)
        for ii in 1:M
            outm[:, ii] .= @views solOde[ii][2] .- xm[:, ii]
        end
        out[end] = Π.Σ(_evolve_flow_prm(Π, xm[:, M], par, tₘ * T⁰)[1][2], T⁰)
    end
    out
end

"""
$(TYPEDSIGNATURES)

Solve the Poincaré return map for a shooting-based periodic orbit. Given a point `xₛ` on the section Σ, find the return point after flow integration by solving `poincaré_functional` with Newton's method.
"""
function _solve(Π::PoincaréMap{ <: PeriodicOrbitFunctionalSh}, xₛ, par)
    @assert (Π.po isa AbstractVector) "The case of a general AbstractArray for the state space is not handled yet."
    # xₛ is close to / belongs to the hyperplane Σ
    # for x near po, this computes the poincare return map
    # get the size of the state space
    sh = get_discretization(Π.probpo)
    M = get_mesh_size(sh)
    N = div(length(Π.po) - 1, M)
    # we construct the initial guess
    x₀ = Π.po[N+1:end] # careful! Don't use views otherwise Π.po is overwritten
    x₀[end] = sh.ds[end]
    mapΠ(x, p) = poincaré_functional(Π, x, p, xₛ)
    ## TODO needs a jacobian
    probΠ = BifurcationProblem(mapΠ,
                                x₀,
                                par)

    solΠ = solve(probΠ, Newton(), Π.options)
    ~solΠ.converged && @error "Newton failed!! We did not succeed in computing the Poincaré return map."
    return solΠ.u
end

"""
    _extend(Π::PoincaréMap{<:PeriodicOrbitFunctionalSh}, solΠ, par, xₛ) -> (u, t)

Extract the return point and return time from the Newton solution `solΠ` of the Poincaré return map. It returns the flow `xₛ` stopped at the section.

- For simple shooting (`sh.M == 1`), the return point is obtained by evolving `xₛ` for `tₘ * T⁰`.
- For multiple shooting (`sh.M > 1`), the return point is obtained by evolving the last intermediate
  mesh point of `solΠ` for `tₘ * T⁰`.
- The return time is `T⁰ + (tₘ - sh.ds[end]) * T⁰`.
"""
function _extend(Π::PoincaréMap{ <: PeriodicOrbitFunctionalSh }, solΠ, par, xₛ)
    sh = get_discretization(Π.probpo)
    T⁰ = getperiod(sh, Π.po)
    tₘ = _extract_period(solΠ)
    tᵣ = T⁰ + (tₘ - sh.ds[end]) * T⁰
    M = get_mesh_size(sh)
    if M == 1
        # xs = copy(get_time_slices(sh, Π.po)[:, 1])
        # xᵣ = _evolve_flow_prm(sh, xs, par, tₘ * T⁰).u
        xᵣ = _evolve_flow_prm(Π, xₛ, par, tₘ * T⁰).u
    elseif ~isparallel(sh)
        xᵣ = _evolve_flow_prm(Π, get_time_slices(Π, solΠ)[:, end], par, tₘ * T⁰).u
    else
        xᵣ = _evolve_flow_prm(Π, get_time_slices(Π, solΠ)[:, end], par, tₘ * T⁰)[1].u
    end
    return (u = xᵣ, t = tᵣ)
end

"""
$(TYPEDSIGNATURES)

Evaluate the Poincaré return map functional for a collocation-based periodic orbit discretization.
The unknown `u = (u₁, …, u_{m⋅Ntst+1}, T)` contains the time slices `uᵢ ∈ ℝᴺ` of the orbit and the return time `T` as its last entry. Given a starting point `x₁` on the section `Σ`, the functional returns the residual of the system:

- the collocation residual of the vector field on the interior mesh points (computed with `po_residual_bare!`), so the orbit satisfies `ẋ = F(x)`;
- the section condition `x₁ - u[:, 1] = 0` at the first time slice, so the orbit starts at `x₁`;
- the section condition `Σ(u[end-N:end-1], T) = 0` at the last time slice, so the return point lies on the hyperplane `Σ`.

The periodicity condition of the bare collocation problem is replaced by these section conditions.
"""
function poincaré_functional(Π::PoincaréMap{ <: PeriodicOrbitFunctionalColl }, u::AbstractVector, par, x₁)
    coll = get_discretization(Π.probpo)
    um = get_time_slices(coll, u)
    T = getperiod(coll, u, nothing) # we know that it hits the section after time T
    𝒯 = promote_type(VI.scalartype(u), VI.scalartype(x₁))
    result = similar(u, 𝒯)
    resultm = get_time_slices(coll, result)
    po_residual_bare!(coll, resultm, um, T, get_Ls(coll.mesh_cache), par)
    resultm[:, end] .= x₁ .- (@views um[:, 1])
    return vcat(vec(resultm), Π.Σ(um[:, end], T))
end

function _solve(Π::PoincaréMap{ <: PeriodicOrbitFunctionalColl }, xₛ, par)
    # xₛ is close to / belongs to the hyperplane Σ
    # for x near po, this function computes the poincare return map
    # we construct the initial guess
    x₀ = _copy(Π.po)
    mapΠ(x, p) = poincaré_functional(Π, x, p, xₛ)
    probΠ = BifurcationProblem(mapΠ,
                                x₀,
                                par)
    solΠ = solve(probΠ, Newton(), Π.options)
    ~solΠ.converged && @error "Newton failed!! We did not succeed in computing the Poincaré return map. Residuals = $(solΠ.residuals)"
    return solΠ.u
end

"""
    _extend(Π::PoincaréMap{<:PeriodicOrbitFunctionalColl}, solΠ, par, xₛ) -> (u, t)

Extract the return point and return time from the Newton solution `solΠ` of the Poincaré return map computed by collocation.

- The return point is the last time slice of the orbit: `solΠ[end-N:end-1]` where `N` is the state space dimension.
- The return time is the last entry `tₘ = solΠ[end]` of the solution, which is the period of the collocation solution (the time needed to go from `xₛ` back to the section `Σ`).
"""
function _extend(Π::PoincaréMap{ <: PeriodicOrbitFunctionalColl }, solΠ::AbstractVector, par, xₛ)
    coll = get_discretization(Π.probpo)
    N, _, _ = size(coll)
    T⁰ = getperiod(coll, Π.po)
    tₘ = _extract_period(solΠ)
    tᵣ = tₘ
    return (u = solΠ[end-N:end-1], t = tᵣ)
end

_evolve_flow_prm(Π::PoincaréMap{ <: PeriodicOrbitFunctionalSh}, args...) = evolve(get_discretization(Π.probpo).flow, args...)
_R01_evolve_flow_prm(Π::PoincaréMap{ <: PeriodicOrbitFunctionalSh}, x, pars, t, lens, p₀)      = R01(get_discretization(Π.probpo).flow, x, pars, t, lens, p₀)
_R11_evolve_flow_prm(Π::PoincaréMap{ <: PeriodicOrbitFunctionalSh}, x, pars, dx, t, lens, p₀)  = R11(get_discretization(Π.probpo).flow, x, pars, dx, t, lens, p₀)
_R20_evolve_flow_prm(Π::PoincaréMap{ <: PeriodicOrbitFunctionalSh}, x, pars, dx1, dx2, t)      = R20(get_discretization(Π.probpo).flow, x, pars, dx1, dx2, t)
_R30_evolve_flow_prm(Π::PoincaréMap{ <: PeriodicOrbitFunctionalSh}, x, pars, dx1, dx2, dx3, t) = R30(get_discretization(Π.probpo).flow, x, pars, dx1, dx2, dx3, t)
# JVP of ::PoincaréMap
function d1F(Π::PoincaréMap, x, pars, h)
    @assert length(x) == length(h)
    disc = get_discretization(Π.probpo)
    normal = Π.Σ.normal
    Πx, tΣ = Π(x, pars)
    Fx = _vector_field_prm(disc, Πx, pars)
    y = _evolve_flow_prm(Π, Val(:SerialdFlow), x, pars, h, tΣ).du
    # differential of return time
    ∂th = - LA.dot(normal, y) / LA.dot(normal, Fx)
    out = @. y + ∂th * Fx
    return (u = out, t = ∂th)
end

"""
$(TYPEDSIGNATURES)

Derivative ``\\frac{\\partial}{\\partial p}`` of the Poincaré return map ``\\Pi`` and return time `t_\\Sigma`.

Returns `(∂Π/∂p, ∂t/∂p)`.
"""
function R01(Π::PoincaréMap, x, pars)
    disc = get_discretization(Π.probpo)
    normal = Π.Σ.normal
    lens = getlens(disc)
    p₀ = _get(pars, lens)

    Πx, tΣ = Π(x, pars)
    Fx₀ = _vector_field_prm(disc, Πx, pars)

    ∂ϕ_∂p = _R01_evolve_flow_prm(Π, x, pars, tΣ, lens, p₀)
    ∂t_∂p = -LA.dot(normal, ∂ϕ_∂p) / LA.dot(normal, Fx₀)
    ∂Π_∂p = @. ∂ϕ_∂p + Fx₀ * ∂t_∂p
    return (u = ∂Π_∂p, t = ∂t_∂p)
end

"""
$(TYPEDSIGNATURES)

Mixed partial derivative ``\\frac{\\partial}{\\partial p} [\\frac{\\partial \\Pi}{\\partial x} \\cdot h_1]`` of the Poincaré return map.
"""
function R11(Π::PoincaréMap{ <: PeriodicOrbitFunctionalSh }, x, pars, h₁)
    disc = get_discretization(Π.probpo)
    normal = Π.Σ.normal
    lens = getlens(disc)
    p₀ = _get(pars, lens)

    Πx, tΣ = Π(x, pars)
    Fx₀ = _vector_field_prm(disc, Πx, pars)
    VF(z) = _vector_field_prm(disc, z, pars)
    dvf(z,h) = ForwardDiff.derivative(t -> VF(z .+ t .* h), 0)

    # ∂Π/∂p and ∂t/∂p
    r01 = R01(Π, x, pars)
    ∂Π_∂p = r01.u; ∂t_∂p = r01.t

    # d1F at the base point
    y₀ = _evolve_flow_prm(Π, Val(:SerialdFlow), x, pars, h₁, tΣ).du
    ∂th₀ = -LA.dot(normal, y₀) / LA.dot(normal, Fx₀)

    # ∂y/∂p = ∂²ϕ/∂x∂p·h₁ + dvf(Πx, y₀) * ∂t/∂p
    ∂²ϕ_∂x∂p_h₁ = _R11_evolve_flow_prm(Π, x, pars, h₁, tΣ, lens, p₀)
    ∂y_∂p = ∂²ϕ_∂x∂p_h₁ .+ dvf(Πx, y₀) .* ∂t_∂p

    # ∂Fx/∂p = dvf(Πx, ∂Π_∂p) + ∂VF/∂p
    # ∂VF_∂p = ForwardDiff.derivative(p -> _vector_field_prm(disc, Πx, set(pars, lens, p)), p₀)
    δ = getdelta(disc)
    ∂VF_∂p = (_vector_field_prm(disc, Πx, set(pars, lens, p₀ + δ)) .- 
              _vector_field_prm(disc, Πx, set(pars, lens, p₀ - δ))) ./ (2δ)
    ∂Fx_∂p = dvf(Πx, ∂Π_∂p) .+ ∂VF_∂p

    # ∂(∂th)/∂p = -(n·∂y/∂p * n·Fx₀ - n·y₀ * n·∂Fx_∂p) / (n·Fx₀)²
    n_Fx₀ = LA.dot(normal, Fx₀)
    ∂∂th_∂p = -(LA.dot(normal, ∂y_∂p) * n_Fx₀ - LA.dot(normal, y₀) * LA.dot(normal, ∂Fx_∂p)) / n_Fx₀^2

    return (u = (@. ∂y_∂p + ∂∂th_∂p * Fx₀ + ∂th₀ * ∂Fx_∂p), t = nothing)
end

"""
$(TYPEDSIGNATURES)

Compute the monodromy matrix of the Poincaré Return Map. It returns a dense matrix `Matrix{𝒯p}` where `𝒯p` is the promotion of the type `𝒯` of `x` with the type of the parameter value.
"""
function jacobian(Π::PoincaréMap{ <: PeriodicOrbitFunctionalSh }, x::AbstractVector{𝒯}, pars) where {𝒯}
    disc = get_discretization(Π.probpo)
    normal = Π.Σ.normal

    Πx, tΣ = Π(x, pars)
    Fx = _vector_field_prm(disc, Πx, pars)
    # monodromy matrix
    N = length(x)
    𝒯p = promote_type(𝒯, typeof(_get(pars, getlens(disc))))
    Mono = zeros(𝒯p, N, N)
    h = zeros(𝒯p, N)
    for i in eachindex(h)
        h[i] += 1
        y = _evolve_flow_prm(Π, Val(:SerialdFlow), x, pars, h, tΣ).du
        # differential of return time
        ∂th = - LA.dot(normal, y) / LA.dot(normal, Fx)
        out = @. y + ∂th * Fx
        Mono[:, i] .= out
        h[i] -= 1
    end
    return Mono
end

function d2F(Π::PoincaréMap{ <: PeriodicOrbitFunctionalSh }, x, pars, h₁, h₂)
    @assert length(x) == length(h₁) == length(h₂)
    disc = get_discretization(Π.probpo)
    normal = Π.Σ.normal
    VF(z) = _vector_field_prm(disc, z, pars)
    dvf(z, h) = ForwardDiff.derivative(t -> VF(z .+ t .* h), 0)

    Πx, tΣ = Π(x, pars)
    Fx = _vector_field_prm(disc, Πx, pars)
    ∂Πh2, ∂th2 = d1F(Π, x, pars, h₂) # not good, we recompute a lot

    ∂ϕ(z,h) = _evolve_flow_prm(Π, Val(:SerialdFlow), z, pars, h, tΣ).du
    ∂2ϕ(z,h1,h2) = _R20_evolve_flow_prm(Π, z, pars, h1, h2, tΣ)

    ∂ϕh1 = ∂ϕ(x,h₁)
    ∂2ϕh12 = ∂2ϕ(x,h₁,h₂)

    # differentials of return times
    ∂th1 = -LA.dot(normal, ∂ϕh1) / LA.dot(normal, Fx)
    y = dvf(Πx, ∂Πh2) .* ∂th1 .+ ∂2ϕh12 .+ dvf(Πx, ∂ϕh1) .* ∂th2
    ∂2t = -LA.dot(normal, y) / LA.dot(normal, Fx)
    y .+= ∂2t .* Fx

    abs(LA.dot(normal, y)) > 1e-10 && @error "This dot product <normal, y> is not zero, $(abs(LA.dot(normal, y)))"

    return (u = y, t = ∂2t)
end

function d3F(Π::PoincaréMap{ <: PeriodicOrbitFunctionalSh }, x, pars, h₁, h₂, h₃)
    @assert length(x) == length(h₁) == length(h₂) == length(h₃)
    disc = get_discretization(Π.probpo)
    normal = Π.Σ.normal
    Πx, tΣ = Π(x, pars)

    VF(z) = _vector_field_prm(disc, z, pars)
    dvf(z,h) = ForwardDiff.derivative(t -> VF(z .+ t .* h), 0)
    d2vf(z,h1,h2) = ForwardDiff.derivative(t -> dvf(z .+ t .* h2, h1), 0)

    ∂ϕ(z,h) = _evolve_flow_prm(Π, Val(:SerialdFlow), z, pars, h, tΣ).du
    ∂2ϕ(z,h1,h2) = _R20_evolve_flow_prm(Π, z, pars, h1, h2, tΣ)
    ∂3ϕ(z,h1,h2,h3) = _R30_evolve_flow_prm(Π, z, pars, h1, h2, h3, tΣ)

    _, ∂th1 = d1F(Π, x, pars, h₁)
    ∂Πh2, ∂th2 = d1F(Π, x, pars, h₂)
    ∂Πh3, ∂th3 = d1F(Π, x, pars, h₃)

    ∂2Πh23, ∂2t23 = d2F(Π, x, pars, h₂, h₃)
    ∂2t12  = d2F(Π, x, pars, h₁, h₂).t
    ∂2t13  = d2F(Π, x, pars, h₁, h₃).t

    Fx = VF(Πx)
    ∂2FΠh23 = d2vf(Πx, ∂Πh2, ∂Πh3)

    ∂ϕh1 = ∂ϕ(x,h₁)
    ∂ϕh2 = ∂ϕ(x,h₂)
    ∂ϕh3 = ∂ϕ(x,h₃)

    ∂2ϕ12  = ∂2ϕ(x, h₁, h₂)
    ∂3ϕ123 = ∂3ϕ(x, h₁, h₂, h₃)

    ∂2ϕt13 = ∂2ϕ(x, h₁, h₃) .+ dvf(Πx, ∂ϕh1) .* ∂th3

    y = ∂2FΠh23 .* ∂th1 .+
            dvf(Πx, ∂2Πh23) .* ∂th1 .+
            dvf(Πx, ∂Πh2) .* ∂2t13

    y .+= dvf(Πx, ∂Πh3) .* ∂2t12

    # differential ∂(d2ϕ)|t=t(x)
    y .+= ∂3ϕ123 .+ (d2vf(Πx, ∂ϕh1, ∂ϕh2) .+ dvf(Πx, ∂2ϕ12)) .* ∂th3

    # last bit
    y .+= d2vf(Πx, ∂ϕh1, ∂Πh3) .* ∂th2 .+
           dvf(Πx, ∂2ϕt13) .* ∂th2 .+
           dvf(Πx, ∂ϕh1) .* ∂2t23

    # we compute dτ(x)[h₁, h₂, h₃]
    ∂3t = -LA.dot(normal, y) / LA.dot(normal, Fx)
    out = y .+ ∂3t .* Fx

    abs(LA.dot(normal, out)) > 1e-10 && @error "This product <normal, out> is not zero $(abs(LA.dot(normal, out))) > 1e-10"
    return (u = out, t = ∂3t)
end
