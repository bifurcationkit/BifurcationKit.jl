"""
$(TYPEDEF)

Construct a Poincaré return map `Π` to an hyperplane `Σ` from an `AbstractPeriodicOrbitProblem`.
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

@inline get_mesh_size(Π::PoincaréMap{ <: WrapPOSh}) = get_mesh_size(Π.probpo.prob) - 1

@views function get_time_slices(Π::PoincaréMap{ <: WrapPOSh}, x::AbstractVector)
    M = get_mesh_size(Π)
    if M == 0
        return x
    end
    N = div(length(x) - 1, M)
    return reshape(x[begin:end-1], N, M)
end

function (Π::PoincaréMap)(xₛ, par)
    solΠ = _solve(Π, xₛ, par)
    _extend(Π, solΠ, par)
end

"""
$(TYPEDSIGNATURES)

Constructor for the Poincaré return map. Return a `PoincaréMap`.
"""
function PoincareMap(wrap::WrapPOSh, po, par, optn)
    sh = wrap.prob
    Π = PoincaréMap(wrap, po, deepcopy(wrap.prob.section), optn)
    poc = get_time_slices(sh, po)
    @views update!(Π.Σ, vf(sh.flow, poc[:, begin], par), poc[:, begin])
    Π.Σ.normal ./= norm(sh.section.normal)
    return Π
end

"""
$(TYPEDSIGNATURES)

Constructor for the Poincaré return map. Return a `PoincaréMap`.
"""
function PoincareMap(wrap::WrapPOColl, po, par, optn)
    coll = wrap.prob
    N, m, Ntst = size(coll)
    Σ = SectionSS(rand(N), rand(N))
    poc = get_time_slices(coll, po)
    @views update!(Σ, residual(coll.prob_vf, po[1:N], par), po[1:N])
    Σ.normal ./= norm(Σ.normal)
    return PoincaréMap(wrap, po, Σ, optn)
end

function poincaré_functional(Π::PoincaréMap{ <: WrapPOSh }, x, par, x₁)
    sh = Π.probpo.prob

    M = get_mesh_size(Π)
    N = div(length(Π.po) - 1, M+1)
    T⁰ = getperiod(sh, Π.po)  # period of the reference periodic orbit
    tₘ = _extract_period(x)   # estimate of the last bit for the return time

    # extract the orbit guess and reshape it into a matrix as it's more convenient to handle
    poc = get_time_slices(sh, Π.po)
    # unknowns are po₁, po₂, ..., poₘ, period
    @assert size(poc) == (N, M+1)

    xc = get_time_slices(Π, x)
    # unknowns are x₂,...,xₘ,tΣ

    # variable to hold the computed result
    out = similar(x, typeof(x[1] * x₁[1] * _get(par, getlens(sh))))
    outc = get_time_slices(Π, out)

    if M == 0
        𝒯 = typeof(x[1] * x₁[1])
        # this type promotion is for ForwardDiff
        out[1] = Π.Σ(evolve(sh.flow, 𝒯.(x₁), par, tₘ * T⁰).u, T⁰)
        return out
    end

    if ~isparallel(sh)
        outc[:, 1] .= evolve(sh.flow, x₁, par, sh.ds[1] * T⁰).u .- xc[:, 1]
        for ii in 1:M-1
            outc[:, ii+1] .= evolve(sh.flow, xc[:, ii], par, sh.ds[ii] * T⁰).u .- xc[:, ii+1]
        end
        out[end] = Π.Σ(evolve(sh.flow, xc[:, M], par, tₘ * T⁰).u, T⁰)
    else
        # call jacobian of the flow
        solOde = evolve(sh.flow, hcat(x₁, xc), par, sh.ds .* T⁰)
        for ii in 1:M
            outc[:, ii] .= @views solOde[ii][2] .- xc[:, ii]
        end
        out[end] = Π.Σ(evolve(sh.flow, xc[:, M], par, tₘ * T⁰)[1][2], T⁰)
    end
    out
end

function _solve(Π::PoincaréMap{ <: WrapPOSh}, xₛ, par)
    @assert (Π.po isa AbstractVector) "The case of a general AbstractArray for the state space is not handled yet."
    # xₛ is close to / belongs to the hyperplane Σ
    # for x near po, this computes the poincare return map
    # get the size of the state space
    sh = Π.probpo.prob
    M = get_mesh_size(sh)
    N = div(length(Π.po) - 1, M)
    # we construct the initial guess
    x₀ = Π.po[N+1:end]
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

function _extend(Π::PoincaréMap{ <: WrapPOSh }, solΠ, par)
    sh = Π.probpo.prob
    # we get the return time
    T⁰ = getperiod(sh, Π.po)
    tₘ = _extract_period(solΠ)
    tᵣ = getperiod(sh, Π.po) + (tₘ - sh.ds[end]) * T⁰
    # we get the return point
    M = get_mesh_size(sh)
    if M == 1
        xᵣ = evolve(sh.flow, xₛ, par, tₘ * T⁰).u
    elseif ~isparallel(sh)
        xᵣ = evolve(sh.flow, get_time_slices(Π, solΠ)[:, end], par, tₘ * T⁰).u
    else
        xᵣ = evolve(sh.flow, get_time_slices(Π, solΠ)[:, end], par, tₘ * T⁰)[1].u
    end
    return (u = xᵣ, t = tᵣ)
end

@views function poincaré_functional(Π::PoincaréMap{ <: WrapPOColl }, u, par, x₁)
    # collocation problem
    coll = Π.probpo.prob
    N,_,_ = size(coll)

    uc = get_time_slices(coll, u)
    T = getperiod(coll, u, nothing)
    𝒯 = promote_type(VI.scalartype(u), VI.scalartype(x₁))
    result = 𝒯.(u)
    resultc = get_time_slices(coll, result)
    po_residual_bare!(coll, resultc, uc, T, get_Ls(coll.mesh_cache), par)
    resultc[:, end] .= x₁ .- uc[:, 1]
    return vcat(vec(resultc), Π.Σ(u[end-N:end-1], T))
end

function _solve(Π::PoincaréMap{ <: WrapPOColl }, xₛ, par)
    # xₛ is close to / belongs to the hyperplane Σ
    # for x near po, this computes the poincare return map
    # we construct the initial guess
    x₀ = Π.po

    mapΠ(x, p) = poincaré_functional(Π, x, p, xₛ)
    probΠ = BifurcationProblem(mapΠ,
                                x₀,
                                par)
    solΠ = solve(probΠ, Newton(), NewtonPar())
    ~solΠ.converged && @error "Newton failed!! We did not succeed in computing the Poincaré return map. Residuals = $(solΠ.residuals)"
    return solΠ.u
end

function _extend(Π::PoincaréMap{ <: WrapPOColl }, solΠ, par)
    coll = Π.probpo.prob
    N,_,_ = size(coll)
    T⁰ = getperiod(coll, Π.po)
    tₘ = _extract_period(solΠ)
    tᵣ = tₘ
    return (u = solΠ[end-N:end-1], t = tᵣ)
end

function d1F(Π::PoincaréMap{ <: WrapPOSh }, x, pars, h)
    @assert length(x) == length(h)
    sh = Π.probpo.prob
    normal = Π.Σ.normal

    Πx, tΣ = Π(x, pars)
    Fx = vf(sh.flow, Πx, pars)
    y = evolve(sh.flow, Val(:SerialdFlow), x, pars, h, tΣ).du
    # differential of return time
    ∂th = - LA.dot(normal, y) / LA.dot(normal, Fx)
    out = @. y + ∂th * Fx
    return (u = out, t = ∂th)
end

"""
$(TYPEDSIGNATURES)

Compute the monodromy matrix of the Poincaré Return Map. It returns a `Matrix{𝒯}`.
"""
function jacobian(Π::PoincaréMap{ <: WrapPOSh }, x::AbstractVector{𝒯}, pars) where {𝒯}
    sh = Π.probpo.prob
    normal = Π.Σ.normal

    Πx, tΣ = Π(x, pars)
    Fx = vf(sh.flow, Πx, pars)
    # monodromy matrix
    N = length(x)
    𝒯p = promote_type(𝒯, typeof(_get(pars, getlens(sh))))
    Mono = zeros(𝒯p, N, N)
    h = zeros(𝒯p, N)
    for i in eachindex(h)
        h[i] += 1
        y = evolve(sh.flow, Val(:SerialdFlow), x, pars, h, tΣ).du
        # differential of return time
        ∂th = - LA.dot(normal, y) / LA.dot(normal, Fx)
        out = @. y + ∂th * Fx
        Mono[:, i] .= out
        h[i] -= 1
    end
    return Mono
end

function d2F(Π::PoincaréMap{ <: WrapPOSh }, x, pars, h₁, h₂)
    @assert length(x) == length(h₁) == length(h₂)
    sh = Π.probpo.prob
    normal = Π.Σ.normal
    VF(z) = vf(sh.flow, z, pars)
    dvf(z,h) = ForwardDiff.derivative(t -> VF(z .+ t .* h), 0)

    Πx, tΣ = Π(x, pars)
    Fx = vf(sh.flow, Πx, pars)
    ∂Πh2, ∂th2 = d1F(Π, x, pars, h₂) # not good, we recompute a lot

    ∂ϕ(z,h) = evolve(sh.flow, Val(:SerialdFlow), z, pars, h, tΣ).du
    ∂2ϕ(z,h1,h2) = ForwardDiff.derivative(t -> ∂ϕ(z .+ t .* h2, h1), 0)

    ∂ϕh1 = ∂ϕ(x,h₁)
    ∂2ϕh12 = ∂2ϕ(x,h₁,h₂)

    # differentials of return times
    ∂th1 = -LA.dot(normal, ∂ϕh1) / LA.dot(normal, Fx)
    y = ∂ϕ(x,h₂)

    y = dvf(Πx, ∂Πh2) .* ∂th1 .+
        ∂2ϕh12 .+ dvf(Πx, ∂ϕh1) .* ∂th2
    ∂2t = -LA.dot(normal, y) / LA.dot(normal, Fx)
    y .+= ∂2t .* Fx

    abs(LA.dot(normal, y)) > 1e-10 && @error "This dot product is not zero, $(abs(LA.dot(normal, y)))"

    return (u = y, t = ∂2t)
end

function d3F(Π::PoincaréMap{ <: WrapPOSh }, x, pars, h₁, h₂, h₃)
    @assert length(x) == length(h₁) == length(h₂) == length(h₃)
    sh = Π.probpo.prob
    normal = Π.Σ.normal
    Πx, tΣ = Π(x, pars)

    VF(z) = vf(sh.flow, z, pars)
    dvf(z,h) = ForwardDiff.derivative(t -> VF(z .+ t .* h), 0)
    d2vf(z,h1,h2) = ForwardDiff.derivative(t -> dvf(z .+ t .* h2, h1), 0)

    ∂ϕ(z,h) = evolve(sh.flow, Val(:SerialdFlow), z, pars, h, tΣ).du
    ∂2ϕ(z,h1,h2) = ForwardDiff.derivative(t -> ∂ϕ(z .+ t .* h2, h1), 0)
    ∂3ϕ(z,h1,h2,h3) = ForwardDiff.derivative(t -> ∂2ϕ( z .+ t .* h3, h1, h2), 0)

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

    abs(LA.dot(normal, out)) > 1e-10 && @error "This product is not zero $(abs(LA.dot(normal, out))) > 1e-10"
    return (u = out, t = ∂3t)
end
