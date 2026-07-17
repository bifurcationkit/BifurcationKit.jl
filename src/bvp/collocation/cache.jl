import FastGaussQuadrature
using LinearAlgebra
import DocStringExtensions: TYPEDFIELDS, TYPEDSIGNATURES
import ForwardDiff
using PreallocationTools: DiffCache, get_tmp


"""
    cache = MeshCollocationCache(Ntst::Int, m::Int, Ty = Float64)

Structure to hold the cache for the collocation method. More precisely, it starts from a partition of [0, 1] based on the mesh points:

    0 = τ₁ < τ₂ < ... < τₙₜₛₜ₊₁ = 1

On each mesh interval [τⱼ, τⱼ₊₁] mapped to [-1, 1], a Legendre polynomial of degree m is formed.

# Internal fields
$(TYPEDFIELDS)

# Constructor

    MeshCollocationCache(Ntst::Int, m::Int, 𝒯 = Float64)

- `Ntst` number of time steps.
- `m` degree of the collocation polynomials.
- `Ty` type of the time variable.
"""
struct MeshCollocationCache{𝒯}
    "Coarse mesh size."
    Ntst::Int
    "Collocation degree, usually named `m`."
    degree::Int
    "Lagrange matrix."
    lagrange_vals::Matrix{𝒯}
    "Lagrange matrix for derivative."
    lagrange_∂::Matrix{𝒯}
    "Gauss nodes."
    gauss_nodes::Vector{𝒯}
    "Gauss weights. Useful to compute integrals."
    gauss_weight::Vector{𝒯}
    "Values of the coarse mesh, named τj. This can be adapted."
    τs::Vector{𝒯}
    "Values of collocation points, named σj. These are fixed."
    σs::Vector{𝒯}
    "Full mesh containing both the coarse mesh and the collocation points."
    full_mesh::Vector{𝒯}
end

@inline n_mesh_pts(m::Int, Ntst::Int) = 1 + m * Ntst
@inline n_mesh_pts(cache::MeshCollocationCache) = n_mesh_pts(cache.degree, cache.Ntst)


function MeshCollocationCache(Ntst::Int, m::Int, 𝒯 = Float64)
    τs = LinRange{𝒯}( 0, 1, Ntst + 1) |> collect
    σs = LinRange{𝒯}(-1, 1, m + 1) |> collect
    L, ∂L, zg, wg = compute_legendre_matrices(σs)
    cache = MeshCollocationCache{𝒯}(Ntst, m, L, ∂L, zg, wg, τs, σs, zeros(𝒯, n_mesh_pts(m, Ntst)))
    # save the mesh where we removed redundant timing
    cache.full_mesh .= get_times(cache)
    return cache
end

@inline Base.eltype(::MeshCollocationCache{𝒯}) where {𝒯} = 𝒯
@inline Base.size(cache::MeshCollocationCache) = (cache.degree, cache.Ntst)
@inline get_Ls(cache::MeshCollocationCache) = (cache.lagrange_vals, cache.lagrange_∂)
@inline getmesh(cache::MeshCollocationCache) = cache.τs
@inline get_mesh_coll(cache::MeshCollocationCache) = cache.σs
@inline get_full_mesh(cache::MeshCollocationCache) = cache.full_mesh
@inline get_gauss_nodes(cache::MeshCollocationCache) = cache.gauss_nodes
@inline get_gauss_weight(cache::MeshCollocationCache) = cache.gauss_weight
get_max_time_step(cache::MeshCollocationCache) = maximum(diff(getmesh(cache)))
@inline _τj(σ, τⱼ₊₁, τⱼ) = τⱼ + (1 + σ)/2 * (τⱼ₊₁ - τⱼ) # for σ ∈ [-1, 1], τj ∈ [τⱼ, τs[j+1]]
@inline τj(σ, τs, j) = _τj(σ, τs[j+1], τs[j])
"Get the `σ` corresponding to `τ` in the interval (τs[j], τs[j+1])"
@inline σj(τ, τs, j) = (2*τ - τs[j] - τs[j + 1])/(τs[j + 1] - τs[j]) # for τ ∈ [τs[j], τs[j+1]], σj ∈ [-1, 1]

"""
$(TYPEDSIGNATURES)

Evaluate Lagrange polynomial at `x`.
"""
function lagrange(i::Int, x, z)
    nz = length(z)
    l = one(z[1])
    for k in 1:(i-1)
        l = l * (x - z[k]) / (z[i] - z[k])
    end
    for k in (i+1):nz
        l = l * (x - z[k]) / (z[i] - z[k])
    end
    return l
end

dlagrange(i, x, z) = ForwardDiff.derivative(x -> lagrange(i, x, z), x)

# should accept a range, ie σs = LinRange(-1, 1, m + 1)
function compute_legendre_matrices(σs::AbstractVector{𝒯}) where {𝒯}
    m = length(σs) - 1
    zg, wg = FastGaussQuadrature.gausslegendre(m)
    @assert length(zg) == m
    L  = zeros(𝒯, m + 1, m)
    ∂L = zeros(𝒯, m + 1, m)
    for j in 1:m+1
        for (i,z) in pairs(zg)
             L[j, i] =  lagrange(j, z, σs)
            ∂L[j, i] = dlagrange(j, z, σs)
        end
    end
    return (;L, ∂L, zg, wg)
end

"""
$(TYPEDSIGNATURES)

Return the times at which the problem is evaluated.

!!! danger "This is a bit tricky"
    In order to remove the obvious continuity conditions at the coarse mesh border, we form the mesh by evaluating `_τj(σs[l], τs[j+1], τs[j])` for `l=2:m+1`. The only point not present is then `t=0` which is added to the list.
"""
function get_times(cache::MeshCollocationCache{𝒯}) where {𝒯}
    m, Ntst = size(cache)
    tsvec = zeros(𝒯, n_mesh_pts(m, Ntst))
    τs = cache.τs
    σs = cache.σs
    ind = 2
    @inbounds for j in 1:Ntst
        for l in 2:m+1
            t = _τj(σs[l], τs[j+1], τs[j])
            tsvec[ind] = t
            ind += 1
        end
    end
    return tsvec
end

function update_mesh!(cache::MeshCollocationCache, τs)
    cache.τs .= τs
    cache.full_mesh .= get_times(cache)
end

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
Cache to remove allocations from Collocation
"""
struct CollocationCache{𝒯}
    gj::DiffCache{Matrix{𝒯},  Vector{𝒯}}
    gi::DiffCache{Matrix{𝒯},  Vector{𝒯}}
    ∂gj::DiffCache{Matrix{𝒯}, Vector{𝒯}}
    uj::DiffCache{Matrix{𝒯},  Vector{𝒯}}
    vj::DiffCache{Matrix{𝒯},  Vector{𝒯}}
    tmp::DiffCache{Vector{𝒯}, Vector{𝒯}}
    ∇phase::Vector{𝒯}
    In::Matrix{Bool}
end

"""
$(TYPEDSIGNATURES)

[Internal] In case `save_mem = true`, we do not allocate the identity matrix. Indeed think about `n = 100_000`.
"""
function CollocationCache(𝒯::Type, Ntst::Int, n::Int, m::Int, save_mem = false)
    gj  = DiffCache(zeros(𝒯, n, m))
    gi  = DiffCache(zeros(𝒯, n, m))
    ∂gj = DiffCache(zeros(𝒯, n, m))
    uj  = DiffCache(zeros(𝒯, n, m + 1))
    vj  = DiffCache(zeros(𝒯, n, m + 1))
    tmp = DiffCache(zeros(𝒯, n))
    ∇phase = zeros(𝒯, n * n_mesh_pts(m, Ntst))
    In = Array(I(save_mem ? 1 : n))
    return CollocationCache(gj, gi, ∂gj, uj, vj, tmp, ∇phase, In)
end
