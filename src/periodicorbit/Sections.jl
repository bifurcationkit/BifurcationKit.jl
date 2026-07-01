#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function _section_hyp!(out, x, normals, centers, radius)
    for ii in eachindex(normals)
        if norm(x-centers[ii]) < radius
            out[ii] = VI.inner(normals[ii], x) - VI.inner(normals[ii], centers[ii])
        else
            out[ii] = 1
        end
    end
    out
end

"""
$(TYPEDEF)

This composite type (named for SectionPoincaréShooting) encodes a type of Poincaré sections implemented by hyperplanes. It can be used in conjunction with [`PoincareShooting`](@ref). Each hyperplane is defined par a point (one example in `centers`) and a normal (one example in `normals`). See [1] for more details.

# Internal fields
$(TYPEDFIELDS)

# Constructor(s)
    SectionPS(normals, centers)

# Ref(s)
[1] J., M. Net, B. Garcı́a-Archilla, and C. Simó. “Newton–Krylov Continuation of Periodic Orbits for Navier–Stokes Flows.” Journal of Computational Physics 201, no. 1 (November 20, 2004): 13–33. https://doi.org/10.1016/j.jcp.2004.04.018.

"""
struct SectionPS{Tn, Tc, Tnb, Tcb, Tr} <: AbstractSection
    "number of hyperplanes"
    M::Int64
    "normals to define hyperplanes"
    normals::Tn
    "representative point on each hyperplane"
    centers::Tc
    "indices to be removed in the operator Ek"
    indices::Vector{Int64}

    normals_bar::Tnb
    centers_bar::Tcb

    radius::Tr

    function SectionPS(normals, centers; radius = Inf)
        @assert length(normals) == length(centers)
        M = length(normals)
        indices = zeros(Int64, M)
        for ii in 1:M
            indices[ii] = _select_index(normals[ii])
        end
        nbar = [R(normals[ii], indices[ii]) for ii in 1:M]
        cbar = [R(centers[ii], indices[ii]) for ii in 1:M]

        return new{typeof(normals), typeof(centers), typeof(nbar), typeof(cbar), typeof(radius)}(M, normals, centers, indices, nbar, cbar, radius)
    end

    SectionPS(M = 0) = new{Nothing, Nothing, Nothing, Nothing, Float64}(M, nothing, nothing, Int64[], nothing, nothing, 100.)
end

get_normals(sect::SectionPS) = sect.normals
get_centers(sect::SectionPS) = sect.centers
get_normals_bar(sect::SectionPS) = sect.normals_bar
get_centers_bar(sect::SectionPS) = sect.centers_bar

(hyp::SectionPS)(out, u) = _section_hyp!(out, u, hyp.normals, hyp.centers, hyp.radius)
_isempty(::SectionPS{Tn, Tc, Tnb, Tcb}) where {Tn, Tc, Tnb, Tcb} = (Tn == Nothing) || (Tc == Nothing)

_select_index(v) = argmax(abs.(v))
# ==================================================================================================
function _duplicate!(x::AbstractVector)
    n = length(x)
    for ii in 1:n
        push!(x, _copy(x[ii]))
    end
    x
end
_duplicate(x::AbstractVector) = _duplicate!(_copy(x))
_duplicate(hyp::SectionPS) = SectionPS(_duplicate(hyp.normals), _duplicate(hyp.centers))
# ==================================================================================================
"""
$(TYPEDSIGNATURES)

Update the hyperplanes saved in `hyp`.
"""
function update!(hyp::SectionPS, normals, centers)
    M = hyp.M
    @assert length(normals) == M "Wrong number of normals!"
    @assert length(centers) == M "Wrong number of centers!"
    for ii in 1:M
        hyp.normals[ii] .= normals[ii]
        hyp.centers[ii] .= centers[ii]
        k = _select_index(normals[ii])
        hyp.indices[ii] = k
        R!(hyp.normals_bar[ii], normals[ii], k)
        R!(hyp.centers_bar[ii], centers[ii], k)
    end
    return hyp
end

# Operateur Rk from the paper above
@views function R!(out, x::AbstractVector, k::Int)
    out[1:k-1] .= x[1:k-1]
    out[k:end] .= x[k+1:end]
    return out
end

R!(hyp::SectionPS, out, x::AbstractVector, k::Int) = R!(out, x, hyp.indices[k])
R(x::AbstractVector, k::Int) = R!(similar(x, length(x) - 1), x, k)
R(hyp::SectionPS, x::AbstractVector, k::Int) = R!(hyp, similar(x, length(x) - 1), x, k)

# differential of R
dR!(hyp::SectionPS, out, dx::AbstractVector, k::Int) = R!(hyp, out, dx, k)
dR(hyp::SectionPS, dx::AbstractVector, k::Int) = R(hyp, dx, k)

# Operateur Ek from the paper above
function E!(hyp::SectionPS, out, xbar::AbstractVector, ii::Int)
    @assert length(xbar) == length(hyp.normals[1]) - 1 "Wrong size for the projector / expansion operators, length(xbar) = $(length(xbar)) and length(normal) = $(length(hyp.normals[1]))"
    k = hyp.indices[ii]
    nbar  = hyp.normals_bar[ii]
    xcbar = hyp.centers_bar[ii]
    coord_k = hyp.centers[ii][k] - (VI.inner(nbar, xbar) - VI.inner(nbar, xcbar)) / hyp.normals[ii][k]

    @views out[1:k-1] .= xbar[1:k-1]
    @views out[k+1:end] .= xbar[k:end]
    out[k] = coord_k
    return out
end

function E(hyp::SectionPS, xbar::AbstractVector, ii::Int)
    out = similar(xbar, length(xbar) + 1)
    E!(hyp, out, xbar, ii)
end

# differential of E!
function dE!(hyp::SectionPS, out, dxbar::AbstractVector, ii::Int)
    k = hyp.indices[ii]
    nbar = hyp.normals_bar[ii]
    coord_k = - VI.inner(nbar, dxbar) / hyp.normals[ii][k]
    @views out[1:k-1]   .= dxbar[1:k-1]
    @views out[k+1:end] .= dxbar[k:end]
    out[k] = coord_k
    return out
end

function dE(hyp::SectionPS, dxbar::AbstractVector, ii::Int)
    out = similar(dxbar, length(dxbar) + 1)
    dE!(hyp, out, dxbar, ii)
end
