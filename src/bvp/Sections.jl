abstract type AbstractSection end

update!(sect::AbstractSection) = error("update! is not implemented for $(typeof(sect)).")

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SectionSS for Standard Shooting
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function sectionShooting(x::AbstractArray,
                        T,
                        normal::AbstractArray,
                        center::AbstractArray)
    # we only constrain the first point to lie on a specific hyperplane
    # this avoids the temporary xc - centers
    return (VI.inner(x, normal) - VI.inner(center, normal)) * T
end

"""
$(TYPEDEF)

This composite type (named for Section Standard Shooting) encodes a type of section implemented by a single hyperplane. It can be used in conjunction with [`Shooting`](@ref). The hyperplane is defined by a point `center` and a `normal`.

# Internal fields
$(TYPEDFIELDS)

# Constructor(s)
    SectionSS(normal, center)
"""
struct SectionSS{Tn} <: AbstractSection
    "Normal to define hyperplane."
    normal::Tn

    "Representative point on hyperplane."
    center::Tn
end

# Functor definitions
(sect::SectionSS)(u, T) = sectionShooting(u, T, sect.normal, sect.center)
function (sect::SectionSS)(u, T::𝒯, du, dT::𝒯) where 𝒯
    return sect(u, one(𝒯)) * dT + VI.inner(du, sect.normal) * T
end

_isempty(::SectionSS{Tn}) where {Tn} = (Tn == Nothing)

"""
$(TYPEDSIGNATURES)

Update the field of `SectionSS`, useful during continuation procedure for updating the section.
"""
function update!(sect::SectionSS, normal, center)
    _copyto!(sect.normal, normal)
    _copyto!(sect.center, center)
    sect
end

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SectionTrapeze for Trapeze method
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
$(TYPEDEF)

Phase condition section for the Trapeze method.

# Internal fields
$(TYPEDFIELDS)
"""
struct SectionTrapeze{Tvectype} <: AbstractSection
    "Previous periodic orbit or reference vector for the phase constraint equation."
    ϕ::Tvectype

    "Used in the section for the phase constraint equation (e.g. previous point on branch)."
    xπ::Tvectype
end

"""
$(TYPEDSIGNATURES)

Update the field of `SectionTrapeze`, useful during continuation procedure for updating the section.
"""
function update!(sect::SectionTrapeze, ϕ, xπ)
    _copyto!(sect.ϕ, ϕ)
    _copyto!(sect.xπ, xπ)
    return sect
end

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SectionCollocation for Collocation method
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
$(TYPEDEF)

Phase condition section for the Collocation method.

# Internal fields
$(TYPEDFIELDS)
"""
struct SectionCollocation{Tϕ, T∂ϕ} <: AbstractSection
    "Used to set a section for the phase constraint equation."
    ϕ::Tϕ

    "Derivative of ϕ, cached to avoid recomputation."
    ∂ϕ::T∂ϕ
end

"""
$(TYPEDSIGNATURES)

Update the field of `SectionCollocation`, useful during continuation procedure for updating the section.
"""
function update!(sect::SectionCollocation, ϕ, ∂ϕ)
    _copyto!(sect.ϕ, ϕ)
    _copyto!(sect.∂ϕ, ∂ϕ)
    return sect
end
