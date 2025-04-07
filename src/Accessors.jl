const AllOpticTypes = Union{PropertyLens, IndexLens, ComposedOptic, typeof(identity)}

@inline _get(par, optic) = optic(par)

function _set(obj, optics::Tuple{<:AllOpticTypes, <:AllOpticTypes}, val::Tuple)
    obj2 = set(obj,  optics[1], val[1])
    obj2 = set(obj2, optics[2], val[2])
    return obj2
end

@inline get_lens_symbol(o) = :p
@inline get_lens_symbol(o::IndexLens{Tuple{Int64}}) = Symbol("p"*"$(o.indices[1])")
@inline get_lens_symbol(::PropertyLens{F}) where F = F
get_lens_symbol(o::ComposedOptic) = get_lens_symbol(o.outer)

function get_lens_symbol(lens1, lens2)
    p1 = get_lens_symbol(lens1)
    p2 = get_lens_symbol(lens2)
    out = p1 == p2 ? (Symbol(String(p1) * "1"), Symbol(String(p2) * "2")) : (p1, p2)
end