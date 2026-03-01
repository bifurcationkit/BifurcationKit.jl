abstract type AbstractCodim2BifurcationPointOfPO <: AbstractBifurcationPoint end

for op in (:CuspPO, :R1, :R2, :R3, :R4, :GPD, :FoldNS, :FoldPD)
    @eval begin
        """
        $(TYPEDEF)

        # Internal fields
        $(TYPEDFIELDS)

        # Associated methods

        ## Predictor

        You can call `predictor(bp, ds; kwargs...)` on such bifurcation point `bp`
        to find the zeros of the normal form polynomials.
        """
        mutable struct $op{Tprob, Tv, 𝒯, Tpar, Tlens, Tevr, Tevl, Tnf} <: AbstractCodim2BifurcationPointOfPO
            "Bifurcation point (periodic orbit)"
            po::Tv

            "Period"
            T::𝒯

            "Parameters used by the vector field."
            params::Tpar

            "Parameter axis used to compute the branch on which this cusp point was detected."
            lens::Tlens

            "Right eigenvector(s)"
            ζ::Tevr

            "Left eigenvector(s)"
            ζ★::Tevl

            "Normal form"
            nf::Tnf

            "Periodic orbit problem"
            prob::Tprob

            "Normal form computed using Poincaré return map"
            prm::Bool
        end
    end
end

function Base.show(io::IO, bp::AbstractCodim2BifurcationPointOfPO)
    lens1, lens2 = bp.lens
    p1 = _get(bp.params, lens1)
    p2 = _get(bp.params, lens2)
    type = typeof(bp).name.name
    if type == :GPD
        type = "Generalized Period-doubling"
    elseif type == :FoldNS
        type = "Fold-Neimark-Sacker"
    elseif type == :FoldPD
        type = "Fold-Period-doubling"
    elseif type == :CuspPO
        type = "Cusp"
    end
    printstyled(io, "$type", color=:cyan, bold = true)
    println(io, " bifurcation point of periodic orbit ")
    println(io, "├─ ", get_lens_symbol(lens1, lens2)," ≈ ($p1, $p2)")
    println(io, "├─ Period = ", abs(bp.T))
    println(io, "└─ Problem : ", typeof(bp.prob).name.name)
end