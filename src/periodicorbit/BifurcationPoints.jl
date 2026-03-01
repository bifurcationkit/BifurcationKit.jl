abstract type AbstractBifurcationPointOfPO <: AbstractBifurcationPoint end
abstract type AbstractSimpleBifurcationPointPO <: AbstractBifurcationPointOfPO end
####################################################################################################
# types for bifurcation point with 1d kernel for the jacobian
for op in (:BranchPointPO, :PeriodDoublingPO,)
    @eval begin
        """
        $(TYPEDEF)

        # Internal fields

        $(TYPEDFIELDS)

        # Predictor

        You can call `predictor(bp, ds; kwargs...)` on such bifurcation point `bp`
        to find the zeros of the normal form polynomials.
        """
        mutable struct $op{Tprob, Tv, 𝒯, Tevr, Tevl, Tnf} <: AbstractSimpleBifurcationPointPO
            "Bifurcation point (periodic orbit)."
            po::Tv

            "Period."
            T::𝒯

            "Right eigenvector(s)."
            ζ::Tevr

            "Left eigenvector(s)."
            ζ★::Tevl

            "Normal form."
            nf::Tnf

            "Periodic orbit problem."
            prob::Tprob

            "Is normal form computed using Poincaré return map?"
            prm::Bool
        end
    end
end

type(bp::PeriodDoublingPO) = :PeriodDoubling
type(bp::BranchPointPO) = :BranchPoint

function Base.show(io::IO, pd::PeriodDoublingPO)
    printstyled(io, "Period-Doubling", color=:cyan, bold = true)
    println(io, " bifurcation point of periodic orbit")
    println(io, "├─ Period = ", abs(pd.T), " -> ", 2abs(pd.T))
    print(io, "├─ Problem: ")
    printstyled(io, typeof(pd.prob).name.name, "\n", bold = true)
    if pd.prob isa ShootingProblem
        show(io, pd.nf)
    else
        if ~pd.prm
            println(io, "├─ ", get_lens_symbol(pd.nf.lens)," ≈ $(pd.nf.p)")
            print(io, "├─ type: ")
            printstyled("$(pd.nf.type)\n", color=:cyan, bold = true)
            println(io, "├─ Normal form (Iooss):\n├\t∂τ = 1 + a₀₁⋅δp + a₂⋅ξ²\n├\t∂ξ =  ξ⋅(c₁₁⋅δp + c₃⋅ξ²)")
            if get(pd.nf.nf, :a₀₁, nothing) != nothing
                println(io, "├─── a₀₁ = ", pd.nf.nf.a₀₁,
                          "\n├─── a₂  = ", pd.nf.nf.a,
                          "\n├─── c₁₁ = ", pd.nf.nf.c₁₁,
                          "\n└─── c₃  = ", pd.nf.nf.b3)
            end
        else
            show(io, pd.nf)
        end
    end
end

function Base.show(io::IO, bp::BranchPointPO)
    printstyled(io, type(bp), color=:cyan, bold = true)
    println(io, " bifurcation point of periodic orbit\n┌─ ", get_lens_symbol(bp.nf.lens)," ≈ $(bp.nf.p)")
    println(io, "├─ Period = ", abs(bp.T))
    print(io, "├─ Problem: ")
    printstyled(io, typeof(bp.prob).name.name, "\n", bold = true)
    println(io, "└─ Normal form =")
    show(io, bp.nf; prefix = "\t")
end

####################################################################################################
# type for Neimark-Sacker bifurcation point

"""
$(TYPEDEF)

# Internal fields

$(TYPEDFIELDS)

# Predictor

You can call `predictor(bp::NeimarkSackerPO, ds)` on such bifurcation point `bp` to get the guess for the periodic orbit.
"""
mutable struct NeimarkSackerPO{Tprob, Tv, 𝒯, Tω, Tevr, Tevl, Tnf} <: AbstractSimpleBifurcationPointPO
    "Bifurcation point (periodic orbit)."
    po::Tv

    "Period."
    T::𝒯

    "Parameter value at the Neimark-Sacker point."
    p::𝒯

    "Frequency of the Neimark-Sacker point."
    ω::Tω

    "Right eigenvector(s)."
    ζ::Tevr

    "Left eigenvector(s)."
    ζ★::Tevl

    "Normal form."
    nf::Tnf

    "Periodic orbit problem."
    prob::Tprob

    "Normal form computed using Poincaré return map."
    prm::Bool
end

type(bp::NeimarkSackerPO) = type(bp.nf)

function Base.show(io::IO, ns::NeimarkSackerPO)
    printstyled(io, ns.nf.type, " - Neimark-Sacker", color=:cyan, bold = true)
    println(io, " bifurcation point of periodic orbit\n┌─ ", get_lens_symbol(ns.nf.lens)," ≈ $(ns.p).")
    println(io, "├─ Frequency θ ≈ ", ns.ω)
    println(io, "├─ Period at the periodic orbit T ≈ ", abs(ns.T))
    println(io, "├─ Second period of the bifurcated torus ≈ ", abs(2pi*ns.ω*ns.T))
    print(io, "├─ Problem: ")
    printstyled(io, typeof(ns.prob).name.name, "\n", bold = true)
    if ns.prm
        println(io, "├─ Normal form z ─▶ z⋅eⁱᶿ(1 + a⋅δp + b⋅|z|²)")
    else
        println(io, "├─ Normal form:\n├\t∂τ = 1 + a⋅|ξ|²\n├\t∂ξ = iθ/T⋅ξ + d⋅ξ⋅|ξ|²")
    end
    if ~isnothing(ns.nf.nf)
        if ns.prm
            println(io,"├─── a = ", ns.nf.nf.a, "\n├─── b = ", ns.nf.nf.b)
        else
            println(io,"├─── a = ", ns.nf.nf.a, "\n├─── d = ", ns.nf.nf.d)
        end
    end
    println(io, "└─ Periodic orbit problem: \n")
    show(io, ns.prob)
end
