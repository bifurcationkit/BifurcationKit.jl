abstract type AbstractBifurcationPointOfPO <: AbstractBifurcationPoint end
abstract type AbstractSimpleBifurcationPointPO <: AbstractBifurcationPointOfPO end
####################################################################################################
# types for bifurcation point with 1d kernel for the jacobian

for op in (:BranchPointPO, :PeriodDoublingPO,)
    @eval begin
        """
        $(TYPEDEF)

        $(TYPEDFIELDS)

        ## Predictor

        You can call `predictor(bp, ds; kwargs...)` on such bifurcation point `bp`
        to find the zeros of the normal form polynomials.
        """
        mutable struct $op{Tprob, Tv, T, Tevr, Tevl, Tnf} <: AbstractSimpleBifurcationPointPO
            "Bifurcation point (periodic orbit)."
            po::Tv

            "Period"
            T::T

            "Right eigenvector(s)."
            ζ::Tevr

            "Left eigenvector(s)."
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

type(bp::PeriodDoublingPO) = :PeriodDoubling
type(bp::BranchPointPO) = :BranchPoint

function Base.show(io::IO, pd::PeriodDoublingPO)
    printstyled(io, "Period-Doubling", color=:cyan, bold = true)
    println(io, " bifurcation point of periodic orbit at\n┌─ ", getLensSymbol(pd.nf.lens)," ≈ $(pd.nf.p)")
    println(io, "├─ Period = ", abs(pd.T), " -> ", 2abs(pd.T))
    println(io, "├─ Problem: ", typeof(pd.prob).name.name)
    if pd.prob isa ShootingProblem
        show(io, pd.nf)
    else
        if ~pd.prm
            println(io, "├─ Normal form:\n├\t∂τ = 1 + a⋅ξ²\n├\t∂ξ = c⋅ξ³\n└─ ", pd.nf.nf)
        else
            show(io, pd.nf)
        end
    end
end

function Base.show(io::IO, bp::BranchPointPO)
    println(io, type(bp), " bifurcation point of periodic orbit at\n┌─ ", getLensSymbol(bp.nf.lens)," ≈ $(bp.nf.p)")
    println(io, "├─ Period = ", abs(bp.T))
    println(io, "└─ Problem: ", typeof(bp.prob).name.name)
end

####################################################################################################
# type for Neimark-Sacker bifurcation point

"""
$(TYPEDEF)

$(TYPEDFIELDS)

# Associated methods

## Predictor

You can call `predictor(bp::NeimarkSackerPO, ds)` on such bifurcation point `bp` to get the guess for the periodic orbit.
"""
mutable struct NeimarkSackerPO{Tprob, Tv, T, Tω, Tevr, Tevl, Tnf} <: AbstractSimpleBifurcationPointPO
    "Bifurcation point (periodic orbit)"
    po::Tv

    "Period"
    T::T

    "Parameter value at the Neimark-Sacker point"
    p::T

    "Frequency of the Neimark-Sacker point"
    ω::Tω

    "Right eigenvector(s)."
    ζ::Tevr

    "Left eigenvector(s)."
    ζ★::Tevl

    "Underlying normal form for Poincaré return map"
    nf::Tnf

    "Periodic orbit problem"
    prob::Tprob

    "Normal form computed using Poincaré return map"
    prm::Bool
end

type(bp::NeimarkSackerPO) = type(bp.nf)

function Base.show(io::IO, ns::NeimarkSackerPO)
    println(io, ns.nf.type, " - ",type(ns), " bifurcation point of periodic orbit at\n┌─ ", getLensSymbol(ns.nf.lens)," ≈ $(ns.p).")
    println(io, "├─ Frequency θ ≈ ", abs(ns.ω))
    println(io, "├─ Period at the periodic orbit ≈ ", abs(ns.T))
    println(io, "├─ Second frequency of the bifurcated torus ≈ ", abs(2pi/ns.ω))
    println(io, "├─ Normal form z⋅eⁱᶿ(1 + a⋅δp + b⋅|z|²)")
    println(io, "└─ Periodic orbit problem: \n")
    show(io, ns.prob)
    if ~isnothing(ns.nf.nf)
        println(io,"\n- a = ", ns.nf.nf.a)
        println(io,"- b = ", ns.nf.nf.b)
    end
end
