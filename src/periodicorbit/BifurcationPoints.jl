abstract type AbstractBifurcationPointOfPO <: AbstractBifurcationPoint end
abstract type AbstractSimpleBifurcationPointPO <: AbstractBifurcationPointOfPO end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# types for bifurcation point with 1d kernel for the jacobian
for op in (:BranchPointPO, :PeriodDoublingPO, :PitchforkPO, :TranscriticalPO, :FoldPO)
    @eval begin
        """
        $(TYPEDEF)

        # Internal fields

        $(TYPEDFIELDS)

        # Predictor

        You can call `predictor(bp, δp, ampfactor; override = false)` on such bifurcation point `bp`
        to get a guess for the periodic orbit on the bifurcated branch.
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

        function Base.show(io::IO, bp::$op)
            if !bp.prm && bp.nf isa BranchPoint &&
                bp.nf.nf isa NamedTuple && haskey(bp.nf.nf, :c01)
                return _show_po_iooss(io, bp, type(bp))
            end
            printstyled(io, type(bp), color=:cyan, bold = true)
            println(io, " bifurcation point of periodic orbit")
            println(io, "├─ Period = ", abs(bp.T))
            print(io, "├─ Problem: ")
            printstyled(io, typeof(bp.prob).name.name, "\n", bold = true)
        end
    end
end

type(::PeriodDoublingPO) = :PeriodDoubling
type(::BranchPointPO) = :BranchPoint
type(::PitchforkPO) = :Pitchfork
type(::TranscriticalPO) = :Transcritical
type(::FoldPO) = :Fold

function Base.show(io::IO, pd::PeriodDoublingPO)
    printstyled(io, "Period-Doubling", color=:cyan, bold = true)
    println(io, " bifurcation point of periodic orbit")
    println(io, "├─ Period = ", abs(pd.T), " -> ", 2abs(pd.T))
    print(io, "├─ Problem: ")
    printstyled(io, typeof(pd.prob).name.name, "\n", bold = true)
    if pd.prob isa Shooting
        show(io, pd.nf)
    else
        if ~pd.prm
            println(io, "├─ ", get_lens_symbol(pd.nf.lens)," ≈ $(pd.nf.p)")
            print(io, "├─ type: ")
            printstyled("$(pd.nf.type)\n", color=:cyan, bold = true)
            println(io, "├─ Normal form (Iooss):\n├\t∂τ = 1 + a₀₁⋅δp + a₂⋅ξ²\n├\t∂ξ =  ξ⋅(c₁₁⋅δp + c₃⋅ξ²)")
            if get(pd.nf.nf, :a₀₁, nothing) !== nothing
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

function Base.show(io::IO, bp::FoldPO)
    if !bp.prm && bp.nf isa BranchPoint &&
       bp.nf.nf isa NamedTuple && haskey(bp.nf.nf, :c01)
        return _show_po_iooss(io, bp, "Fold (Iooss)")
    end
    printstyled(io, "Fold (LPC)", color=:cyan, bold = true)
    println(io, " bifurcation point of periodic orbit")
    println(io, "├─ Period = ", abs(bp.T))
    print(io, "├─ Problem: ")
    printstyled(io, typeof(bp.prob).name.name, "\n", bold = true)
    if !isnothing(bp.nf)
        println(io, "└─ Normal form coefficient a = ", bp.nf.a)
    end
end

function _show_po_iooss(io::IO, bp, title::String)
    nf = bp.nf.nf
    printstyled(io, title, color=:cyan, bold = true)
    println(io, " bifurcation point of periodic orbit")
    println(io, "┌─ ", get_lens_symbol(bp.nf.lens), " ≈ ", bp.nf.p)
    println(io, "├─ Period = ", abs(bp.T))
    print(io, "├─ Problem: ")
    printstyled(io, typeof(bp.prob).name.name, "\n", bold = true, color=:cyan)
    if bp.nf isa BranchPoint && bp.nf.type in (:SuperCritical, :SubCritical)
        print(io, "├─ type: ");printstyled(io, bp.nf.type, "\n", bold = true, color=:cyan)
    end
    println(io, "├─ Normal form [Iooss:")
    println(io, "├\t∂τ = 1 + α⋅ξ + (a₁₁⋅ξ + a₀₁)⋅δp + a₂₀⋅ξ² + ...")
    println(io, "├\t∂ξ = c₀₁⋅δp + c₁₁⋅δp⋅ξ + c₂⋅ξ² + a₃₀⋅ξ³ + ...")
    println(io, "├─ Coefficients a:")
    println(io, "│  ├─ a₀₁ = ", nf.a01)
    println(io, "│  ├─ a₁₁ = ", nf.a11)
    println(io, "│  ├─ a₂₀ = ", nf.a20)
    println(io, "│  └─ a₃₀ = ", nf.a30)
    println(io, "└─ Coefficients c:")
    println(io, "   ├─ c₀₁ = ", nf.c01)
    println(io, "   ├─ c₁₁ = ", nf.c11)
    println(io, "   ├─ c₂  = ", nf.c2)
    println(io, "   └─ c₃  = ", nf.c3)
    return nothing
end

function Base.show(io::IO, bp::BranchPointPO)
    if !bp.prm && bp.nf isa BranchPoint &&
       bp.nf.nf isa NamedTuple && haskey(bp.nf.nf, :a11)
        return _show_po_iooss(io, bp, "BranchPoint (Iooss)")
    end
    printstyled(io, type(bp), color=:cyan, bold = true)
    println(io, " bifurcation point of periodic orbit\n┌─ ", get_lens_symbol(bp.nf.lens)," ≈ $(bp.nf.p)")
    println(io, "├─ Period = ", abs(bp.T))
    print(io, "├─ Problem: ")
    printstyled(io, typeof(bp.prob).name.name, "\n", bold = true)
    println(io, "└─ Normal form =")
    show(io, bp.nf; prefix = "\t")
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
$(TYPEDEF)

# Internal fields

$(TYPEDFIELDS)

# Predictor

You can call `predictor(bp::NeimarkSackerPO, δp, ampfactor)` on such bifurcation point `bp` to get the guess for the periodic orbit.
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
