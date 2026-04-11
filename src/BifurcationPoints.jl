abstract type AbstractBifurcationPoint end
abstract type AbstractBranchPoint <: AbstractBifurcationPoint end
abstract type AbstractSimpleBranchPoint <: AbstractBranchPoint end
abstract type AbstractSimpleBranchPointForMaps <: AbstractSimpleBranchPoint end

istranscritical(::AbstractBranchPoint) = false
####################################################################################################
"""
$(TYPEDEF)

Structure to record special points on a curve. There are two types of special points that are recorded in this structure: bifurcation points and events (see https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/EventCallback/).

# Internal fields

$(TYPEDFIELDS)

# Associated methods
- `BifurcationKit.type(::SpecialPoint)` returns the bifurcation type (`::Symbol`)
"""
@with_kw struct SpecialPoint{T, Tp, Tv, Tvτ} <: AbstractBifurcationPoint
    "Description of the special points. In case of `Events`, this field records the user passed name to the event, or the default `:userD`, `:userC`. In case of bifurcation points, it can be one of the following names:

    - :bp Bifurcation point, simple eigenvalue crossing the imaginary axis
    - :fold Fold point
    - :hopf Hopf point
    - :nd not documented bifurcation point. Detected by multiple eigenvalues crossing. Generally occurs in problems with symmetries or in cases where the continuation step size is too large and merge two different bifurcation points.
    - :cusp Cusp point
    - :gh Generalized Hopf point (also called Bautin point)
    - :bt Bogdanov-Takens point
    - :zh Zero-Hopf point
    - :hh Hopf-Hopf point
    - :ns Neimark-Sacker point
    - :pd Period-doubling point
    - :R1 Strong resonance 1:1 of periodic orbits
    - :R2 Strong resonance 1:2 of periodic orbits
    - :R3 Strong resonance 1:3 of periodic orbits
    - :R4 Strong resonance 1:4 of periodic orbits
    - :foldFlip Fold / Flip of periodic orbits
    - :foldNS Fold / Neimark-Sacker of periodic orbits
    - :pdNS  Period-Doubling / Neimark-Sacker of periodic orbits
    - :gpd Generalized Period-Doubling of periodic orbits
    - :nsns Double Neimark-Sacker of periodic orbits
    - :ch Chenciner bifurcation of periodic orbits
    "
    type::Symbol = :none

    "Index in `br.branch` or `br.eig` (see [`ContResult`](@ref)) for which the bifurcation occurs."
    idx::Int64 = 0

    "Parameter value at the special point (this is an estimate)."
    param::T = 0.

    "Norm of the equilibrium at the special point."
    norm::T  = 0.

    "`printsol = record_from_solution(x, param)` where `record_from_solution` is one of the arguments to [`continuation`](@ref)."
    printsol::Tp = 0.

    "Solution at the special point."
    x::Tv = Vector{T}(undef, 0)

    "Tangent along the branch at the special point."
    τ::BorderedArray{Tvτ, T} = BorderedArray(x, zero(T))

    "Eigenvalue index responsible for detecting the special point (if applicable)."
    ind_ev::Int64 = 0

    "Continuation step at which the special occurs."
    step::Int64 = 0

    "`status ∈ {:converged, :guess, :guessL}` indicates whether the bisection algorithm was successful in detecting the special (bifurcation) point. If `status == :guess`, the bisection algorithm failed to meet the requirements given in `::ContinuationPar`. Same for `status == :guessL` but the bisection algorithm stopped on the left of the bifurcation point."
    status::Symbol = :guess

    "`δ = (δr, δi)` where δr indicates the change in the number of unstable eigenvalues and δi indicates the change in the number of unstable eigenvalues with nonzero imaginary part. `abs(δr)` is thus an estimate of the dimension of the kernel of the Jacobian at the special (bifurcation) point."
    δ::Tuple{Int64, Int64} = (0, 0)

    "Precision in the location of the special point"
    precision::T = -1

    "Interval parameter containing the special point"
    interval::Tuple{T, T} = (0, 0)
end

_getvectortype(::Vector{SpecialPoint{T, Tp, Tv, Tvτ}}) where {T, Tp, Tv, Tvτ} = Tvτ
type(bp::SpecialPoint) = bp.type

"""
$(TYPEDSIGNATURES)

Return the dimension of the kernel of the special point.
"""
@inline kernel_dimension(bp::SpecialPoint) = abs(bp.δ[1])

function SpecialPoint(x0, τ, T::Type, printsol)
    return SpecialPoint(type = :none,
                        idx = 0,
                        param = zero(T),
                        norm  = zero(T),
                        printsol = printsol,
                        x = x0,
                        τ = τ,
                        ind_ev = 0,
                        step = 0,
                        status = :guess,
                        δ = (0, 0),
                        precision = T(-1),
                        interval = (zero(T), zero(T)))
end

function SpecialPoint(it::ContIterable,
                    state::ContState,
                    type::Symbol,
                    status::Symbol,
                    interval; 
                    ind_ev = 0,
                    δ = (0, 0),
                    idx = state.step )
    return SpecialPoint(;
                        type,
                        idx,
                        param = getp(state),
                        norm = it.normC(getx(state)),
                        printsol = _namedrecordfromsol(record_from_solution(it)(getx(state), getp(state); iter=it, state)),
                        x = save_solution(it.prob, _copy(getx(state)), getparam(it.prob)),
                        τ = _copy(state.τ),
                        ind_ev,
                        step = state.step,
                        status,
                        δ,
                        precision = abs(interval[2] - interval[1]),
                        interval)
end


function _show(io::IO, bp::SpecialPoint, ii::Int, p::String = "p")
    if type(bp) == :none ; return; end
    @printf(io, "- #%3i, ", ii)
    if type(bp) == :endpoint
        printstyled(io, @sprintf("%8s", type(bp)); bold=true)
        @printf(io, " at %s ≈ %+4.8f,                                                                     step = %3i\n", p, bp.param, bp.step)
    else
        printstyled(io, @sprintf("%8s", type(bp)); bold=true, color=:blue)
        @printf(io, " at %s ≈ %+4.8f ∈ (%+4.8f, %+4.8f), |δp|=%1.0e, [", p, bp.param, bp.interval..., bp.precision)
        printstyled(io, @sprintf("%9s", bp.status); bold=true, color=(bp.status == :converged) ? :green : :red)
        @printf(io, "], δ = (%2i, %2i), step = %3i\n", bp.δ..., bp.step)
    end
end

function is_bifurcation(sp::SpecialPoint)
    type(sp) in (:bp, :fold, :hopf, :nd, :cusp, :gh, :bt, :zh, :hh, :ns, :pd,)
end
####################################################################################################
# types for bifurcation point with 1d kernel for the jacobian

for (op, opt) in ((:BranchPoint, AbstractSimpleBranchPoint),
                  (:Fold, AbstractSimpleBranchPoint),
                  (:Pitchfork, AbstractSimpleBranchPoint),
                  (:Transcritical, AbstractSimpleBranchPoint),
                  (:PeriodDoubling, AbstractSimpleBranchPointForMaps),
                  (:BranchPointMap, AbstractSimpleBranchPointForMaps),
                  (:PitchforkMap, AbstractSimpleBranchPointForMaps),
                  (:TranscriticalMap, AbstractSimpleBranchPointForMaps),
                  (:NdBranchPoint, AbstractBranchPoint),
                  )
    @eval begin
        """
        $(TYPEDEF)

        ## Fields

        $(TYPEDFIELDS)

        ## Predictor

        You can call `predictor(bp, ds; kwargs...)` on such bifurcation point `bp`
        to find the zeros of the normal form polynomials.

        ## Manipulating the normal form for `NdBranchPoint`

        - You can use `bp(Val(:reducedForm), x, p)` to evaluate the normal form polynomials on the vector `x` for (scalar) parameter `p`.

        - You can use `bp(x, δp::Real)` to get the (large dimensional guess) associated to the low dimensional vector `x`. Note that we must have `length(x) == length(bp)`.

        - You can use `BifurcationKit.nf(bp; kwargs...)` to pretty print the normal form with a string."
        """
        mutable struct $op{Tv, Tτ, T, Tpar, Tlens <: AllOpticTypes, Tevl, Tevr, Tnf} <: $opt
            "Bifurcation point."
            x0::Tv

            "Tangent of the curve at the bifurcation point."
            τ::Tτ

            "Parameter value at the bifurcation point."
            p::T

            "Parameters used by the vector field."
            params::Tpar

            "Parameter axis used to compute the branch on which this bifurcation point was detected."
            lens::Tlens

            "Right eigenvector(s)."
            ζ::Tevr

            "Left eigenvector(s)."
            ζ★::Tevl

            "Normal form coefficients."
            nf::Tnf

            "Type of bifurcation point"
            type::Symbol
        end
    end
end

for op in (:Pitchfork, :PitchforkMap)
    @eval begin
    $op(x0, τ, p, params, lens, ζ, ζ★, nf) = $op(x0, τ, p, params, lens, ζ, ζ★, nf, real(nf.b11) * real(nf.b30) < 0 ? :SuperCritical : :SubCritical)
    end
end

istranscritical(bp::AbstractSimpleBranchPoint) = bp isa Transcritical
type(::BranchPoint) = :BranchPoint
type(::Pitchfork) = :Pitchfork
type(::PitchforkMap) = :Pitchfork
type(::Fold) = :Fold
type(::Transcritical) = :Transcritical
type(::TranscriticalMap) = :Transcritical
type(::PeriodDoubling) = :PeriodDoubling
type(::Nothing) = nothing

function printnf1d(io, nf; prefix = "")
    println(io, prefix * "┌─ a01 = ", nf.a01)
    println(io, prefix * "├─ a02 = ", nf.a02)
    println(io, prefix * "├─ b11 = ", nf.b11)
    println(io, prefix * "├─ b20 = ", nf.b20)
    println(io, prefix * "└─ b30 = ", nf.b30)
end

function Base.show(io::IO, bp::AbstractBifurcationPoint; prefix = "")
    printstyled(io, prefix*string(type(bp)), color=:cyan, bold = true)
    if bp isa AbstractSimpleBranchPointForMaps
        printstyled(io, " (Maps)", color=:cyan, bold = true)
    end
    plens = get_lens_symbol(bp.lens)
    println(io, " bifurcation point at $plens ≈ $(bp.p)")
    if bp isa AbstractSimpleBranchPointForMaps
        println(io, prefix*"Normal form x ─▶ x + (a01⋅δ$plens + a02⋅δ$(plens)²/2 + b10⋅x⋅δ$plens + b20⋅x²/2 + b30⋅x³/6)")
    else
        println(io, prefix*"Normal form (a01⋅δ$plens + a02⋅δ$(plens)²/2 + b10⋅x⋅δ$plens + b20⋅x²/2 + b30⋅x³/6)")
    end
    if ~isnothing(bp.nf)
        printnf1d(io, bp.nf; prefix)
    end
end

function Base.show(io::IO, bp::Union{Pitchfork, PitchforkMap}; prefix = "")
    printstyled(io, prefix, bp.type, " - Pitchfork", color=:cyan, bold = true)
    if bp isa PitchforkMap
        printstyled(io, " (Maps)", color=:cyan, bold = true)
    end
    plens = get_lens_symbol(bp.lens)
    println(io, " bifurcation point at $plens ≈ $(bp.p)")
    if bp isa PitchforkMap
        println(io, prefix*"Normal form x ─▶ x + a01⋅δ$plens + a02⋅δ$(plens)² + x⋅(b11⋅δ$plens + b30⋅x²/6)")
    else
        println(io, prefix*"Normal form a01⋅δ$plens + a02⋅δ$(plens)² + x⋅(b11⋅δ$plens + b30⋅x²/6)")
    end
    if ~isnothing(bp.nf)
        printnf1d(io, bp.nf; prefix)
    end
end

function Base.show(io::IO, bp::PeriodDoubling)
    plens = get_lens_symbol(bp.lens)
    printstyled(io, bp.type, " - Period-Doubling ", color=:cyan, bold = true)
    println("bifurcation point at $plens ≈ $(bp.p)")
    println(io, "┌─ Normal form:\n├\t x ─▶ x⋅(a⋅δ$plens - 1 + c⋅x²)")
    if ~isnothing(bp.nf)
        println(io, "├─ a = ", bp.nf.a)
        println(io, "└─ c = ", bp.nf.b3)
    else
        println(io, "├─ a = ", missing)
        println(io, "└─ c = ", missing)
    end
end

function Base.show(io::IO, bp::BranchPointMap) #a⋅(p - pbif) + x⋅(b1⋅(p - pbif) + b2⋅x/2 + b3⋅x^2/6)
    plens = get_lens_symbol(bp.lens)
    printstyled(io, bp.type, " - Branch point ", color=:cyan, bold = true)
    println(io, "(Maps) bifurcation point at $plens ≈ $(bp.p)")
    println(io, "Normal form x ─▶ a01⋅δ$plens + x⋅(b11⋅δ$plens + b30⋅x²/6):")
    if ~isnothing(bp.nf)
        printnf1d(io, bp.nf)
    end
end

####################################################################################################
# type for bifurcation point Nd kernel for the jacobian
type(::NdBranchPoint) = :NonSimpleBranchPoint
Base.length(bp::NdBranchPoint) = length(bp.ζ)

function Base.show(io::IO, bp::NdBranchPoint; prefix = "")
    plens = get_lens_symbol(bp.lens)
    printstyled(io, prefix, bp.type, color=:cyan, bold = true)
    println(io, " (non simple) bifurcation point at ", plens, " ≈ $(bp.p). \nKernel dimension = ", length(bp))
    println(io, prefix, "Normal form:")
    println(io, prefix, mapreduce(x -> x * "\n", *, _get_string(bp, "δ$plens")) )
end
####################################################################################################
for (op, opt) in ((:Hopf, AbstractSimpleBranchPoint),
                  (:NeimarkSacker, AbstractSimpleBranchPointForMaps)
                )
    @eval begin
        """
        $(TYPEDEF)

        ## Fields

        $(TYPEDFIELDS)

        # Associated methods

        ## Predictor

        You can call `predictor(bp, ds)` on such bifurcation point `bp` to get the guess for the periodic orbit.
        """
        mutable struct $op{Tv, Tτ, T, Tω, Tpar, Tlens <: AllOpticTypes, Tevr, Tevl, Tnf} <: $opt
            "Bifurcation point"
            x0::Tv

            "Tangent of the curve at the bifurcation point."
            τ::Tτ

            "Parameter value at the bifurcation point"
            p::T

            "Frequency at the bifurcation point"
            ω::Tω

            "Parameters used by the vector field."
            params::Tpar

            "Parameter axis used to compute the branch on which this bifurcation point was detected."
            lens::Tlens

            "Right eigenvector"
            ζ::Tevr

            "Left eigenvector"
            ζ★::Tevl

            "Normal form coefficient ex: (a = 0., b = 1 + 1im)"
            nf::Tnf

            "Type bifurcation"
            type::Symbol
        end
    end
end

type(::Hopf) = :Hopf
type(::NeimarkSacker) = :NeimarkSacker

Hopf(x0, p, ω, params, lens, ζ, ζ★, nf) = Hopf(x0, p, ω, params, lens, ζ, ζ★, nf, real(nf.b1) * real(nf.b3) < 0 ? :SuperCritical : :SubCritical)

function Base.show(io::IO, bp::Hopf)
    plens = get_lens_symbol(bp.lens)
    printstyled(io, bp.type, " - ", type(bp), color=:cyan, bold = true)
    println(io, " bifurcation point at $plens ≈ $(bp.p).")
    println(io, "Frequency ω ≈ ", abs(bp.ω))
    println(io, "Period of the periodic orbit ≈ ", abs(2pi/bp.ω))
    println(io, "Normal form z⋅(iω + a⋅δ$plens + b⋅|z|²):")
    if ~isnothing(bp.nf)
        println(io,"┌─ a = ", bp.nf.a)
        println(io,"└─ b = ", bp.nf.b)
    end
end

function Base.show(io::IO, bp::NeimarkSacker)
    printstyled(io, bp.type, " - ", type(bp), color=:cyan, bold = true)
    plens = get_lens_symbol(bp.lens)
    println(io, " bifurcation point at $plens ≈ $(bp.p).")
    println(io, "Frequency θ ≈ ", abs(bp.ω))
    println(io, "Period of the periodic orbit ≈ ", abs(2pi/bp.ω))
    println(io, "Normal form z ─▶ z⋅eⁱᶿ(1 + a⋅δ$plens + b⋅|z|²)")
    if ~isnothing(bp.nf)
        println(io,"┌─ a = ", bp.nf.a)
        println(io,"└─ b = ", bp.nf.b)
    end
end
