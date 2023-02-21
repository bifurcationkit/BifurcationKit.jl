abstract type AbstractBifurcationPoint end
abstract type AbstractBranchPoint <: AbstractBifurcationPoint end
abstract type AbstractSimpleBranchPoint <: AbstractBranchPoint end
abstract type AbstractSimpleBranchPointForMaps <: AbstractSimpleBranchPoint end

istranscritical(bp::AbstractBranchPoint) = false
####################################################################################################
"""
$(TYPEDEF)

Structure to record a generic special (bifurcation) point.

$(TYPEDFIELDS)
"""
@with_kw struct SpecialPoint{T, Tp, Tv, Tvτ} <: AbstractBifurcationPoint
	"Bifurcation type, `:hopf, :bp...`."
	type::Symbol = :none

	"Index in `br.eig` (see [`ContResult`](@ref)) for which the bifurcation occurs."
	idx::Int64 = 0

	"Parameter value at the special (bifurcation) point, this is an estimate."
	param::T = 0.

	"Norm of the equilibrium at the special (bifurcation) point"
	norm::T  = 0.

	"`printsol = recordFromSolution(x, param)` where `recordFromSolution` is one of the arguments to [`continuation`](@ref)"
	printsol::Tp = 0.

	"Equilibrium at the special (bifurcation) point"
	x::Tv = Vector{T}(undef, 0)

	"Tangent along the branch at the special (bifurcation) point"
	τ::BorderedArray{Tvτ, T} = BorderedArray(x, T(0))

	"Eigenvalue index responsible for the special (bifurcation) (if applicable)"
	ind_ev::Int64 = 0

	"Continuation step at which the special (bifurcation) occurs"
	step::Int64 = 0

	"`status ∈ {:converged, :guess}` indicates whether the bisection algorithm was successful in detecting the special (bifurcation) point"
	status::Symbol = :guess

	"`δ = (δr, δi)` where δr indicates the change in the number of unstable eigenvalues and δi indicates the change in the number of unstable eigenvalues with nonzero imaginary part. `abs(δr)` is thus an estimate of the dimension of the kernel of the Jacobian at the special (bifurcation) point."
	δ::Tuple{Int64, Int64} = (0,0)

	"Precision in the location of the special (bifurcation) point"
	precision::T = -1

	"Interval containing the special (bifurcation) point"
	interval::Tuple{T, T} = (0, 0)
end

getVectorType(::Type{SpecialPoint{T, Tp, Tv, Tvτ}}) where {T, Tp, Tv, Tvτ} = Tvτ
type(bp::SpecialPoint) = bp.type

"""
$(SIGNATURES)

Return the dimension of the kernel of the special point.
"""
@inline kernelDim(bp::SpecialPoint) = abs(bp.δ[1])

# constructors
SpecialPoint(x0, τ, T::Type, printsol) = SpecialPoint(type = :none, idx = 0, param = T(0), norm  = T(0), printsol = printsol, x = x0, τ = τ, ind_ev = 0, step = 0, status = :guess, δ = (0, 0), precision = T(-1), interval = (T(0), T(0)))

SpecialPoint(it::ContIterable, state::ContState, type::Symbol, status::Symbol, interval; ind_ev = 0, δ = (0,0), idx = state.step ) = SpecialPoint(
				type = type,
				idx = idx,
				param = getp(state),
				norm = it.normC(getx(state)),
				printsol = namedprintsol(recordFromSolution(it)(getx(state), getp(state))),
				x = getSolution(it.prob, _copy(getx(state))),
				τ = copy(state.τ),
				ind_ev = ind_ev,
				step = state.step,
				status = status,
				δ = δ,
				precision = abs(interval[2] - interval[1]),
				interval = interval)


function _show(io::IO, bp::SpecialPoint, ii::Int, p::String = "p")
	if bp.type == :none ; return; end
	if bp.type == :endpoint
		@printf(io, "- #%3i, \033[1m%5s\033[0m at %s ≈ %+4.8f,                                                                     step = %3i\n", ii, "endpoint", p, bp.param, bp.step)
		return
	end
	if bp.status == :converged
		@printf(io, "- #%3i,\e[1;34m %8s\e[0m at %s ≈ %+4.8f ∈ (%+4.8f, %+4.8f), |δp|=%1.0e, [\e[1;32m%9s\e[0m], δ = (%2i, %2i), step = %3i, eigenelements in eig[%3i], ind_ev = %3i\n", ii, bp.type, p, bp.param, bp.interval..., bp.precision, bp.status, bp.δ..., bp.step, bp.idx, bp.ind_ev)
	else
		@printf(io, "- #%3i,\e[1;34m %8s\e[0m at %s ≈ %+4.8f ∈ (%+4.8f, %+4.8f), |δp|=%1.0e, [\e[1;31m%9s\e[0m], δ = (%2i, %2i), step = %3i, eigenelements in eig[%3i], ind_ev = %3i\n", ii, bp.type, p, bp.param, bp.interval..., bp.precision, bp.status, bp.δ..., bp.step, bp.idx, bp.ind_ev)
	end
end

####################################################################################################
# types for bifurcation point with 1d kernel for the jacobian

for (op, opt) in ((:BranchPoint, AbstractSimpleBranchPoint),
					(:Pitchfork, AbstractSimpleBranchPoint),
					(:Fold, AbstractSimpleBranchPoint),
					(:Transcritical, AbstractSimpleBranchPoint),
					(:PeriodDoubling, AbstractSimpleBranchPointForMaps),
					)
	@eval begin
		"""
		$(TYPEDEF)

		$(TYPEDFIELDS)

		## Predictor

		You can call `predictor(bp, ds; kwargs...)` on such bifurcation point `bp`
		to find the zeros of the normal form polynomials.
		"""
		mutable struct $op{Tv, T, Tpar, Tlens <: Lens, Tevl, Tevr, Tnf} <: $opt
			"Bifurcation point."
			x0::Tv

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

Pitchfork(x0, p, params, lens, ζ, ζ★, nf) = Pitchfork(x0, p, params, lens, ζ, ζ★, nf, real(nf.b1) * real(nf.b3) < 0 ? :SuperCritical : :SubCritical)

isTranscritical(bp::AbstractSimpleBranchPoint) = bp isa Transcritical
type(bp::BranchPoint) = :BranchPoint
type(bp::Pitchfork) = :Pitchfork
type(bp::Fold) = :Fold
type(bp::Transcritical) = :Transcritical
type(bp::PeriodDoubling) = :PeriodDoubling
type(::Nothing) = nothing

function Base.show(io::IO, bp::AbstractBifurcationPoint)
	println(io, type(bp), " bifurcation point at ", getLensSymbol(bp.lens)," ≈ $(bp.p)")
	println(io, "Normal form (aδμ + b1⋅x⋅δμ + b2⋅x^2/2 + b3⋅x^3/6): \n", bp.nf)
end

function Base.show(io::IO, bp::Pitchfork) #a⋅(p - pbif) + x⋅(b1⋅(p - pbif) + b2⋅x/2 + b3⋅x^2/6)
	print(io, bp.type, " - ")
	println(io, type(bp), " bifurcation point at ", getLensSymbol(bp.lens)," ≈ $(bp.p)")
	println(io, "Normal form x⋅(b1⋅δp + b3⋅x²/6): \n", bp.nf)
end

function Base.show(io::IO, bp::PeriodDoubling)
	print(io, bp.type, " - ")
	println(io, type(bp), " bifurcation point at ", getLensSymbol(bp.lens)," ≈ $(bp.p)")
	println(io, "Normal form x⋅(a⋅δp + c⋅x³)")
	if ~isnothing(bp.nf)
		println(io,"- a = ", bp.nf.a)
		println(io,"- c = ", bp.nf.b3)
	end
	end


####################################################################################################
# type for bifurcation point Nd kernel for the jacobian

"""
This is a type which holds information for the bifurcation points of equilibria with dim(Ker)>1.

$(TYPEDEF)

$(TYPEDFIELDS)

## Associated methods

You can call `type(bp::NdBranchPoint), length(bp::NdBranchPoint)`.

## Predictor

You can call `predictor(bp, ds)` on such bifurcation point `bp` to find the zeros of the normal form polynomials.

## Manipulating the normal form

- You can use `bp(Val(:reducedForm), x, p)` to evaluate the normal form polynomials on the vector `x` for (scalar) parameter `p`.

- You can use `bp(x, δp::Real)` to get the (large dimensional guess) associated to the low dimensional vector `x`. Note that we must have `length(x) == length(bp)`.

- You can use `BifurcationKit.nf(bp; kwargs...)` to pretty print the normal form with a string.
"""
mutable struct NdBranchPoint{Tv, T, Tpar, Tlens <: Lens, Tevl, Tevr, Tnf} <: AbstractBranchPoint
	"Bifurcation point"
	x0::Tv

	"Parameter value at the bifurcation point"
	p::T

	"Parameters used by the vector field."
	params::Tpar

	"Parameter axis used to compute the branch on which this bifurcation point was detected."
	lens::Tlens

	"Right eigenvectors"
	ζ::Tevr

	"Left eigenvectors"
	ζ★::Tevl

	"Normal form coefficients"
	nf::Tnf

	"Type of bifurcation point"
	type::Symbol
end

type(bp::NdBranchPoint) = :NonSimpleBranchPoint
Base.length(bp::NdBranchPoint) = length(bp.ζ)

function Base.show(io::IO, bp::NdBranchPoint)
	println(io, "Non simple bifurcation point at ", getLensSymbol(bp.lens), " ≈ $(bp.p). \nKernel dimension = ", length(bp))
	println(io, "Normal form:")
	println(io, mapreduce(x -> x * "\n", *, nf(bp)) )
end
####################################################################################################
# type for Hopf bifurcation point

"""
$(TYPEDEF)

$(TYPEDFIELDS)

# Associated methods

## Predictor

You can call `predictor(bp::Hopf, ds)` on such bifurcation point `bp` to get the guess for the periodic orbit.
"""
mutable struct Hopf{Tv, T, Tω, Tpar, Tlens <: Lens, Tevr, Tevl, Tnf} <: AbstractSimpleBranchPoint
	"Hopf point"
	x0::Tv

	"Parameter value at the Hopf point"
	p::T

	"Frequency at the Hopf point"
	ω::Tω

	"Parameters used by the vector field."
	params::Tpar

	"Parameter axis used to compute the branch on which this Hopf point was detected."
	lens::Tlens

	"Right eigenvector"
	ζ::Tevr

	"Left eigenvector"
	ζ★::Tevl

	"Normal form coefficient ex: (a = 0., b = 1 + 1im)"
	nf::Tnf

	"Type of Hopf bifurcation"
	type::Symbol
end

type(bp::Hopf) = :Hopf
Hopf(x0, p, ω, params, lens, ζ, ζ★, nf) = Hopf(x0, p, ω, params, lens, ζ, ζ★, nf, real(nf.b1) * real(nb.b3) < 0 ? :SuperCritical : :SubCritical)

function Base.show(io::IO, bp::Hopf)
	print(io, bp.type, " - ")
	println(io, type(bp), " bifurcation point at ", getLensSymbol(bp.lens)," ≈ $(bp.p).")
	println(io, "Frequency ω ≈ ", abs(bp.ω))
	println(io, "Period of the periodic orbit ≈ ", abs(2pi/bp.ω))
	println(io, "Normal form z⋅(iω + a⋅δp + b⋅|z|²): \n", bp.nf)
end
####################################################################################################
# type for Neimark-Sacker bifurcation (of Maps)
"""
$(TYPEDEF)

$(TYPEDFIELDS)

# Associated methods

## Predictor

You can call `predictor(bp::NeimarkSacker, ds)` on such bifurcation point `bp` to get the guess for the periodic orbit.
"""
mutable struct NeimarkSacker{Tv, T, Tω, Tpar, Tlens <: Lens, Tevr, Tevl, Tnf} <: AbstractSimpleBranchPointForMaps
	"Neimark-Sacker point"
	x0::Tv

	"Parameter value at the Neimark-Sacker point"
	p::T

	"Frequency at the Neimark-Sacker point"
	ω::Tω

	"Parameters used by the vector field."
	params::Tpar

	"Parameter axis used to compute the branch on which this Neimark-Sacker point was detected."
	lens::Tlens

	"Right eigenvector"
	ζ::Tevr

	"Left eigenvector"
	ζ★::Tevl

	"Normal form coefficient ex: (a = 0., b = 1 + 1im)"
	nf::Tnf

	"Type of Hopf bifurcation"
	type::Symbol
end

type(bp::NeimarkSacker) = :NeimarkSacker

function Base.show(io::IO, bp::NeimarkSacker)
	print(io, bp.type, " - ")
	println(io, type(bp), " bifurcation point at ", getLensSymbol(bp.lens)," ≈ $(bp.p).")
	println(io, "Frequency θ ≈ ", abs(bp.ω))
	println(io, "Period of the periodic orbit ≈ ", abs(2pi/bp.ω))
	println(io, "Normal form z⋅eⁱᶿ(1 + a⋅δp + b⋅|z|²)")
	if ~isnothing(bp.nf)
		println(io,"- a = ", bp.nf.a)
		println(io,"- b = ", bp.nf.b)
	end
end
