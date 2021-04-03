abstract type AbstractBifurcationPoint end
abstract type AbstractBranchPoint <: AbstractBifurcationPoint end
abstract type AbstractSimpleBranchPoint <: AbstractBranchPoint end

istranscritical(bp::AbstractBranchPoint) = false
####################################################################################################
"""
$(TYPEDEF)

Structure to record a generic bifurcation point.

$(TYPEDFIELDS)
"""
@with_kw struct GenericBifPoint{T, Tp, Tv} <: AbstractBifurcationPoint
	"Bifurcation type, `:hopf, :bp...`."
	type::Symbol = :none

	"Index in `br.eig` (see [`ContResult`](@ref)) for which the bifurcation occurs."
	idx::Int64 = 0

	"Parameter value at the bifurcation point, this is an estimate."
	param::T = 0.

	"Norm of the equilibrium at the bifurcation point"
	norm::T  = 0.

	"`printsol = printSolution(x, param)` where `printSolution` is one of the arguments to [`continuation`](@ref)"
	printsol::Tp = 0.

	"Equilibrium at the bifurcation point"
	x::Tv = Vector{T}(undef, 0)

	"Tangent along the branch at the bifurcation point"
	tau::BorderedArray{Tv, T} = BorderedArray(x, T(0))

	"Eigenvalue index responsible for the bifurcation (if applicable)"
	ind_ev::Int64 = 0

	"Continuation step at which the bifurcation occurs"
	step::Int64 = 0

	"`status ∈ {:converged, :guess}` indicates whether the bisection algorithm was successful in detecting the bifurcation point"
	status::Symbol = :guess

	"`δ = (δr, δi)` where δr indicates the change in the number of unstable eigenvalues and δi indicates the change in the number of unstable eigenvalues with nonzero imaginary part. `abs(δr)` is thus an estimate of the dimension of the kernel of the Jacobian at the bifurcation point."
	δ::Tuple{Int64, Int64} = (0,0)

	"Precision in the location of the bifurcation point"
	precision::T = -1

	"Interval containing the bifurcation point"
	interval::Tuple{T, T} = (0, 0)
end

getVectorType(::Type{GenericBifPoint{T, Tp, Tv}}) where {T, Tp, Tv} = Tv

# constructor
GenericBifPoint(x0, T, printsol) = GenericBifPoint(type = :none, idx = 0, param = T(0), norm  = T(0), printsol = namedprintsol(printsol), x = x0, tau = BorderedArray(x0, T(0)), ind_ev = 0, step = 0, status = :guess, δ = (0, 0), precision = T(-1), interval = (T(0), T(0)))

function _show(io::IO, bp::GenericBifPoint, ii::Int, p::String = "p")
	if bp.type == :none ; return; end
	if bp.status == :converged
		@printf(io, "- #%3i,\e[1;34m %5s\e[0m at %s ≈ %+4.8f ∈ (%+4.8f, %+4.8f), |δp|=%1.0e, [\e[1;32m%9s\e[0m], δ = (%2i, %2i), step = %3i, eigenelements in eig[%3i], ind_ev = %3i\n", ii, bp.type, p, bp.param, bp.interval..., bp.precision, bp.status, bp.δ..., bp.step, bp.idx, bp.ind_ev)
	else
		@printf(io, "- #%3i,\e[1;34m %5s\e[0m at %s ≈ %+4.8f ∈ (%+4.8f, %+4.8f), |δp|=%1.0e, [\e[1;31m%9s\e[0m], δ = (%2i, %2i), step = %3i, eigenelements in eig[%3i], ind_ev = %3i\n", ii, bp.type, p, bp.param, bp.interval..., bp.precision, bp.status, bp.δ..., bp.step, bp.idx, bp.ind_ev)
	end
end

@inline kernelDim(bp::GenericBifPoint) = abs(bp.δ[1])
####################################################################################################
# types for bifurcation point with 1d kernel for the jacobian

for op in (:Pitchfork, :Fold, :Transcritical)
	@eval begin
		"""
		$(TYPEDEF)
		$(TYPEDFIELDS)

		## Associated methods

		You can call `istranscritical(bp::AbstractSimpleBranchPoint), type(bp::AbstractSimpleBranchPoint)`

		## Predictor

		You can call `predictor(bp, ds; kwargs...)` on such bifurcation point `bp`
		to find the zeros of the normal form polynomials.
		"""
		mutable struct $op{Tv, T, Tpar, Tlens <: Lens, Tevl, Tevr, Tnf} <: AbstractSimpleBranchPoint
			"bifurcation point."
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
			ζstar::Tevl

			"Normal form coefficients."
			nf::Tnf

			"Type of bifurcation point"
			type::Symbol
		end
	end
end

Pitchfork(x0, p, params, lens, ζ, ζstar, nf) = Pitchfork(x0, p, params, lens, ζ, ζstar, nf, real(nf.b1) * real(nf.b3) < 0 ? :SuperCritical : :SubCritical)

isTranscritical(bp::AbstractSimpleBranchPoint) = bp isa Transcritical
type(bp::Pitchfork) = :Pitchfork
type(bp::Fold) = :Fold
type(bp::Transcritical) = :Transcritical
type(::Nothing) = nothing
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
	"bifurcation point"
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
	ζstar::Tevl

	"Normal form coefficients"
	nf::Tnf

	"Type of bifurcation point"
	type::Symbol
end

type(bp::NdBranchPoint) = :NonSimpleBranchPoint
Base.length(bp::NdBranchPoint) = length(bp.ζ)

function Base.show(io::IO, bp::NdBranchPoint)
	if bp isa Pitchfork || bp isa HopfBifPoint
		print(io, bp.type, " - ")
	end
	println(io, "Non simple bifurcation point at ", getLensParam(bp.lens), " ≈ $(bp.p). \nKernel dimension = ", length(bp))
	println(io, "Normal form :")
	println(io, mapreduce(x -> x * "\n", *, nf(bp)) )
end
####################################################################################################
# type for Hopf bifurcation point

"""
$(TYPEDEF)
$(TYPEDFIELDS)

# Associated methods

## Predictor

You can call `predictor(bp, ds)` on such bifurcation point `bp` to get to find the guess for the periodic orbit.
"""
mutable struct HopfBifPoint{Tv, T, Tω, Tpar, Tlens <: Lens, Tevr, Tevl, Tnf} <: AbstractSimpleBranchPoint
	"Hopf point"
	x0::Tv

	"Parameter value at the Hopf point"
	p::T

	"Frequency of the Hopf point"
	ω::Tω

	"Parameters used by the vector field."
	params::Tpar

	"Parameter axis used to compute the branch on which this Hopf point was detected."
	lens::Tlens

	"Right eigenvector"
	ζ::Tevr

	"Left eigenvector"
	ζstar::Tevl

	"Normal form coefficient ex: (a = 0., b = 1 + 1im)"
	nf::Tnf

	"Type of Hopf bifurcation"
	type::Symbol
end

type(bp::HopfBifPoint) = :Hopf
HopfBifPoint(x0, p, ω, params, lens, ζ, ζstar, nf) = HopfBifPoint(x0, p, ω, params, lens, ζ, ζstar, nf, real(nf.b1) * real(nb.b3) < 0 ? :SuperCritical : :SubCritical)

function Base.show(io::IO, bp::AbstractBifurcationPoint)
	if bp isa Pitchfork || bp isa HopfBifPoint
		print(io, bp.type, " - ")
	end
	println(io, type(bp), " bifurcation point at ", getLensParam(bp.lens)," ≈ $(bp.p).")
	bp isa HopfBifPoint && println(io, "Period of the periodic orbit ≈ ", (2pi/bp.ω))
	println(io, "Normal form: ", bp.nf)
end
