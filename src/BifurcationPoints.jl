abstract type BifurcationPoint end
abstract type BranchPoint <: BifurcationPoint end
abstract type SimpleBranchPoint <: BranchPoint end

istranscritical(bp::BranchPoint) = false
####################################################################################################
"""
$(TYPEDEF)

Structure to record a generic bifurcation point which was only detected by a change in the number of stable eigenvalues.

$(TYPEDFIELDS)
"""
@with_kw struct GenericBifPoint{T, Tp, Tv} <: BifurcationPoint
	"Bifurcation type, `:hopf, :bp...`,"
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

getvectortype(::Type{GenericBifPoint{T, Tp, Tv}}) where {T, Tp, Tv} = Tv

function _show(io::IO, bp::GenericBifPoint, ii::Int)
	if bp.type == :none || bp.precision < 0; return; end
	if bp.status == :converged
		@printf(io, "- #%3i,\e[1;34m %5s\e[0m at p ≈ %+4.8f ∈ (%+4.8f, %+4.8f), |δp|=%1.0e, [\e[1;32m%9s\e[0m], δ = (%2i, %2i), step = %3i, eigenelements in eig[%3i], ind_ev = %3i\n", ii, bp.type, bp.param, bp.interval..., bp.precision, bp.status, bp.δ..., bp.step, bp.idx, bp.ind_ev)
	else
		@printf(io, "- #%3i,\e[1;34m %5s\e[0m at p ≈ %+4.8f ∈ (%+4.8f, %+4.8f), |δp|=%1.0e, [\e[1;31m%9s\e[0m], δ = (%2i, %2i), step = %3i, eigenelements in eig[%3i], ind_ev = %3i\n", ii, bp.type, bp.param, bp.interval..., bp.precision, bp.status, bp.δ..., bp.step, bp.idx, bp.ind_ev)
	end
end

function _showFold(io::IO, bp::GenericBifPoint, ii::Int)
	# if bp.precision <= 0 return nothing; end
	@printf(io, "- #%3i,\e[1;34m fold\e[0m at p ≈ %4.8f ∈ (%4.8f, %4.8f), |δp|=%1.0e, [\e[1;34m%9s\e[0m], δ = (%2i, %2i), step = %3i, eigenelements in eig[%3i], ind_ev = %3i\n", ii, bp.param, bp.interval..., bp.precision, bp.status, bp.δ..., bp.step, bp.idx, bp.ind_ev)
end
@inline kerneldim(bp::GenericBifPoint) = abs(bp.δ[1])
####################################################################################################
# types for bifurcation point 1d kernel for the jacobian

for op in (:Pitchfork, :Fold, :Transcritical)
	@eval begin
		"""
		$(TYPEDEF)
		$(TYPEDFIELDS)

		## Associated methods

		You can call `istranscritical(bp::SimpleBranchPoint), type(bp::SimpleBranchPoint)`

		## Predictor

		You can call `predictor(bp, ds; kwargs...)` on such bifurcation point `bp` to get to find zeros of the normal form polynomials.
		"""
		mutable struct $op{Tv, T, Tpar, Tlens <: Lens, Tevl, Tevr, Tnf} <: SimpleBranchPoint
			"bifurcation point."
			x0::Tv

			"Parameter value at the bifurcation point."
			p::T

			"Parameters used by the vector field."
			params::Tpar

			"Parameter axis used to compute the branch on which this Branch point was detected."
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

istranscritical(bp::SimpleBranchPoint) = bp isa Transcritical
type(bp::Pitchfork) = :Pitchfork
type(bp::Fold) = :Fold
type(bp::Transcritical) = :Transcritical
type(::Nothing) = nothing
####################################################################################################
# type for bifurcation point Nd kernel for the jacobian

"""
This is a type which holds information for the bifurcation points of equilibria.

$(TYPEDEF)
$(TYPEDFIELDS)

## Associated methods

You can call `type(bp::NdBranchPoint), length(bp::NdBranchPoint)`.

## Predictor

You can call `predictor(bp, ds)` on such bifurcation point `bp` to get to find zeros of the normal form polynomials.

## Manipulating the normal form

- You can use `bp(Val(:reducedForm), x, p)` to evaluate the normal form polynomials on thw vector `x` for (scalar) parameter `p`.

- You can use `bp(x, δp::Real)` to get the (large dimensional guess) associated to the low dimensional vector `x`. Note that we must have `length(x) == length(bp)`.

- You can use `BifurcationKit.nf(bp; kwargs...)` to print the normal form with a nice string.
"""
mutable struct NdBranchPoint{Tv, T, Tpar, Tlens <: Lens, Tevl, Tevr, Tnf} <: BranchPoint
	"bifurcation point"
	x0::Tv

	"Parameter value at the bifurcation point"
	p::T

	"Parameters used by the vector field."
	params::Tpar

	"Parameter axis used to compute the branch on which this Branch point was detected."
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
	println(io, "Non simple bifurcation point at p ≈ $(bp.p). \nKernel dimension = ", length(bp))
	println(io, "Normal form :")
	println(io, mapreduce(x->x*"\n",*, nf(bp)) )
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
mutable struct HopfBifPoint{Tv, T, Tω, Tpar, Tlens <: Lens, Tevr, Tevl, Tnf} <: SimpleBranchPoint
	"Hopf point"
	x0::Tv

	"Parameter value at the Hopf point"
	p::T

	"Frequency of the Hopf point"
	ω::Tω

	"Parameters used by the vector field."
	params::Tpar

	"Parameter axis used to compute the branch on which this Branch point was detected."
	lens::Tlens

	"Right eigenvector"
	ζ::Tevr

	"Left eigenvector"
	ζstar::Tevl

	"Normal form coefficient (a = 0., b = 1 + 1im)"
	nf::Tnf

	"Type of Hopf bifurcation"
	type::Symbol
end

type(bp::HopfBifPoint) = :Hopf
HopfBifPoint(x0, p, ω, params, lens, ζ, ζstar, nf) = HopfBifPoint(x0, p, ω, params, lens, ζ, ζstar, nf, real(nf.b1) * real(nb.b3) < 0 ? :SuperCritical : :SubCritical)

function Base.show(io::IO, bp::BifurcationPoint)
	if bp isa Pitchfork || bp isa HopfBifPoint
		print(io, bp.type, " - ")
	end
	println(io, type(bp), " bifurcation point at p ≈ $(bp.p).")
	bp isa HopfBifPoint && println(io, "Period of the periodic orbit ≈ $(2pi/bp.ω).")
	println(io, "Normal form: ", bp.nf)
end
