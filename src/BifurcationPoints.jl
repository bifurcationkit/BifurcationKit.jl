import Base: getproperty
abstract type BifurcationPoint end
abstract type BranchPoint <: BifurcationPoint end
abstract type SimpleBranchPoint <: BranchPoint end

istranscritical(bp::BranchPoint) = false
####################################################################################################
# types for bifurcation point 1d kernel for the jacobian

for op in (:Pitchfork, :Fold, :Transcritical)
	@eval begin
		"""
		$(TYPEDEF)
		$(TYPEDFIELDS)
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

####################################################################################################
# type for Hopf bifurcation point

"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
mutable struct HopfBifPoint{Tv, T, Tω, Tpar, Tlens <: Lens, Tevr, Tevl, Tnf} <: BifurcationPoint
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
