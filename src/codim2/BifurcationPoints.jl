####################################################################################################
"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
@with_kw_noshow mutable struct Cusp{Tv, Tpar, Tlens, Tevr, Tevl, Tnf} <: AbstractBifurcationPoint
	"Cusp point"
	x0::Tv

	"Parameters used by the vector field."
	params::Tpar

	"Parameter axis used to compute the branch on which this cusp point was detected."
	lens::Tlens

	"Right eigenvector"
	ζ::Tevr

	"Left eigenvector"
	ζstar::Tevl

	"Normal form coefficients"
	nf::Tnf

	"Type of bifurcation"
	type::Symbol
end

type(bp::Cusp) = :Cusp

function Base.show(io::IO, bp::Cusp)
	lens1, lens2 = bp.lens
	p1 = get(bp.params, lens1)
	p2 = get(bp.params, lens2)
	println(io, "Cusp bifurcation point at ", getLensSymbol(lens1, lens2)," ≈ ($p1, $p2).")
	# avoid aliasing with user defined parameters
	p1 = :β1 == getLensSymbol(lens1) ? :p1 : :β1
	p2 = :β2 == getLensSymbol(lens2) ? :p2 : :β2
	println(io, "Normal form: $p1 + $p2⋅A + c⋅A³)")
	@unpack c = bp.nf
	println(io, "Normal form coefficients:\n c = $c")
end
####################################################################################################
"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
@with_kw_noshow mutable struct BogdanovTakens{Tv, Tpar, Tlens, Tevr, Tevl, Tnf, Tnf2} <: AbstractBifurcationPoint
	"Bogdanov-Takens point"
	x0::Tv

	"Parameters used by the vector field."
	params::Tpar

	"Parameter axis used to compute the branch on which this BT point was detected."
	lens::Tlens

	"Right eigenvectors"
	ζ::Tevr

	"Left eigenvectors"
	ζstar::Tevl

	"Normal form coefficients (basic)"
	nf::Tnf

	"Normal form coefficients (detailed)"
	nfsupp::Tnf2

	"Type of bifurcation"
	type::Symbol
end

type(bp::BogdanovTakens) = :BogdanovTakens

function Base.show(io::IO, bp::BogdanovTakens)
	lens1, lens2 = bp.lens
	p1 = get(bp.params, lens1)
	p2 = get(bp.params, lens2)
	println(io, "Bogdanov-Takens bifurcation point at ", getLensSymbol(lens1, lens2)," ≈ ($p1, $p2).")
	# avoid aliasing with user defined parameters
	p1 = :β1 == getLensSymbol(lens1) ? :p1 : :β1
	p2 = :β2 == getLensSymbol(lens2) ? :p2 : :β2
	println(io, "Normal form (B, $p1 + $p2⋅B + b⋅A⋅B + a⋅A²)")
	@unpack a,b = bp.nf
	println(io, "Normal form coefficients:\n a = $a\n b = $b")
end
####################################################################################################
"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
mutable struct Bautin{Tv, Tpar, Tlens, Tevr, Tevl, Tnf} <: AbstractBifurcationPoint
	"Bautin point"
	x0::Tv

	"Parameters used by the vector field."
	params::Tpar

	"Parameter axis used to compute the branch on which this Bautin point was detected."
	lens::Tlens

	"Right eigenvectors"
	ζ::Tevr

	"Left eigenvectors"
	ζstar::Tevl

	"Normal form coefficients"
	nf::Tnf

	"Type of Bautin bifurcation"
	type::Symbol
end

type(bp::Bautin) = :Bautin

function Base.show(io::IO, bp::Bautin)
	lens1, lens2 = bp.lens
	p1 = get(bp.params, lens1)
	p2 = get(bp.params, lens2)
	println(io, "Bautin bifurcation point at ", getLensSymbol(lens1, lens2)," ≈ ($p1, $p2).")
	println(io, "ω = ", bp.nf.ω)
	println(io, "Second lyapunov coefficient l2 = ", bp.nf.l2)
	println(io, "Normal form: i⋅ω⋅u + l2⋅u⋅|u|⁴")
	println(io, "Normal form coefficients (detailed):")
	println(io, bp.nf)
end
