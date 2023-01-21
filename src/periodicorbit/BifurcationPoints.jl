abstract type AbstractBifurcationPointOfPO <: AbstractBifurcationPoint end
abstract type AbstractSimpleBifurcationPointPO <: AbstractBifurcationPointOfPO end
####################################################################################################
# types for bifurcation point with 1d kernel for the jacobian

for op in (:BranchPointPeriodicOrbit, :PeriodDoubling,)
	@eval begin
		"""
		$(TYPEDEF)

		$(TYPEDFIELDS)

		## Predictor

		You can call `predictor(bp, ds; kwargs...)` on such bifurcation point `bp`
		to find the zeros of the normal form polynomials.
		"""
		mutable struct $op{Tprob, Tv, T, Tpar, Tlens <: Lens, Tevl, Tevr, Tnf} <: AbstractSimpleBifurcationPointPO
			"Bifurcation point (periodic orbit)."
			po::Tv

			"Period"
			T::T

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

			"Periodic orbit problem"
			prob::Tprob
		end
	end
end

# PeriodDoubling(x0, p, params, lens, ζ, ζstar, nf) = PeriodDoubling(x0, p, params, lens, ζ, ζstar, nf, real(nf.b1) * real(nf.b3) < 0 ? :SuperCritical : :SubCritical)

type(bp::PeriodDoubling) = :PeriodDoubling
type(bp::BranchPointPeriodicOrbit) = :BranchPointPeriodicOrbit

function Base.show(io::IO, pd::PeriodDoubling)
	print(io, pd.type, " - ")
	println(io, " Period-Doubling bifurcation point at ", getLensSymbol(pd.lens)," ≈ $(pd.p)")
	println(io, "Period = ", abs(pd.T), " -> ", 2abs(pd.T))
	println(io, "Problem : ", typeof(pd.prob).name.name)
	println(io, "Normal form:\n∂τ = 1 + a⋅ξ²\n∂ξ = c⋅ξ³\n", pd.nf)
end

function Base.show(io::IO, bp::BranchPointPeriodicOrbit)
	print(io, bp.type, " - ")
	println(io, type(bp), " bifurcation point at ", getLensSymbol(bp.lens)," ≈ $(bp.p)")
	println(io, "Period = ", abs(bp.T))
	println(io, "Problem : ", typeof(bp.prob).name.name)
end

####################################################################################################
# type for Neimark-Sacker bifurcation point

"""
$(TYPEDEF)

$(TYPEDFIELDS)

# Associated methods

## Predictor

You can call `predictor(bp::Hopf, ds)` on such bifurcation point `bp` to get the guess for the periodic orbit.
"""
mutable struct NeimarkSacker{Tprob, Tv, T, Tω, Tpar, Tlens <: Lens, Tevr, Tevl, Tnf} <: AbstractSimpleBifurcationPointPO
	"Bifurcation point (periodic orbit)"
	po::Tv

	"Period"
	T::T

	"Parameter value at the Hopf point"
	p::T

	"Frequency of the Neimark-Sacker point"
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

	"Periodic orbit problem"
	prob::Tprob
end

type(bp::NeimarkSacker) = :NeimarkSacker
NeimarkSacker(x0, p, ω, params, lens, ζ, ζ★, nf) = NeimarkSacker(x0, p, ω, params, lens, ζ, ζ★, nf, real(nf.b1) * real(nb.b3) < 0 ? :SuperCritical : :SubCritical)

function Base.show(io::IO, bp::NeimarkSacker)
	print(io, bp.type, " - ")
	println(io, type(bp), " bifurcation point at ", getLensSymbol(bp.lens)," ≈ $(bp.p).")
	println(io, "Frequency ω ≈ ", abs(bp.ω))
	println(io, "Period at the periodic orbit ≈ ", abs(bp.T))
	println(io, "Second frequency of the bifurcated torus ≈ ", abs(bp.ω))
	println(io, "Computed using", typeof(bp.prob).name)
end
