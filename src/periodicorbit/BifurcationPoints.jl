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

			"Underlying normal form for Poincaré return map"
			nf::Tnf

			"Periodic orbit problem"
			prob::Tprob
		end
	end
end

type(bp::PeriodDoublingPO) = :PeriodDoubling
type(bp::BranchPointPO) = :BranchPoint

function Base.show(io::IO, pd::PeriodDoublingPO)
	println(io, "Period-Doubling bifurcation point of periodic orbit at\n", getLensSymbol(pd.nf.lens)," ≈ $(pd.nf.p)")
	println(io, "Period = ", abs(pd.T), " -> ", 2abs(pd.T))
	println(io, "Problem: ", typeof(pd.prob).name.name)
	if pd.prob isa ShootingProblem
		show(io, pd.nf)
	else
		println(io, "Normal form:\n∂τ = 1 + a⋅ξ²\n∂ξ = c⋅ξ³\n", pd.nf.nf)
	end
end

function Base.show(io::IO, bp::BranchPointPO)
	println(io, type(bp), " bifurcation point of periodic orbit at\n", getLensSymbol(bp.nf.lens)," ≈ $(bp.nf.p)")
	println(io, "Period = ", abs(bp.T))
	println(io, "Problem: ", typeof(bp.prob).name.name)
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
end

type(bp::NeimarkSackerPO) = :NeimarkSacker

function Base.show(io::IO, bp::NeimarkSackerPO)
	println(io, type(bp), " bifurcation point of periodic orbit at\n", getLensSymbol(bp.nf.lens)," ≈ $(bp.p).")
	println(io, "Frequency ω ≈ ", abs(bp.ω))
	println(io, "Period at the periodic orbit ≈ ", abs(bp.T))
	println(io, "Second frequency of the bifurcated torus ≈ ", abs(bp.ω))
	println(io, "Problem: \n", bp.prob)
end
