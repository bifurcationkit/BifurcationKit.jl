function d2PO(f, x, dx1, dx2)
   return ForwardDiff.derivative(t2 -> ForwardDiff.derivative( t1 -> f(x .+ t1 .* dx1 .+ t2 .* dx2,), 0.), 0.)
end

struct FloquetWrapperBLS{T} <: AbstractBorderedLinearSolver
	solver::T # use solver as a field is good for BLS
end
(ls::FloquetWrapperBLS)(J, args...; k...) = ls.solver(J, args...; k...)
(ls::FloquetWrapperBLS)(J::FloquetWrapper, args...; k...) = ls.solver(J.jacpb, args...; k...)
Base.transpose(J::FloquetWrapper) = transpose(J.jacpb)

for op in (:NeimarkSackerProblemMinimallyAugmented,
			:PeriodDoublingProblemMinimallyAugmented)
	@eval begin
		"""
		$(TYPEDEF)

		Structure to encode functional based on a Minimally Augmented formulation.

		# Fields

		$(FIELDS)
		"""
		mutable struct $op{Tprob <: AbstractBifurcationProblem, vectype, T <: Real, S <: AbstractLinearSolver, Sa <: AbstractLinearSolver, Sbd <: AbstractBorderedLinearSolver, Sbda <: AbstractBorderedLinearSolver, Tmass} <: AbstractProblemMinimallyAugmented
			"Functional F(x, p) - vector field - with all derivatives"
			prob_vf::Tprob
			"close to null vector of Jᵗ"
			a::vectype
			"close to null vector of J"
			b::vectype
			"vector zero, to avoid allocating it many times"
			zero::vectype
			"Lyapunov coefficient"
			l1::Complex{T}
			"Cusp test value"
			CP::T
			"Bogdanov-Takens test value"
			FOLDNS::T
			"Generalised period douling test value"
			GPD::T
			"Fold-NS test values"
			FLIPNS::Int
			"linear solver. Used to invert the jacobian of MA functional"
			linsolver::S
			"linear solver for the jacobian adjoint"
			linsolverAdjoint::Sa
			"bordered linear solver"
			linbdsolver::Sbd
			"linear bordered solver for the jacobian adjoint"
			linbdsolverAdjoint::Sbda
			"wether to use the hessian of prob_vf"
			usehessian::Bool
			"wether to use a mass matrix M for studying M⋅∂tu = F(u), default = I"
			massmatrix::Tmass
		end

		@inline hasHessian(pb::$op) = hasHessian(pb.prob_vf)
		@inline isSymmetric(pb::$op) = isSymmetric(pb.prob_vf)
		@inline hasAdjoint(pb::$op) = hasAdjoint(pb.prob_vf)
		@inline hasAdjointMF(pb::$op) = hasAdjointMF(pb.prob_vf)
		@inline isInplace(pb::$op) = isInplace(pb.prob_vf)
		@inline getLens(pb::$op) = getLens(pb.prob_vf)
		jad(pb::$op, args...) = jad(pb.prob_vf, args...)

		# constructor
		function $op(prob, a, b, linsolve::AbstractLinearSolver, linbdsolver = MatrixBLS(); usehessian = true, massmatrix = LinearAlgebra.I)
			# determine scalar type associated to vectors a and b
			α = norm(a) # this is valid, see https://jutho.github.io/KrylovKit.jl/stable/#Package-features-and-alternatives-1
			Ty = eltype(α)
			return $op(prob, a, b, 0*a,
						complex(zero(Ty)),  # l1
						real(one(Ty)),		# cp
						real(one(Ty)),		# fold-ns
						real(one(Ty)),		# gpd
						1,					# flip-ns
						linsolve, linsolve, linbdsolver, linbdsolver, usehessian, massmatrix)
		end
	end
end

function correctBifurcation(contres::ContResult{<: Union{FoldPeriodicOrbitCont, PDPeriodicOrbitCont}})
	if contres.prob.prob isa FoldProblemMinimallyAugmented
		conversion = Dict(:bp => :R1, :hopf => :foldNS, :fold => :cusp, :nd => :nd, :pd => :foldpd)
	elseif contres.prob.prob isa PeriodDoublingProblemMinimallyAugmented
		conversion = Dict(:bp => :foldFlip, :hopf => :pdNS, :pd => :R2,)
	elseif contres.prob.prob isa NeimarkSackerProblemMinimallyAugmented
		conversion = Dict(:bp => :foldNS, :hopf => :nsns, :pd => :pdNS,)
	else
		throw("Error! this should not occur. Please open an issue on the website of BifurcationKit.jl")
	end
	for (ind, bp) in pairs(contres.specialpoint)
		if bp.type in keys(conversion)
			@set! contres.specialpoint[ind].type = conversion[bp.type]
		end
	end
	return contres
end
