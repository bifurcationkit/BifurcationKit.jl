using DiffEqBase: ODEProblem, DAEProblem, EnsembleProblem, terminate!, solve, VectorContinuousCallback
const ODEType = Union{ODEProblem, DAEProblem}

function getVectorField(prob::Union{ODEProblem, DAEProblem})
	if isinplace(prob)
		return (x, p) -> (out = similar(x); prob.f(out, x, p, prob.tspan[1]); return out)
	else
		return (x, p) -> prob.f(x, p, prob.tspan[1])
	end
end
getVectorField(pb::EnsembleProblem) = getVectorField(pb.prob)
####################################################################################################
### 									STANDARD SHOOTING
####################################################################################################
# this constructor takes into accound a parameter passed to the vector field
# if M = 1, we disable parallel processing
function ShootingProblem(prob::ODEType, alg, ds, section; parallel = false, kwargs...)
	_M = length(ds)
	parallel = _M == 1 ? false : parallel
	_pb = parallel ? EnsembleProblem(prob) : prob
	return ShootingProblem(M = _M, flow = Flow(_pb, alg; kwargs...),
			ds = ds, section = section, parallel = parallel)
end

ShootingProblem(prob::ODEType, alg, M::Int, section; parallel = false, kwargs...) = ShootingProblem(prob, alg, diff(LinRange(0, 1, M + 1)), section; parallel = parallel, kwargs...)

function ShootingProblem(prob::ODEType, alg, centers::AbstractVector; parallel = false, kwargs...)
	p = prob.p # parameters
	F = getVectorField(prob)
	ShootingProblem(prob, alg, diff(LinRange(0, 1, length(centers) + 1)), SectionSS(F(centers[1], p)./ norm(F(centers[1], p)), centers[1]); parallel = parallel, kwargs...)
end

# this is the "simplest" constructor to use in automatic branching from Hopf
ShootingProblem(M::Int, prob::ODEType, alg; parallel = false, kwargs...) = ShootingProblem(prob, alg, M, nothing; parallel = parallel, kwargs...)

ShootingProblem(M::Int, prob1::ODEType, alg1, prob2::ODEType, alg2; parallel = false, kwargs...) = ShootingProblem(prob1, alg1, prob2, alg2, M, nothing; parallel = parallel, kwargs...)

# idem but with an ODEproblem to define the derivative of the flow
function ShootingProblem(prob1::ODEType, alg1, prob2::ODEType, alg2, ds, section; parallel = false, kwargs...)
	_M = length(ds)
	parallel = _M == 1 ? false : parallel
	_pb1 = parallel ? EnsembleProblem(prob1) : prob1
	_pb2 = parallel ? EnsembleProblem(prob2) : prob2
	ShootingProblem(M = _M, flow = Flow(_pb1, alg1, _pb2, alg2; kwargs...), ds = ds, section = section, parallel = parallel)
end

ShootingProblem(prob1::ODEType, alg1, prob2::ODEType, alg2, M::Int, section; parallel = false, kwargs...) = ShootingProblem(prob1, alg1, prob2, alg2, diff(LinRange(0, 1, M + 1)), section; parallel = parallel, kwargs...)

function ShootingProblem(prob1::ODEType, alg1, prob2::ODEType, alg2, centers::AbstractVector; parallel = false, kwargs...)
	F = getVectorField(prob1)
	p = prob1.p # parameters
	ShootingProblem(prob1, alg1, prob2, alg2, diff(LinRange(0, 1, length(centers) + 1)), SectionSS(F(centers[1], p)./ norm(F(centers[1], p)), centers[1]); parallel = parallel, kwargs...)
end
####################################################################################################
### 									POINCARE SHOOTING
####################################################################################################
function PoincareShootingProblem(prob::ODEProblem, alg,
			hyp::SectionPS;
			δ = 1e-8, interp_points = 50, parallel = false, kwargs...)
	p = prob.p # parameters
	pSection(out, u, t, integrator) = (hyp(out, u); out .*= integrator.iter > 1)
	affect!(integrator, idx) = terminate!(integrator)
	# we put nothing option to have an upcrossing
	cb = VectorContinuousCallback(pSection, affect!, hyp.M; interp_points = interp_points, affect_neg! = nothing)
	# change the ODEProblem -> EnsembleProblem for the parallel case
	_M = hyp.M
	parallel = _M == 1 ? false : parallel
	_pb = parallel ? EnsembleProblem(prob) : prob
	return PoincareShootingProblem(flow = Flow(_pb, alg; callback = cb, kwargs...), M = hyp.M, section = hyp, δ = δ, parallel = parallel)
end

# this is the "simplest" constructor to use in automatic branching from Hopf
# this is a Hack to pass the arguments to construct a Flow. Indeed, we need to provide the
# appropriate callback for Poincare Shooting to work
PoincareShootingProblem(M::Int, prob::ODEProblem, alg; parallel = false, section = SectionPS(M), kwargs...) = PoincareShootingProblem(M = M, flow = (par = prob.p, prob = prob, alg = alg, kwargs = kwargs), parallel = (M == 1 ? false : parallel), section = section)

PoincareShootingProblem(M::Int, prob1::ODEProblem, alg1, prob2::ODEProblem, alg2; parallel = false, section = SectionPS(M), kwargs...) = PoincareShootingProblem(M = M, flow = (par = prob1.p, prob1 = prob1, alg1 = alg1, prob2 = prob2, alg2 = alg2, kwargs = kwargs), parallel = parallel, section = section)

function PoincareShootingProblem(prob::ODEProblem, alg,
			normals::AbstractVector, centers::AbstractVector;
			δ = 1e-8, interp_points = 50, parallel = false, kwargs...)
	return PoincareShootingProblem(prob, alg,
					SectionPS(normals, centers);
					δ = δ, interp_points = interp_points, parallel = parallel, kwargs...)
end

function PoincareShootingProblem(prob1::ODEProblem, alg1,
				prob2::ODEProblem, alg2,
				hyp::SectionPS;
				δ = 1e-8, interp_points = 50, parallel = false, kwargs...)
	p = prob1.p # parameters
	pSection(out, u, t, integrator) = (hyp(out, u); out .*= integrator.iter > 1)
	affect!(integrator, idx) = terminate!(integrator)
	# we put nothing option to have an upcrossing
	cb = VectorContinuousCallback(pSection, affect!, hyp.M; interp_points = interp_points, affect_neg! = nothing)
	# change the ODEProblem -> EnsembleProblem for the parallel case
	_M = hyp.M
	parallel = _M == 1 ? false : parallel
	_pb1 = parallel ? EnsembleProblem(prob1) : prob1
	_pb2 = parallel ? EnsembleProblem(prob2) : prob2
	return PoincareShootingProblem(flow = Flow(_pb1, alg1, _pb2, alg2; callback = cb, kwargs...), M = hyp.M, section = hyp, δ = δ, parallel = parallel)
end

function PoincareShootingProblem(prob1::ODEProblem, alg1,
				prob2::ODEProblem, alg2,
				normals::AbstractVector, centers::AbstractVector;
				δ = 1e-8, interp_points = 50, parallel = false, kwargs...)
	return PoincareShootingProblem(prob1, alg2, prob2, alg2,
					SectionPS(normals, centers);
					δ = δ, interp_points = interp_points, parallel = parallel, kwargs...)
end
