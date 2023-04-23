using SciMLBase: ODEProblem, DAEProblem, EnsembleProblem, terminate!, solve, VectorContinuousCallback
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
_sync_jacobian!(sh) = @set! sh.flow.jacobian = sh.jacobian

# this constructor takes into account a parameter passed to the vector field
# if M = 1, we disable parallel processing
function ShootingProblem(prob::ODEType, alg, ds, section; parallel = false, par = prob.p, kwargs...)
	_M = length(ds)
	parallel = _M == 1 ? false : parallel
	_pb = parallel ? EnsembleProblem(prob) : prob
	kwargsSh = [k for k in kwargs if first(k) ∈ fieldnames(ShootingProblem)]
	kwargsDE = setdiff(kwargs, kwargsSh)
	sh = ShootingProblem(;M = _M, flow = Flow(_pb, alg; kwargsDE...), kwargsSh..., ds = ds, section = section, parallel = parallel, par = par)
	# set jacobian for the flow too
	_sync_jacobian!(sh)
end

ShootingProblem(prob::ODEType, alg, M::Int, section; kwargs...) = ShootingProblem(prob, alg, diff(LinRange(0, 1, M + 1)), section; kwargs...)

function ShootingProblem(prob::ODEType, alg, centers::AbstractVector; parallel = false, par = prob.p, kwargs...)
	F = getVectorField(prob)
	sh = ShootingProblem(prob, alg, diff(LinRange(0, 1, length(centers) + 1)), SectionSS(F(centers[1], par) ./ norm(F(centers[1], par)), centers[1]); parallel = parallel, par = par, kwargs...)
	# set jacobian for the flow too
	_sync_jacobian!(sh)
end

# this is the "simplest" constructor to use in automatic branching from Hopf
ShootingProblem(M::Int, prob::ODEType, alg; kwargs...) = ShootingProblem(prob, alg, M, nothing; kwargs...)

# idem but with an ODEproblem to define the derivative of the flow
function ShootingProblem(prob1::ODEType, alg1, prob2::ODEType, alg2, ds, section; parallel = false, par = prob1.p, kwargs...)
	_M = length(ds)
	parallel = _M == 1 ? false : parallel
	_pb1 = parallel ? EnsembleProblem(prob1) : prob1
	_pb2 = parallel ? EnsembleProblem(prob2) : prob2
	kwargsSh = [k for k in kwargs if first(k) ∈ fieldnames(ShootingProblem)]
	kwargsDE = setdiff(kwargs, kwargsSh)
	sh = ShootingProblem(;M = _M, flow = Flow(_pb1, alg1, _pb2, alg2; kwargsDE...), kwargsSh..., ds = ds, section = section, parallel = parallel, par = par)
	# set jacobian for the flow too
	_sync_jacobian!(sh)
end

ShootingProblem(M::Int, prob1::ODEType, alg1, prob2::ODEType, alg2; kwargs...) = ShootingProblem(prob1, alg1, prob2, alg2, M, nothing; kwargs...)

ShootingProblem(prob1::ODEType, alg1, prob2::ODEType, alg2, M::Int, section; kwargs...) = ShootingProblem(prob1, alg1, prob2, alg2, diff(LinRange(0, 1, M + 1)), section; kwargs...)

function ShootingProblem(prob1::ODEType, alg1, prob2::ODEType, alg2, centers::AbstractVector; kwargs...)
	F = getVectorField(prob1)
	p = prob1.p # parameters
	sh = ShootingProblem(prob1, alg1, prob2, alg2, diff(LinRange(0, 1, length(centers) + 1)), SectionSS(F(centers[1], p)./ norm(F(centers[1], p)), centers[1]); kwargs...)
	# set jacobian for the flow too
	_sync_jacobian!(sh)
end
####################################################################################################
### 									POINCARE SHOOTING
####################################################################################################
function PoincareShootingProblem(prob::ODEProblem,
								alg,
								hyp::SectionPS;
								δ = 1e-8,
								interp_points = 50,
								parallel = false,
								par = prob.p,
								kwargs...)
	pSection(out, u, t, integrator) = (hyp(out, u); out .*= integrator.iter > 1)
	affect!(integrator, idx) = terminate!(integrator)
	# we put nothing option to have an upcrossing
	cb = VectorContinuousCallback(pSection, affect!, hyp.M; interp_points = interp_points, affect_neg! = nothing)
	# change ODEProblem -> EnsembleProblem in the parallel case
	_M = hyp.M
	parallel = _M == 1 ? false : parallel
	_pb = parallel ? EnsembleProblem(prob) : prob

	kwargsSh = [k for k in kwargs if first(k) ∈ fieldnames(PoincareShootingProblem)]
	kwargsDE = setdiff(kwargs, kwargsSh)

	psh = PoincareShootingProblem(;
				flow = Flow(_pb, alg; callback = cb, kwargsDE...),
				kwargsSh...,
				M = hyp.M,
				section = hyp,
				parallel = parallel,
				par = par)
	# set jacobian for the flow too
	_sync_jacobian!(psh)
end

# this is the "simplest" constructor to use in automatic branching from Hopf
# this is a Hack to pass the arguments to construct a Flow. Indeed, we need to provide the
# appropriate callback for Poincare Shooting to work
function PoincareShootingProblem(M::Int,
							prob::ODEProblem,
							alg;
							parallel = false,
							section = SectionPS(M),
							par = prob.p,
							kwargs...)
	kwargsSh = [k for k in kwargs if first(k) ∈ fieldnames(PoincareShootingProblem)]
	kwargsDE = setdiff(kwargs, kwargsSh)
	psh = PoincareShootingProblem(;
				M = M,
				flow = (par = par, prob = prob, alg = alg, kwargs = kwargsDE),
				kwargsSh...,
				parallel = (M == 1 ? false : parallel),
				section = section,
				par = par)
end

function PoincareShootingProblem(M::Int,
					prob1::ODEProblem, alg1,
					prob2::ODEProblem, alg2;
					parallel = false,
					section = SectionPS(M),
					lens = nothing,
					updateSectionEveryStep = 0,
					jacobian = :autodiffDenseAnalytical,
					par = prob1.p,
					kwargs...)
	kwargsSh = [k for k in kwargs if first(k) ∈ fieldnames(PoincareShootingProblem)]
	kwargsDE = setdiff(kwargs, kwargsSh)

	psh = PoincareShootingProblem(M = M, flow = (par = prob1.p, prob1 = prob1, alg1 = alg1, prob2 = prob2, alg2 = alg2, kwargs = kwargsDE), kwargsSh..., parallel = parallel, section = section, par = par)
end

function PoincareShootingProblem(prob::ODEProblem,
								alg,
								normals::AbstractVector,
								centers::AbstractVector;
								δ = 1e-8,
								interp_points = 50,
								parallel = false,
								radius = Inf,
								par = prob.p,
								kwargs...)

	psh = PoincareShootingProblem(prob, alg,
					SectionPS(normals, centers); # radius = radius);
					δ = δ, interp_points = interp_points, parallel = parallel, par = par, kwargs...)
	# set jacobian for the flow too
	_sync_jacobian!(psh)
end

function PoincareShootingProblem(prob1::ODEProblem, alg1,
								prob2::ODEProblem, alg2,
								hyp::SectionPS;
								δ = 1e-8,
								interp_points = 50,
								parallel = false,
								par = prob1.p,
								kwargs...)
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

	kwargsSh = [k for k in kwargs if first(k) ∈ fieldnames(PoincareShootingProblem)]
	kwargsDE = setdiff(kwargs, kwargsSh)

	psh = PoincareShootingProblem(;
		M = hyp.M,
		flow = Flow(_pb1, alg1, _pb2, alg2; callback = cb, kwargsDE...), kwargsSh...,
		section = hyp,
		δ = δ,
		parallel = parallel,
		par = par)
	# set jacobian for the flow too
	_sync_jacobian!(psh)
end

function PoincareShootingProblem(prob1::ODEProblem, alg1,
								prob2::ODEProblem, alg2,
								normals::AbstractVector, centers::AbstractVector;
								δ = 1e-8,
								interp_points = 50,
								parallel = false,
								radius = Inf,
								kwargs...)
	psh = PoincareShootingProblem(prob1, alg2, prob2, alg2,
					SectionPS(normals, centers); #radius = radius);
					δ = δ, interp_points = interp_points, parallel = parallel, kwargs...)
	# set jacobian for the flow too
	_sync_jacobian!(psh)
end
