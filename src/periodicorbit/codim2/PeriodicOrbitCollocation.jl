@inline hasAdjoint(::WrapPOColl) = false
@inline hasAdjointMF(::WrapPOColl) = false
@inline hasHessian(::WrapPOColl) = false
@inline getDelta(::WrapPOColl) = 1e-8

d2F(pbwrap::WrapPOColl, x, p, dx1, dx2) = d2PO(z -> pbwrap.prob(z, p), x, dx1, dx2)

function Base.transpose(J::FloquetWrapper{ <: PeriodicOrbitOCollProblem})
	@set J.jacpb = transpose(J.jacpb)
end

function Base.adjoint(J::FloquetWrapper{ <: PeriodicOrbitOCollProblem})
	@set J.jacpb = adjoint(J.jacpb)
end

function jacobianPeriodDoubling(pbwrap::WrapPOColl, x, par)
	N, m, Ntst = size(pbwrap.prob)
	Jac = jacobian(pbwrap, x, par)
	# put the PD boundary condition
	@set Jac.jacpb = copy(Jac.jacpb)
	J = Jac.jacpb
	J[end-N:end-1, 1:N] .= I(N)
	@set Jac.jacpb = J[1:end-1,1:end-1]
end

function jacobianNeimarkSacker(pbwrap::WrapPOColl, x, par, ω)
	N, m, Ntst = size(pbwrap.prob)
	Jac = jacobian(pbwrap, x, par)
	# put the NS boundary condition
	J = Complex.(copy(Jac.jacpb))
	J[end-N:end-1, end-N:end-1] .= UniformScaling(cis(ω))(N)
	@set Jac.jacpb = J[1:end-1,1:end-1]
end

function continuation(br::AbstractResult{Tkind, Tprob},
					ind_bif::Int64,
					lens2::Lens,
					options_cont::ContinuationPar = br.contparams ;
					bdlinsolver = MatrixBLS(),
					detectCodim2Bifurcation::Int = 0,
					kwargs...) where {Tkind <: PeriodicOrbitCont, Tprob <: WrapPOColl}
	biftype = br.specialpoint[ind_bif].type

	# options to detect codim2 bifurcations
	computeEigenElements = options_cont.detectBifurcation > 0
	_options_cont = detectCodim2Parameters(detectCodim2Bifurcation, options_cont; kwargs...)

	if biftype == :bp
		return continuationCOLL_Fold(br, ind_bif, lens2, options_cont; kwargs... )
	elseif biftype == :pd
		return continuationCOLL_PD(br, ind_bif, lens2, _options_cont; kwargs... )
	elseif biftype == :ns
		return continuationCOLL_NS(br, ind_bif, lens2, _options_cont; kwargs... )
	else
		throw("We continue only Fold / PD / NS points of periodic orbits for now")
	end
	nothing
end

function continuationCOLL_Fold(br::AbstractResult{Tkind, Tprob},
					ind_bif::Int64,
					lens2::Lens,
					options_cont::ContinuationPar = br.contparams ;
					startWithEigen = false,
					detectCodim2Bifurcation::Int = 0,
					bdlinsolver = MatrixBLS(),
					kwargs...) where {Tkind <: PeriodicOrbitCont, Tprob <: WrapPOColl}
	biftype = br.specialpoint[ind_bif].type
	bifpt = br.specialpoint[ind_bif]

	# we get the collocation problem
	probco = getProb(br).prob

	probcoFold = BifurcationProblem((x, p) -> probco(x, p), bifpt, getParams(br), getLens(br);
				J = (x, p) -> FloquetWrapper(probco, ForwardDiff.jacobian(z -> probco(z, p), x), x, p),
				d2F = (x, p, dx1, dx2) -> d2PO(z -> probco(z, p), x, dx1, dx2),
				plotSolution = (x,p;k...) -> br.prob.plotSolution(x.u,p;k...)
				)

	options_foldpo = @set options_cont.newtonOptions.linsolver = FloquetWrapperLS(options_cont.newtonOptions.linsolver)

	# perform continuation
	continuationFold(probcoFold,
		br, ind_bif, lens2,
		options_foldpo;
		startWithEigen = startWithEigen,
		bdlinsolver = FloquetWrapperBLS(bdlinsolver),
		kind = FoldPeriodicOrbitCont(),
		detectCodim2Bifurcation = detectCodim2Bifurcation,
		kwargs...
		)
end

function continuationCOLL_PD(br::AbstractResult{Tkind, Tprob},
					ind_bif::Int64,
					lens2::Lens,
					options_cont::ContinuationPar = br.contparams ;
					alg = br.alg,
					startWithEigen = false,
					detectCodim2Bifurcation::Int = 0,
					bdlinsolver = MatrixBLS(),
					kwargs...) where {Tkind <: PeriodicOrbitCont, Tprob <: WrapPOColl}
	bifpt = br.specialpoint[ind_bif]
	biftype = bifpt.type

	@assert biftype == :pd "We continue only PD points of Periodic orbits for now"

	pdpointguess = PDPoint(br, ind_bif)

	# we copy the problem for not mutating the one passed by the user
	coll = deepcopy(br.prob.prob)
	N, m, Ntst = size(coll)

	# get the PD eigenvectors
	par = setParam(br, bifpt.param)
	jac = jacobian(br.prob, bifpt.x, par)
	J = jac.jacpb
	nj = size(J, 1)
	J[end, :] .= rand(nj) #must be close to kernel
	J[:, end] .= rand(nj)
	J[end, end] = 0
	# enforce PD boundary condition
	J[end-N:end-1, 1:N] .= I(N)
	rhs = zeros(nj); rhs[end] = 1
	q = J  \ rhs; q = q[1:end-1]; q ./= norm(q) #≈ ker(J)
	p = J' \ rhs; p = p[1:end-1]; p ./= norm(p)

	# perform continuation
	continuationPD(br.prob, alg,
		pdpointguess, setParam(br, pdpointguess.p),
		getLens(br), lens2,
		p, q,
		options_cont;
		kwargs...,
		detectCodim2Bifurcation = detectCodim2Bifurcation,
		kind = PDPeriodicOrbitCont(),
		)
end

function continuationCOLL_NS(br::AbstractResult{Tkind, Tprob},
					ind_bif::Int64,
					lens2::Lens,
					options_cont::ContinuationPar = br.contparams ;
					alg = br.alg,
					startWithEigen = false,
					detectCodim2Bifurcation::Int = 0,
					bdlinsolver = MatrixBLS(),
					kwargs...) where {Tkind <: PeriodicOrbitCont, Tprob <: WrapPOColl}
	bifpt = br.specialpoint[ind_bif]
	biftype = bifpt.type

	@assert biftype == :ns "We continue only NS points of Periodic orbits for now"

	nspointguess = NSPoint(br, ind_bif)

	# we copy the problem for not mutating the one passed by the user
	coll = deepcopy(br.prob.prob)
	N, m, Ntst = size(coll)

	# get the NS eigenvectors
	par = setParam(br, bifpt.param)
	jac = jacobian(br.prob, bifpt.x, par)
	J = Complex.(copy(jac.jacpb))
	nj = size(J, 1)
	J[end, :] .= rand(nj) #must be close to eigensapce
	J[:, end] .= rand(nj)
	J[end, end] = 0
	# enforce NS boundary condition
	λₙₛ = br.eig[bifpt.idx].eigenvals[bifpt.ind_ev]
	J[end-N:end-1, end-N:end-1] .= UniformScaling(exp(λₙₛ))(N)

	rhs = zeros(nj); rhs[end] = 1
	q = J  \ rhs; q = q[1:end-1]; q ./= norm(q) #≈ ker(J)
	p = J' \ rhs; p = p[1:end-1]; p ./= norm(p)

	# perform continuation
	continuationNS(br.prob, alg,
		nspointguess, setParam(br, nspointguess.p[1]),
		getLens(br), lens2,
		p, q,
		options_cont;
		kwargs...,
		detectCodim2Bifurcation = detectCodim2Bifurcation,
		kind = NSPeriodicOrbitCont(),
		)
end
