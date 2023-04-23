# if the jacobian is matrix based, use transpose
@inline hasAdjoint(::WrapPOSh{ <: ShootingProblem{Tp, Tj} }) where {Tp, Tj} = ~(Tj <: AbstractJacobianMatrix)
@inline getDelta(::WrapPOSh) = 1e-8
@inline hasHessian(::WrapPOSh) = true
@inline hasJvp(wrap::WrapPOSh) = hasJvp(wrap.prob)

function Base.transpose(J::FloquetWrapper{ <: ShootingProblem })
	@set J.jacpb = transpose(J.jacpb)
end

function Base.adjoint(J::FloquetWrapper{ <: ShootingProblem })
	@set J.jacpb = adjoint(J.jacpb)
end

# this function is necessary for pdtest to work in PDMinimallyAugmented problem
function jacobianPeriodDoubling(pbwrap::WrapPOSh{ <: ShootingProblem{Tp, Tj} }, x, par) where {Tp, Tj}
	dx -> jacobianPD_NFMatrixFree(pbwrap::WrapPOSh{ <: ShootingProblem }, x, par, 1, dx)
end

# this function is necessary for the jacobian of a PDMinimallyAugmented problem
function jacobianAdjointPeriodDoubling(pbwrap::WrapPOSh{ <: ShootingProblem }, x, par)
	dx -> jacobianAdjointPeriodDoublingMatrixFree(pbwrap, x, par, dx)
end

jacobianAdjointPeriodDoublingMatrixFree(pbwrap::WrapPOSh{ <: ShootingProblem }, x, par, dx) = jacobianAdjointPD_NFMatrixFree(pbwrap, x, par, 1, dx)

jacobianAdjointNeimarkSackerMatrixFree(pbwrap::WrapPOSh{ <: ShootingProblem }, x, par, ω, dx) = jacobianAdjointPD_NFMatrixFree(pbwrap, x, par, -cis(-ω), dx)

# same as above but matrix based
function jacobianPeriodDoubling(pbwrap::WrapPOSh{ <: ShootingProblem{Tp, Tj} }, x, par) where {Tp, Tj <: AbstractJacobianMatrix}
	M = getMeshSize(pbwrap.prob)
	N = div(length(x) - 1, M)
	Jac = jacobian(pbwrap, x, par)
	# put the PD boundary condition
	@set Jac.jacpb = copy(Jac.jacpb)
	J = Jac.jacpb
	J[end-N:end-1, 1:N] .= I(N)
	@set Jac.jacpb = J[1:end-1, 1:end-1]
end

# matrix free linear operator associated to the monodromy whose zeros are used to detect PD/NS points
function jacobianPD_NFMatrixFree(pbwrap::WrapPOSh{ <: ShootingProblem }, x, par, α::𝒯, dx) where 𝒯
	sh = pbwrap.prob
	T  = getPeriod(sh, x)
	M  = getMeshSize(sh)
	N  = div(length(x) - 1, M)

	xc = getTimeSlices(sh, x)
	dxc = reshape(dx, N, M)

	# variable to hold the computed result
	out = similar(dx, promote_type(eltype(dx), 𝒯))
	outc = reshape(out, N, M)

	# jacobian of the flow
	dflowDE(_x, _dx, _T) = ForwardDiff.derivative(z -> evolve(sh.flow, _x .+ z .* _dx, par, _T).u, 0)
	dflow(_x, _dx, _T) = dflowDE(_x, real.(_dx), _T) .+ im .* dflowDE(_x, imag.(_dx), _T)
	dflow(_x, _dx::T, _T) where {T <: AbstractArray{<: Real}} = dflowDE(_x, _dx, _T)
	if ~isParallel(sh)
		for ii in 1:M
			ip1 = (ii == M) ? 1 : ii+1
			# call jacobian of the flow, jacobian-vector product
			tmp = dflow(xc[:, ii], dxc[:, ii], sh.ds[ii] * T)

			if ii<M
				outc[:, ii] .= @views tmp .- dxc[:, ip1]
			else
				outc[:, ii] .= @views tmp .+ α .* dxc[:, ip1]
			end
		end
	else
		@assert 1==0
		# call jacobian of the flow, jacobian-vector product
		solOde = jvp(sh.flow, xc, par, dxc, sh.ds .* T)
		for ii in 1:M
			ip1 = (ii == M) ? 1 : ii+1
			outc[:, ii] .= solOde[ii].du .+ vf(sh.flow, solOde[ii].u, par) .* sh.ds[ii] .* dT .- dxc[:, ip1]
		end
	end
	return out
end

# matrix free adjoint linear operator associated to the monodromy whose zeros are used to detect PD/NS points
function jacobianAdjointPD_NFMatrixFree(pbwrap::WrapPOSh{ <: ShootingProblem }, x, par, α::𝒯, dx) where 𝒯
	sh = pbwrap.prob
	T  = getPeriod(sh, x)
	M  = getMeshSize(sh)
	N  = div(length(x) - 1, M)

	xc = getTimeSlices(sh, x)
	dxc = reshape(dx, N, M)

	# variable to hold the computed result
	out = similar(dx, promote_type(eltype(dx), 𝒯))
	outc = reshape(out, N, M)

	# jacobian of the flow
	dflowDE(_x, _dx, _T) = vjp(sh.flow, _x, par, _dx, _T)
	dflow(_x, _dx, _T) = dflowDE(_x, real.(_dx), _T) .+ im .* dflowDE(_x, imag.(_dx), _T)
	dflow(_x, _dx::T, _T) where {T <: AbstractArray{<: Real}} = dflowDE(_x, _dx, _T)

	if ~isParallel(sh)
		for ii in 1:M
			im1 = (ii == 1) ? M : ii-1
			# call jacobian of the flow, jacobian-vector product
			tmp = dflow(xc[:, ii], dxc[:, ii], sh.ds[ii] * T)

			if ii==1
				outc[:, ii] .= @views tmp .+ α .* dxc[:, im1]
			else
				outc[:, ii] .= @views tmp .- dxc[:, im1]
			end
		end
	else
		@assert 1==0
		# call jacobian of the flow, jacobian-vector product
		solOde = jvp(sh.flow, xc, par, dxc, sh.ds .* T)
		for ii in 1:M
			ip1 = (ii == M) ? 1 : ii+1
			outc[:, ii] .= solOde[ii].du .+ vf(sh.flow, solOde[ii].u, par) .* sh.ds[ii] .* dT .- dxc[:, ip1]
		end
	end
	return out
end

function jacobianNeimarkSacker(pbwrap::WrapPOSh{ <: ShootingProblem{Tp, Tj} }, x, par, ω) where {Tp, Tj}
	dx -> jacobianPD_NFMatrixFree(pbwrap::WrapPOSh{ <: ShootingProblem }, x, par, -cis(ω), dx)
end

function jacobianNeimarkSacker(pbwrap::WrapPOSh{ <: ShootingProblem{Tp, Tj} }, x, par, ω) where {Tp, Tj <: AbstractJacobianMatrix}
	M = getMeshSize(pbwrap.prob)
	N = div(length(x) - 1, M)
	Jac = jacobian(pbwrap, x, par)
	# put the NS boundary condition
	J = Complex.(copy(Jac.jacpb))
	J[end-N:end-1, 1:N] .*= cis(ω)
	@set Jac.jacpb = J[1:end-1, 1:end-1]
end

# this function is necessary for the jacobian of a PDMinimallyAugmented problem
function jacobianAdjointNeimarkSacker(pbwrap::WrapPOSh{ <: ShootingProblem }, x, par, ω)
	dx -> jacobianAdjointNeimarkSackerMatrixFree(pbwrap, x, par, ω, dx)
end

function continuation(br::AbstractResult{Tkind, Tprob},
					ind_bif::Int64,
					lens2::Lens,
					options_cont::ContinuationPar = br.contparams ;
					startWithEigen = false,
					detectCodim2Bifurcation::Int = 0,
					kwargs...) where {Tkind <: PeriodicOrbitCont, Tprob <: WrapPOSh}
	biftype = br.specialpoint[ind_bif].type

	# options to detect codim2 bifurcations
	computeEigenElements = options_cont.detectBifurcation > 0
	_options_cont = detectCodim2Parameters(detectCodim2Bifurcation, options_cont; kwargs...)

	if biftype == :bp
		return continuationSH_Fold(br, ind_bif, lens2, options_cont; kwargs... )
	elseif biftype == :pd
		return continuationSH_PD(br, ind_bif, lens2, _options_cont; kwargs... )
	elseif biftype == :ns
		return continuationSH_NS(br, ind_bif, lens2, _options_cont; kwargs... )
	end
	throw("You passed the bifurcation type = $biftype. We continue only Fold / PD / NS points of periodic orbits for now.")
end

function continuationSH_Fold(br::AbstractResult{Tkind, Tprob},
					ind_bif::Int64,
					lens2::Lens,
					options_cont::ContinuationPar = br.contparams ;
					startWithEigen = false,
					detectCodim2Bifurcation::Int = 0,
					bdlinsolver = MatrixBLS(),
					Jᵗ = nothing,
					kwargs...) where {Tkind <: PeriodicOrbitCont, Tprob <: WrapPOSh}
	biftype = br.specialpoint[ind_bif].type
	bifpt = br.specialpoint[ind_bif]

	pbwrap = getProb(br)
	probsh = pbwrap.prob

	probshFold = BifurcationProblem((x, p) -> residual(pbwrap, x, p), bifpt, getParams(br), getLens(br);
				J = (x, p) -> jacobian(pbwrap, x, p),
				Jᵗ = Jᵗ,
				d2F = (x, p, dx1, dx2) -> d2PO(z -> probsh(z, p), x, dx1, dx2)
				)

	options_foldpo = @set options_cont.newtonOptions.linsolver = FloquetWrapperLS(options_cont.newtonOptions.linsolver)

	# perform continuation
	br_fold_po = continuationFold(probshFold,
		br, ind_bif, lens2,
		options_cont;
		startWithEigen = startWithEigen,
		detectCodim2Bifurcation = detectCodim2Bifurcation,
		bdlinsolver = FloquetWrapperBLS(bdlinsolver),
		kind = FoldPeriodicOrbitCont(),
		kwargs...)
end

function continuationSH_PD(br::AbstractResult{Tkind, Tprob},
					ind_bif::Int64,
					lens2::Lens,
					options_cont::ContinuationPar = br.contparams ;
					alg = br.alg,
					startWithEigen = false,
					detectCodim2Bifurcation::Int = 0,
					Jᵗ = nothing,
					kwargs...) where {Tkind <: PeriodicOrbitCont, Tprob <: WrapPOSh}
		verbose = get(kwargs, :verbosity, 0) > 0

		bifpt = br.specialpoint[ind_bif]
		bptype = bifpt.type
		pdpointguess = PDPoint(br, ind_bif)

		# copy the problem for not mutating the one passed by the user
		pbwrap = br.prob
		sh = deepcopy(pbwrap.prob)

		# get the parameters
		par_pd = setParam(br, pdpointguess.p)

		# let us compute the eigenspace
		λ = (br.eig[bifpt.idx].eigenvals[bifpt.ind_ev])
		verbose && print("├─ computing nullspace of Periodic orbit problem...")
		ζ = geteigenvector(br.contparams.newtonOptions.eigsolver, br.eig[bifpt.idx].eigenvecs, bifpt.ind_ev)
		# we normalize it by the sup norm because it could be too small/big in L2 norm
		# TODO: user defined scaleζ
		ζ ./= norm(ζ, Inf)
		verbose && println(" Done!")

		# # compute the full eigenvector
		# floquetsolver = br.contparams.newtonOptions.eigsolver
		# ζ_a = floquetsolver(Val(:ExtractEigenVector), br.prob, bifpt.x, setParam(br, bifpt.param), real.(ζ))
		# ζs = reduce(vcat, ζ_a)
		# ζs_ad = copy(ζs)

		# compute the full eigenvector, version with bordered problem
		ls = options_cont.newtonOptions.linsolver
		J = jacobianPeriodDoubling(pbwrap, bifpt.x, par_pd)
		rhs = zero(bifpt.x)[1:end-1]; rhs[end] = 1
		q, = ls(J, rhs); q ./= norm(q) #≈ ker(J)
		# p, = ls(transpose(J), rhs); p ./= norm(p)
		p = copy(q)

		# perform continuation
		continuationPD(br.prob, alg,
			pdpointguess, setParam(br, pdpointguess.p),
			getLens(br), lens2,
			# ζs, ζs_ad,
			p, q,
			options_cont;
			kwargs...,
			kind = PDPeriodicOrbitCont(),
			)
end

function continuationSH_NS(br::AbstractResult{Tkind, Tprob},
					ind_bif::Int64,
					lens2::Lens,
					options_cont::ContinuationPar = br.contparams ;
					alg = br.alg,
					startWithEigen = false,
					detectCodim2Bifurcation::Int = 0,
					bdlinsolver = MatrixBLS(),
					kwargs...) where {Tkind <: PeriodicOrbitCont, Tprob <: WrapPOSh}
	bifpt = br.specialpoint[ind_bif]
	biftype = bifpt.type

	@assert biftype == :ns "We continue only NS points of Periodic orbits for now"

	nspointguess = NSPoint(br, ind_bif)

	# copy the problem for not mutating the one passed by the user
	pbwrap = br.prob
	sh = deepcopy(br.prob.prob)

	M = getMeshSize(sh)
	N = div(length(bifpt.x) - 1, M)

	par_ns = setParam(br, bifpt.param)
	period = getPeriod(sh, bifpt.x, par_ns)

	# compute the eigenspace
	λₙₛ = br.eig[bifpt.idx].eigenvals[bifpt.ind_ev]
	ωₙₛ = λₙₛ/1im

	J = jacobianNeimarkSacker(pbwrap, bifpt.x, par_ns, ωₙₛ)
	nj = length(bifpt.x)-1
	q, = bdlinsolver(J, Complex.(rand(nj)), Complex.(rand(nj)), 0, Complex.(zeros(nj)), 1)
	q ./= norm(q)
	p = conj(q)

	###########
	# ζ = geteigenvector(br.contparams.newtonOptions.eigsolver, br.eig[bifpt.idx].eigenvecs, bifpt.ind_ev)
	# # compute the full eigenvector
	# floquetsolver = br.contparams.newtonOptions.eigsolver
	# ζ_a = floquetsolver(Val(:ExtractEigenVector), br.prob, bifpt.x, setParam(br, bifpt.param), real.(ζ))
	# ζs = reduce(vcat, ζ_a)
	# ζs_ad = copy(ζs)

	# perform continuation
	continuationNS(br.prob, alg,
		nspointguess, setParam(br, nspointguess.p[1]),
		getLens(br), lens2,
		p, q,
		# ζs, copy(ζs_ad),
		options_cont;
		kwargs...,
		bdlinsolver = bdlinsolver,
		kind = NSPeriodicOrbitCont(),
		)
end
