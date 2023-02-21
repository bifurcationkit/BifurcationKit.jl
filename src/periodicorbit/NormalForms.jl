function getNormalForm(prob::AbstractBifurcationProblem,
			br::ContResult{ <: PeriodicOrbitCont}, id_bif::Int ;
			nev = length(eigenvalsfrombif(br, id_bif)),
			verbose = false,
			ζs = nothing,
			lens = getLens(br),
			Teigvec = getvectortype(br),
			scaleζ = norm,
			prm = false,
			δ = 1e-8,
			detailed = true,
			autodiff = true)
	bifpt = br.specialpoint[id_bif]

	@assert !(bifpt.type in (:endpoint,)) "Normal form for $(bifpt.type) not implemented"

	# parameters for normal form
	kwargs_nf = (nev = nev, verbose = verbose, lens = lens, Teigvec = Teigvec, scaleζ = scaleζ)

	if bifpt.type == :pd
		return perioddoublingNormalForm(prob, br, id_bif; kwargs_nf...)
	elseif bifpt.type == :bp
		return branchNormalForm(prob, br, id_bif; kwargs_nf...)
	elseif bifpt.type == :ns
		return neimarksackerNormalForm(prob, br, id_bif; kwargs_nf...)
	end

	throw("Bifurcation point not yet implemented.")
end

####################################################################################################
function branchNormalForm(pbwrap,
							br,
							ind_bif::Int;
							nev = length(eigenvalsfrombif(br, ind_bif)),
							verbose = false,
							lens = getLens(br),
							Teigvec = vectortype(br),
							kwargs_nf...)
	pb = pbwrap.prob
	bifpt = br.specialpoint[ind_bif]
	bptype = bifpt.type
	par = setParam(br, bifpt.param)
	period = getPeriod(pb, bifpt.x, par)

	# let us compute the kernel
	λ = (br.eig[bifpt.idx].eigenvals[bifpt.ind_ev])
	verbose && print("├─ computing nullspace of Periodic orbit problem...")
	ζ = geteigenvector(br.contparams.newtonOptions.eigsolver, br.eig[bifpt.idx].eigenvecs, bifpt.ind_ev)
	# we normalize it by the sup norm because it could be too small/big in L2 norm
	# TODO: user defined scaleζ
	ζ ./= norm(ζ, Inf)
	verbose && println("Done!")

	# compute the full eigenvector
	floquetsolver = br.contparams.newtonOptions.eigsolver
	ζ_a = floquetsolver(Val(:ExtractEigenVector), pbwrap, bifpt.x, setParam(br, bifpt.param), real.(ζ))
	ζs = reduce(vcat, ζ_a)

	# normal form for Poincaré map
	nf = BranchPoint(nothing, bifpt.param, par, getLens(br), nothing, nothing, nothing, :none)

	return BranchPointPO(bifpt.x, period, real.(ζs), nothing, nf, pb)
end
####################################################################################################
function perioddoublingNormalForm(pbwrap,
								br,
								ind_bif::Int;
								nev = length(eigenvalsfrombif(br, ind_bif)),
								verbose = false,
								lens = getLens(br),
								Teigvec = vectortype(br),
								kwargs_nf...)
	pb = pbwrap.prob
	bifpt = br.specialpoint[ind_bif]
	bptype = bifpt.type
	pars = setParam(br, bifpt.param)
	period = getPeriod(pb, bifpt.x, pars)

	# let us compute the kernel
	λ = (br.eig[bifpt.idx].eigenvals[bifpt.ind_ev])
	verbose && print("├─ computing nullspace of Periodic orbit problem...")
	ζ = geteigenvector(br.contparams.newtonOptions.eigsolver, br.eig[bifpt.idx].eigenvecs, bifpt.ind_ev)
	# we normalize it by the sup norm because it could be too small/big in L2 norm
	# TODO: user defined scaleζ
	ζ ./= norm(ζ, Inf)
	verbose && println("Done!")

	# compute the full eigenvector
	floquetsolver = br.contparams.newtonOptions.eigsolver
	ζ_a = floquetsolver(Val(:ExtractEigenVector), pbwrap, bifpt.x, setParam(br, bifpt.param), real.(ζ))
	ζs = reduce(vcat, ζ_a)

	# normal form for Poincaré map
	nf = PeriodDoubling(nothing, bifpt.param, pars, getLens(br), nothing, nothing, nothing, :none)

	return PeriodDoublingPO(bifpt.x, period, real.(ζs), nothing, nf, pb)

end

function perioddoublingNormalForm(pbwrap::WrapPOColl,
								br,
								ind_bif::Int;
								nev = length(eigenvalsfrombif(br, ind_bif)),
								verbose = false,
								lens = getLens(br),
								Teigvec = vectortype(br),
								kwargs_nf...)
	# Kuznetsov, Yu. A., W. Govaerts, E. J. Doedel, and A. Dhooge. “Numerical Periodic Normalization for Codim 1 Bifurcations of Limit Cycles.” SIAM Journal on Numerical Analysis 43, no. 4 (January 2005): 1407–35. https://doi.org/10.1137/040611306.
	# on page 1243

	# first, get the bifurcation point parameters
	coll = pbwrap.prob
	N, m, Ntst = size(coll)
	@assert coll isa PeriodicOrbitOCollProblem "Something is wrong. Please open an issue on the website"
	verbose && println("#"^53*"\n--> Neimark-Sacker normal form computation")

	# bifurcation point
	bifpt = br.specialpoint[ind_bif]
	bptype = bifpt.type
	pars = setParam(br, bifpt.param)
	T = getPeriod(coll, bifpt.x, pars)

	# get the eigenvalue
	eigRes = br.eig
	λₙₛ = eigRes[bifpt.idx].eigenvals[bifpt.ind_ev]
	ωₙₛ = imag(λₙₛ)

	# we first try to get the floquet eigenvectors for μ = -1
	jac = jacobian(pbwrap, bifpt.x, pars)
	# remove borders
	J = Complex.(jac.jacpb)
	nj = size(J, 1)
	J[end, :] .= rand(nj)
	J[:, end] .= rand(nj)

	# enforce NS boundary condition
	J[end-N:end-1, end-N:end-1] .= UniformScaling(exp(λₙₛ))(N)

	rhs = zeros(eltype(J), nj); rhs[end] = 1
	q = J  \ rhs; q = q[1:end-1]; q ./= norm(q)
	p = J' \ rhs; p = p[1:end-1]; p ./= norm(p)

	J[end, 1:end-1] .= q
	J[1:end-1, end] .= p

	wext = J' \ rhs
	vext = J  \ rhs

	v₁ = @view vext[1:end-1]
	v₁★ = @view wext[1:end-1]

	ζₛ = getTimeSlices(coll, vext)
	vext ./= sqrt(∫(coll, ζₛ, ζₛ, 1))

	ζ★ₛ = getTimeSlices(coll, wext)
	v₁★ ./= 2∫(coll, ζ★ₛ, ζₛ, 1)

	v₁ₛ = getTimeSlices(coll, vcat(v₁,0))
	v₁★ₛ = getTimeSlices(coll, vcat(v₁★,0))

	# return PeriodDoublingPO(bifpt.x, T, bifpt.param, par, getLens(br), v₁, v₁★, nf, :none, coll)
	# normal form for Poincaré map
	nf = PeriodDoubling(nothing, bifpt.param, pars, getLens(br), nothing, nothing, nothing, :none)

	return PeriodDoublingPO(bifpt.x, T, v₁, v₁★, nf, coll)

end
####################################################################################################
function neimarksackerNormalForm(pbwrap,
							br,
							ind_bif::Int;
							nev = length(eigenvalsfrombif(br, ind_bif)),
							verbose = false,
							lens = getLens(br),
							Teigvec = vectortype(br),
							kwargs_nf...)
	pb = pbwrap.prob
	bifpt = br.specialpoint[ind_bif]
	bptype = bifpt.type
	pars = setParam(br, bifpt.param)
	period = getPeriod(pb, bifpt.x, pars)

	# get the eigenvalue
	eigRes = br.eig
	λₙₛ = eigRes[bifpt.idx].eigenvals[bifpt.ind_ev]
	ωₙₛ = imag(λₙₛ)

	ns0 =  NeimarkSacker(bifpt.x, bifpt.param, ωₙₛ, pars, getLens(br), nothing, nothing, nothing, :none)

	return NeimarkSackerPO(bifpt.x, period, bifpt.param, ωₙₛ, nothing, nothing, ns0, pbwrap)

end
####################################################################################################
function predictor(nf::PeriodDoublingPO{ <: PeriodicOrbitTrapProblem}, δp, ampfactor)
	pb = nf.prob

	M, N = size(pb)
	orbitguess0 = nf.po[1:end-1]
	orbitguess0c = getTimeSlices(pb, nf.po)
	ζc = reshape(nf.ζ, N, M)
	orbitguess_c = orbitguess0c .+ ampfactor .*  ζc
	orbitguess_c = hcat(orbitguess_c, orbitguess0c .- ampfactor .*  ζc)
	orbitguess = vec(orbitguess_c[:,1:2:end])
	# we append twice the period
	orbitguess = vcat(orbitguess, 2nf.T)
	return (orbitguess = orbitguess, pnew = nf.nf.p + δp, prob = pb)
end

function predictor(nf::BranchPointPO{ <: PeriodicOrbitTrapProblem}, δp, ampfactor)
	orbitguess = copy(nf.po)
	orbitguess[1:end-1] .+= ampfactor .*  nf.ζ
	return (orbitguess = orbitguess, pnew = nf.nf.p + δp, prob = nf.prob)
end

function predictor(nf::NeimarkSackerPO, δp, ampfactor)
	orbitguess = copy(nf.po)
	return (orbitguess = orbitguess, pnew = nf.nf.p + δp, prob = nf.prob)
end
####################################################################################################
function predictor(nf::PeriodDoublingPO{ <: PeriodicOrbitOCollProblem }, δp, ampfactor)
	pbnew = deepcopy(nf.prob)
	N, m, Ntst = size(nf.prob)

	# we update the problem by doubling the Ntst
	pbnew = setCollocationSize(pbnew, 2Ntst, m)

	orbitguess0 = nf.po[1:end-1]

	orbitguess_c = orbitguess0 .+ ampfactor .*  nf.ζ
	orbitguess = vcat(orbitguess_c[1:end-N], orbitguess0 .- ampfactor .*  nf.ζ)

	pbnew.xπ .= orbitguess
	pbnew.ϕ .= circshift(orbitguess, length(orbitguess)÷1)

	# we append twice the period
	orbitguess = vcat(orbitguess, 2nf.T)

	# no need to change pbnew.cache
	return (orbitguess = orbitguess, pnew = nf.nf.p + δp, prob = pbnew)
end
####################################################################################################
function predictor(nf::PeriodDoublingPO{ <: ShootingProblem }, δp, ampfactor)
	pbnew = deepcopy(nf.prob)
	ζs = nf.ζ
	orbitguess = copy(nf.po)[1:end-1] .+ ampfactor .* ζs
	orbitguess = vcat(orbitguess, copy(nf.po)[1:end-1] .- ampfactor .* ζs, nf.po[end])

	@set! pbnew.M = 2nf.prob.M
	@set! pbnew.ds = _duplicate(pbnew.ds) ./ 2
	orbitguess[end] *= 2
	return (orbitguess = orbitguess, pnew = nf.nf.p + δp, prob = pbnew)
end

function predictor(nf::BranchPointPO{ <: ShootingProblem }, δp, ampfactor)
	ζs = nf.ζ
	orbitguess = copy(nf.po)
	orbitguess[1:length(ζs)] .+= ampfactor .* ζs
	return (orbitguess = orbitguess, pnew = nf.nf.p + δp, prob = nf.prob)
end
####################################################################################################
function predictor(nf::PeriodDoublingPO{ <: PoincareShootingProblem }, δp, ampfactor)
	pbnew = deepcopy(nf.prob)
	ζs = nf.ζ

	@set! pbnew.section = _duplicate(pbnew.section)
	@set! pbnew.M = pbnew.section.M
	orbitguess = copy(nf.po) .+ ampfactor .* ζs
	orbitguess = vcat(orbitguess, orbitguess .- ampfactor .* ζs)

	return (orbitguess = orbitguess, pnew = nf.nf.p + δp, prob = pbnew)
end

function predictor(nf::BranchPointPO{ <: PoincareShootingProblem}, δp, ampfactor)
	ζs = nf.ζ
	orbitguess = copy(nf.po)
	orbitguess .+= ampfactor .* ζs
	return (orbitguess = orbitguess, pnew = nf.nf.p + δp, prob = nf.prob)
end
