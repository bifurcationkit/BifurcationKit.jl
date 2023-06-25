"""
$(SIGNATURES)

Compute the normal form of periodic orbits. Same arguments as the function `getNormalForm` for equilibria. We detail the additional keyword arguments specific to periodic orbits

# Optional arguments
- `prm = true` compute the normal form using Poincaré return map. For collocation, there will be another way to compute the normal form in the future.
"""
function getNormalForm(prob::AbstractBifurcationProblem,
			br::ContResult{ <: PeriodicOrbitCont}, id_bif::Int ;
			nev = length(eigenvalsfrombif(br, id_bif)),
			verbose = false,
			ζs = nothing,
			lens = getLens(br),
			Teigvec = getvectortype(br),
			scaleζ = norm,
			prm = true,
			δ = 1e-8,
			detailed = true, # to get detailed normal form
			)
	bifpt = br.specialpoint[id_bif]

	@assert !(bifpt.type in (:endpoint,)) "Normal form for $(bifpt.type) not implemented"

	# parameters for normal form
	kwargs_nf = (nev = nev, verbose = verbose, lens = lens, Teigvec = Teigvec, scaleζ = scaleζ)

	if bifpt.type == :pd
		return perioddoublingNormalForm(prob, br, id_bif; prm = prm, detailed = detailed, δ = δ, kwargs_nf...)
	elseif bifpt.type == :bp
		return branchNormalForm(prob, br, id_bif; kwargs_nf...)
	elseif bifpt.type == :ns
		return neimarksackerNormalForm(prob, br, id_bif; δ = δ, detailed = detailed, kwargs_nf...)
	end

	throw("Normal form for $(bifpt.type) not yet implemented.")
end

####################################################################################################
"""
[WIP] Note that the computation of this normal form is not implemented yet.
"""
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
	nf = BranchPoint(nothing, nothing, bifpt.param, par, getLens(br), nothing, nothing, nothing, :none)

	return BranchPointPO(bifpt.x, period, real.(ζs), nothing, nf, pb, true)
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
	nf = PeriodDoubling(nothing, nothing, bifpt.param, pars, getLens(br), nothing, nothing, nothing, :none)
	PeriodDoublingPO(bifpt.x, period, real.(ζs), nothing, nf, pb, true)
end

function perioddoublingNormalForm(pbwrap::WrapPOSh,
								br,
								ind_bif::Int;
								nev = length(eigenvalsfrombif(br, ind_bif)),
								verbose = false,
								lens = getLens(br),
								Teigvec = vectortype(br),
								detailed = true,
								kwargs_nf...)
	verbose && println("━"^53*"\n──▶ Period-doubling normal form computation")
	bifpt = br.specialpoint[ind_bif]
	bptype = bifpt.type
	pars = setParam(br, bifpt.param)

	# let us compute the kernel
	λ = (br.eig[bifpt.idx].eigenvals[bifpt.ind_ev])
	verbose && print("├─ computing nullspace of Periodic orbit problem...")
	ζ₋₁ = geteigenvector(br.contparams.newtonOptions.eigsolver, br.eig[bifpt.idx].eigenvecs, bifpt.ind_ev) .|> real
	# we normalize it by the sup norm because it could be too small/big in L2 norm
	# TODO: user defined scaleζ
	ζ₋₁ ./= norm(ζ₋₁, Inf)
	verbose && println("Done!")

	# compute the full eigenvector
	floquetsolver = br.contparams.newtonOptions.eigsolver
	ζ_a = floquetsolver(Val(:ExtractEigenVector), pbwrap, bifpt.x, setParam(br, bifpt.param), real.(ζ₋₁))
	ζs = reduce(vcat, ζ_a)

	pd0 = PeriodDoubling(bifpt.x, nothing, bifpt.param, pars, getLens(br), nothing, nothing, nothing, :none)
	if ~detailed
		period = getPeriod(pbwrap.prob, pd0.x0, pd0.params)
		return PeriodDoublingPO(pd0.x0, period, real.(ζs), nothing, pd0, pbwrap.prob, true)
	end

	# newton parameter
	optn = br.contparams.newtonOptions
	perioddoublingNormalForm(pbwrap, pd0, (ζ₋₁, ζs), optn; verbose = verbose, nev = nev, kwargs_nf...)
end

function perioddoublingNormalForm(pbwrap::WrapPOSh{ <: PoincareShootingProblem },
								pd0::PeriodDoubling,
								(ζ₋₁, ζs),
								optn::NewtonPar;
								nev = 3,
								verbose = false,
								lens = getLens(pbwrap),
								kwargs_nf...)
	psh = pbwrap.prob
	period = getPeriod(psh, pd0.x0, pd0.params)
	PeriodDoublingPO(pd0.x0, period, real.(ζs), nothing, pd0, psh, true)
end

function perioddoublingNormalForm(pbwrap::WrapPOSh{ <: ShootingProblem },
								pd0::PeriodDoubling,
								(ζ₋₁, ζs),
								optn::NewtonPar;
								nev = 3,
								verbose = false,
								lens = getLens(pbwrap),
								δ = 1e-9,
								kwargs_nf...)
	sh = pbwrap.prob
	pars = pd0.params
	period = getPeriod(sh, pd0.x0, pars)
	# compute the Poincaré return map, the section is on the first time slice
	Π = PoincareMap(pbwrap, pd0.x0, pars, optn)
	# Π = PoincareCallback(pbwrap, pd0.x0, pars; radius = 0.1)
	xₛ = getTimeSlices(sh, Π.po)[:, 1]
	# ζ₁ = getVectorField(br.prob.prob.flow.prob)(xₛ,pars) |> normalize

	# If M is the monodromy matrix and E = x - <x,e>e with e the eigen
	# vector of M for the eigenvalue 1, then, we find that
	# eigenvector(P) = E ∘ eigenvector(M)
	# E(x) = x .- dot(ζ₁, x) .* ζ₁

	_nrm = norm(Π(xₛ, pars).u - xₛ, Inf)
	_nrm > 1e-10 && @warn "Residual seems large = $_nrm"

	# dP = ForwardDiff.jacobian( x -> Π(x,pars).u, xₛ)
	dP = finiteDifferences(x -> Π(x,pars).u, xₛ; δ = δ)
	J = jacobian(pbwrap, pd0.x0, pars)
	M = MonodromyQaD(J)

	Fₘ = eigen(M)
	F = eigen(dP)

	# N = length(xₛ)
	# q = rand(N); p = rand(N)
	# rhs = vcat(zeros(N), 1)
	#
	# Pbd = zeros(N+1, N+1)
	# Pbd[1:N, 1:N] .= dP + I;
	# Pbd[end, 1:N] .= p
	# Pbd[1:N, end] .= q
	# ψ = Pbd \ rhs
	# ϕ = Pbd' \ rhs
	#
	# ev₋₁ = ψ[1:end-1]; normalize!(ev₋₁)
	# ev₋₁p = ϕ[1:end-1]; normalize!(ev₋₁p)

	####
	ind₋₁ = argmin(abs.(F.values .+ 1))
	ev₋₁ = F.vectors[:, ind₋₁]
	Fp = eigen(dP')
	ind₋₁ = argmin(abs.(Fp.values .+ 1))
	ev₋₁p = Fp.vectors[:, ind₋₁]
	####

	@debug "" Fₘ.values F.values Fp.values

	# @info "Essai de VP"
	# dP * ζ₋₁ + ζ₋₁ |> display # not good, need projector E
	# dP * ev₋₁ + ev₋₁ |> display
	# dP' * ev₋₁p + ev₋₁p |> display
	# e = Fₘ.vectors[:,end]; e ./= norm(e)

	# normalize eigenvectors
	ev₋₁ ./= sqrt(dot(ev₋₁, ev₋₁))
	ev₋₁p ./= dot(ev₋₁, ev₋₁p)

	probΠ = BifurcationProblem(
			(x,p) -> Π(x,p).u,
			xₛ, pars, lens ;
			J = (x,p) -> finiteDifferences(z -> Π(z,p).u, x; δ = δ),
			d2F = (x,p,h1,h2) -> d2F(Π,x,p,h1,h2).u,
			d3F = (x,p,h1,h2,h3) -> d3F(Π,x,p,h1,h2,h3).u
			)

	pd1 = PeriodDoubling(xₛ, nothing, pd0.p, pars, lens, ev₋₁, ev₋₁p, nothing, :none)
	# normal form computation
	pd = periodDoublingNormalForm(probΠ, pd1, DefaultLS(); verbose = verbose)
	return PeriodDoublingPO(pd0.x0, period, real.(ζs), nothing, pd, sh, true)
end

function perioddoublingNormalForm(pbwrap::WrapPOColl,
								br,
								ind_bif::Int;
								verbose = false,
								nev = length(eigenvalsfrombif(br, ind_bif)),
								prm = true,
								kwargs_nf...)
	# first, get the bifurcation point parameters
	verbose && println("━"^53*"\n──▶ Period-Doubling normal form computation")
	bifpt = br.specialpoint[ind_bif]
	bptype = bifpt.type
	par = setParam(br, bifpt.param)

	if bifpt.x isa NamedTuple
		# the solution is mesh adapted, we need to restore the mesh.
		pbwrap = deepcopy(pbwrap)
		updateMesh!(pbwrap.prob, bifpt.x._mesh )
		bifpt = @set bifpt.x = bifpt.x.sol
	end
	pd0 = PeriodDoubling(bifpt.x, nothing, bifpt.param, par, getLens(br), nothing, nothing, nothing, :none)
	if prm
		# newton parameter
		optn = br.contparams.newtonOptions
		return perioddoublingNormalFormPRM(pbwrap, pd0, optn; verbose = verbose, nev = nev, kwargs_nf...)
	end

	return perioddoublingNormalForm(pbwrap, pd0; verbose = verbose, nev = nev, kwargs_nf...)

end

function perioddoublingNormalFormPRM(pbwrap::WrapPOColl,
								pd0::PeriodDoubling,
								optn::NewtonPar;
								nev = 3,
								δ = 1e-7,
								verbose = false,
								lens = getLens(pbwrap),
								kwargs_nf...)
	@debug "method PRM"
	coll = pbwrap.prob
	N, m, Ntst = size(coll)
	pars = pd0.params
	@debug pars typeof(pd0.x0)
	T = getPeriod(coll, pd0.x0, pars)

	Π = PoincareMap(pbwrap, pd0.x0, pars, optn)
	xₛ = pd0.x0[1:N]
	dP = finiteDifferences(x -> Π(x,pars).u, xₛ)
	F = eigen(dP)

	####
	ind₋₁ = argmin(abs.(F.values .+ 1))
	ev₋₁ = F.vectors[:, ind₋₁]
	Fp = eigen(dP')
	ind₋₁ = argmin(abs.(Fp.values .+ 1))
	ev₋₁p = Fp.vectors[:, ind₋₁]
	####
	# Π(xₛ, pars).u - xₛ |> display
	# dP * ev₋₁ + ev₋₁ |> display
	# dP' * ev₋₁p + ev₋₁p |> display

	# normalize eigenvectors
	ev₋₁ ./= sqrt(dot(ev₋₁, ev₋₁))
	ev₋₁p ./= dot(ev₋₁, ev₋₁p)

	δ2 = √δ
	δ3 = δ^(1/3)
	d1Π(x,p,dx) = (Π(x .+ δ .* dx, p).u .- Π(x .- δ .* dx, p).u) ./ (2δ)
	d2Π(x,p,dx1,dx2) = (d1Π(x .+ δ2 .* dx2, p, dx1) .- d1Π(x .- δ2 .* dx2, p, dx1)) ./ (2δ2)
	d3Π(x,p,dx1,dx2,dx3) = (d2Π(x .+ δ3 .* dx3, p, dx1, dx2) .- d2Π(x .- δ3 .* dx3, p, dx1, dx2)) ./ (2δ3)

	probΠ = BifurcationProblem(
			(x,p) -> Π(x,p).u,
			xₛ, pars, lens ;
			J = (x,p) -> finiteDifferences(z -> Π(z,p).u, x),
			# d2F = (x,p,h1,h2) -> d2F(Π,x,p,h1,h2).u,
			# d3F = (x,p,h1,h2,h3) -> d3F(Π,x,p,h1,h2,h3).u
			d2F = d2Π,
			d3F = d3Π,
			)

	pd1 = PeriodDoubling(xₛ, nothing, pd0.p, pars, lens, ev₋₁, ev₋₁p, nothing, :none)
	pd = periodDoublingNormalForm(probΠ, pd1, DefaultLS(); verbose = verbose)
	return PeriodDoublingPO(pd0.x0, pd0.x0[end], nothing, nothing, pd, coll, true)
end
####################################################################################################
function neimarksackerNormalForm(pbwrap::WrapPOColl,
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
	verbose && println("━"^53*"\n──▶ Period-doubling normal form computation")

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

	vl = J' \ rhs
	vr = J  \ rhs

	v₁ = @view vr[1:end-1]
	v₁★ = @view vl[1:end-1]

	ζₛ = getTimeSlices(coll, vr)
	vr ./= sqrt(∫(coll, ζₛ, ζₛ, 1))

	ζ★ₛ = getTimeSlices(coll, vl)
	v₁★ ./= 2∫(coll, ζ★ₛ, ζₛ, 1)

	v₁ₛ = getTimeSlices(coll, vcat(v₁,0))
	v₁★ₛ = getTimeSlices(coll, vcat(v₁★,0))

	ns0 =  NeimarkSacker(bifpt.x, nothing, bifpt.param, ωₙₛ, pars, getLens(br), nothing, nothing, nothing, :none)

	# newton parameter
	optn = br.contparams.newtonOptions
	return neimarksackerNormalFormPRM(pbwrap, ns0, optn; verbose = verbose, nev = nev, kwargs_nf...)
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
	return NeimarkSackerPO(bifpt.x, period, bifpt.param, ωₙₛ, nothing, nothing, ns0, pbwrap, true)
end

function neimarksackerNormalFormPRM(pbwrap::WrapPOColl,
								ns0::NeimarkSacker,
								optn::NewtonPar;
								nev = 3,
								δ = 1e-7,
								verbose = false,
								lens = getLens(pbwrap),
								kwargs_nf...)
	@debug "methode PRM"
	coll = pbwrap.prob
	N, m, Ntst = size(coll)
	pars = ns0.params
	T = getPeriod(coll, ns0.x0, pars)

	Π = PoincareMap(pbwrap, ns0.x0, pars, optn)
	xₛ = ns0.x0[1:N]
	dP = finiteDifferences(x -> Π(x,pars).u, xₛ)
	F = eigen(dP)

	_nrm = norm(Π(xₛ, pars).u - xₛ, Inf)
	_nrm > 1e-12 && @warn  "$_nrm"

	####
	ind = argmin(abs.(log.(complex.(F.values)) .- Complex(0, ns0.ω )))
	ev = F.vectors[:, ind]
	Fp = eigen(dP')
	indp = argmin(abs.(log.(complex.(Fp.values)) .+ Complex(0, ns0.ω )))
	evp = Fp.vectors[:, indp]

	# normalize eigenvectors
	ev ./= sqrt(dot(ev, ev))
	evp ./= dot(ev, evp)

	δ2 = √δ
	δ3 = δ^(1/3)
	d1Π(x,p,dx) = ((Π(x .+ δ .* dx, p).u .- Π(x .- δ .* dx, p).u) ./ (2δ))
	d2Π(x,p,dx1,dx2) = ((d1Π(x .+ δ2 .* dx2, p, dx1) .- d1Π(x .- δ2 .* dx2, p, dx1)) ./ (2δ2))
	d3Π(x,p,dx1,dx2,dx3) = ((d2Π(x .+ δ3 .* dx3, p, dx1, dx2) .- d2Π(x .- δ3 .* dx3, p, dx1, dx2)) ./ (2δ3))

	probΠ = BifurcationProblem(
			(x,p) -> Π(x,p).u,
			xₛ, pars, lens ;
			J = (x,p) -> finiteDifferences(z -> Π(z,p).u, x),
			d2F = d2Π,
			d3F = d3Π,
			)

	ns1 = NeimarkSacker(xₛ, nothing, ns0.p, ns0.ω, pars, lens, ev, evp, nothing, :none)
	ns = neimarkSackerNormalForm(probΠ, ns1, DefaultLS(); verbose = verbose)
	return NeimarkSackerPO(ns0.x0, T, 0., 0., real.(1), nothing, ns, coll, true)
end

function neimarksackerNormalForm(pbwrap::WrapPOSh{ <: ShootingProblem },
								br,
								ind_bif::Int;
								nev = length(eigenvalsfrombif(br, ind_bif)),
								verbose = false,
								lens = getLens(br),
								Teigvec = vectortype(br),
								kwargs_nf...)

	# first, get the bifurcation point parameters
	sh = pbwrap.prob
	@assert sh isa ShootingProblem "Something is wrong. Please open an issue on the website"
	verbose && println("━"^53*"\n──▶ Neimark-Sacker normal form computation")

	# bifurcation point
	bifpt = br.specialpoint[ind_bif]
	bptype = bifpt.type
	pars = setParam(br, bifpt.param)
	T = getPeriod(sh, bifpt.x, pars)

	# get the eigenvalue
	eigRes = br.eig
	λₙₛ = eigRes[bifpt.idx].eigenvals[bifpt.ind_ev]
	ωₙₛ = imag(λₙₛ)

	ns0 =  NeimarkSacker(bifpt.x, nothing, bifpt.param, ωₙₛ, pars, getLens(br), nothing, nothing, nothing, :none)

	# newton parameter
	optn = br.contparams.newtonOptions
	return neimarksackerNormalForm(pbwrap, ns0, (1, 1), optn; verbose = verbose, nev = nev, kwargs_nf...)
end

function neimarksackerNormalForm(pbwrap::WrapPOSh{ <: ShootingProblem },
								ns0::NeimarkSacker,
								(ζ₋₁, ζs),
								optn::NewtonPar;
								nev = 3,
								verbose = false,
								lens = getLens(pbwrap),
								kwargs_nf...)
	sh = pbwrap.prob
	pars = ns0.params
	period = getPeriod(sh, ns0.x0, pars)
	# compute the Poincaré return map, the section is on the first time slice
	Π = PoincareMap(pbwrap, ns0.x0, pars, optn)
	xₛ = getTimeSlices(sh, Π.po)[:, 1]

	_nrm = norm(Π(xₛ, pars).u - xₛ, Inf)
	_nrm > 1e-12 && @warn  "$_nrm"

	dP = finiteDifferences(x -> Π(x,pars).u, xₛ)
	# dP = ForwardDiff.jacobian(x -> Π(x,pars).u, xₛ)
	J = jacobian(pbwrap, ns0.x0, pars)
	M = MonodromyQaD(J)

	Fₘ = eigen(M)
	F = eigen(dP)

	ind = argmin(abs.(log.(complex.(F.values)) .- Complex(0, ns0.ω )))
	ev = F.vectors[:, ind]
	Fp = eigen(dP')
	indp = argmin(abs.(log.(complex.(Fp.values)) .+ Complex(0, ns0.ω )))
	evp = Fp.vectors[:, indp]

	# normalize eigenvectors
	ev ./= sqrt(dot(ev, ev))
	evp ./= dot(evp, ev)

	@debug "" xₛ ev evp dP _nrm pars F.values[ind] Fp.values[indp]
	@debug "" F.values ns0.x0

	probΠ = BifurcationProblem(
			(x,p) -> Π(x,p).u,
			xₛ, pars, lens ;
			J = (x,p) -> finiteDifferences(z -> Π(z,p).u, x),
			d2F = (x,p,h1,h2) -> d2F(Π,x,p,h1,h2).u,
			d3F = (x,p,h1,h2,h3) -> d3F(Π,x,p,h1,h2,h3).u
			)

	ns1 = NeimarkSacker(xₛ, nothing, ns0.p, ns0.ω, pars, lens, ev, evp, nothing, :none)
	# normal form computation
	ns = neimarkSackerNormalForm(probΠ, ns1, DefaultLS(); verbose = verbose)

	return NeimarkSackerPO(ns0.x0, period, ns0.p, ns0.ω, real.(ζs), nothing, ns, sh, true)
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
