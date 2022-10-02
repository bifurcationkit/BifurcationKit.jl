"""
$(SIGNATURES)

Compute the Cusp normal form.

# Arguments
- `prob` bifurcation problem
- `pt::Cusp` Cusp bifurcation point
- `ls` linear solver

# Optional arguments
- `δ = 1e-8` used for finite differences
- `verbose` bool to print information
"""
function cuspNormalForm(_prob,
		br::AbstractBranchResult, ind_bif::Int;
		δ = 1e-8,
		nev = length(eigenvalsfrombif(br, ind_bif)),
		verbose = false,
		ζs = nothing,
		lens = getLens(br),
		Teigvec = getvectortype(br),
		scaleζ = norm)
	@assert getvectortype(br) <: BorderedArray
	@assert br.specialpoint[ind_bif].x isa BorderedArray
	@assert br.specialpoint[ind_bif].type == :cusp "The provided index does not refer to a Cusp Point"

	verbose && println("#"^53*"\n--> Cusp Normal form computation")

	# MA problem formulation
	prob_ma = _prob.prob

	# get the vector field
	prob_vf = prob_ma.prob_vf

	# scalar type
	T = eltype(Teigvec)
	ϵ2 = T(δ)

	# linear solvers
	ls = prob_ma.linsolver
	bls = prob_ma.linbdsolver

	# kernel dimension
	N = 1

	# in case nev = 0 (number of unstable eigenvalues), we increase nev to avoid bug
	nev = min(2N, nev)

	# Newton parameters
	options = br.contparams.newtonOptions

	# bifurcation point
	bifpt = br.specialpoint[ind_bif]
	eigRes = br.eig

	# eigenvalue
	if bifpt.ind_ev > 0
		λ = eigRes[bifpt.idx].eigenvals[bifpt.ind_ev]
	else
		λ = rightmost(eigRes[bifpt.idx].eigenvals)[1]
	end

	# parameters for vector field
	p = bifpt.param
	parbif = set(getParams(br), lens, p)
	parbif = set(parbif, getLens(prob_ma), get(bifpt.printsol, getLens(prob_ma)))

	# jacobian at bifurcation point
	x0 = convert(Teigvec.parameters[1], bifpt.x.u)
	L = jacobian(prob_vf, x0, parbif)

	# eigenvectors
	# we recompute the eigen-elements if there were not saved during the computation of the branch
	@info "Eigen-elements not saved in the branch. Recomputing them..."
	eigsolver = getsolver(options.eigsolver)
	_λ0, _ev0, _ = eigsolver(L, nev)
	Ivp = sortperm(_λ0, by = abs)
	_λ = _λ0[Ivp]
	if norm(_λ[1:N] .- 0, Inf) > br.contparams.tolStability
		@warn "We did not find the correct eigenvalues. We found the eigenvalues:\n $(display(( _λ[1:N]))).\n Difference between the eigenvalues:"
		display(_λ[1:N] .- 0)
	end
	ζ = real.(geteigenvector(eigsolver, _ev0, Ivp[1]))
	ζ ./= scaleζ(ζ)

	# extract eigen-elements for adjoint(L), needed to build spectral projector
	if isSymmetric(prob_vf)
		λstar = br.eig[bifpt.idx].eigenvals[bifpt.ind_ev]
		ζstar = copy(ζ)
	else
		_Jt = hasAdjoint(prob_vf) ? jad(prob_vf, x0, parbif) : adjoint(L)
		ζstar, λstar = getAdjointBasis(_Jt, conj(λ), eigsolver; nev = nev, verbose = verbose)
	end

	ζstar = real.(ζstar); λstar = real.(λstar)

	@assert abs(dot(ζ, ζstar)) > 1e-10 "We got ζ⋅ζstar = $((dot(ζ, ζstar))). This dot product should not be zero. Perhaps, you can increase `nev` which is currently $nev."
	ζstar ./= dot(ζ, ζstar)

	# Kuznetsov, Yu. A. “Numerical Normalization Techniques for All Codim 2 Bifurcations of Equilibria in ODE’s.” SIAM Journal on Numerical Analysis 36, no. 4 (January 1, 1999): 1104–24. https://doi.org/10.1137/S0036142998335005.
	# notations from this paper
	B(dx1, dx2) = d2F(prob_vf, x0, parbif, dx1, dx2)
	C(dx1, dx2, dx3) = d3F(prob_vf, x0, parbif, dx1, dx2, dx3)
	q = ζ; p = ζstar

	h2 = B(q, q)
	h2 .= dot(p, h2) .* q .- h2
	H2, = bls(L, q, p, zero(T), h2, zero(T))

	c = dot(p, C(q, q, q)) + 3dot(p, B(q, H2))
	c /= 6

	pt = Cusp(
		x0, parbif,
		(getLens(prob_ma), lens),
		ζ, ζstar,
		(c = c, ),
		:none
	)
end

"""
$(SIGNATURES)

Compute the Bogdanov-Takens normal form.

# Arguments
- `prob_ma` a `FoldProblemMinimallyAugmented` or `HopfProblemMinimallyAugmented`
- `pt::BogdanovTakens` BogdanovTakens bifurcation point
- `ls` linear solver

# Optional arguments
- `δ = 1e-8` used for finite differences
- `verbose` bool to print information
- `autodiff = true` only for Bogdanov-Takens point. Whether to use ForwardDiff for the many differentiations that are required to compute the normal form.
- `detailed = true` only for Bogdanov-Takens point. Whether to compute only a simplified normal form.
"""
function bogdanovTakensNormalForm(prob_ma, L,
							pt::BogdanovTakens;
							δ = 1e-8,
							verbose = false,
							detailed = true,
							autodiff = true)
	x0 = pt.x0
	parbif = pt.params
	Ty = eltype(x0)

	# vector field
	VF = prob_ma.prob_vf
	F(x, p) = residual(VF, x, p)

	# for finite differences
	ϵ = convert(Ty, δ)
	ϵ2 = sqrt(ϵ) # this one is for second order differential

	# linear solvers
	ls = prob_ma.linsolver
	bls = prob_ma.linbdsolver

	lens1, lens2 = pt.lens

	getp(l::Lens) = get(parbif, l)
	setp(l::Lens, p::Number) = set(parbif, l, p)
	setp(p1::Number, p2::Number) = set(set(parbif, lens1, p1), lens2, p2)

	ζ0, ζ1 = pt.ζ
	ζs0, ζs1 = pt.ζstar

	G = [dot(xs, x) for xs in pt.ζstar, x in pt.ζ]
	norm(G-I(2), Inf) > 1e-5 && @warn "G == I(2) is not valid. We built a basis such that G = $G"

	G = [dot(xs, apply(L,x)) for xs in pt.ζstar, x in pt.ζ]
	norm(G-[0 1;0 0], Inf) > 1e-5 && @warn "G is not close to the Jordan block of size 2. We built a basis such that G = $G. The norm of the difference is $(norm(G-[0 1;0 0], Inf))"

	# second differential
	R2(dx1, dx2) = d2F(VF, x0, parbif, dx1, dx2) ./2

	# quadratic coefficients
	R20 = R2(ζ0, ζ0)
	a = dot(ζs1, R20)
	b = 2dot(ζs0, R20) + 2dot(ζs1, R2(ζ0, ζ1))

	# return the normal form coefficients
	pt.nf = (; a, b)
	if detailed == false
		return pt
	end

	###########################
	# computation of the unfolding. We follow the procedure described in Al-Hdaibat et al. 2016

	# Al-Hdaibat, B., W. Govaerts, Yu. A. Kuznetsov, and H. G. E. Meijer. “Initialization of Homoclinic Solutions near Bogdanov--Takens Points: Lindstedt--Poincaré Compared with Regular Perturbation Method.” SIAM Journal on Applied Dynamical Systems 15, no. 2 (January 2016): 952–80. https://doi.org/10.1137/15M1017491.
	###########################
	# to have the same notations as in the paper above
	q0 = ζ0 ; q1 = ζ1;
	p0 = ζs0; p1 = ζs1;

	# second differential notations, to be in agreement with Kuznetsov et al.
	B(dx1, dx2) = d2F(VF, x0, parbif, dx1, dx2)
	Ainv(dx) = bls(L, p1, q0, zero(Ty), dx, zero(Ty))

	H2000, = Ainv(2 .* a .* q1 .- B(q0, q0))
	γ = (-2dot(p0, H2000) + 2dot(p0, B(q0, q1)) + dot(p1, B(q1, q1))) / 2
	H2000 .+= γ .* q0

	H1100, = Ainv(b .* q1 .+ H2000 .- B(q0, q1))
	H0200, = Ainv(2 .* H1100 .- B(q1, q1))

	# first order drivatives
	pBq(p, q) = 2 .* (applyJacobian(VF, x0 .+ ϵ .* q, parbif, p, true) .-
					  applyJacobian(VF, x0, parbif, p, true)) ./ ϵ
	A1(q, lens) = (applyJacobian(VF, x0, setp(lens, get(parbif, lens) + ϵ), q) .-
	 				  applyJacobian(VF, x0, parbif, q)) ./ϵ
	pAq(p, q, lens) =  dot(p, A1(q, lens))

	# second order derivative
	p10 = get(parbif, lens1); p20 = get(parbif, lens2);

	if autodiff
		Jp(p, l)  = ForwardDiff.derivative( P -> F(x0, setp(l, P)), p)
		Jpp(p, l) = ForwardDiff.derivative( P -> Jp(P, l), p)
		Fp(p1, p2)  = F(x0, setp(p1, p2))
		Jp1p2(p1, p2) = ForwardDiff.derivative(P1 -> ForwardDiff.derivative(P2 -> Fp(P1, P2) , p2), p1)

		J2_11 = Jpp(p10, lens1)
		J2_22 = Jpp(p20, lens2)
		J2_12 = Jp1p2(p10, p20)
	else #finite  differences. We need to be carreful here because (1e-8)^2 is really small!!
		J2_11 = (F(x0, setp(lens1, p10 + ϵ2)) .- 2 .* F(x0, setp(lens1, p10)) .+
				    F(x0, setp(lens1, p10 - ϵ2)) ) ./ ϵ2^2

		J2_22 = (F(x0, setp(lens2, p20 + ϵ2)) .- 2 .* F(x0, setp(lens2, p20)) .+
				   F(x0, setp(lens2, p20 - ϵ2)) )./ ϵ2^2

		J2_12 = (F(x0, setp(p10 + ϵ2, p20 + ϵ2)) .- F(x0, setp(lens1, p10 + ϵ2)) .-
													F(x0, setp(lens2, p20 + ϵ2)) .+ F(x0, parbif))./ ϵ2^2
	end

	# build the big matrix of size (n+2) x (n+2) A = [L J1s; A12 A22]
	J1 = lens -> F(x0, setp(lens, get(parbif, lens) + ϵ)) ./ ϵ
	J1s = (J1(lens1), J1(lens2))

	A12_1 = pBq(p1, q0) ./2
	A12_2 = (pBq(p0, q0) .+ pBq(p1, q1)) ./2
	A22 = [[pAq(p1, q0, lens1), pAq(p0, q0, lens1)+pAq(p1, q1, lens1)] [pAq(p1, q0, lens2), pAq(p0, q0, lens2)+pAq(p1, q1, lens2)] ]

	# solving the linear system of size n+2
	# TODO REMOVE THIS HACK FOR MATRIX-FREE
	Anp2 = [L hcat(J1s...); hcat(A12_1, A12_2)' A22]
	c = 3dot(p0, H1100) - dot(p0, B(q1, q1))

	sol = Anp2 \ vcat(q1, dot(p1, B(q1, q1))/2, c)
	H0010 = sol[1:end-2]
	@assert size(H0010) == size(x0)
	K10 = sol[end-1:end]

	sol = Anp2 \ vcat(zero(q1), zero(Ty), one(Ty))
	H0001 = sol[1:end-2]
	@assert size(H0001) == size(x0)
	K11 = sol[end-1:end]

	# computation of K2
	κ1 = dot(p1, B(H0001, H0001))
	κ2 = pAq(p1, H0001, lens1) * K11[1] +
		 pAq(p1, H0001, lens2) * K11[2]
	J2K = @. J2_11 * K11[1]^2 + 2J2_12 * K11[1] * K11[2] + J2_11 * K11[2]^2
	κ3 = dot(p1, J2K)
	K2 = -( κ1 + 2κ2 + κ3 ) .* K10

	# computation of H0002
	h0002 = B(H0001, H0001)
	h0002 .+= A1(H0001, lens1) .* (2K11[1]) .+ A1(H0001, lens2) .* (2K11[2])
	h0002 .+= J2K
	h0002 .+= J1s[1] .* K2[1] .+ J1s[2] .* K2[2]
	H0002, = Ainv(h0002)
	H0002 .*= -1

	# computation of H1001
	h1001 = B(q0, H0001)
	h1001 .+= A1(q0, lens1) .* K11[1] .+ A1(q0, lens2) .* K11[2]
	H1001, = Ainv(h1001)
	H1001 .*= -1

	# computation of H0101
	h0101 = B(q1, H0001)
	h0101 .+= A1(q1, lens1) .* K11[1] .+ A1(q1, lens2) .* K11[2]
	h0101 .-= H1001 .+ q1
	H0101, = Ainv(h0101)
	H0101 .*= -1

	# computation of H3000 and d
	h3000 = d3F(VF, x0, parbif, q0, q0, q0) .+ 3 .* B(q0, H2000) .- (6a) .* H1100
	d = dot(p1, h3000)/6
	h3000 .-= (6d) .* q1
	H3000, = Ainv(h3000)
	H3000 .*= -1

	# computation of e
	e = dot(p1, d3F(VF, x0, parbif, q0, q0, q0)) + 2dot(p1, B(q0, H1100)) + dot(p1, B(q1, H2000))
	e += -2b * dot(p1, H1100) -2a * dot(p1, H0200) - dot(p1, H3000)
	e /= 2

	# computation of H2001 and a1
	B1(q, p, l) = (d2F(VF, x0, setp(l, getp(l) + ϵ), q, p) .- d2F(VF, x0, parbif, q, p)) ./ ϵ
	h2001 = d3F(VF, x0, parbif, q0, q0, H0001) .+ 2 .* B(q0, H1001) .+ B(H0001, H2000)
	h2001 .+= B1(q0, q0, lens1) .* K11[1] .+ B1(q0, q0, lens2) .* K11[2]
	h2001 .+= A1(H2000, lens1)  .* K11[1] .+ A1(H2000, lens2)  .* K11[2]
	h2001 .-= (2a) .* H0101
	a1 = dot(p1, h2001) / 2
	h2001 .-= (2a1) .* q1
	H2001, = Ainv(h2001)
	H2001 .*= -1

	# computation of b1
	b1 = dot(p1, d3F(VF, x0, parbif, q0, q1, H0001)) +
		 dot(p1, B1(q0, q1, lens1)) * K11[1] +
		 dot(p1, B1(q0, q1, lens2)) * K11[2] +
		 dot(p1, B(q1, H1001)) +
		 dot(p1, B(H0001, H1100)) +
		 dot(p1, B(q0, H0101)) +
		 dot(p1, A1(H1100, lens1)) * K11[1] + dot(p1, A1(H1100, lens2)) * K11[2] -
		 b * dot(p1, H0101) - dot(p1, H1100) - dot(p1, H2001)

	verbose && println(pt.nf)
	return @set pt.nfsupp = (; γ, c, K10, K11, K2, d, e, a1, b1, H0001, H0010, H0002, H1001, H2000)
end

function predictor(bt::BogdanovTakens, ::Val{:HopfCurve}, ds::T; verbose = false, ampfactor = T(1)) where T
	# If we write the normal form [y2, β1 + β2 y2 + a y1^2 + b y1 y2]
	# equilibria y2 = 0, 0 = β1 + a y1^2
	# Characteristic polynomial: t^2 + (-x*b - β2)*t - 2*x*a
	# the fold curve is β1 / a < 0 with x± := ±√(-β1/a)v
	# the Hopf curve is 0 = -x*b - β2, -x⋅a > 0
	# ie β2 = -bx with ±b√(-β1/a)
	@unpack a, b = bt.nf
	@unpack K10, K11, K2 = bt.nfsupp
	lens1, lens2 = bt.lens
	p1 = get(bt.params, lens1)
	p2 = get(bt.params, lens2)
	par0 = [p1, p2]
	getx(s) = a > 0 ? -sqrt(abs(s) / a) : sqrt(abs(s) / abs(a))

	function HopfCurve(s)
		# x = getx(s)
		if a > 0
			x = -sqrt(abs(s) / a)
			β1 = -abs(s)
		else
			x = sqrt(abs(s) / abs(a))
			β1 = abs(s)
		end
		β2 = -b * x
		ω = sqrt(-2x*a)
		return (pars = par0 .+ K10 .* β1 .+ K11 .* β2 .+ K2 .* (β2^2/2), ω = ω)
	end

	# compute eigenvector corresponding to the Hopf branch
	function EigenVec(s)
		x = getx(s)
		# the jacobian is [0 1; 2x*a b*X+β2] with b*X+β2 = 0
		A = [0 1; 2x*a 0]
		F = eigen(A)
		ind = findall(imag.(F.values) .> 0)
		hopfvec = F.vectors[:, ind]
		return bt.ζ[1] .* hopfvec[1] .+ bt.ζ[2] .* hopfvec[2]
	end

	function EigenVecAd(s)
		x = getx(s)
		# the jacobian is [0 1; 2x*a b*X+β2] with b*X+β2 = 0
		A = [0 1; 2x*a 0]'
		F = eigen(A)
		ind = findall(imag.(F.values) .< 0)
		hopfvec = F.vectors[:, ind]
		return bt.ζstar[1] .* hopfvec[1] .+ bt.ζstar[2] .* hopfvec[2]
	end

	# compute point on the Hopf curve
	x0 = getx(ds)

	return (hopf = t->HopfCurve(t).pars,
			ω = t->HopfCurve(t).ω,
			EigenVec = EigenVec,
			EigenVecAd = EigenVecAd,
			x0 = t -> getx(t) .* bt.ζ[1])
end

function predictor(bt::BogdanovTakens, ::Val{:FoldCurve}, ds::T; verbose = false, ampfactor = T(1)) where T
	# If we write the normal form [y2, β1 + β2 y2 + a y1^2 + b y1 y2]
	# equilibria y2 = 0, 0 = β1 + a y1^2
	# the fold curve is β1 / a < 0 with x± := ±√(-β1/a)
	# the Hopf curve is 0 = -x*b - β2, x⋅a > 0
	# ie β2 = -bx with ±b√(-β1/a)
	@unpack a, b = bt.nf
	@unpack K10, K11, K2 = bt.nfsupp
	lens1, lens2 = bt.lens
	p1 = get(bt.params, lens1)
	p2 = get(bt.params, lens2)
	par0 = [p1, p2]
	getx(s) = a > 0 ? -sqrt(abs(s) / a) : sqrt(abs(s) / abs(a))
	function FoldCurve(s)
		β1 = 0
		β2 = s
		return par0 .+ K10 .* β1 .+ K11 .* β2 .+ K2 .* (β2^2/2)
	end
	return (fold = FoldCurve,
			EigenVec = t -> (bt.ζ[1]),
			EigenVecAd = t -> (bt.ζstar[2]),
			x0 = t -> getx(t) .* bt.ζ[1])
end

function predictor(bt::BogdanovTakens, ::Val{:HomoclinicCurve}, ds::T; verbose = false, ampfactor = 	one(T)) where T
	# we follow
	# Al-Hdaibat, B., W. Govaerts, Yu. A. Kuznetsov, and H. G. E. Meijer. “Initialization of Homoclinic Solutions near Bogdanov--Takens Points: Lindstedt--Poincaré Compared with Regular Perturbation Method.” SIAM Journal on Applied Dynamical Systems 15, no. 2 (January 2016): 952–80. https://doi.org/10.1137/15M1017491.

	@unpack a, b = bt.nf
	@unpack K10, K11, K2, b1, e, d, a1 = bt.nfsupp
	@unpack H0001, H0010, H0002, H1001, H2000 = bt.nfsupp

	lens1, lens2 = bt.lens
	p1 = get(bt.params, lens1)
	p2 = get(bt.params, lens2)
	par0 = [p1, p2]

	# formula 63
	τ2 = 4/a * (25/49*b1 - e/b) + 2/(49a^2) * (144/49b^2 - 25b*a1 + 73d)

	# formula 69
	α(ϵ) = @. par0 + (10b*ϵ^2 / (7a)) * K11 + ϵ^4/a * ( -4*K10 + 50b^2/(49a) * K2 + b * τ2 * K11)

	# formula 71
	q0, q1 = bt.ζ

	u0(ξ) = -6sech(ξ)^2 + 2
	v0(ξ) = 12sech(ξ)^2 * tanh(ξ)
	u1(ξ) = 0
	v1(ξ) = -6b/(7a) * tanh(ξ) * v0(ξ)
	u2(ξ) = -3/(49a^2) * (6b^2 - 70b*a1 + 49d) * sech(ξ)^2 - 2(5a1*b + 7d)/(7a^2)

	function xLP(t, ϵ)
		ξ = ϵ * t
		return @. bt.x0 + (ϵ^2/a) * ( (10b/7) * H0001 + u0(ξ) * q0) +
			(ϵ^3/a) * ( v0(ξ) * q1 + u1(ξ) * q0) +
			(ϵ^4/a) * ( -4 * H0010 + 50 * b^2/(49a) * H0002 + b*τ2 * H0001 +
						u2(ξ) * q0 + v1(ξ) * q1 +
						1/(2a) * u0(ξ)^2 * H2000 + 10b/(7a) * u0(ξ) * H1001)
	end

	return (α = α, orbit = xLP)
end

"""
$(SIGNATURES)

Compute the Bogdanov-Takens normal form.

# Arguments
- `prob` bifurcation problem, typically `br.prob`
- `br` branch result from a call to [`continuation`](@ref)
- `ind_bif` index of the bifurcation point in `br`
- `options` options for the Newton solver

# Optional arguments
- `δ = 1e-8` used for finite differences
- `nev = 5` number of eigenvalues to compute to estimate the spectral projector
- `verbose` bool to print information
- `autodiff = true` only for Bogdanov-Takens point. Whether to use ForwardDiff for the many differentiations that are required to compute the normal form.
- `detailed = true` only for Bogdanov-Takens point. Whether to compute only a simplified normal form.
"""
function bogdanovTakensNormalForm(_prob,
		br::AbstractBranchResult, ind_bif::Int;
		δ = 1e-8,
		nev = length(eigenvalsfrombif(br, ind_bif)),
		verbose = false,
		ζs = nothing,
		lens = getLens(br),
		Teigvec = getvectortype(br),
		scaleζ = norm,
		detailed = true,
		autodiff = true)
	@assert getvectortype(br) <: BorderedArray
	@assert br.specialpoint[ind_bif].type == :bt "The provided index does not refer to a Bogdanov-Takens Point"

	# functional
	# get the MA problem
	prob_ma = _prob.prob

	# get the initial vector field
	prob_vf = prob_ma.prob_vf

	@assert prob_ma isa AbstractProblemMinimallyAugmented

	# bordered linear solver
	bls = prob_ma.linbdsolver

	# kernel dimension
	N = 2

	# in case nev = 0 (number of requested eigenvalues), we increase nev to avoid bug
	nev = max(2N, nev)
	verbose && println("#"^53*"\n--> Bogdanov-Takens Normal form computation")

	# Newton parameters
	optionsN = br.contparams.newtonOptions

	# bifurcation point
	bifpt = br.specialpoint[ind_bif]
	eigRes = br.eig

	# parameters for vector field
	p = bifpt.param
	parbif = set(getParams(br), lens, p)
	parbif = set(parbif, getLens(prob_ma), get(bifpt.printsol, getLens(prob_ma)))

	# jacobian at bifurcation point
	if Teigvec <: BorderedArray
		x0 = convert(Teigvec.parameters[1], getVec(bifpt.x, prob_ma))
	else
		x0 = convert(Teigvec, getVec(bifpt.x , prob_ma))
	end
	Ty = eltype(Teigvec)
	L = jacobian(prob_vf, x0, parbif)

	# "zero" eigenvalues at bifurcation point
	rightEv = br.eig[bifpt.idx].eigenvals
	indev = br.specialpoint[ind_bif].ind_ev

	# and corresponding eigenvectors
	eigsolver = getsolver(optionsN.eigsolver)
	if isnothing(ζs) # do we have a basis for the kernel?
		if haseigenvector(br) == false # are the eigenvector saved in the branch?
			@info "No eigenvector recorded, computing them on the fly"
			# we recompute the eigen-elements if there were not saved during the computation of the branch
			_λ0, _ev, _ = eigsolver(L, nev)
			Ivp = sortperm(_λ0, by = abs)
			_λ = _λ0[Ivp]
			verbose && (println("--> (λs, λs (recomputed)) = "); display(( _λ[1:N])))
			if norm(_λ[1:N] .- 0, Inf) > br.contparams.tolStability
				@warn "We did not find the correct eigenvalues (see 1st col). We found the eigenvalues displayed in the second column:\n $(display(( _λ[1:N]))).\n Difference between the eigenvalues:"
				display(_λ[1:N] .- 0)
			end
			ζs = [copy(geteigenvector(eigsolver, _ev, ii)) for ii in Ivp[1:N]]
		else
			# find the 2 eigenvalues closest to zero
			Ind = sortperm(abs.(rightEv))
			ind0 = Ind[1]
			ind1 = Ind[2]
			verbose && (println("----> eigenvalues = ", rightEv[Ind[1:2]]))
			ζs = [copy(geteigenvector(eigsolver, br.eig[bifpt.idx].eigenvecs, ii)) for ii in (ind0, ind1)]
		end
	end
	###########################
	# Construction of the basis (ζ0, ζ1), (ζstar0, ζstar1). We follow the procedure described in Al-Hdaibat et al. 2016 on page 972.

	# Al-Hdaibat, B., W. Govaerts, Yu. A. Kuznetsov, and H. G. E. Meijer. “Initialization of Homoclinic Solutions near Bogdanov--Takens Points: Lindstedt--Poincaré Compared with Regular Perturbation Method.” SIAM Journal on Applied Dynamical Systems 15, no. 2 (January 2016): 952–80. https://doi.org/10.1137/15M1017491.
	###########################
	vext = real.(ζs[1])
	Lᵗ = hasAdjoint(prob_vf) ? jad(prob_vf, x0, parbif) : transpose(L)
	_λstar, _evstar, _ = eigsolver(Lᵗ, nev)
	Ivp = sortperm(_λstar, by = abs)
	# in case the prob is HopfMA, we real it
	zerov = real.(prob_ma.zero)
	wext = real.(geteigenvector(eigsolver, _evstar, Ivp[1]))
	q0, = bls(L, wext, vext, zero(Ty), zerov, one(Ty))
	p1, = bls(Lᵗ, vext, wext, zero(Ty), zerov, one(Ty))
	q1, = bls(L, p1, q0, zero(Ty), q0, zero(Ty))
	p0, = bls(Lᵗ, q0, p1, zero(Ty), p1, zero(Ty))
	# we want
	# A⋅q0 = 0, A⋅q1 = q0
	# At⋅p1 = 0, At⋅p0 = p1

	μ = √(abs(dot(q0, q0)))
	q0 ./= μ
	q1 ./= μ
	q1 .= q1 .- dot(q0, q1) .* q0
	ν = dot(q0, p0)
	p1 ./= ν
	p0 .= p0 .- dot(p0, q1) .* p1
	p0 ./= ν

	pt = BogdanovTakens(
		x0, parbif, (getLens(prob_ma), lens),
		(;q0, q1), (;p0, p1),
		(a = zero(Ty), b = zero(Ty) ),
		(K2 = zero(Ty),),
		:none
	)
	return bogdanovTakensNormalForm(prob_ma, L, pt; δ = δ, verbose = verbose, detailed = detailed, autodiff = autodiff)
end
####################################################################################################
function bautinNormalForm(_prob,
		br::AbstractBranchResult, ind_bif::Int;
		δ = 1e-8,
		nev = length(eigenvalsfrombif(br, ind_bif)),
		verbose = false,
		ζs = nothing,
		lens = getLens(br),
		Teigvec = getvectortype(br),
		scaleζ = norm)
	@assert br.specialpoint[ind_bif].type == :gh "The provided index does not refer to a Bautin Point"

	verbose && println("#"^53*"\n--> Bautin Normal form computation")

	# get the MA problem
	prob_ma = _prob.prob
	# get the initial vector field
	prob_vf = prob_ma.prob_vf

	# scalar type
	T = eltype(Teigvec)
	ϵ2 = T(δ)

	# functional
	@assert prob_ma isa HopfProblemMinimallyAugmented
	ls = prob_ma.linsolver
	bls = prob_ma.linbdsolver

	# ``kernel'' dimension
	N = 2

	# in case nev = 0 (number of unstable eigenvalues), we increase nev to avoid bug
	nev = max(N, nev)

	# Newton parameters
	optionsN = br.contparams.newtonOptions

	# bifurcation point
	bifpt = br.specialpoint[ind_bif]
	eigRes = br.eig

	# eigenvalue
	ω = abs(getP(bifpt.x, prob_ma)[2])
	λ = Complex(0, ω)

	# parameter for vector field
	p = bifpt.param
	parbif = set(getParams(br), lens, p)
	parbif = set(parbif, getLens(prob_ma), get(bifpt.printsol, getLens(prob_ma)))

	# jacobian at bifurcation point
	if Teigvec <: BorderedArray
		x0 = convert(Teigvec.parameters[1], getVec(bifpt.x, prob_ma))
	else
		x0 = convert(Teigvec, getVec(bifpt.x, prob))
	end
	L = jacobian(prob_vf, x0, parbif)

	# right eigenvector
	# TODO IMPROVE THIS
	if 1==1#haseigenvector(br) == false
		# we recompute the eigen-elements if there were not saved during the computation of the branch
		@info "Recomputing eigenvector on the fly"
		_λ, _ev, _ = optionsN.eigsolver.eigsolver(L, nev)
		_ind = argmin(abs.(_λ .- λ))
		@info "The eigenvalue is $(_λ[_ind])"
		@warn abs(_λ[_ind] - λ) < br.contparams.newtonOptions.tol "We did not find the correct eigenvalue $λ. We found $(_λ[_ind])"
		ζ = geteigenvector(optionsN.eigsolver, _ev, _ind)
	else
		ζ = copy(geteigenvector(optionsN.eigsolver, br.eig[bifpt.idx].eigenvecs, bifpt.ind_ev))
	end
	ζ ./= scaleζ(ζ)

	# left eigen-elements
	_Jt = hasAdjoint(prob_vf) ? jad(prob_vf, x0, parbif) : adjoint(L)
	ζstar, λstar = getAdjointBasis(_Jt, conj(_λ[_ind]), optionsN.eigsolver.eigsolver; nev = nev, verbose = verbose)

	# check that λstar ≈ conj(λ)
	abs(λ + λstar) > 1e-2 && @warn "We did not find the left eigenvalue for the Hopf point to be very close to the imaginary part, $λ ≈ $(λstar) and $(abs(λ + λstar)) ≈ 0?\n You can perhaps increase the number of computed eigenvalues, the number is nev = $nev"

	# normalise left eigenvector
	ζstar ./= dot(ζ, ζstar)
	@assert dot(ζ, ζstar) ≈ 1

	# parameters for vector field
	p = bifpt.param
	parbif = set(getParams(br), lens, p)
	parbif = set(parbif, getLens(prob_ma), get(bifpt.printsol, getLens(prob_ma)))

	# second order differential, to be in agreement with Kuznetsov et al.
	B = BilinearMap( (dx1, dx2) -> d2F(prob_vf, x0, parbif, dx1, dx2) )
	C = TrilinearMap((dx1, dx2, dx3) -> d3F(prob_vf, x0, parbif, dx1, dx2, dx3) )

	q0 = ζ; p0 = ζstar
	cq0 = conj.(q0)

	H20, = ls(L, B(q0, q0); a₀ = Complex(0, 2ω), a₁ = -1)
	H11, = ls(L, -B(q0, cq0))
	H30, = ls(L, C(q0, q0, q0) .+ 3 .* B(q0, H20); a₀ = Complex(0, 3ω), a₁ = -1)

	h21 = C(q0, q0, cq0) .+ B(cq0, H20) .+ 2 .* B(q0, H11)
	G21 = dot(p0, h21)
	h21 .= G21 .* q0 .- h21
	H21, = bls(L, q0, p0, zero(T), h21, zero(T); shift = Complex{T}(0, -ω))
	# sol = [L-λ*I q0; p0' 0] \ [h21..., 0]

	# 4-th order coefficient
	d4F(x0, dx1, dx2, dx3, dx4) = (d3F(prob_vf, x0 .+ ϵ2 .* dx4, parbif, dx1, dx2, dx3) .-
								   d3F(prob_vf, x0 .- ϵ2 .* dx4, parbif, dx1, dx2, dx3)) ./(2ϵ2)

	# implement 4th order differential with finite differences
	function D(x0, dx1, dx2, dx3, dx4)
		dx4r = real.(dx4); dx4i = imag.(dx4);
		# C(dx, dx4r) + i * C(dx, dx4i)
		trilin_r = TrilinearMap((_dx1, _dx2, _dx3) -> d4F(x0, _dx1, _dx2, _dx3, dx4r) )
		out1 = trilin_r(dx1, dx2, dx3)
		trilin_i = TrilinearMap((_dx1, _dx2, _dx3) -> d4F(x0, _dx1, _dx2, _dx3, dx4i) )
		out2 = trilin_i(dx1, dx2, dx3)
		return out1 .+ im .* out2
	end

	h31 = D(x0, q0, q0, q0, cq0) .+ 3 .* C(q0, q0, H11) .+ 3 .* C(q0, cq0, H20) .+ 3 .* B(H20, H11)
	h31 .+= B(cq0, H30) .+ 3 .* B(q0, H21) .- (3 * G21) .* H20
	H31, = ls(L, h31; a₀ = Complex(0, 2ω), a₁ = -1)

	h22 = D(x0, q0, q0, cq0, cq0) .+
		4 .* C(q0, cq0, H11) .+ C(cq0, cq0, H20) .+ C(q0, q0, conj.(H20)) .+
		2 .* B(H11, H11) .+ 2 .* B(q0, conj.(H21)) .+ 2 .* B(cq0, H21) .+ B(conj.(H20), H20) .-
		(2G21 + 2conj(G21)) .* H11
	H22, = ls(L, h22)
	H22 .*= -1

	# 5-th order coefficient
	# implement 5th order differential with finite differences
	function E(dx1, dx2, dx3, dx4, dx5)
		dx5r = real.(dx5); dx5i = imag.(dx5);
		out1 = (D(x0 .+ ϵ2 .* dx5r, dx1, dx2, dx3, dx4) .-
			    D(x0 .- ϵ2 .* dx5r, dx1, dx2, dx3, dx4)) ./(2ϵ2)
	    out2 = (D(x0 .+ ϵ2 .* dx5i, dx1, dx2, dx3, dx4) .-
			    D(x0 .- ϵ2 .* dx5i, dx1, dx2, dx3, dx4)) ./(2ϵ2)
		return out1 .+ im .* out2
	end

	G32 = dot(p0, E(q0, q0, q0, cq0, cq0))
	G32 += dot(p0, D(x0, q0, q0, q0, conj.(H20))) +
		  3dot(p0, D(x0, q0, cq0, cq0, H20)) +
		  6dot(p0, D(x0, q0, q0, cq0, H11))

	G32 += dot(p0, C(cq0, cq0, H30)) +
		  3dot(p0, C(q0, q0, conj.(H21))) +
		  6dot(p0, C(q0, cq0, H21)) +
		  3dot(p0, C(q0, conj.(H20), H20)) +
		  6dot(p0, C(q0, H11, H11)) +
		  6dot(p0, C(cq0, H20, H11))

	G32 += 2dot(p0, B(cq0, H31)) +
		   3dot(p0, B(q0, H22)) +
		    dot(p0, B(conj(H20), H30)) +
		   3dot(p0, B(conj(H21), H20)) +
		   6dot(p0, B(H11, H21))

	# second Lyapunov coefficient
	l2 = real(G32) / 12

	# the unfolding are in meijer. “Switching to Nonhyperbolic Cycles from Codim 2 Bifurcations of Equilibria in ODEs,” 2005. https://doi.org/10.1016/j.physd.2008.06.006.

	pt = Bautin(
		x0, parbif,
		(getLens(prob_ma), lens),
		ζ, ζstar,
		(;ω, G21, G32, l2 ),
		:none
	)
end
####################################################################################################
function zeroHopfNormalForm(_prob,
		br::AbstractBranchResult, ind_bif::Int;
		δ = 1e-8,
		nev = length(eigenvalsfrombif(br, ind_bif)),
		verbose = false,
		ζs = nothing,
		lens = getLens(br),
		Teigvec = getvectortype(br),
		scaleζ = norm,
		autodiff = true)
	@assert br.specialpoint[ind_bif].type == :zh "The provided index does not refer to a Zero-Hopf Point"

	verbose && println("#"^53*"\n--> Zero-Hopf Normal form computation")

	# scalar type
	T = eltype(Teigvec)
	ϵ2 = T(δ)

	# get the MA problem
	prob_ma = _prob.prob

	# get the initial vector field
	prob_vf = prob_ma.prob_vf

	@assert prob_ma isa AbstractProblemMinimallyAugmented

	# linear solver
	ls = prob_ma.linsolver

	# bordered linear solver
	bls = prob_ma.linbdsolver

	# kernel dimension
	N = 3

	# in case nev = 0 (number of unstable eigenvalues), we increase nev to avoid bug
	nev = max(N, nev)

	# Newton parameters
	optionsN = br.contparams.newtonOptions

	# bifurcation point
	bifpt = br.specialpoint[ind_bif]
	eigRes = br.eig

	# eigenvalue
	# ω = abs(getP(bifpt.x, prob)[2])
	# λ = Complex(0, ω)

	# parameter for vector field
	p = bifpt.param
	parbif = set(getParams(br), lens, p)
	parbif = set(parbif, getLens(prob_ma), get(bifpt.printsol, getLens(prob_ma)))

	# jacobian at bifurcation point
	if Teigvec <: BorderedArray
		x0 = convert(Teigvec.parameters[1], getVec(bifpt.x, prob_ma))
	else
		x0 = convert(Teigvec, getVec(bifpt.x, prob_ma))
	end
	L = jacobian(prob_vf, x0, parbif)

	# right eigenvector
	# TODO IMPROVE THIS
	if 1==1#haseigenvector(br) == false
		# we recompute the eigen-elements if there were not saved during the computation of the branch
		@info "Recomputing eigenvector on the fly"
		_λ, _ev, _ = optionsN.eigsolver.eigsolver(L, nev)
		# null eigenvalue
		_ind0 = argmin(abs.(_λ))
		@info "The eigenvalue is $(_λ[_ind0])"
		@warn abs(_λ[_ind0]) < br.contparams.newtonOptions.tol "We did not find the correct eigenvalue 0. We found $(_λ[_ind0])"
		q0 = geteigenvector(optionsN.eigsolver, _ev, _ind0)
		# imaginary eigenvalue
		_indIm = argmin(real.((_λ[ii] for ii = eachindex(_λ) if ii!=_ind0)))
		λI = _λ[_indIm]
		q1 = geteigenvector(optionsN.eigsolver, _ev, _indIm)
	else
		@assert 1==0 "Not done"
		ζ = copy(geteigenvector(optionsN.eigsolver ,br.eig[bifpt.idx].eigenvec, bifpt.ind_ev))
	end
	q0 ./= scaleζ(q0)

	# left eigen-elements
	_Jt = hasAdjoint(prob_vf) ? jad(prob_vf, x0, parbif) : adjoint(L)
	p0, λstar = getAdjointBasis(_Jt, conj(_λ[_ind0]), optionsN.eigsolver.eigsolver; nev = nev, verbose = verbose)
	p1, λstar1 = getAdjointBasis(_Jt, conj(λI), optionsN.eigsolver.eigsolver; nev = nev, verbose = verbose)

	# normalise left eigenvectors
	p0 ./= dot(p0, q0)
	@show dot(p1, q1)
	p1 ./= dot(q1, p1)
	@assert dot(p0, q0) ≈ 1
	@assert dot(p1, q1) ≈ 1

	# parameters for vector field
	p = bifpt.param
	parbif = set(getParams(br), lens, p)
	parbif = set(parbif, getLens(prob_ma), get(bifpt.printsol, getLens(prob_ma)))

	# parameters
	lenses = (getLens(prob_ma), lens)
	lens1, lens2 = lenses
	p10 = get(parbif, lens1); p20 = get(parbif, lens2);

	getp(l::Lens) = get(parbif, l)
	setp(l::Lens, p::Number) = set(parbif, l, p)
	setp(p1::Number, p2::Number) = set(set(parbif, lens1, p1), lens2, p2)
	if autodiff
		Jp = (p, l) -> ForwardDiff.derivative( P -> residual(prob_vf, x0, setp(l, P)) , p)
	else
		# finite differencess
		Jp = (p, l) -> (residual(prob_vf, x0, setp(l, p + ϵ2)) .- residual(prob_vf, x0, setp(l, p - ϵ2)) ) ./ (2ϵ2)
	end


	dFp = [dot(p0, Jp(p10, lens1)) dot(p0, Jp(p20, lens2)); dot(p1, Jp(p10, lens1)) dot(p1, Jp(p20, lens2))]

	pt = ZeroHopf(
		x0, parbif,
		lenses,
		(;q0, q1), (;p0, p1),
		(;ω = λI, λ0 = _λ[_ind0], dFp),
		:none
	)
end

function predictor(zh::ZeroHopf, ::Val{:HopfCurve}, ds::T; verbose = false, ampfactor = T(1)) where T
	@unpack ω, λ0 = zh.nf
	lens1, lens2 = zh.lens
	p1 = get(zh.params, lens1)
	p2 = get(zh.params, lens2)
	par0 = [p1, p2]
	function HopfCurve(s)
		return (pars = par0 , ω = abs(ω))
	end
	# compute eigenvector corresponding to the Hopf branch
	function EigenVec(s)
		return zh.ζ.q1
	end
	function EigenVecAd(s)
		return zh.ζstar.p1
	end

	return (hopf = t -> HopfCurve(t).pars,
			ω    = t -> HopfCurve(t).ω,
			EigenVec = EigenVec,
			EigenVecAd = EigenVecAd,
			x0 = t -> 0)
end
