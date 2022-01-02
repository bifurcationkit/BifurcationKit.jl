"""
$(SIGNATURES)

Compute the Cusp normal form.

# Arguments
- `F, dF, d2F, d3F`: function `(x, p) -> F(x, p)` and its differentials `(x, p, dx) -> d1F(x, p, dx)`, `(x, p, dx1, dx2) -> d2F(x, p, dx1, dx2)`...
- `pt::Cusp` Cusp bifurcation point
- `ls` linear solver

# Optional arguments
- `δ = 1e-8` used for finite differences
- `verbose` bool to print information
"""
function cuspNormalForm(F, dF, d2F, d3F,
		br::AbstractBranchResult, ind_bif::Int;
		δ = 1e-8,
		nev = length(eigenvalsfrombif(br, ind_bif)),
		Jᵗ = nothing,
		verbose = false,
		ζs = nothing,
		lens = br.lens,
		Teigvec = getvectortype(br),
		scaleζ = norm,
		issymmetric = false)
	@assert getvectortype(br) <: BorderedArray
	@assert br.specialpoint[ind_bif].x isa BorderedArray
	@assert br.specialpoint[ind_bif].type == :cusp "The provided index does not refer to a Cusp Point"

	verbose && println("#"^53*"\n--> Cusp Normal form computation")

	# scalar type
	T = eltype(Teigvec)
	ϵ2 = T(δ)

	# functional
	prob = br.functional

	# linear solvers
	ls = prob.linsolver
	bls = prob.linbdsolver

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
	parbif = set(br.params, lens, p)
	parbif = set(parbif, prob.lens, get(bifpt.printsol, prob.lens))

	# jacobian at bifurcation point
	x0 = convert(Teigvec.parameters[1], bifpt.x.u)
	L = dF(x0, parbif)

	# eigenvectors
	# we recompute the eigen-elements if there were not saved during the computation of the branch
	@info "Eigen-elements not saved in the branch. Recomputing them..."
	eigsolver = getsolver(options.eigsolver)
	_λ0, _ev0, _ = eigsolver(L, nev)
	Ivp = sortperm(_λ0, by = abs)
	_λ = _λ0[Ivp]
	if norm(_λ[1:N] .- 0, Inf) > br.contparams.precisionStability
		@warn "We did not find the correct eigenvalues. We found the eigenvalues:\n $(display(( _λ[1:N]))).\n Difference between the eigenvalues:"
		display(_λ[1:N] .- 0)
	end
	ζ = real.(geteigenvector(eigsolver, _ev0, Ivp[1]))
	ζ ./= scaleζ(ζ)

	# extract eigen-elements for adjoint(L), needed to build spectral projector
	if issymmetric
		λstar = br.eig[bifpt.idx].eigenvals[bifpt.ind_ev]
		ζstar = copy(ζ)
	else
		_Jt = isnothing(Jᵗ) ? adjoint(L) : Jᵗ(x0, parbif)
		ζstar, λstar = getAdjointBasis(_Jt, conj(λ), eigsolver; nev = nev, verbose = verbose)
	end

	ζstar = real.(ζstar); λstar = real.(λstar)

	@assert abs(dot(ζ, ζstar)) > 1e-10 "We got ζ⋅ζstar = $((dot(ζ, ζstar))). This dot product should not be zero. Perhaps, you can increase `nev` which is currently $nev."
	ζstar ./= dot(ζ, ζstar)

	# Kuznetsov, Yu. A. “Numerical Normalization Techniques for All Codim 2 Bifurcations of Equilibria in ODE’s.” SIAM Journal on Numerical Analysis 36, no. 4 (January 1, 1999): 1104–24. https://doi.org/10.1137/S0036142998335005.
	# notations from this paper
	B(dx1, dx2) = d2F(x0, parbif, dx1, dx2)
	C(dx1, dx2, dx3) = d3F(x0, parbif, dx1, dx2, dx3)
	q = ζ; p = ζstar

	h2 = B(q, q)
	h2 .= dot(p, h2) .* q .- h2
	H2, = bls(L, q, p, zero(T), h2, zero(T))

	c = dot(p, C(q, q, q)) + 3dot(p, B(q, H2))
	c /= 6

	pt = Cusp(
		x0, parbif,
		(lens, prob.lens),
		ζ, ζstar,
		(c = c, ),
		:none
	)
end

"""
$(SIGNATURES)

Compute the Bogdanov-Takens normal form.

# Arguments
- `F, dF, d2F, d3F`: function `(x, p) -> F(x, p)` and its differentials `(x, p, dx) -> d1F(x, p, dx)`, `(x, p, dx1, dx2) -> d2F(x, p, dx1, dx2)`...
- `pt::BogdanovTakens` BogdanovTakens bifurcation point
- `ls` linear solver

# Optional arguments
- `δ = 1e-8` used for finite differences
- `verbose` bool to print information
- `autodiff = true` only for Bogdanov-Takens point. Whether to use ForwardDiff for the many differentiations that are required to compute the normal form.
- `detailed = true` only for Bogdanov-Takens point. Whether to compute only a simplified normal form.
"""
function bogdanovTakensNormalForm(prob, L, d2F, d3F,
							pt::BogdanovTakens;
							δ = 1e-8,
							verbose = false,
							detailed = true,
							autodiff = true)
	x0 = pt.x0
	parbif = pt.params
	Ty = eltype(x0)

	# for finite differences
	ϵ = convert(Ty, δ)
	ϵ2 = sqrt(ϵ) # this one is for second order differential

	F = prob.F
	J = prob.J

	# linear solvers
	ls = prob.linsolver
	bls = prob.linbdsolver

	lens1, lens2 = pt.lens

	getp(l::Lens) = get(parbif, l)
	setp(l::Lens, p::Number) = set(parbif, l, p)
	setp(p1::Number, p2::Number) = set(set(parbif, lens1, p1), lens2, p2)

	ζ0, ζ1 = pt.ζ
	ζs0, ζs1 = pt.ζstar

	G = [dot(xs, x) for xs in pt.ζstar, x in pt.ζ]
	norm(G-I(2), Inf) > 1e-5 && @warn "G == I(2) is not valid. We built a basis such that G = $G"

	G = [dot(xs, L*x) for xs in pt.ζstar, x in pt.ζ]
	norm(G-[0 1;0 0], Inf) > 1e-5 && @warn "G is not close to the Jordan block of size 2. We built a basis such that G = $G. The norm of the difference is $(norm(G-[0 1;0 0], Inf))"

	# second differential
	R2(dx1, dx2) = d2F(x0, parbif, dx1, dx2) ./2

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
	B(dx1, dx2) = d2F(x0, parbif, dx1, dx2)
	Ainv(dx) = bls(L, p1, q0, zero(Ty), dx, zero(Ty))

	H2000, = Ainv(2 .* a .* q1 .- B(q0, q0))
	γ = (-2dot(p0, H2000) + 2dot(p0, B(q0, q1)) + dot(p1, B(q1, q1))) / 2
	H2000 .+= γ .* q0

	H1100, = Ainv(b .* q1 .+ H2000 .- B(q0, q1))
	H0200, = Ainv(2 .* H1100 .- B(q1, q1))

	# first order drivatives
	pBq(p, q) = 2 .* (applyJacobian(prob, x0 + ϵ * q, parbif, p, true) .-
					  applyJacobian(prob, x0, parbif, p, true)) ./ ϵ
	A1(q, lens) = (applyJacobian(prob, x0, setp(lens, get(parbif, lens) + ϵ), q) .-
	 				  applyJacobian(prob, x0, parbif, q)) ./ϵ
	pAq(p, q, lens) =  dot(p, A1(q, lens))

	# second order derivative
	p10 = get(parbif, lens1); p20 = get(parbif, lens2);

	if autodiff
		Jp(p, l)  = ForwardDiff.derivative( P -> F(x0, setp(l, P)) , p)
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
	h3000 = d3F(x0, parbif, q0, q0, q0) .+ 3 .* B(q0, H2000) .- (6a) .* H1100
	d = dot(p1, h3000)/6
	h3000 .-= (6d) .* q1
	H3000, = Ainv(h3000)
	H3000 .*= -1

	# computation of e
	e = dot(p1, d3F(x0, parbif, q0, q0, q0)) + 2dot(p1, B(q0, H1100)) + dot(p1, B(q1, H2000))
	e += -2b * dot(p1, H1100) -2a * dot(p1, H0200) - dot(p1, H3000)
	e /= 2

	# computation of H2001 and a1
	B1(q, p, l) = (d2F(x0, setp(l, getp(l) + ϵ), q, p) .- d2F(x0, parbif, q, p)) ./ ϵ
	h2001 = d3F(x0, parbif, q0, q0, H0001) .+ 2 .* B(q0, H1001) .+ B(H0001, H2000)
	h2001 .+= B1(q0, q0, lens1) .* K11[1] .+ B1(q0, q0, lens2) .* K11[2]
	h2001 .+= A1(H2000, lens1)  .* K11[1] .+ A1(H2000, lens2)  .* K11[2]
	h2001 .-= (2a) .* H0101
	a1 = dot(p1, h2001) / 2
	h2001 .-= (2a1) .* q1
	H2001, = Ainv(h2001)
	H2001 .*= -1

	# computation of b1
	b1 = dot(p1, d3F(x0, parbif, q0, q1, H0001)) +
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
	# compute point on the Fold curve
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

function predictor(bt::BogdanovTakens, ::Val{:HomoclinicCurve}, ds::T; verbose = false, ampfactor = T(1)) where T
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
		return @. (ϵ^2/a) * ( (10b/7) * H0001 + u0(ξ) * q0) +
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
- `F, dF, d2F, d3F`: function `(x, p) -> F(x, p)` and its differentials `(x, p, dx) -> d1F(x, p, dx)`, `(x, p, dx1, dx2) -> d2F(x, p, dx1, dx2)`...
- `br` branch result from a call to [`continuation`](@ref)
- `ind_bif` index of the bifurcation point in `br`
- `options` options for the Newton solver

# Optional arguments
- `Jᵗ` is the jacobian adjoint, used for computation of the eigen-elements of the jacobian adjoint, needed to compute the spectral projector
- `δ = 1e-8` used for finite differences
- `nev = 5` number of eigenvalues to compute to estimate the spectral projector
- `verbose` bool to print information
- `autodiff = true` only for Bogdanov-Takens point. Whether to use ForwardDiff for the many differentiations that are required to compute the normal form.
- `detailed = true` only for Bogdanov-Takens point. Whether to compute only a simplified normal form.
"""
function bogdanovTakensNormalForm(F, dF, d2F, d3F,
		br::AbstractBranchResult, ind_bif::Int;
		δ = 1e-8,
		nev = length(eigenvalsfrombif(br, ind_bif)),
		Jᵗ = nothing,
		verbose = false,
		ζs = nothing,
		lens = br.lens,
		Teigvec = getvectortype(br),
		scaleζ = norm,
		detailed = true,
		autodiff = true)
	@assert getvectortype(br) <: BorderedArray
	@assert br.specialpoint[ind_bif].type == :bt "The provided index does not refer to a Bogdanov-Takens Point"

	# functional
	prob = br.functional
	@assert prob isa AbstractProblemMinimallyAugmented

	# bordered linear solver
	bls = prob.linbdsolver

	# kernel dimension:
	N = 2

	# in case nev = 0 (number of requested eigenvalues), we increase nev to avoid bug
	nev = max(2N, nev)
	verbose && println("#"^53*"\n--> Bogdanov-Takens Normal form computation")

	# Newton parameters
	optionsN = br.contparams.newtonOptions

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
	parbif = set(br.params, lens, p)
	parbif = set(parbif, prob.lens, get(bifpt.printsol, prob.lens))

	# jacobian at bifurcation point
	@assert bifpt.x isa BorderedArray "Need to fill THIS !!!!!!!!!!!"
	x0 = convert(Teigvec.parameters[1], bifpt.x.u)
	Ty = eltype(Teigvec)
	L = dF(x0, parbif)

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
			if norm(_λ[1:N] .- 0, Inf) > br.contparams.precisionStability
				@warn "We did not find the correct eigenvalues (see 1st col). We found the eigenvalues displayed in the second column:\n $(display(( _λ[1:N]))).\n Difference between the eigenvalues:"
				display(_λ[1:N] .- 0)
			end
			ζs = [copy(geteigenvector(eigsolver, _ev, ii)) for ii in Ivp[1:N]]
		else
			ζs = [copy(geteigenvector(eigsolver, br.eig[bifpt.idx].eigenvec, ii)) for ii in indev-N+1:indev]
		end
	end
	###########################
	# Construction of the basis (ζ0, ζ1), (ζstar0, ζstar1). We follow the procedure described in Al-Hdaibat et al. 2016 on page 972.

	# Al-Hdaibat, B., W. Govaerts, Yu. A. Kuznetsov, and H. G. E. Meijer. “Initialization of Homoclinic Solutions near Bogdanov--Takens Points: Lindstedt--Poincaré Compared with Regular Perturbation Method.” SIAM Journal on Applied Dynamical Systems 15, no. 2 (January 2016): 952–80. https://doi.org/10.1137/15M1017491.
	###########################
	vext = real.(ζs[1])
	_λstar, _evstar, _ = eigsolver(transpose(L), nev)
	Ivp = sortperm(_λstar, by = abs)
	# in case the prob is HopfMA, we real it
	zerov = real.(prob.zero)
	wext = real.(geteigenvector(eigsolver, _evstar, Ivp[1]))
	q0, = bls(L, wext, vext, zero(Ty), zerov, one(Ty))
	p1, = bls(transpose(L), vext, wext, zero(Ty), zerov, one(Ty))
	q1, = bls(L, p1, q0, zero(Ty), q0, zero(Ty))
	p0, = bls(transpose(L), q0, p1, zero(Ty), p1, zero(Ty))
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
		x0, parbif, (prob.lens, lens),
		(q0, q1), (p0, p1),
		(a = zero(Ty), b = zero(Ty) ),
		(K2 = zero(Ty),),
		:none
	)
	return bogdanovTakensNormalForm(prob, L, d2F, d3F, pt; δ = δ, verbose = verbose, detailed = detailed, autodiff = autodiff)
end
