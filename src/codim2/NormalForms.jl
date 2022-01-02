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

