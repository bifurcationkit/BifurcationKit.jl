function getAdjointBasis(Lstar, λs, eigsolver; nev = 3, verbose = false)
	# we compute the eigen-elements of the adjoint of L
	λstar, evstar = eigsolver(Lstar, nev)
	verbose && Base.display(λstar)
	# vectors to hold eigen-elements for the adjoint of L
	λstars = Vector{ComplexF64}()
	ζstars = Vector{typeof(geteigenvector(eigsolver, evstar, 1))}()

	for (idvp, λ) in enumerate(λs)
		I = argmin(abs.(λstar .- λ))
		abs(real(λstar[I])) > 1e-2 && @warn "Did not converge to the requested eigenvalues. We found $(real(λstar[I])) !≈ 0. This might lead to unprecise normal form computation. You can perhaps increase the argument `nev`."
		verbose && println("--> VP[$idvp] paired with VPstar[$I]")
		ζstar = geteigenvector(eigsolver, evstar, I)
		push!(ζstars, copy(ζstar))
		push!(λstars, λstar[I])
		# we change λstar so that it is not used twice
		λstar[I] = 1e9
	end

	return ζstars, λstars
end

"""
	getAdjointBasis(Lstar, λ::Number, options::NewtonPar; nev = 3)

Return a left eigenvector for an eigenvalue closest to λ. `nev` indicates how many eigenvalues must be computed by the eigensolver.
"""
function getAdjointBasis(Lstar, λ::Number, eigsolver; nev = 3, verbose = false)
	λstar, evstar = eigsolver(Lstar, nev)
	I = argmin(abs.(λstar .- λ))
	verbose && (println("--> left eigenvalues = ");Base.display(λstar))
	verbose && println("--> right eigenvalue = ", λ, ", left eigenvalue = ", λstar[I])
	abs(real(λstar[I])) > 1e-2 && @warn "The bifurcating eigenvalue is not that close to Re=0. We found $(real(λstar[I])) !≈ 0.  You can perhaps increase the argument `nev`."
	ζstar = geteigenvector(eigsolver ,evstar, I)
	return copy(ζstar), λstar[I]
end

# the following structs are a machinery to extend multilinear mapping from Real valued to Complex valued Arrays
# this is done so as to use AD (ForwardDiff.jl,...) to provide the differentials which only works on reals (usually).

# struct for bilinear map
struct BilinearMap{Tm}
	bl::Tm
end

function (R2::BilinearMap)(dx1, dx2)
	dx1r = real.(dx1); dx2r = real.(dx2)
	dx1i = imag.(dx1); dx2i = imag.(dx2)
	return R2(dx1r, dx2r) .- R2(dx1i, dx2i) .+ im .* (R2(dx1r, dx2i) .+ R2(dx1i, dx2r))
end

(b::BilinearMap)(dx1::T, dx2::T) where {T <: AbstractArray{<: Real}} = b.bl(dx1, dx2)

# struct for trilinear map
struct TrilinearMap{Tm}
	tl::Tm
end

function (R3::TrilinearMap)(dx1, dx2, dx3)
	dx1r = real.(dx1); dx2r = real.(dx2); dx3r = real.(dx3)
	dx1i = imag.(dx1); dx2i = imag.(dx2); dx3i = imag.(dx3)
	outr =  R3(dx1r, dx2r, dx3r) .- R3(dx1r, dx2i, dx3i) .-
			R3(dx1i, dx2r, dx3i) .- R3(dx1i, dx2i, dx3r)
	outi =  R3(dx1r, dx2r, dx3i) .+ R3(dx1r, dx2i, dx3r) .+
			R3(dx1i, dx2r, dx3r) .- R3(dx1i, dx2i, dx3i)
	return Complex.(outr, outi)
end

(b::TrilinearMap)(dx1::T, dx2::T, dx3::T) where {T <: AbstractArray{<: Real}} = b.tl(dx1, dx2, dx3)

"""
$(SIGNATURES)

Compute a normal form based on Golubitsky, Martin, David G Schaeffer, and Ian Stewart. Singularities and Groups in Bifurcation Theory. New York: Springer-Verlag, 1985, VI.1.d page 295.
"""
function computeNormalForm1d(F, dF, d2F, d3F, br::ContResult, ind_bif::Int; δ = 1e-8, nev = 5, Jt = nothing, verbose = false, lens = br.param_lens, issymmetric = false, Teigvec = vectortype(br), tolFold = 1e-3, scaleζ = norm)
	bifpt = br.bifpoint[ind_bif]
	@assert bifpt.type == :bp "The provided index does not refer to a Branch Point with 1d kernel. The type of the bifurcation is $(bifpt.type). The bifurcation point is $bifpt."
	@assert abs(bifpt.δ[1]) == 1 "We only provide normal form computation for simple bifurcation points e.g when the kernel of the jacobian is 1d. Here, the dimension of the kernel is $(abs(bifpt.δ[1]))."

	verbose && println("#"^53*"\n--> Normal form Computation for 1d kernel")
	verbose && println("--> analyse bifurcation at p = ", bifpt.param)

	# Newton parameters
	options = br.contparams.newtonOptions

	# bifurcation point
	# we need this conversion for when running on GPU and loading the branch from the disk
	x0 = convert(Teigvec, bifpt.x)
	p = bifpt.param

	# parameter for vector field
	parbif = set(br.params, lens, p)

	# jacobian at bifurcation point
	L = dF(x0, parbif)

	# linear solver
	ls = options.linsolver

	# "zero" eigenvalue at bifurcation point, it must be real
	λ = real(br.eig[bifpt.idx].eigenvals[bifpt.ind_ev])
	verbose && println("--> smallest eigenvalue at bifurcation = ", λ)

	# corresponding eigenvector, it must be real
	if haseigenvector(br) == false
		# we recompute the eigen-elements if there were not saved during the computation of the branch
		@info "Eigen-elements no saved in the branch. Recomputing them..."
		_λ, _ev, _ = options.eigsolver(L, bifpt.ind_ev + 2)
		@assert _λ[bifpt.ind_ev] ≈ λ "We did not find the correct eigenvalue $λ. We found $(_λ)"
		ζ = real.(geteigenvector(options.eigsolver, _ev, bifpt.ind_ev))
	else
		ζ = real.(geteigenvector(options.eigsolver, br.eig[bifpt.idx].eigenvec, bifpt.ind_ev))
	end
	ζ ./= scaleζ(ζ)

	# extract eigen-elements for adjoint(L), needed to build spectral projector
	if issymmetric
		λstar = br.eig[bifpt.idx].eigenvals[bifpt.ind_ev]
		ζstar = copy(ζ)
	else
		_Jt = isnothing(Jt) ? adjoint(L) : Jt(x0, parbif)
		ζstar, λstar = getAdjointBasis(_Jt, conj(λ), options.eigsolver; nev = nev, verbose = verbose)
	end

	ζstar = real.(ζstar); λstar = real.(λstar)

	@assert abs(dot(ζ, ζstar)) > 1e-10 "We got ζ⋅ζstar = $((dot(ζ, ζstar))). This dot product should not be zero. Perhaps, you can increase nev which is currently $nev."
	ζstar ./= dot(ζ, ζstar)

	# differentials and projector on Range(L), there are real valued
	R2 = ((dx1, dx2)      -> d2F(x0, parbif, dx1, dx2))
	R3 = ((dx1, dx2, dx3) -> d3F(x0, parbif, dx1, dx2, dx3))
	E = x -> x .- dot(x, ζstar) .* ζ

	# we compute the reduced equation: a⋅(p - pbif) + x⋅(b1⋅(p - pbif) + b2⋅x/2 + b3⋅x^2/6)
	# coefficient of p
	R01 = (F(x0, set(parbif, lens, p + δ)) .- F(x0, set(parbif, lens, p - δ))) ./ (2δ)
	a = dot(R01, ζstar)
	verbose && println("--> a = ", a)

	# coefficient of x*p
	R11 = (apply(dF(x0, set(parbif, lens, p + δ)), ζ) - apply(dF(x0, set(parbif, lens, p - δ)), ζ)) ./ (2δ)
	Ψ01, _ = ls(L, E(R01))

	b1 = dot(R11 .- R2(ζ, Ψ01), ζstar)
	verbose && println("--> b1 = ", b1)

	# coefficient of x^2
	b2v = R2(ζ, ζ)
	b2 = dot(b2v, ζstar)
	verbose && println("--> b2/2 = ", b2/2)

	# coefficient of x^3, recall b2v = R2(ζ, ζ)
	wst, _ = ls(L, E(b2v))
	b3v = R3(ζ, ζ, ζ) .- 3 .* R2(ζ, wst)
	b3 = dot(b3v, ζstar)
	verbose && println("--> b3/6 = ", b3/6)

	bp = (x0, p, parbif, lens, ζ, ζstar, (a=a, b1=b1, b2=b2, b3=b3), :NA)
	if abs(a) < tolFold
		return 100abs(b2/2) < abs(b3/6) ? Pitchfork(bp[1:end-1]...) : Transcritical(bp...)
	else
		return Fold(bp...)
	end
	return
end

computeNormalForm1d(F, dF, d2F, d3F, br::Branch, ind_bif::Int; kwargs...) = computeNormalForm1d(F, dF, d2F, d3F, getContResult(br), ind_bif; kwargs...)

function predictor(bp::Transcritical, ds::T; verbose = false, ampfactor = T(1)) where T
	nf = bp.nf
	a, b1, b2, b3 = nf
	pnew = bp.p + ds
	# we solve b1 * ds + b2 * amp / 2 = 0
	amp = -2ds * b1 / b2 * ampfactor
	dsfactor = T(1)

	verbose && println("--> Prediction from Normal form, δp = $(pnew - bp.p), amp = $amp")
	return (x = bp.x0 .+ amp .* real.(bp.ζ), p = pnew, dsfactor = dsfactor)
end

function predictor(bp::Pitchfork, ds::T; verbose = false, ampfactor = T(1)) where T
	nf = bp.nf
	a, b1, b2, b3 = nf

	# we need to find the type, supercritical or subcritical
	dsfactor = b1 * b3 < 0 ? T(1) : T(-1)
	if 1==1
		# we solve b1 * ds + b3 * amp^2 / 6 = 0
		amp = ampfactor * sqrt(-6abs(ds) * dsfactor * b1 / b3)
		pnew = bp.p + abs(ds) * dsfactor
	# else
	# 	# we solve b1 * ds + b3 * amp^2 / 6 = 0
	# 	amp = ampfactor * abs(ds)
	# 	pnew = bp.p + dsfactor * ds^2 * abs(b3/b1/6)
	end
	verbose && println("--> Prediction from Normal form, δp = $(pnew - bp.p), amp = $amp")
	return (x = bp.x0 .+ amp .* real.(bp.ζ), p = pnew, dsfactor = dsfactor)
end

function predictor(bp::Fold, ds::T; verbose = false, ampfactor = T(1)) where T
	@error "It seems the point is a Saddle-Node bifurcation. The normal form is $(bp.nf)."
	return nothing
end
####################################################################################################
# type for bifurcation point Nd kernel for the jacobian
function factor3d(i,j,k)
	if i == j == k
		return 1/6
	else
		_power = length(unique((i,j,k)))
		if _power == 1
			factor = 1/6 /2
		elseif _power == 2
			factor = 1/2 / 3
		else
			factor = 1.0
		end
		return factor
	end
end

function (bp::NdBranchPoint)(::Val{:reducedForm}, x, p)
	# formula from https://fr.qwe.wiki/wiki/Taylor's_theorem
	# dimension of the kernel
	N = length(bp.ζ)
	# for the output
	out = zero(x)
	# normal form
	nf = bp.nf
	# coefficient p
	out .= p .* nf.a
	# factor to account for factorials
	factor = 1.0

	@inbounds for ii in 1:N
		factor = 1.
		out[ii] = 0
		# coefficient x*p
		for jj in 1:N
			# coefficient x*p
			out[ii] += p * nf.b1[ii , jj] * x[jj]

			for kk in 1:N
				# coefficients of x^2
				factor = jj == kk ? 1/2 : 1
				out[ii] += nf.b2[ii, jj, kk] * x[jj] * x[kk] * factor / 2

				for ll in 1:N
					# coefficients of x^3
					factor = factor3d(ii, jj, kk)
					out[ii] += nf.b3[ii, jj, kk, ll] * x[jj] * x[kk]  * x[ll] * factor
				end
			end
		end
	end
	return out
end

function (bp::NdBranchPoint)(x, δp::Real)
	out = bp.x0 .+ x[1] .* bp.ζ[1]
	for ii in 2:length(x)
		out .+= x[ii] .* bp.ζ[ii]
	end
	return out
end

"""
	 nf(bp::NdBranchPoint; tol = 1e-6, digits = 4)

Print the normal form `bp` with a nice string.
"""
function nf(bp::NdBranchPoint; tol = 1e-6, digits = 4)
	superDigits = [c for c in "⁰ ²³⁴⁵⁶⁷⁸⁹"]

	nf = bp.nf
	N = length(nf.a)
	out = ["" for _ in 1:N]

	for ii = 1:N
		if abs(nf.a[ii]) > tol
			out[ii] *= "$(round(nf.a[ii],digits=digits)) ⋅ p"
		end
		for jj in 1:N
			coeff = round(nf.b1[ii,jj],digits=digits)
			if abs(coeff) > tol
				out[ii] *= " + $coeff * x$jj ⋅ p"
			end
		end

		for jj in 1:N
			for kk in jj:N
				coeff = round(nf.b2[ii,jj,kk] / 2,digits=digits)
				if abs(coeff) > tol
					if jj == kk
						out[ii] *= " + $coeff ⋅ x$(jj)²"
					else
						out[ii] *= " + $(round(2coeff,digits=digits)) ⋅ x$jj ⋅ x$kk"
					end
				end

				for ll in kk:N
					coeff = round(nf.b3[ii,jj,kk,ll] / 6,digits=digits)
					_pow = zeros(Int64,N)
					_pow[jj] += 1;_pow[kk] += 1;_pow[ll] += 1;

					if abs(coeff) > tol
						if jj == kk == ll
							out[ii] *= " + $coeff"
						else
							out[ii] *= " + $(round(3coeff,digits=digits))"
						end
						for mm in 1:N
							if _pow[mm] > 1
								out[ii] *= " ⋅ x$mm" * (superDigits[_pow[mm]+1])
							elseif _pow[mm] == 1
								out[ii] *= " ⋅ x$mm"
							end
						end
					end
				end
			end
		end
	end
	return out
end

"""
$(SIGNATURES)

Bi-orthogonalise the arguments.
"""
function biorthogonalise(ζs, ζstars, verbose)
	# change only the ζstars to have bi-orthogonal left/right eigenvectors
	# we could use projector P=A(A^{T}A)^{-1}A^{T}
	# we use Gram-Schmidt algorithm instead

	G = [ dot(ζ, ζstar) for ζ in ζs, ζstar in ζstars]
	@assert abs(det(G)) >1e-14 "The Gram matrix is not invertible! det(G) = $(det(G))"

	# save those in case the first algo fails
	_ζs = copy.(ζs)
	_ζstars = copy.(ζstars)

	# first algo
	tmp = copy(ζstars[1])
	for ii in eachindex(ζstars)
		tmp .= ζstars[ii]
		for jj in eachindex(ζs)
			if ii != jj
				tmp .-= dot(tmp, ζs[jj]) .* ζs[jj] ./ dot(ζs[jj], ζs[jj])
			end
		end
		ζstars[ii] .= tmp ./ dot(tmp, ζs[ii])
	end

	G = [ dot(ζ, ζstar) for ζ in ζs, ζstar in ζstars]

	# we switch to another algo if the above fails
	if norm(G - LinearAlgebra.I, Inf) >= 1e-5
		G = [ dot(ζ, ζstar) for ζ in _ζs, ζstar in _ζstars]
		@warn "Gram matrix not equal to idendity. Switching to LU algorithm."
		display(G)
		@show det(G)
		_F = lu(G)
		display(inv(_F.L) * G * inv(_F.U)')
		ζs = inv(_F.L) * _ζs
		ζstars = inv(_F.U)' * _ζstars
	end

	# test the bi-orthogonalization
	G = [ dot(ζ, ζstar) for ζ in ζs, ζstar in ζstars]
	verbose && (printstyled(color=:green, "--> Gram matrix = \n");Base.display(G))
	@assert norm(G - LinearAlgebra.I, Inf) < 1e-5 "Failure in bi-orthogonalisation of the right / left eigenvectors. The left eigenvectors do not form a basis. You may want to increase `nev`."
	return ζs, ζstars
end

"""
$(SIGNATURES)

Compute the normal form of the bifurcation point located at `br.bifpoint[ind_bif]`.

# Arguments
- `F, dF, d2F, d3F` vector field `(x, p) -> F(x, p)` and its derivatives w.r.t. `x`.
- `br` result from a call to [`continuation`](@ref)
- `ind_bif` index of the bifurcation point in `br.bifpoint`

# Optional arguments
- `δ` used to compute ∂pF with finite differences
- `nev` number of eigenvalues used to compute the spectral projection. This number has to be adjusted when used with iterative methods.
- `Jt = (x,p) -> ...` jacobian adjoint, it should be implemented in an efficient manner. For matrix-free methods, `transpose` is not readily available and the user must provide a dedicated method. In the case of sparse based jacobian, `Jt` should not be passed as it is computed internally more efficiently, i.e. it avoid recomputing the jacobian as it would be if you pass `Jt = (x, p) -> transpose(dF(x, p))`.
- `verbose` whether to display information
- `ζs` list of vectors spanning the kernel of `dF` at the bifurcation point. Useful to enforce the basis for the normal form.
- `lens::Lens` specify which parameter to take the partial derivative ∂pF
- `issymmetric` whether the Jacobian is Symmetric, avoid computing the left eigenvectors.
- `scaleζ` function to normalise the kernel basis. Indeed, when used with large vectors and `norm`, it results in ζs and the normal form coeffocient being super small.

Based on Golubitsky, Martin, David G Schaeffer, and Ian Stewart. Singularities and Groups in Bifurcation Theory. New York: Springer-Verlag, 1985, VI.1.d page 295.
"""
function computeNormalForm(F, dF, d2F, d3F, br::ContResult, id_bif::Int ; δ = 1e-8, nev = br.n_unstable[br.bifpoint[id_bif].idx], Jt = nothing, verbose = false, ζs = nothing, lens = br.param_lens, issymmetric = false, Teigvec = vectortype(br), scaleζ = norm)
	bifpt = br.bifpoint[id_bif]
	if abs(bifpt.δ[2]) > 0 # we try a Hopf point
		return hopfNormalForm(F, dF, d2F, d3F, br, id_bif; δ = δ, nev = nev, Jt = Jt, verbose = verbose, lens = lens, Teigvec = Teigvec, scaleζ = scaleζ)
	elseif abs(bifpt.δ[1]) == 1 # simple branch point
		return computeNormalForm1d(F, dF, d2F, d3F, br, id_bif ; δ = δ, nev = nev, Jt = Jt, verbose = verbose, lens = lens, issymmetric = issymmetric, Teigvec = Teigvec, scaleζ = scaleζ)
	end
	# kernel dimension:
	N = abs(bifpt.δ[1])
	# in case nev = 0 (number of unstable eigenvalues), we increase nev to avoid bug
	nev = max(2N, nev)
	verbose && println("#"^53*"\n--> Normal form Computation for a $N-d kernel")
	verbose && println("--> analyse bifurcation at p = ", bifpt.param)

	# Newton parameters
	options = br.contparams.newtonOptions

	# bifurcation point
	x0 = convert(Teigvec, bifpt.x)
	p = bifpt.param

	# parameter for vector field
	parbif = set(br.params, br.param_lens, p)

	# jacobian at bifurcation point
	L = dF(x0, parbif)

	# we only invert L repeatdly, so we try to factorize it
	Linv = L isa AbstractMatrix ? factorize(L) : L

	# linear solver
	ls = options.linsolver


	# "zero" eigenvalues at bifurcation point
	rightEv = br.eig[bifpt.idx].eigenvals
	indev = br.bifpoint[id_bif].ind_ev
	λs = rightEv[indev-N+1:indev]
	verbose && println("--> smallest eigenvalues at bifurcation = ", real.(λs))

	# and corresponding eigenvectors
	if isnothing(ζs) # do we have a basis for the kernel?
		if haseigenvector(br) == false # are the eigenvector saved in the branch?
			@info "No eigenvector recorded, computing them on the fly"
			# we recompute the eigen-elements if there were not saved during the computation of the branch
			_λ, _ev, _ = options.eigsolver(L, length(rightEv))
			verbose && (println("--> λs = ");Base.display(rightEv))
			verbose && (println("--> λs (recomputed) = ");Base.display(_λ))
			@assert _λ[1:length(rightEv)] ≈ rightEv "We did not find the correct eigenvalues $(rightEv). We found $(_λ)"
			ζs = [copy(geteigenvector(options.eigsolver, _ev, ii)) for ii in indev-N+1:indev]
		else
			ζs = [copy(geteigenvector(options.eigsolver, br.eig[bifpt.idx].eigenvec, ii)) for ii in indev-N+1:indev]
		end
	end

	# extract eigen-elements for transpose(L), needed to build spectral projector
	# it is OK to re-scale at this stage as the basis ζs is not touched anymore, we
	# only adjust ζstars
	for ζ in ζs;ζ ./= scaleζ(ζ);end
	if issymmetric
		λstars = copy(λs)
		ζstars = copy.(ζs)
	else
		_Jt = isnothing(Jt) ? transpose(L) : Jt(x0, parbif)
		ζstars, λstars = getAdjointBasis(_Jt, conj.(λs), options.eigsolver; nev = nev, verbose = verbose)
	end
	ζstars = real.(ζstars); λstars = real.(λstars)
	ζs = real.(ζs); λs = real.(λs)
	verbose && println("--> VP     = ", λs, "\n--> VPstar = ", λstars)

	ζs, ζstars = biorthogonalise(ζs, ζstars, verbose)

	# differentials should work as we are looking at reals
	R2 = ((dx1, dx2)      -> d2F(x0, parbif, dx1, dx2))
	R3 = ((dx1, dx2, dx3) -> d3F(x0, parbif, dx1, dx2, dx3))

	# projector on Range(L)
	function E(x)
		out = copy(x)
		for ii in 1:N
			out .= out .- dot(x, ζstars[ii]) .* ζs[ii]
		end
		out
	end

	# coefficients of p
	dgidp = Vector{Float64}(undef, N)
	R01 = (F(x0, set(parbif, lens, p + δ)) .- F(x0, set(parbif, lens, p - δ))) ./ (2δ)
	for ii in 1:N
		dgidp[ii] = dot(R01, ζstars[ii])
	end
	verbose && printstyled(color=:green,"--> a (∂/∂p) = ", dgidp, "\n")

	# coefficients of x*p
	d2gidxjdpk = zeros(Float64, N, N)
	for ii in 1:N, jj in 1:N
		R11 = (apply(dF(x0, set(parbif, lens, p + δ)), ζs[jj]) .- apply(dF(x0, set(parbif, lens, p - δ)), ζs[jj])) ./ (2δ)
		Ψ01, flag = ls(Linv, E(R01))
		~flag && @warn "linear solver did not converge"
		d2gidxjdpk[ii,jj] = dot(R11 .- R2(ζs[jj], Ψ01), ζstars[ii])
	end
	verbose && (printstyled(color=:green, "\n--> b1 (∂²/∂x∂p)  = \n"); Base.display( d2gidxjdpk ))

	# coefficients of x^2
	d2gidxjdxk = zeros(Float64, N, N, N)
	for ii in 1:N, jj in 1:N, kk in 1:N
		b2v = R2(ζs[jj], ζs[kk])
		d2gidxjdxk[ii,jj,kk] = dot(b2v, ζstars[ii])
	end

	if verbose
		printstyled(color=:green, "\n--> b2 (∂²/∂x²) = \n")
		for ii in 1:N
			printstyled(color=:blue, "--> component $ii\n")
			Base.display( d2gidxjdxk[ii,:,:] ./ 2)
		end
	end

	# coefficient of x^3
	d3gidxjdxkdxl = zeros(Float64, N, N, N, N)
	for jj in 1:N, kk in 1:N, ll in 1:N
		b3v = R3(ζs[jj], ζs[kk], ζs[ll])
		# d3gidxjdxkdxl[ii,jj,kk,ll] = dot(b3v, ζstars[ii])

		wst, flag = ls(Linv, E(R2(ζs[ll], ζs[kk])))
		~flag && @warn "linear solver did not converge"
		b3v .-= R2(ζs[jj], wst)
		# d3gidxjdxkdxl[ii,jj,kk,ll] -= dot(R2(ζs[jj], wst), ζstars[ii])

		wst, flag = ls(Linv, E(R2(ζs[ll], ζs[jj])))
		~flag && @warn "linear solver did not converge"
		b3v .-= R2(ζs[kk], wst)
		# d3gidxjdxkdxl[ii,jj,kk,ll] -= dot(R2(ζs[kk], wst), ζstars[ii])

		wst, flag = ls(Linv, E(R2(ζs[kk], ζs[jj])))
		~flag && @warn "linear solver did not converge"
		b3v .-= R2(ζs[ll], wst)
		# d3gidxjdxkdxl[ii,jj,kk,ll] -= dot(R2(ζs[ll], wst), ζstars[ii])
		for ii in 1:N
			d3gidxjdxkdxl[ii,jj,kk,ll] = dot(b3v, ζstars[ii])
		end
	end
	if verbose
		printstyled(color=:green, "\n--> b3 (∂³/∂x³) = \n")
		for ii in 1:N
			printstyled(color=:blue, "--> component $ii\n")
			Base.display( d3gidxjdxkdxl[ii,:,:, :] ./ 6 )
		end
	end

	return NdBranchPoint(x0, p, parbif, lens, ζs, ζstars, (a=dgidp, b1=d2gidxjdpk, b2=d2gidxjdxk, b3=d3gidxjdxkdxl), Symbol("$N-d"))

end

computeNormalForm(F, dF, d2F, d3F, br::Branch, id_bif::Int; kwargs...) = computeNormalForm(F, dF, d2F, d3F, getContResult(br), id_bif; kwargs...)

function predictor(bp::NdBranchPoint, δp::T; verbose = false, ampfactor = T(1), nbfailures = 30, maxiter = 100, perturb = identity, jac = nothing, normN = x->norm(x,Inf), optn = NewtonPar(maxIter = maxiter, verbose = verbose)) where T
	# dimension of the kernel
	n = length(bp.ζ)

	# find zeros for the normal on each side of the bifurcation point
	function getRootsNf(_ds)
		deflationOp = DeflationOperator(2.0, dot, 1.0, [zeros(n)])
		failures = 0
		# we allow for 10 failures of nonlinear deflation
		outdef1 = rand(n)
		while failures < nbfailures
			if isnothing(jac)
				outdef1, hist, flag, _ = newton((x, p) -> perturb(bp(Val(:reducedForm), x, p)), outdef1 .+ 0.1rand(n), _ds, optn, deflationOp; normN = normN)
			else
				outdef1, hist, flag, _ = newton((x, p) -> perturb(bp(Val(:reducedForm), x, p)),jac, outdef1 .+ 0.1rand(n), _ds, optn, deflationOp; normN = normN)
			end
			flag && push!(deflationOp, outdef1)
			~flag && (failures += 1)
		end
		return deflationOp.roots
	end
	rootsNFm =  getRootsNf(-abs(δp))
	rootsNFp =  getRootsNf(abs(δp))
	println("\n--> BS from Non simple branch point")
	printstyled(color=:green, "--> we find $(length(rootsNFm)) (resp. $(length(rootsNFp))) roots on the left (resp. right) of the bifurcation point counting the trivial solution (Reduced equation).\n")
	return (before = rootsNFm, after = rootsNFp)
end
####################################################################################################

function hopfNormalForm(F, dF, d2F, d3F, pt::HopfBifPoint, ls; δ = 1e-8, verbose = false)
	x0 = pt.x0
	p = pt.p
	lens = pt.lens
	parbif = set(pt.params, lens, p)
	ω = pt.ω
	ζ = pt.ζ
	cζ = conj.(pt.ζ)
	ζstar = pt.ζstar

	# jacobian at the bifurcation point
	L = dF(x0, parbif)

	# we use BilinearMap to be able to call on complex valued arrays
	R2 = BilinearMap( (dx1, dx2)      -> d2F(x0, parbif, dx1, dx2) ./2)
	R3 = TrilinearMap((dx1, dx2, dx3) -> d3F(x0, parbif, dx1, dx2, dx3) ./6 )

	# −LΨ001 = R01
	R01 = (F(x0, set(parbif, lens, p + δ)) .- F(x0, set(parbif, lens, p - δ))) ./ (2δ)
	Ψ001, _ = ls(L, -R01)

	# (2iω−L)Ψ200 = R20(ζ,ζ)
	R20 = R2(ζ, ζ)
	Ψ200, _ = ls(L, R20; a₀ = Complex(0, 2ω), a₁ = -1)
	# @assert Ψ200 ≈ (Complex(0, 2ω)*I - L) \ R20

	# −LΨ110 = 2R20(ζ,cζ).
	R20 = 2 .* R2(ζ, cζ)
	Ψ110, _ = ls(L, -R20)

	# a = ⟨R11(ζ) + 2R20(ζ,Ψ001),ζ∗⟩
	av = (apply(dF(x0, set(parbif, lens, p + δ)), ζ) .- apply(dF(x0, set(parbif, lens, p - δ)), ζ)) ./ (2δ)
	av .+= 2 .* R2(ζ, Ψ001)
	a = dot(av, ζstar)

	# b = ⟨2R20(ζ,Ψ110) + 2R20(cζ,Ψ200) + 3R30(ζ,ζ,cζ), ζ∗⟩)
	bv = 2 .* R2(ζ, Ψ110) .+ 2 .* R2(cζ, Ψ200) .+ 3 .* R3(ζ, ζ, cζ)
	b = dot(bv, ζstar)

	# return coefficients of the normal form
	verbose && println((a = a, b = b))
	pt.nf = (a = a, b = b)
	pt.type = real(a) * real(b) < 0 ? :SuperCritical : :SubCritical
	printstyled(color = :red,"--> Hopf bifurcation point is: ", pt.type, "\n")
	return pt
end

"""
$(SIGNATURES)

Compute the Hopf normal form.

# Arguments
- `F, dF, d2F, d3F`: function `(x,p) -> F(x,p)` and its differencials `(x,p,dx) -> d1F(x,p,dx)`, `(x,p,dx1,dx2) -> d2F(x,p,dx1,dx2)`...
- `br` branch result from a call to [`continuation`](@ref)
- `ind_hopf` index of the bifurcation point in `br`
- `options` options for the Newton solver

# Optional argument
- `Jt` is the jacobian adjoint, used for computation of the eigen-elements of the jacobian adjoint, needed to compute the spectral projector
- `δ = 1e-8` used for finite differences
- `nev = 5` number of eigenvalues to compute to estimate the spectral projector
- `verbose` bool to print information
"""
function hopfNormalForm(F, dF, d2F, d3F, br::ContResult, ind_hopf::Int; Jt = nothing, δ = 1e-8, nev = 5, verbose = false, lens = br.param_lens, Teigvec = vectortype(br), scaleζ = norm)
	@assert br.bifpoint[ind_hopf].type == :hopf "The provided index does not refer to a Hopf Point"
	println("#"^53*"\n--> Hopf Normal form computation")

	# Newton parameters
	options = br.contparams.newtonOptions

	# bifurcation point
	bifpt = br.bifpoint[ind_hopf]
	eigRes = br.eig

	# eigenvalue
	λ = eigRes[bifpt.idx].eigenvals[bifpt.ind_ev]
	ω = imag(λ)

	# parameter for vector field
	p = bifpt.param
	parbif = set(br.params, lens, p)

	# jacobian at bifurcation point
	L = dF(convert(Teigvec, bifpt.x), parbif)

	# right eigenvector
	if haseigenvector(br) == false
		# we recompute the eigen-elements if there were not saved during the computation of the branch
		_λ, _ev, _ = options.eigsolver(L, bifpt.ind_ev + 2)
		@assert _λ[bifpt.ind_ev] ≈ λ "We did not find the correct eigenvalue $λ. We found $(_λ)"
		ζ = geteigenvector(options.eigsolver, _ev, bifpt.ind_ev)
	else
		ζ = copy(geteigenvector(options.eigsolver ,br.eig[bifpt.idx].eigenvec, bifpt.ind_ev))
	end
	ζ ./= scaleζ(ζ)

	# left eigen-elements
	_Jt = isnothing(Jt) ? adjoint(L) : Jt(x, p)
	ζstar, λstar = getAdjointBasis(_Jt, conj(λ), options.eigsolver; nev = nev, verbose = verbose)

	# check that λstar ≈ conj(λ)
	abs(λ + λstar) > 1e-2 && @warn "We did not find the left eigenvalue for the Hopf point to be very close to the imaginary part, $λ ≈ $(λstar) and $(abs(λ + λstar)) ≈ 0?\n You can perhaps increase the number of computed eigenvalues, the number is nev = $nev"

	# normalise left eigenvector
	ζstar ./= dot(ζ, ζstar)
	@assert dot(ζ, ζstar) ≈ 1

	hopfpt = HopfBifPoint(
		bifpt.x,
		bifpt.param,
		ω,
		parbif,
		lens,
		ζ,
		ζstar,
		(a = 0. + 0im, b = 0. + 0im),
		:SuperCritical
	)
	return hopfNormalForm(F, dF, d2F, d3F, hopfpt, options.linsolver ; δ = δ, verbose = verbose)
end

function predictor(hp::HopfBifPoint, ds::T; verbose = false, ampfactor = T(1) ) where T
	# get the normal form
	nf = hp.nf
	a = nf.a
	b = nf.b

	# we need to find the type, supercritical or subcritical
	dsfactor = real(a) * real(b) < 0 ? T(1) : T(-1)
	pnew = hp.p + abs(ds) * dsfactor

	# we solve a * ds + b * amp^2 = 0
	amp = ampfactor * sqrt(-abs(ds) * dsfactor * real(a) / real(b))

	# o(1) correction to Hopf Frequency
	ω = hp.ω + (imag(a) - imag(b) * real(a) / real(b)) * ds

	return (orbit = t -> hp.x0 .+ 2amp .* real.(hp.ζ .* exp(complex(0, t))),
			amp = 2amp,
			ω = ω,
			p = pnew,
			dsfactor = dsfactor)
end
