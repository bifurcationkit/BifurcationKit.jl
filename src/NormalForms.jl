abstract type BifurcationPoint end
abstract type BranchPoint <: BifurcationPoint end


function getAdjointBasis(Lstar, λs, eigsolver; nev = 3, verbose = false)
	# we compute the eigen-elements of the adjoint of L
	λstar, evstar = eigsolver(Lstar, nev)
	# vectors to hold eigen-elements for the adjoint of L
	λstars = Vector{ComplexF64}()
	ζstars = Vector{typeof(geteigenvector(eigsolver, evstar, 1))}()

	for λ in λs
		I = argmin(abs.(λstar .- λ))
		@assert abs(real(λstar[I])) < 1e-2 "Did not converge to the requested eigenvalue. We found $(λstar[I]) ≈ 0"
		ζstar = geteigenvector(eigsolver, evstar, I)
		push!(ζstars, ζstar)
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
	@assert abs(real(λstar[I])) < 1e-2 "Did not converge to the requested eigenvalue. We found $(λstar[I]) ≈ 0"
	ζstar = geteigenvector(eigsolver ,evstar, I)
	return ζstar, λstar[I]
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
####################################################################################################
# type for bifurcation point 1d kernel for the jacobian

"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
mutable struct SimpleBranchPoint{Tv, T, Tpar, Tlens <: Lens, Tevl, Tevr, Tnf} <: BranchPoint
	"bifurcation point."
	x0::Tv

	"Parameter value at the bifurcation point."
	p::T

	"Parameters used by the vector field."
	params::Tpar

	"Parameter axis used to compute the branch on which this Branch point was detected."
	lens::Tlens

	"Right eigenvector(s)."
	ζ::Tevr

	"Left eigenvector(s)."
	ζstar::Tevl

	"Normal form coefficients."
	nf::Tnf

	"Type of bifurcation point."
	type::Symbol
end

"""
Compute a normal form based on Golubitsky, Martin, David G Schaeffer, and Ian Stewart. Singularities and Groups in Bifurcation Theory. New York: Springer-Verlag, 1985, VI.1.d page 295.

"""
function computeNormalForm1d(F, dF, d2F, d3F, br::ContResult, ind_bif::Int; δ = 1e-8, nev = 5, Jt = nothing, verbose = false, lens = br.param_lens)
	bifpt = br.bifpoint[ind_bif]
	@assert bifpt.type == :bp "The provided index does not refer to a Branch Point"
	@assert abs(bifpt.δ[1]) == 1 "We only provide analysis for simple bifurcation points for which the kernel of the jacobian is 1d. Here, the dimension of the BP is $(abs(bifpt.δ[1]))"

	verbose && println("#"^53*"\n--> Normal form Computation for 1d kernel")
	verbose && println("--> analyse bifurcation at p = ", bifpt.param)

	# Newton parameters
	options = br.contparams.newtonOptions

	# bifurcation point
	x0 = bifpt.x
	p = bifpt.param

	# parameter for vector field
	parbif = set(br.params, lens, p)

	# jacobian at bifurcation point
	L = dF(x0, parbif)

	# linear solver
	ls = options.linsolver

	# "zero" eigenvalue at bifurcation point, it must be real
	λ = real(br.eig[bifpt.idx].eigenvals[bifpt.ind_bif])
	verbose && println("--> smallest eigenvalue at bifurcation = ", λ)


	# corresponding eigenvector, it must be real
	ζ = real.(geteigenvector(options.eigsolver ,br.eig[bifpt.idx].eigenvec, bifpt.ind_bif))
	ζ ./= norm(ζ)

	# extract eigenelements for adjoint(L), needed to build spectral projector
	if isnothing(Jt)
		ζstar, λstar = getAdjointBasis(adjoint(L), conj(λ), options.eigsolver; nev = nev, verbose = verbose)
	else
		ζstar, λstar = getAdjointBasis(Jt(x, p), conj(λ), options.eigsolver; nev = nev, verbose = verbose)
	end

	ζstar = real.(ζstar); λstar = real.(λstar)

	@assert abs(dot(ζ, ζstar)) > 1e-12
	ζstar ./= dot(ζ, ζstar)


	# differentials and projector on Range(L), there are real valued
	R2 = ((dx1, dx2)      -> d2F(x0, parbif, dx1, dx2))
	R3 = ((dx1, dx2, dx3) -> d3F(x0, parbif, dx1, dx2, dx3))
	E = x -> x .- dot(x, ζstar) .* ζ

	# we compute the reduced equation: a⋅(p - pbif) + x⋅(b1⋅(p - pbif) + b2⋅x/2 + b3⋅x^2/6)
	# coefficient of p
	R01 = (F(x0, set(parbif, lens, p + δ)) .- F(x0, parbif)) ./ δ
	a = dot(R01, ζstar)
	verbose && println("--> a = ", a)

	# coefficient of x*p
	R11 = (apply(dF(x0, set(parbif, lens, p + δ)), ζ) - apply(L, ζ)) ./ δ
	Ψ01, _ = ls(L, E(R01))

	b1 = dot(R11 .- R2(ζ, Ψ01), ζstar)
	verbose && println("--> b1 = ", b1)

	# coefficient of x^2
	b2v = R2(ζ, ζ)
	b2 = dot(b2v, ζstar)
	verbose && println("--> b2/2 = ", b2/2)

	# coefficient of x^3
	wst, _ = ls(L, E(R2(ζ, ζ)))
	b3v = R3(ζ, ζ, ζ) .- 3 .* R2(ζ, wst)
	b3 = dot(b3v, ζstar)
	verbose && println("--> b3/6 = ", b3/6)

	if abs(a) < 1e-5
		type = abs(b2) < 1e-9 ? :Pitchfork : :Transcritical
	else
		type = :ProbablySaddleNode
	end
	return SimpleBranchPoint(x0, p, parbif, lens, ζ, ζstar, (a=a, b1=b1, b2=b2, b3=b3), type)
end

function predictor(bp::SimpleBranchPoint, ds::T; verbose = false) where T
	@assert bp.type != :ProbablySaddleNode "It seems to be a Saddle-Node bifurcation, not applicable here."
	nf = bp.nf
	if bp.type == :Transcritical
		pnew = bp.p + ds
		# we solve b1 * ds + b2 * amp / 2 = 0
		amp = -2ds * nf.b1 / nf.b2
		dsfactor = T(1)
	else
		# case of the Pitchfork bifurcation
		# we need to find the type, supercritical or subcritical
		dsfactor = nf.b1 * nf.b3 < 0 ? T(1) : T(-1)
		pnew = bp.p + ds * dsfactor
		# we solve b1 * ds + b3 * amp^2 / 6 = 0
		amp = sqrt(-6abs(ds) * dsfactor * nf.b1 / nf.b3)
	end
	verbose && println("--> Prediction from Normal form, δp = $(pnew - bp.p), amp = $amp")
	return (x = bp.x0 .+ amp .* real.(bp.ζ), p = pnew, dsfactor = dsfactor)
end
####################################################################################################
# type for bifurcation point Nd kernel for the jacobian

"""
This is a type which holds information for the bifurcation points of equilibria.

$(TYPEDEF)
$(TYPEDFIELDS)
"""
mutable struct NdBranchPoint{Tv, T, Tpar, Tlens <: Lens, Tevl, Tevr, Tnf} <: BranchPoint
	"bifurcation point"
	x0::Tv

	"Parameter value at the bifurcation point"
	p::T

	"Parameters used by the vector field."
	params::Tpar

	"Parameter axis used to compute the branch on which this Branch point was detected."
	lens::Tlens

	"Right eigenvectors"
	ζ::Tevr

	"Left eigenvectors"
	ζstar::Tevl

	"Normal form coefficients"
	nf::Tnf

	"Type of bifurcation point"
	type::Symbol
end

function (bp::NdBranchPoint)(::Val{:reducedForm}, x, p::Real)
	# dimension of the kernel
	N = length(bp.ζ)
	# for the output
	out = zero(x)
	# normal form
	nf = bp.nf
	# coefficient p
	out = p .* nf.a

	@inbounds for ii in 1:N
		# coefficient x*p
		for jj in 1:N
			# coefficient x*p
			out[ii] += p * nf.b1[ii , jj] * x[jj]

			for kk in 1:N
				# coefficients of x^2
				out[ii] += nf.b2[ii, jj, kk] * x[jj] * x[kk]

				for ll in 1:N
					# coefficients of x^3
					out[ii] += nf.b3[ii, jj, kk, ll] * x[jj] * x[kk]  * x[ll]
				end
			end
		end
	end
	return out
end

function (bp::NdBranchPoint)(x, δp::Real)
	out = bp.x0 .+ δp .* x[1] .* bp.ζ[1]
	for ii in 2:length(x)
		out .+= δp .* x[ii] .* bp.ζ[ii]
	end
	return out
end

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
				out[ii] *= " + ($coeff) * x$jj ⋅ p"
			end

			for kk in jj:N
				coeff = round(nf.b2[ii,jj,kk] / 2,digits=digits)
				if abs(coeff) > tol
					if jj == kk
						out[ii] *= " + ($coeff) ⋅ x$jj²"
					else
						out[ii] *= " + ($(round(2coeff,digits=digits))) ⋅ x$jj ⋅ x$kk"
					end
				end

				for ll in kk:N
					coeff = round(nf.b3[ii,jj,kk,ll] / 6,digits=digits)
					_pow = zeros(Int64,N)
					_pow[jj] += 1;_pow[kk] += 1;_pow[ll] += 1;

					if abs(coeff) > tol
						if jj == kk == ll
							out[ii] *= " + ($coeff)"
						else
							out[ii] *= " + ($(round(3coeff,digits=digits)))"
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
"""
function computeNormalForm(F, dF, d2F, d3F, br::ContResult, ind_bif::Int ; δ = 1e-8, nev = 5, Jt = nothing, verbose = false, ζs = nothing, lens = br.param_lens)
	bifpt = br.bifpoint[ind_bif]
	if abs(bifpt.δ[2]) > 0 # we try a Hopf point
		return hopfNormalForm(F, dF, d2F, d3F, br, ind_bif; δ = δ, nev = nev, Jt = Jt, verbose = verbose, lens = lens)
	elseif abs(bifpt.δ[1]) == 1 # simple branch point
		return computeNormalForm1d(F, dF, d2F, d3F, br, ind_bif ; δ = δ, nev = nev, Jt = Jt, verbose = verbose, lens = lens)
	end
	N = abs(bifpt.δ[1])
	verbose && println("#"^53*"\n--> Normal form Computation for a $N-d kernel")
	verbose && println("--> analyse bifurcation at p = ", bifpt.param)

	# Newton parameters
	options = br.contparams.newtonOptions

	# bifurcation point
	x0 = bifpt.x
	p = bifpt.param

	# parameter for vector field
	parbif = set(br.params, br.param_lens, p)

	# jacobian at bifurcation point
	L = dF(x0, parbif)

	# linear solver
	ls = options.linsolver

	# "zero" eigenvalues at bifurcation point
	rightEv = br.eig[bifpt.idx].eigenvals
	I = sortperm(abs.(real.(rightEv)))
	λs = rightEv[I][1:N]
	verbose && println("--> smallest eigenvalues at bifurcation = ", real.(λs))

	# and corresponding eigenvectors
	if isnothing(ζs)
		ζs = [geteigenvector(options.eigsolver, br.eig[bifpt.idx].eigenvec, ii) for ii in I[1:N]]
	end

	# extract eigenelements for transpose(L), needed to build spectral projector
	if isnothing(Jt)
		ζstars, λstars = getAdjointBasis(transpose(L), conj.(λs), options.eigsolver; nev = nev, verbose = verbose)
	else
		ζstars, λstars = getAdjointBasis(Jt(x, p), conj.(λs), options.eigsolver; nev = nev, verbose = verbose)
	end
	ζstars = real.(ζstars); λstars = real.(λstars)
	ζs = real.(ζs); λs = real.(λs)
	verbose && println("--> VP     = ", λs, "\n--> VPstar = ", λstars)

	# change only the ζstars to have bi-orthogonal left/right eigenvectors
	# we could use projector P=A(A^{T}A)^{-1}A^{T}
	# we use Gram-Schmidt algorithm instead
	tmp = copy(ζstars[1])
	for ii in 1:N
		tmp .= ζstars[ii]
		for jj in 1:N
			if !(ii==jj)
				tmp .-= dot(tmp, ζs[jj]) .* ζs[jj]
			end
		end
		ζstars[ii] .= tmp ./ dot(tmp, ζs[ii])
	end

	# test the bi-orthogonalization
	G = [ dot(ζ, ζstar) for ζ in ζs, ζstar in ζstars]
	verbose && (printstyled(color=:green, "--> Gram matrix = \n");Base.display(G))
	@assert norm(G - LinearAlgebra.I(N), Inf) < 1e-5 "Failure in bi-orthogonalisation of the right /left eigenvectors. The left eigenvectors do not form a basis."

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
	R01 = (F(x0, set(parbif, lens, p + δ)) .- F(x0, parbif)) ./ δ
	for ii in 1:N
		dgidp[ii] = dot(R01, ζstars[ii])
	end
	verbose && printstyled(color=:green,"--> a = ", dgidp,"\n")

	# coefficients of x*p
	d2gidxjdpk = zeros(Float64, N, N)
	for ii in 1:N, jj in 1:N
		R11 = (apply(dF(x0, set(parbif, lens, p + δ)), ζs[jj]) - apply(L, ζs[jj])) ./ δ
		Ψ01, _ = ls(L, E(R01))
		d2gidxjdpk[ii,jj] = dot(R11 .- R2(ζs[jj], Ψ01), ζstars[ii])
	end
	verbose && (printstyled(color=:green, "\n--> b1 = \n");Base.display( d2gidxjdpk ))

	# coefficients of x^2
	d2gidxjdxk = zeros(Float64, N, N, N)
	for ii in 1:N, jj in 1:N, kk in 1:N
		b2v = R2(ζs[jj], ζs[kk])
		d2gidxjdxk[ii,jj,kk] = dot(b2v, ζstars[ii])
	end

	if verbose
		printstyled(color=:green, "\n--> b2 = \n")
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

		wst, _ = ls(L, E(R2(ζs[ll], ζs[kk])))
		b3v .-= R2(ζs[jj], wst)
		# d3gidxjdxkdxl[ii,jj,kk,ll] -= dot(R2(ζs[jj], wst), ζstars[ii])

		wst, _ = ls(L, E(R2(ζs[ll], ζs[jj])))
		b3v .-= R2(ζs[kk], wst)
		# d3gidxjdxkdxl[ii,jj,kk,ll] -= dot(R2(ζs[kk], wst), ζstars[ii])

		wst, _ = ls(L, E(R2(ζs[kk], ζs[jj])))
		b3v .-= R2(ζs[ll], wst)
		# d3gidxjdxkdxl[ii,jj,kk,ll] -= dot(R2(ζs[ll], wst), ζstars[ii])
		for ii in 1:N
			d3gidxjdxkdxl[ii,jj,kk,ll] = dot(b3v, ζstars[ii])
		end
	end
	if verbose
		printstyled(color=:green,"\n--> b3 = \n")
		for ii in 1:N
			printstyled(color=:blue, "--> component $ii\n")
			Base.display( d3gidxjdxkdxl[ii,:,:, :] ./ 6 )
		end
	end

	return NdBranchPoint(x0, p, parbif, lens, ζs, ζstars, (a=dgidp, b1=d2gidxjdpk, b2=d2gidxjdxk, b3=d3gidxjdxkdxl), Symbol("$N-d"))

end

####################################################################################################

"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
mutable struct HopfBifPoint{Tv, T, Tω, Tpar, Tlens <: Lens, Tevr, Tevl, Tnf} <: BifurcationPoint
	"Hopf point"
	x0::Tv

	"Parameter value at the Hopf point"
	p::T

	"Frequency of the Hopf point"
	ω::Tω

	"Parameters used by the vector field."
	params::Tpar

	"Parameter axis used to compute the branch on which this Branch point was detected."
	lens::Tlens

	"Right eigenvector"
	ζ::Tevr

	"Left eigenvector"
	ζstar::Tevl

	"Normal form coefficient (a = 0., b = 1 + 1im)"
	nf::Tnf

	"Type of Hopf bifurcation"
	type::Symbol
end

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
	R01 = (F(x0, set(parbif, lens, p + δ)) .- F(x0, parbif)) ./ δ
	Ψ001, _ = ls(L, -R01)

	# (2iω−L)Ψ200 = R20(ζ,ζ)
	R20 = R2(ζ, ζ)
	Ψ200, _ = ls(L, R20; a₀ = Complex(0, 2ω), a₁ = -1)
	@assert Ψ200 ≈ (Complex(0, 2ω)*I - L) \ R20

	# −LΨ110 = 2R20(ζ,cζ).
	R20 = 2 .* R2(ζ, cζ)
	Ψ110, _ = ls(L, -R20)

	# a = ⟨R11(ζ) + 2R20(ζ,Ψ001),ζ∗⟩
	av = (apply(dF(x0, set(parbif, lens, p + δ)), ζ) - apply(L, ζ)) ./ δ
	av .+= 2 .* R2(ζ, Ψ001)
	a = dot(av, ζstar)

	# b = ⟨2R20(ζ,Ψ110) + 2R20(cζ,Ψ200) + 3R30(ζ,ζ,cζ), ζ∗⟩)
	bv = 2 .* R2(ζ, Ψ110) .+ 2 .* R2(cζ, Ψ200) .+ 3 .* R3(ζ, ζ, cζ)
	b = dot(bv, ζstar)

	# return coefficients of the normal form
	supercritical = real(a) * real(b) < 0
	printstyled(color = :red,"--> Hopf bifurcation point is supercritical: ", supercritical, "\n")
	verbose && println((a = a, b = b))
	pt.nf = (a = a, b = b)
	pt.type = real(a) * real(b) < 0 ? :Supercritical : :Subcritical
	return pt
end

"""
	hopfNormalForm(F, dF, d2F, d3F, br::ContResult, ind_hopf::Int; Jt = nothing, δ = 1e-8, nev = 5, verbose = false, lens = br.param_lens)

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
function hopfNormalForm(F, dF, d2F, d3F, br::ContResult, ind_hopf::Int; Jt = nothing, δ = 1e-8, nev = 5, verbose = false, lens = br.param_lens)
	@assert br.bifpoint[ind_hopf].type == :hopf "The provided index does not refer to a Hopf Point"
	println("#"^53*"\n--> Hopf Normal form computation")

	# Newton parameters
	options = br.contparams.newtonOptions

	# bifurcation point
	bifpt = br.bifpoint[ind_hopf]
	eigRes = br.eig

	# eigenvalue
	λ = eigRes[bifpt.idx].eigenvals[bifpt.ind_bif]
	ω = imag(λ)

	# right eigenvector
	ζ = geteigenvector(options.eigsolver ,br.eig[bifpt.idx].eigenvec, bifpt.ind_bif)
	ζ ./= norm(ζ)

	# parameter for vector field
	p = bifpt.param
	parbif = set(br.params, lens, p)

	# jacobian at bifurcation point
	L = dF(bifpt.x, parbif)

	# left eigen-elements
	if isnothing(Jt)
		ζstar, λstar = getAdjointBasis(adjoint(L), conj(λ), options.eigsolver; nev = nev, verbose = verbose)
	else
		ζstar, λstar = getAdjointBasis(Jt(x, p), conj(λ), options.eigsolver; nev = nev, verbose = verbose)
	end

	# check that λstar ≈ conj(λ)
	@assert abs(λ + λstar) < 1e-2 "We did not find the left eigenvalue for the Hopf point, $λ ≈ $(λstar) and $(abs(λ + λstar)) ≈ 0?\n You can perhaps increase the number of computed eigenvalues, the number is nev = $nev"

	# normalise left eigenvector
	ζstar ./= dot(ζ, ζstar)

	hopfpt = HopfBifPoint(
		bifpt.x,
		bifpt.param,
		ω,
		parbif,
		lens,
		ζ,
		ζstar,
		(a = 0. + 0im, b = 0. + 0im),
		:Supercritical
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

####################################################################################################
"""
	continuation(F, dF, d2F, d3F, br::ContResult, ind_bif::Int, optionsCont::ContinuationPar ; Jt = nothing, δ = 1e-8, nev = optionsCont.nev, kwargs...)

Automatic branch switching at branch points based on a computation of the normal form. More information is provided in [Branch switching](@ref). An example of use is provided in [A generalized Bratu–Gelfand problem in two dimensions](@ref).

# Arguments
- `F, dF, d2F, d3F`: function `(x,p) -> F(x,p)` and its differentials `(x,p,dx) -> d1F(x,p,dx)`, `(x,p,dx1,dx2) -> d2F(x,p,dx1,dx2)`...
- `br` branch result from a call to [`continuation`](@ref)
- `ind_bif` index of the bifurcation point in `br` from which you want to branch from
- `optionsCont` options for the call to [`continuation`](@ref)

# Optional arguments
- `Jt` associated jacobian transpose, it should be implemented in an efficient manner. For matrix-free methods, `transpose` is not readily available and the user must provide a dedicated method. In the case of sparse based jacobian, `Jt` should not be passed as it is computed internally more efficiently, i.e. it avoid recomputing the jacobian as it would be if you pass `Jt = (x, p) -> transpose(dF(x, p))`.
- `δ` used internally to compute derivatives w.r.t the parameter `p`.
- `nev` number of eigenvalues to be computed to get the right eigenvector
- `verbose` display information about the bifurcation point (normal form,...)
- `kwargs` optional arguments to be passed to [`continuation`](@ref), the regular `continuation` one.
"""
function continuation(F, dF, d2F, d3F, br::ContResult, ind_bif::Int, optionsCont::ContinuationPar ; Jt = nothing, δ = 1e-8, nev = optionsCont.nev, kwargs...)
	verbose = get(kwargs, :verbosity, 0) > 0 ? true : false

	@assert br.type == :Equilibrium "Error! This bifurcation type is not handled.\n Branch point from $(br.type)"

	# detect bifurcation point type
	if br.bifpoint[ind_bif].type == :hopf
		@error "You need to chose an algorithm for computing the periodic orbit either a Shooting one or one based on Finite differences"
		bifpoint = hopfNF(F, dF, d2F, d3F, br, ind_bif, par, optionsCont.newtonOptions ; Jt = Jt, δ = δ, nev = nev, verbose = verbose)
		return bifpoint
	end

	# compute the normal form of the branch point
	bifpoint = computeNormalForm1d(F, dF, d2F, d3F, br, ind_bif; Jt = Jt, δ = δ, nev = nev, verbose = verbose)

	# compute predictor for a point on new branch
	pred = predictor(bifpoint, optionsCont.ds / 50; verbose = verbose)

	verbose && printstyled(color = :green, "\n--> Start branch switching. \n--> Bifurcation type = ",bifpoint.type, "\n----> newp = ", pred.p, ", δp = ", br.bifpoint[ind_bif].param - pred.p, "\n")

	# perform continuation
	return continuation(F, dF, pred.x, set(br.params, br.param_lens, pred.p), br.param_lens, optionsCont; kwargs...)

end
