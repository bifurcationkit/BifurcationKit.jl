abstract type BifurcationPoint end
abstract type BranchPoint <: BifurcationPoint end

"""
	getAdjointEV(Lstar, λ::Number, options::NewtonPar; nev = 3)

Return an eigenvector for an eigenvalue closest to λ. `nev` indicates how many eigenvalues must be computed by the eigensolver.

More information is provided in [Simple bifurcation branch switching](@ref)
"""
function getAdjointEV(Lstar, λ::Number, options::NewtonPar; nev = 3, verbose = false)
	λstar, evstar = options.eigsolver(Lstar, nev)
	I = argmin(abs.(λstar .- λ))
	verbose && println("--> VPstars = ", λstar)
	verbose && println("--> VP = ", λ, ", VPstar = ", λstar[I])
	@assert abs(real(λstar[I])) < 1e-2 "Did not converge to the requested eigenvalue. We found $(λstar[I]) ≈ 0"
	ζstar = geteigenvector(options.eigsolver ,evstar, I)

	return ζstar, λstar[I]
end

# the following structs are a machinary to extend multilinear mapping from Real valued to Complex valued Arrays
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
mutable struct SimpleBranchPoint{Tv, T, Tevl, Tevr, Tnf} <: BranchPoint
	# "bifurcation point"
	x0::Tv

	# "Parameter value at the bifurcation point"
	p::T

	# "Right eigenvector"
	ζ::Tevr

	# "Left eigenvector"
	ζstar::Tevl

	# "Normal form coefficients"
	nf::Tnf

	# type of bifurcation point
	type::Symbol
end

"""
Compute normal form based on Golubitsky, Martin, David G Schaeffer, and Ian Stewart. Singularities and Groups in Bifurcation Theory. New York: Springer-Verlag, 1985, VI.1.d page 295.

"""
function analyseNF(F, dF, d2F, d3F, br::ContResult, ind_bif::Int, options::NewtonPar ; δ = 1e-8, nev = 5, Jt = nothing, verbose = false)
	bifpt = br.bifpoint[ind_bif]
	@assert bifpt.type == :bp "The provided index does not refer to a Branch Point"
	@assert sum(abs, bifpt.δ) == 1 "We only provide analysis for simple bifurcation points for which the kernel of the jacobian is 1d. Here, the dimension of the BP is $(sum(abs, bifpt.δ))"

	verbose && println("#"^53*"\n--> Normal form Computation")
	verbose && println("--> analyse bifurcation at p = ", bifpt.param)
	# linear solver
	ls = options.linsolver

	# "zero" eigenvalue at bifurcation point
	λ = (br.eig[bifpt.idx].eigenvals[bifpt.ind_bif])
	verbose && println("--> smallest eigenvalue at bifurcation = ", λ)

	# corresponding eigenvector
	ζ = geteigenvector(options.eigsolver ,br.eig[bifpt.idx].eigenvec, bifpt.ind_bif)
	ζ ./= norm(ζ)

	# jacobian at bifurcation point
	L = dF(bifpt.x, bifpt.param)

	# extract eigenelements for adjoint(L), needed to build spectral projector
	if isnothing(Jt)
		ζstar, λstar = getAdjointEV(adjoint(L), conj(λ), options; nev = nev, verbose = verbose)
	else
		ζstar, λstar = getAdjointEV(Jt(x, p), conj(λ), options; nev = nev, verbose = verbose)
	end
	verbose && println("--> VP = ", λ,", VPstar = ",λstar)
	@assert abs(dot(ζ, ζstar)) > 1e-12
	ζstar ./= dot(ζ, ζstar)

	# bifurcation point
	x0 = bifpt.x
	p = bifpt.param

	# differentials and projector on Range(L)
	R2 = BilinearMap( (dx1, dx2)      -> d2F(x0, p, dx1, dx2))
	R3 = TrilinearMap((dx1, dx2, dx3) -> d3F(x0, p, dx1, dx2, dx3))
	E = x -> x .- dot(x, ζstar) .* ζ

	# we compute the reduced equation: a⋅(p - pbif) + x⋅(b1⋅(p - pbif) + b2⋅x/2 + b3⋅x^2/6)
	# coefficient of p
	R01 = (F(x0, p + δ) .- F(x0, p)) ./ δ
	a = dot(R01, ζstar)
	verbose && println("--> a = ", a)

	# coefficient of x*p
	R11 = (apply(dF(x0, p + δ), ζ) - apply(L, ζ)) ./ δ
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
	return SimpleBranchPoint(x0, p, ζ, ζstar, (a=a, b1=b1, b2=b2, b3=b3), type)
end

function predictor(bp::SimpleBranchPoint, ds::T; verbose = false) where T
	@assert bp.type != :ProbablySaddleNode
	nf = bp.nf
	if bp.type == :Transcritical
		pnew = bp.p + ds
		# we solve b1 * ds + b2 * amp / 2 = 0
		amp = -2ds * real(nf.b1 / nf.b2)
		dsfactor = T(1)
	else
		# case of the Pitchfork bifurcation
		# we need to find the type, supercritical or subcritical
		dsfactor = real(nf.b1) * real(nf.b3) < 0 ? T(1) : T(-1)
		pnew = bp.p + ds * dsfactor
		# we solve b1 * ds + b3 * amp^2 / 6 = 0
		amp = sqrt(-6abs(ds) * dsfactor * real(nf.b1 / nf.b3))
	end
	verbose && println("--> Prediction from Normal form, δp = $(bp.p - pnew), amp = $amp")
	return (x = bp.x0 .+ amp .* real.(bp.ζ), p = pnew, dsfactor = dsfactor)
end

####################################################################################################
mutable struct HopfBifPoint <: BifurcationPoint
	# "Hopf point"
	x0

	# "Parameter value at the Hopf point"
	p

	# "Frequency of the Hopf point"
	ω

	# "Right eigenvector"
	ζ

	# "Left eigenvector"
	ζstar

	# "Normal form coefficient (a = 0., b = 1 + 1im)"
	nf
end

function hopfNF(F, dF, d2F, d3F, pt::HopfBifPoint, ls; δ = 1e-8, verbose = false)
	x0 = pt.x0
	p = pt.p
	ω = pt.ω
	ζ = pt.ζ
	cζ = conj.(pt.ζ)
	ζstar = pt.ζstar

	# jacobian at the bifurcation point
	L = dF(x0, p)

	R2 = BilinearMap( (dx1, dx2)      -> d2F(x0, p, dx1, dx2) ./2)
	R3 = TrilinearMap((dx1, dx2, dx3) -> d3F(x0, p, dx1, dx2, dx3) ./6 )

	# −LΨ001 = R01
	R01 = (F(x0, p + δ) .- F(x0, p)) ./ δ
	Ψ001, _ = ls(L, -R01)

	# (2iω−L)Ψ200 = R20(ζ,ζ)
	R20 = R2(ζ, ζ)
	Ψ200, _ = ls(L, R20; a₀ = Complex(0, 2ω), a₁ = -1)
	@assert Ψ200 ≈ (Complex(0, 2ω)*I - L) \ R20

	# −LΨ110 = 2R20(ζ,cζ).
	R20 = 2 .* R2(ζ, cζ)
	Ψ110, _ = ls(L, -R20)

	# a = ⟨R11(ζ) + 2R20(ζ,Ψ001),ζ∗⟩
	av = (apply(dF(x0, p + δ), ζ) - apply(L, ζ)) ./ δ
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
	return pt
end

"""
	hopfNF(F, dF, d2F, d3F, br::ContResult, ind_hopf::Int, options::NewtonPar ; Jt = nothing, δ = 1e-8, nev = 5, verbose = false)

Compute the Hopf normal form.

# Arguments
- `F, dF, d2F, d3F`: function `(x,p) -> F(x,p)` and its differencials `(x,p,dx) -> d1F(x,p,dx)`, `(x,p,dx1,dx2) -> d2F(x,p,dx1,dx2)`...
- `br` branch result from a call to `continuation`
- `ind_hopf` index of the bifurcation point in `br`
- `options` options for the Newton solver

# Optional argument
- `Jt` is the jacobian adjoint, used for computation of the eigen-elements of the jacobian adjoint, needed to compute the spectral projector
- `δ = 1e-8` used for finite differences
- `nev = 5` number of eigenvalues to compute to estimate the spectral projector
- `verbose` bool to print information
"""
function hopfNF(F, dF, d2F, d3F, br::ContResult, ind_hopf::Int, options::NewtonPar ; Jt = nothing, δ = 1e-8, nev = 5, verbose = false)
	@assert br.bifpoint[ind_hopf].type == :hopf "The provided index does not refer to a Hopf Point"
	println("#"^20*"\n--> Hopf Normal form computation")

	# bifurcation point
	bifpt = br.bifpoint[ind_hopf]
	eigRes = br.eig

	# eigenvalue
	λ = eigRes[bifpt.idx].eigenvals[bifpt.ind_bif]
	ω = imag(λ)

	# eigenvector
	ζ = geteigenvector(options.eigsolver ,br.eig[bifpt.idx].eigenvec, bifpt.ind_bif)
	ζ ./= norm(ζ)

	# jacobian at bifurcation point
	L = dF(bifpt.x, bifpt.param)

	@show bifpt.param

	@show eigen(L)

	# eigen-elements of the jacobian adjoint
	if isnothing(Jt)
		ζstar, λstar = getAdjointEV(adjoint(L), conj(λ), options; nev = nev, verbose = verbose)
	else
		ζstar, λstar = getAdjointEV(Jt(x, p), conj(λ), options; nev = nev, verbose = verbose)
	end
	verbose && println("--> VP = ", λ, ", VPstar = ", λstar)

	# check that λstar ≈ conj(λ)
	@assert abs(λ + λstar) < 1e-3 "We did not find the needed eigenvalues for the jacobian adjoint, $λ ≈ $(λstar) and $(abs(λ + λstar)) ≈ 0?"

	# normalise adjoint eigenvector
	ζstar ./= dot(ζ, ζstar)

	# @show abs(dot(ζ, ζstar))
	hopfpt = HopfBifPoint(
		bifpt.x,
		bifpt.param,
		ω,
		ζ,
		ζstar,
		(a = 0 + 0im, b = 0 + 0im)
	)
	return hopfNF(F, dF, d2F, d3F, hopfpt, options.linsolver ; δ = δ, verbose = verbose)
end

####################################################################################################
"""
	continuation(F, dF, d2F, d3F, br::ContResult, ind_bif::Int, optionsCont::ContinuationPar ; Jt = nothing, δ = 1e-8, nev = 5, verbose = false, kwargs...)

Automatic branch switching. More information is provided in [Simple bifurcation branch switching
](@ref). An example is provided in [A generalized Bratu–Gelfand problem in two dimensions](@ref).

# Arguments
- `F, dF, d2F, d3F`: function `(x,p) -> F(x,p)` and its differencials `(x,p,dx) -> d1F(x,p,dx)`, `(x,p,dx1,dx2) -> d2F(x,p,dx1,dx2)`...
- `br` branch result from a call to `continuation`
- `ind_bif` index of the bifurcation point in `br` from which you want to branch from
- `optionsCont` options for the call to `continuation`

# Optional arguments
- `Jt` associated jacobian transpose, it should be implemented in an efficient manner. For matrix-free methods, `transpose` is not readily available and the user must provide a dedicated method. In the case of sparse based jacobian, `Jt` should not be passed as it is computed internally more efficiently, i.e. it avoid recomputing the jacobian as it would be if you pass `Jt = (x, p) -> transpose(dF(x, p))`.
- `δ` used internally to compute derivatives w.r.t the parameter `p`.
- `nev` number of eigenvalues to be computed to get the right eigenvector
- `verbose` display information about the bifurcation point (normal form,...)
- `kwargs` optional arguments to be passed to [`contination`](@ref), the regular `continuation` one.

"""
function continuation(F, dF, d2F, d3F, br::ContResult, ind_bif::Int, optionsCont::ContinuationPar ; Jt = nothing, δ = 1e-8, nev = 5, verbose = false, kwargs...)
	# detect bifurcation point type
	if br.bifpoint[ind_bif].type == :hopf
		bifpoint = hopfNF(F, dF, d2F, d3F, br, ind_bif::Int, optionsCont.newtonOptions ; Jt = Jt, δ = δ, nev = nev, verbose = verbose)
	else
		bifpoint = analyseNF(F, dF, d2F, d3F, br, ind_bif::Int, optionsCont.newtonOptions ; Jt = Jt, δ = δ, nev = nev, verbose = verbose)
	end

	# compute predictor for point on new branch
	pred = predictor(bifpoint, optionsCont.ds / 50; verbose = verbose)

	verbose && printstyled(color = :green, "\n--> Start branch switching. \n--> Bifurcation type = ",bifpoint.type, "\n----> newp = ", pred.p, ", δp = ", br.bifpoint[ind_bif].param - pred.p, "\n")

	# perform continuation
	return continuation(F, dF, pred.x, pred.p, optionsCont; kwargs...)

end
