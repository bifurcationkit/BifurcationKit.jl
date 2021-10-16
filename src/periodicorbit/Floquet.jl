# This function is very important for the computation of Floquet multipliers and checks that the eigensolvers compute the eigenvalues with largest modulus instead of their default behaviour which is with largest real part. If this option is not properly set, bifurcations of periodic orbits will be missed.
function checkFloquetOptions(eigls::AbstractEigenSolver)
	if eigls isa DefaultEig
		return @set eigls.which = abs
	elseif eigls isa EigArpack
		return setproperties(eigls; which = :LM, by = abs)
	elseif eigls isa EigArnoldiMethod
		return setproperties(eigls; which = ArnoldiMethod.LM(), by = abs)
	end
	if eigls isa EigKrylovKit
		return @set eigls.which = :LM
	end
end

# see https://discourse.julialang.org/t/uniform-scaling-inplace-addition-with-matrix/59928/5

####################################################################################################
# Computation of Floquet Coefficients for periodic orbit problems

"""
	floquet = FloquetQaD(eigsolver::AbstractEigenSolver)

This composite type implements the computation of the eigenvalues of the monodromy matrix in the case of periodic orbits problems (based on the Shooting method or Finite Differences (Trapeze method)), also called the Floquet multipliers. The method, dubbed Quick and Dirty (QaD), is not numerically very precise for large / small Floquet exponents when the number of time sections is large because of many matrix products. It allows, nevertheless, to detect bifurcations. The arguments are as follows:
- `eigsolver::AbstractEigenSolver` solver used to compute the eigenvalues.

If `eigsolver == DefaultEig()`, then the monodromy matrix is formed and its eigenvalues are computed. Otherwise, a Matrix-Free version of the monodromy is used.

!!! danger "Floquet multipliers computation"
    The computation of Floquet multipliers is necessary for the detection of bifurcations of periodic orbits (which is done by analyzing the Floquet exponents obtained from the Floquet multipliers). Hence, the eigensolver `eigsolver` needs to compute the eigenvalues with largest modulus (and not with largest real part which is their default behavior). This can be done by changing the option `which = :LM` of `eigsolver`. Nevertheless, note that for most implemented eigensolvers in the current Package, the proper option is set.
"""
struct FloquetQaD{E <: AbstractEigenSolver } <: AbstractFloquetSolver
	eigsolver::E
	function FloquetQaD(eigls::AbstractEigenSolver)
		eigls2 = checkFloquetOptions(eigls)
		return new{typeof(eigls2)}(eigls2)
	end
	FloquetQaD(eigls::FloquetQaD) = eigls
end
geteigenvector(eig::FloquetQaD, vecs, n::Union{Int, Array{Int64,1}}) = geteigenvector(eig.eigsolver, vecs, n)

function (fl::FloquetQaD)(J, nev; kwargs...)
	if fl.eigsolver isa DefaultEig
		# we build the monodromy matrix and compute the spectrum
		monodromy = MonodromyQaD(J)
	else
		# we use a Matrix Free version
		monodromy = dx -> MonodromyQaD(J, dx)
	end
	vals, vecs, cv, info = fl.eigsolver(monodromy, nev)
	Inf in vals && @warn "Detecting infinite eigenvalue during the computation of Floquet coefficients"

	# the `vals` should be sorted by largest modulus, but we need the log of them sorted this way
	logvals = log.(complex.(vals))
	I = sortperm(logvals, by = real, rev = true)
	return logvals[I], geteigenvector(fl.eigsolver, vecs, I), cv, info
end
####################################################################################################
# ShootingProblem
# Matrix free monodromy operator
function MonodromyQaD(JacSH::FloquetWrapper{Tpb, Tjacpb, Torbitguess, Tp}, du::AbstractVector) where {Tpb <: ShootingProblem, Tjacpb, Torbitguess, Tp}
	sh = JacSH.pb
	x = JacSH.x
	p = JacSH.par

	# period of the cycle
	T = extractPeriodShooting(x)

	# extract parameters
	M = getM(sh)
	N = div(length(x) - 1, M)

	# extract the time slices
	xv = @view x[1:end-1]
	xc = reshape(xv, N, M)

	out = copy(du)

	for ii in 1:M
		# call the jacobian of the flow
		@views out .= sh.flow(Val(:SerialdFlow), xc[:, ii], p, out, sh.ds[ii] * T).du
	end
	return out
end

# Compute the monodromy matrix at `x` explicitely, not suitable for large systems
function MonodromyQaD(JacSH::FloquetWrapper{Tpb, Tjacpb, Torbitguess, Tp}) where {Tpb <: ShootingProblem, Tjacpb, Torbitguess, Tp}
	sh = JacSH.pb
	x = JacSH.x
	p = JacSH.par

	# period of the cycle
	T = extractPeriodShooting(x)

	# extract parameters
	M = getM(sh)
	N = div(length(x) - 1, M)

	Mono = zeros(N, N)

	# extract the time slices
	xv = @view x[1:end-1]
	xc = reshape(xv, N, M)
	du = zeros(N)

	for ii in 1:N
		du[ii] = 1
		# call jacobian of the flow
		@views Mono[:, ii] .= sh.flow(Val(:SerialdFlow), xc[:, 1], p, du, T).du
		du[ii] = 0
	end

	return Mono
end

# Compute the monodromy matrix at `x` explicitely, not suitable for large systems
# it is based on a matrix expression of the Jacobian of the shooting functional. We thus
# just extract the blocks needed to compute the monodromy
function MonodromyQaD(JacSH::FloquetWrapper{Tpb, Tjacpb, Torbitguess, Tp}) where {Tpb <: ShootingProblem, Tjacpb <: AbstractMatrix, Torbitguess, Tp}
	J = JacSH.jacpb
	sh = JacSH.pb
	M = getM(sh)
	N = div(length(JacSH.x) - 1, M)
	mono = copy(J[1:N, 1:N])
	if M == 1
		return mono + I
	end
	tmp = similar(mono)
	r = N
	for ii = 1:M-1
		# mono .= J[r+1:r+N, r+1:r+N] * mono
		@views mul!(tmp, J[r+1:r+N, r+1:r+N], mono)
		mono .= tmp
		r += N
	end
	return mono
end

# This function is used to reconstruct the spatio-temporal eigenvector of the shooting functional sh
# at position x from the Floquet eigenvector ζ
@views function MonodromyQaD(::Val{:ExtractEigenVector}, sh::ShootingProblem, x::AbstractVector, par, ζ::AbstractVector)

	# period of the cycle
	T = getPeriod(sh, x)

	# extract parameters
	M = getM(sh)
	N = div(length(x) - 1, M)

	# extract the time slices
	xv = x[1:end-1]
	xc = reshape(xv, N, M)

	out = sh.flow(Val(:SerialdFlow), xc[:, 1], par, ζ, sh.ds[1] * T).du
	out_a = [copy(out)]

	for ii in 2:M
		# call the jacobian of the flow
		out .= sh.flow(Val(:SerialdFlow), xc[:, ii], par, out, sh.ds[ii] * T).du
		push!(out_a, copy(out))
	end
	return out_a
end
####################################################################################################
# PoincareShooting

# matrix free evaluation of monodromy operator
function MonodromyQaD(JacSH::FloquetWrapper{Tpb, Tjacpb, Torbitguess, Tp}, dx_bar::AbstractVector) where {Tpb <: PoincareShootingProblem, Tjacpb, Torbitguess, Tp}
	psh = JacSH.pb
	x_bar = JacSH.x
	p = JacSH.par

	M = getM(psh)
	Nm1 = div(length(x_bar), M)

	# reshape the period orbit guess into a Matrix
	x_barc = reshape(x_bar, Nm1, M)
	@assert length(dx_bar) == Nm1 "Please provide the right dimension to your matrix-free eigensolver, it must be $Nm1."

	xc = similar(x_bar, Nm1 + 1)
	outbar = copy(dx_bar)
	outc = similar(dx_bar, Nm1 + 1)

	for ii in 1:M
		E!(psh.section,  xc,  view(x_barc, :, ii), ii)
		dE!(psh.section, outc, outbar, ii)
		outc .= diffPoincareMap(psh, xc, p, outc, ii)
		# check to <outc, normals[ii]> = 0
		# println("--> ii=$ii, <out, normali> = ", dot(outc, sh.section.normals[ii]))
		dR!(psh.section, outbar, outc, ii)
	end
	return outbar

end

# matrix based formulation of monodromy operator, not suitable for large systems
# it is based on a matrix expression of the Jacobian of the shooting functional. We thus
# just extract the blocks needed to compute the monodromy
function MonodromyQaD(JacSH::FloquetWrapper{Tpb, Tjacpb, Torbitguess, Tp}) where {Tpb <: PoincareShootingProblem, Tjacpb <: AbstractMatrix, Torbitguess, Tp}
	J = JacSH.jacpb
	sh = JacSH.pb
	T = eltype(J)

	M = getM(sh)
	Nj = length(JacSH.x)
	N = div(Nj, M)

	if M == 1
		return I-J
	end

	mono = copy(J[N+1:2N, 1:N])
	tmp = similar(mono)
	r1 = mod(2N, Nj)
	r2 = N
	for ii = 1:M-1
		# mono .= J[r+1:r+N, r+1:r+N] * mono
		@views mul!(tmp, J[r1+1:r1+N, r2+1:r2+N], mono)
		mono .= tmp
		r1 = mod(r1 + N, Nj)
		r2 += N
	end
	# the structure of the functional imposes to take into account the sign
	sgn = iseven(M) ? one(T) : -one(T)
	mono .*= sgn
	return mono
end

# This function is used to reconstruct the spatio-temporal eigenvector of the shooting functional sh
# at position x from the Floquet eigenvector ζ
@views function MonodromyQaD(::Val{:ExtractEigenVector}, psh::PoincareShootingProblem, x_bar::AbstractVector, p, ζ::AbstractVector)
	#  ζ is of size (N-1)
	M = getM(psh)
	Nm1 = length(ζ)
	dx = similar(x_bar, length(ζ) + 1)

	x_barc = reshape(x_bar, Nm1, M)
	xc = similar(x_bar, Nm1 + 1)
	dx_bar = similar(x_bar, Nm1)
	outbar = copy(dx_bar)
	outc = similar(dx_bar, Nm1 + 1)
	out_a = typeof(xc)[]

	for ii in 1:M
		E!(psh.section,  xc,  view(x_barc, :, ii), ii)
		dE!(psh.section, outc, outbar, ii)
		outc .= diffPoincareMap(psh, xc, p, outc, ii)
		# check to <outc, normals[ii]> = 0
		# println("--> ii=$ii, <out, normali> = ", dot(outc, sh.section.normals[ii]))
		dR!(psh.section, outbar, outc, ii)
		push!(out_a, copy(outbar))
	end
	return out_a
end
####################################################################################################
# PeriodicOrbitTrapProblem

# Matrix-Free version of the monodromy operator
function MonodromyQaD(JacFW::FloquetWrapper{Tpb, Tjacpb, Torbitguess, Tp}, du::AbstractVector) where {Tpb <: PeriodicOrbitTrapProblem, Tjacpb, Torbitguess, Tp}
	poPb = JacFW.pb
	u0 = JacFW.x
	par = JacFW.par

	# extraction of various constants
	M, N = size(poPb)

	# period of the cycle
	T = extractPeriodFDTrap(u0)

	# time step
	h =  T * getTimeStep(poPb, 1)
	Typeh = typeof(h)

	out = copy(du)

	u0c = getTimeSlices(u0, N, M)

	@views out .= out .+ h/2 .* apply(poPb.J(u0c[:, M-1], par), out)
	# res = (I - h/2 * poPb.J(u0c[:, 1])) \ out
	@views res, _ = poPb.linsolver(poPb.J(u0c[:, 1], par), out; a₀ = convert(Typeh, 1), a₁ = -h/2)
	out .= res

	for ii in 2:M-1
		h =  T * getTimeStep(poPb, ii)
		@views out .= out .+ h/2 .* apply(poPb.J(u0c[:, ii-1], par), out)
		# res = (I - h/2 * poPb.J(u0c[:, ii])) \ out
		@views res, _ = poPb.linsolver(poPb.J(u0c[:, ii], par), out; a₀ = convert(Typeh, 1), a₁ = -h/2)
		out .= res
	end

	return out
end

# This function is used to reconstruct the spatio-temporal eigenvector of the Trapezoid functional
# at position x from the Floquet eigenvector ζ
function MonodromyQaD(::Val{:ExtractEigenVector}, poPb::PeriodicOrbitTrapProblem, u0::AbstractVector, par, ζ::AbstractVector)
	# extraction of various constants
	M = poPb.M
	N = poPb.N

	# period of the cycle
	T = extractPeriodFDTrap(u0)

	# time step
	h =  T * getTimeStep(poPb, 1)
	Typeh = typeof(h)

	out = copy(ζ)

	u0c = getTimeSlices(u0, N, M)

	@views out .= out .+ h/2 .* apply(poPb.J(u0c[:, M-1], par), out)
	# res = (I - h/2 * poPb.J(u0c[:, 1])) \ out
	@views res, _ = poPb.linsolver(poPb.J(u0c[:, 1], par), out; a₀ = convert(Typeh, 1), a₁ = -h/2)
	out .= res
	out_a = [copy(out)]
	# push!(out_a, copy(out))

	for ii in 2:M-1
		h =  T * getTimeStep(poPb, ii)
		@views out .= out .+ h/2 .* apply(poPb.J(u0c[:, ii-1], par), out)
		# res = (I - h/2 * poPb.J(u0c[:, ii])) \ out
		@views res, _ = poPb.linsolver(poPb.J(u0c[:, ii], par), out; a₀ = convert(Typeh, 1), a₁ = -h/2)
		out .= res
		push!(out_a, copy(out))
	end
	push!(out_a, copy(ζ))

	return out_a
end

# Compute the monodromy matrix at `u0` explicitely, not suitable for large systems
function MonodromyQaD(JacFW::FloquetWrapper{Tpb, Tjacpb, Torbitguess, Tp})  where {Tpb <: PeriodicOrbitTrapProblem, Tjacpb, Torbitguess, Tp}

	poPb = JacFW.pb
	u0 = JacFW.x
	par = JacFW.par

	# extraction of various constants
	M, N = size(poPb)

	# period of the cycle
	T = extractPeriodFDTrap(u0)

	# time step
	h =  T * getTimeStep(poPb, 1)

	u0c = getTimeSlices(u0, N, M)

	@views mono = Array(I - h/2 * (poPb.J(u0c[:, 1], par))) \ Array(I + h/2 * poPb.J(u0c[:, M-1], par))
	temp = similar(mono)

	for ii in 2:M-1
		# for some reason, the next line is faster than doing (I - h/2 * (poPb.J(u0c[:, ii]))) \ ...
		# also I - h/2 .* J seems to hurt (a little) the performances
		h =  T * getTimeStep(poPb, ii)
		@views temp = Array(I - h/2 * (poPb.J(u0c[:, ii], par))) \ Array(I + h/2 * poPb.J(u0c[:, ii-1], par))
		mono .= temp * mono
	end
	return mono
end
####################################################################################################
# simplified version of
# Fairgrieve, Thomas F., and Allan D. Jepson. “O. K. Floquet Multipliers.” SIAM Journal on Numerical Analysis 28, no. 5 (October 1991): 1446–62. https://doi.org/10.1137/0728075.
struct FloquetLUColl{E <: AbstractEigenSolver, Tb} <: AbstractFloquetSolver
	eigsolver::E		# not really needed
	B::Tb
	"""
	## Arguments
	- `eigls` an eigensolver
	- `ntot` total number of unknowns (without countinig the period)
	- `n` space dimension
	"""
	function FloquetLUColl(eigls::AbstractEigenSolver, ntot::Int, n::Int)
		eigls2 = checkFloquetOptions(eigls)
		# build the mass matrix
		B = zeros(ntot, ntot)
		B[end-n+1:end, end-n+1:end] .= I(n)
		return new{typeof(eigls2), typeof(B)}(eigls2, B)
	end
	FloquetLUColl(eigls::FloquetLUColl) = eigls
end

# based on Fairgrieve, Thomas F., and Allan D. Jepson. “O. K. Floquet Multipliers.” SIAM Journal on Numerical Analysis 28, no. 5 (October 1991): 1446–62. https://doi.org/10.1137/0728075.
@views function (fl::FloquetLUColl)(JacColl::FloquetWrapper{Tpb, Tjacpb, Torbitguess, Tp}, nev; kwargs...) where {Tpb <: PeriodicOrbitOCollProblem, Tjacpb <: AbstractMatrix, Torbitguess, Tp}
	prob = JacColl.pb
	_J = JacColl.jacpb
	n, m, Ntst = size(prob)
	J = _J[1:end-1, 1:end-1]
	# case of v(0)
	J[end-n+1:end, 1:n] .= I(n)
	# case of v(1)
	J[end-n+1:end, end-n+1:end] .= -I(n)
	# solve generalized eigenvalue problem
	values, vecs = eigen(J, fl.B)
	# remove infinite eigenvalues
	ind = isinf.(values)
	indvalid = ind .== false
	vals = values[indvalid]
	# this are the Floquet multipliers
	μ = @. Complex(1 / (1 + vals))
	return log.(μ), Complex.(vecs[indvalid, :]), true
end
