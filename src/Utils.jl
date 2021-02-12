closesttozero(ev) = ev[sortperm(ev, by = abs∘real)]
getinterval(a,b) = (min(a,b), max(a,b))
####################################################################################################
function displayIteration(i, residual, itlinear = 0)
	(i == 0) && println("\n Newton Iterations      f(x)      Linear Iterations\n")
	if length(itlinear)==1
		@printf("%11d %19.4e %13d\n", i, residual, itlinear);
	else
		if itlinear isa Tuple{Int64, Int64}
			@printf("%11d %19.4e      (%4d, %4d)\n", i, residual, itlinear[1], itlinear[2]);
		else
			# used for nested linear solves
			@printf("%11d %19.4e  ", i, residual); println(itlinear);
		end
	end
end
####################################################################################################
function computeEigenvalues(it::ContIterable, u0, par, nev = it.contParams.nev; kwargs...)
	return it.contParams.newtonOptions.eigsolver(it.J(u0, par), nev; kwargs...)
end

function computeEigenvalues(iter::ContIterable, state::ContState; kwargs...)
	# we compute the eigen-elements
	n = state.n_unstable[2]
	nev_ = max(n + 5, iter.contParams.nev)
	eiginfo = computeEigenvalues(iter, getx(state), setParam(iter, getp(state)), nev_; kwargs...)
	_isstable, n_unstable, n_imag = isStable(iter.contParams, eiginfo[1])
	return eiginfo, _isstable, n_unstable, n_imag
end

# same as previous but we save the eigen-elements in state
function computeEigenvalues!(iter::ContIterable, state::ContState; saveEigenVec = true, kwargs...)
	eiginfo, _isstable, n_unstable, n_imag = computeEigenvalues(iter, state; kwargs...)
	# we update the state
	updateStability!(state, n_unstable, n_imag)
	if isnothing(state.eigvals) == false
		state.eigvals = eiginfo[1]
	end
	if saveEigenVec && iter.contParams.saveEigenvectors
		state.eigvecs = eiginfo[2]
	end
	# iteration number in eigen-solver
	it_number = eiginfo[end]
	return it_number
end

####################################################################################################
"""
	finiteDifferences(F, x::AbstractVector; δ = 1e-9)

Compute a Jacobian by Finite Differences
"""
function finiteDifferences(F, x::AbstractVector; δ = 1e-9)
	N = length(x)
	Nf = length(F(x))
	J = zeros(eltype(x), Nf, N)
	finiteDifferences!(F, J, x; δ = δ)
	return J
end

"""
	finiteDifferences!(F, J, x::AbstractVector; δ = 1e-9)

Compute a Jacobian by Finite Differences, update J. Use the centered formula (f(x+δ)-f(x-δ))/2δ
"""
function finiteDifferences!(F, J, x::AbstractVector; δ = 1e-9)
	f = F(x)
	x1 = copy(x)
	for i ∈ eachindex(x)
		x1[i] += δ
		J[:, i] .= F(x1)
		x1[i] -= 2δ
		J[:, i] .-= F(x1)
		J[:, i] ./= 2δ
		x1[i] += δ
	end
	return J
end
####################################################################################################
using BlockArrays, SparseArrays
"""
Function waiting to be accepted to BlockArrays.jl
"""
function blockToSparse(J::AbstractBlockArray)
	nl, nc = size(J.blocks)
	# form the first line of blocks
	res = J[Block(1,1)]
	for j in 2:nc
		res = hcat(res, J[Block(1,j)])
	end
	# continue with the other lines
	for i in 2:nl
		line = J[Block(i,1)]
		for j in 2:nc
			line = hcat(line, J[Block(i,j)])
		end
		res = vcat(res,line)
	end
	return res
end
####################################################################################################
# this function extracts the indices of the blocks composing the matrix A. We assume that the blocks have the same sparsity
function getBlocks(A::SparseMatrixCSC, N, M)
	I,J,K = findnz(A)
	out = [Vector{Int}() for i in 1:M+1, j in 1:M+1];
	for k in eachindex(I)
		m, l = div(I[k]-1, N), div(J[k]-1, N)
		push!(out[1+m,1+l], k)
	end
	res = [length(m) for m in out]
	out
end
####################################################################################################
"""
$(SIGNATURES)

This function implements a counter. If `everyN == 0`, it returns false. Otherwise, it returns `true` when `step` is a multiple of `everyN`
"""
function modCounter(step, everyN)
	if step == 0; return false; end
	if everyN == 0; return false; end
	if everyN == 1; return true; end
	return mod(step, everyN) == 0
end
####################################################################################################
# this trick is extracted from KrylovKit. It allows for the Jacobian to be specified as a matrix (sparse / dense) or as a function.
apply(A::AbstractMatrix, x::AbstractVector) = A * x
apply(f, x) = f(x)

# apply!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector) = mul!(y, A, x)
# apply!(y, f, x) = copyto!(y, f(x))

# empty eigenvectors to save memory
# _empty(a::AbstractVector{T}, ::Type{U}=T) where {T,U} = Vector{U}()
# _empty(a::AbstractMatrix{T}, ::Type{U}=T) where {T,U} = similar(a, (0,0))
###################################################################################################
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
