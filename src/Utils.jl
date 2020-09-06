####################################################################################################
function displayIteration(i, funceval, residual, itlinear = 0)
	(i == 0) && println("\n Newton Iterations \n   Iterations      Func-count      f(x)      Linear-Iterations\n")
	if length(itlinear)==1
		@printf("%9d %16d %14.4e %9d\n", i, funceval, residual, itlinear);
	else
		if itlinear isa Tuple{Int64,Int64}
			@printf("%9d %16d %14.4e (%6d, %6d)\n", i, funceval, residual, itlinear[1], itlinear[2]);
		else
			# used for nested linear solves
			@printf("%9d %16d %14.4e  ", i, funceval, residual);println(itlinear);
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
	eiginfo = computeEigenvalues(iter, getx(state), set(iter.par, iter.param_lens, getp(state)), nev_; kwargs...)
	_isstable, n_unstable, n_imag = isstable(iter.contParams, eiginfo[1])
	return eiginfo, _isstable, n_unstable, n_imag
end

# same as previous but we save the eigen-elements in state
function computeEigenvalues!(iter::ContIterable, state::ContState)
	eiginfo, _isstable, n_unstable, n_imag = computeEigenvalues(iter, state)
	# we update the state
	updatestability!(state, n_unstable, n_imag)
	if isnothing(state.eigvals) == false
		state.eigvals = eiginfo[1]
	end
	if iter.contParams.saveEigenvectors
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

Compute a Jacobian by Finite Differences, update J
"""
function finiteDifferences!(F, J, x::AbstractVector; δ = 1e-9)
	f = F(x)
	x1 = copy(x)
	for i ∈ eachindex(x)
		x1[i] += δ
		J[:, i] .= (F(x1) .- F(x)) / δ
		x1[i] -= δ
	end
	return J
end

####################################################################################################
"""
Save solution / data in JLD2 file
- `filename` is for example "example.jld2"
- `sol` is the solution
- `p` is the parameter
- `i` is the index of the solution to be saved
"""
function saveToFile(filename, sol, p, i::Int64, br::ContResult)
	try
		# create a group in the JLD format
		jldopen(filename*".jld2", "a+") do file
			mygroup = JLD2.Group(file, "sol-$i")
			mygroup["sol"] = sol
			mygroup["param"] = p
		end

		jldopen(filename*"-branch.jld2", "w") do file
			file["branch"] = br
			file["contParam"] = br.contparams
		end
	catch
		@error "Could not save branch in the jld2 file"
	end
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
# this trick is extracted from KrylovKit. It allows for the Jacobian to be specified as a matrix (sparse / dense) or as a function.
apply(A::AbstractMatrix, x::AbstractVector) = A * x
apply(f, x) = f(x)

# apply!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector) = mul!(y, A, x)
# apply!(y, f, x) = copyto!(y, f(x))

# empty eigenvectors to save memory
# _empty(a::AbstractVector{T}, ::Type{U}=T) where {T,U} = Vector{U}()
# _empty(a::AbstractMatrix{T}, ::Type{U}=T) where {T,U} = similar(a, (0,0))
