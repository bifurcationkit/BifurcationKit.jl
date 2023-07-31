# plot backends
# this is important because we need to define plotting functions from the user provided ones ; for
# example to plot periodic orbits. However, Plots.jl and Makie.jl have different semantics 
# (plot(x;subplot = 1) instead of plot(ax, x)) which makes the above procedure difficult to implement.
abstract type AbstractPlotBackend end
struct BK_NoPlot <: AbstractPlotBackend end
struct BK_Plots <: AbstractPlotBackend end
struct BK_Makie <: AbstractPlotBackend end
get_plot_backend() = BK_NoPlot()
####################################################################################################
closesttozero(ev) = ev[sortperm(ev, by = abs)]
rightmost(ev) = ev[sortperm(ev, by = abs∘real)]
getinterval(a, b) = (min(a, b), max(a, b))
norm2sqr(x) = dot(x, x)
####################################################################################################
# display eigenvals with color
function printEV(eigenvals, color = :black)
	for r in eigenvals
		printstyled(color = color, r, "\n")
	end
end
####################################################################################################
function printNonlinearStep(step, residual, itlinear = 0, lastRow = false)
	if lastRow
		lastRow && println("└─────────────┴──────────────────────┴────────────────┘")
	else
		if step == 0
			println("\n┌─────────────────────────────────────────────────────┐")
			  println("│ Newton step         residual     linear iterations  │")
			  println("├─────────────┬──────────────────────┬────────────────┤")
		end
		_printLine(step, residual, itlinear)
	end
end 

@inline _printLine(step::Int, residual::Real, itlinear::Tuple{Int, Int}) = @printf("|%8d     │ %16.4e     │ (%4d, %4d)   |\n", step, residual, itlinear[1], itlinear[2])

@inline _printLine(step::Int, residual::Real, itlinear::Int) = @printf("│%8d     │ %16.4e     │ %8d       │\n", step, residual, itlinear)

@inline _printLine(step::Int, residual::Nothing, itlinear::Int) = @printf("│%8d     │                      │ %8d       │\n", step, itlinear)

@inline _printLine(step::Int, residual::Nothing, itlinear::Tuple{Int, Int}) = @printf("│%8d     │                      │ (%4d, %4d)   │\n", step, itlinear[1], itlinear[2])
####################################################################################################
function computeEigenvalues(it::ContIterable, state, u0, par, nev = it.contParams.nev; kwargs...)
	return it.contParams.newtonOptions.eigsolver(jacobian(it.prob, u0, par), nev; iter = it, state = state, kwargs...)
end

function computeEigenvalues(iter::ContIterable, state::ContState; kwargs...)
	# we compute the eigen-elements
	n = state.n_unstable[2]
	nev_ = max(n + 5, iter.contParams.nev)
	eiginfo = computeEigenvalues(iter, state, getx(state), setParam(iter, getp(state)), nev_; kwargs...)
	@unpack isstable, n_unstable, n_imag = isStable(iter.contParams, eiginfo[1])
	return eiginfo, isstable, n_unstable, n_imag, eiginfo[3]
end

# same as previous but we save the eigen-elements in state
function computeEigenvalues!(iter::ContIterable, state::ContState; saveEigenVec = true, kwargs...)
	eiginfo, _isstable, n_unstable, n_imag, cveig = computeEigenvalues(iter, state; kwargs...)
	# we update the state
	updateStability!(state, n_unstable, n_imag, cveig)
	if isnothing(state.eigvals) == false
		state.eigvals = eiginfo[1]
	end
	if saveEigenVec && saveEigenvectors(iter)
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
	@inbounds for i in eachindex(x)
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

function blockToSparse(J::AbstractBlockArray)
	nl, nc = size(J.blocks)
	# form the first line of blocks
	res = J[Block(1,1)]
	@inbounds for j in 2:nc
		res = hcat(res, J[Block(1,j)])
	end
	# continue with the other lines
	@inbounds for i in 2:nl
		line = J[Block(i,1)]
		for j in 2:nc
			line = hcat(line, J[Block(i,j)])
		end
		res = vcat(res,line)
	end
	return res
end
####################################################################################################
"""
$(SIGNATURES)

This function extracts the indices of the blocks composing the matrix A which is a M x M Block matrix where each block N x N has the same sparsity.
"""
function getBlocks(A::SparseMatrixCSC, N, M)
	I,J,K = findnz(A)
	out = [Vector{Int}() for i in 1:M+1, j in 1:M+1];
	for k in eachindex(I)
		m, l = div(I[k]-1, N), div(J[k]-1, N)
		push!(out[1+m, 1+l], k)
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
####################################################################################################
# the following structs are a machinery to extend multilinear mapping from Real valued to Complex valued Arrays
# this is done so as to use AD (ForwardDiff.jl,...) to provide the differentials which only works on reals (usually).

"""
$(TYPEDEF)

This structure wraps a bilinear map to allow evaluation on Complex arguments. This is especially useful when these maps are produced by ForwardDiff.jl.
"""
struct BilinearMap{Tm}
	bl::Tm
end

function (R2::BilinearMap)(dx1, dx2)
	dx1r = real.(dx1); dx2r = real.(dx2)
	dx1i = imag.(dx1); dx2i = imag.(dx2)
	return R2(dx1r, dx2r) .- R2(dx1i, dx2i) .+ im .* (R2(dx1r, dx2i) .+ R2(dx1i, dx2r))
end
(b::BilinearMap)(dx1::T, dx2::T) where {T <: AbstractArray{<: Real}} = b.bl(dx1, dx2)

"""
$(TYPEDEF)

This structure wraps a trilinear map to allow evaluation on Complex arguments. This is especially useful when these maps are produced by ForwardDiff.jl.
"""
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
"""
$(SIGNATURES)

Function to detect continuation branches which loop on themselves.
"""
function detectLoop(br::ContResult, x, p; rtol = 1e-3, verbose = true)
	verbose && printstyled(color = :magenta, "\n    ┌─ Entry in detectLoop, rtol = $rtol\n")
	N = length(br)
	out = false
	for bp in br.specialpoint[1:end-1]
		verbose && printstyled(color = :magenta, "    ├─ bp type = ",bp.type,", norm(δx) = ",norm(minus(bp.x, x), Inf),", norm(δp) = ",abs(bp.param - p)," \n")
		if (norm(minus(bp.x, x), Inf) / norm(x, Inf) < rtol) && isapprox(bp.param , p ; rtol = rtol)
			out = true
			verbose && printstyled(color = :magenta, "    └─ Loop detected!, n = $N\n")
			break
		end
	end
	verbose && printstyled(color = :magenta, "    └─ Loop detected = $out\n")
	return out
end
detectLoop(br::ContResult, u; rtol = 1e-3, verbose = true) = detectLoop(br, u.x, u.param; rtol = rtol, verbose = verbose)
detectLoop(br::ContResult, ::Nothing; rtol = 1e-3, verbose = true) = detectLoop(br, br.specialpoint[end].x, br.specialpoint[end].param; rtol = rtol, verbose = verbose)
