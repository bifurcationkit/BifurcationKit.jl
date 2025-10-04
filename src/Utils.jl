# plot backends
# this is important because we need to define plotting functions from the user provided ones ; for
# example to plot periodic orbits. However, Plots.jl and Makie.jl have different semantics 
# (plot(x;subplot = 1) instead of plot(ax, x)) which makes the above procedure difficult to implement.
abstract type AbstractPlotBackend end
struct BK_NoPlot <: AbstractPlotBackend end
struct BK_Plots <: AbstractPlotBackend end
struct BK_Makie <: AbstractPlotBackend end

"""
Internal function to select the keys out of nt that are valid for the continuation function below.
Can be used like `foo(kw...) = _keep_opts_cont(values(nt))`
"""
function _keep_opts_cont(nt) 
    return NamedTuple{filter(in((:kind,
                            :filename,
                            :plot,
                            :normC,
                            :finalise_solution,
                            :callback_newton,
                            :event,
                            :verbosity,
                            :bothside)), keys(nt))}(nt)
end
####################################################################################################
_empty(x) = empty(x)
_empty(::Nothing) = nothing
_empty(x::Matrix) = similar(x, 0, 0)
####################################################################################################
closesttozero(ev) = ev[sortperm(ev, by = abs)]
rightmost(ev) = ev[sortperm(ev, by = abs∘real)]
getinterval(a, b) = (min(a, b), max(a, b))
norm2sqr(x) = VI.inner(x, x)
####################################################################################################
# display eigenvals with color
function print_ev(eigenvals, color = :black)
    for r in eigenvals
        printstyled(color = color, r, "\n")
    end
end
####################################################################################################
# iterated derivatives
∂(f) = x -> ForwardDiff.derivative(f, x)
∂(f, ::Val{n}) where {n} = n == 0 ? f : ∂(∂(f), Val(n-1))
####################################################################################################
function print_nonlinear_step(step, residual, itlinear = 0, lastRow = false)
    if lastRow
        lastRow && println("└─────────────┴──────────────────────┴────────────────┘")
    else
        if step == 0
            println("\n┌─────────────────────────────────────────────────────┐")
              println("│ Newton step         residual      linear iterations │")
              println("├─────────────┬──────────────────────┬────────────────┤")
        end
        _print_line(step, residual, itlinear)
    end
end 

@inline _print_line(step::Int, residual::Real, itlinear::Tuple{Int, Int}) = @printf("|%8d     │ %16.4e     │ (%4d, %4d)   |\n", step, residual, itlinear[1], itlinear[2])
@inline _print_line(step::Int, residual::Real, itlinear::Int) = @printf("│%8d     │ %16.4e     │ %8d       │\n", step, residual, itlinear)
@inline _print_line(step::Int, residual::Nothing, itlinear::Int) = @printf("│%8d     │                      │ %8d       │\n", step, itlinear)
@inline _print_line(step::Int, residual::Nothing, itlinear::Tuple{Int, Int}) = @printf("│%8d     │                      │ (%4d, %4d)   │\n", step, itlinear[1], itlinear[2])
####################################################################################################
function compute_eigenvalues(iter::ContIterable, state, u0, par, nev = iter.contparams.nev; kwargs...)
    return iter.contparams.newton_options.eigsolver(jacobian(iter.prob, u0, par), nev; iter, state, kwargs...)
end

function compute_eigenvalues(iter::ContIterable, state::ContState; kwargs...)
    # we compute the eigen-elements
    n = state.n_unstable[2]
    nev_ = max(n + 5, iter.contparams.nev)
    @debug "Computing spectrum..."
    eiginfo = compute_eigenvalues(iter, state, getx(state), setparam(iter, getp(state)), nev_; kwargs...)
    (;isstable, n_unstable, n_imag) = is_stable(iter.contparams, eiginfo[1])
    return eiginfo, isstable, n_unstable, n_imag, eiginfo[3]
end

# same as previous but we save the eigen-elements in state
function compute_eigenvalues!(iter::ContIterable, state::ContState; kwargs...)
    eiginfo, _isstable, n_unstable, n_imag, cveig = compute_eigenvalues(iter, state; kwargs...)
    # we update the state
    update_stability!(state, n_unstable, n_imag, cveig)
    if isnothing(state.eigvals) == false
        state.eigvals = eiginfo[1]
    end
    if save_eigenvectors(iter)
        state.eigvecs = eiginfo[2]
    end
    # iteration number in eigen solver
    it_number = eiginfo[end]
    return it_number
end
####################################################################################################
"""
$(TYPEDSIGNATURES)

Compute a Jacobian by Finite Differences. Use the centered formula (f(x+δ)-f(x-δ))/2δ.
"""
function finite_differences(F, x::AbstractVector; δ = 1e-9)
    N = length(x)
    Nf = length(F(x))
    J = zeros(eltype(x), Nf, N)
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

"""
$(TYPEDSIGNATURES)

Same as finite_differences but with inplace `F`
"""
@views function finite_differences!(F, J, x::AbstractVector; δ = 1e-9, tmp = copy(x))
    x1 = copy(x)
    @inbounds for i in eachindex(x)
        x1[i] += δ
        F(J[:, i], x1)
        x1[i] -= 2δ
        F(tmp, x1)
        J[:, i] .= @. (J[:, i] - tmp) / (2δ)
        x1[i] += δ
    end
    return J
end
####################################################################################################
using BlockArrays, SparseArrays

function block_to_sparse(J::AbstractBlockArray)
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
$(TYPEDSIGNATURES)

This function extracts the indices of the blocks composing the matrix A which is a M x M Block matrix where each block N x N has the same sparsity.
"""
function get_blocks(A::SparseMatrixCSC, N, M)
    I, J, K = findnz(A)
    out = [Vector{Int}() for i in 1:M+1, j in 1:M+1];
    for k in eachindex(I)
        m, l = div(I[k]-1, N), div(J[k]-1, N)
        push!(out[1+m, 1+l], k)
    end
    return out
end
####################################################################################################
"""
$(TYPEDSIGNATURES)

This function implements a counter. If `everyN == 0`, it returns false. Otherwise, it returns `true` when `step` is a multiple of `everyN`
"""
function mod_counter(step, everyN)
    if step == 0; return false; end
    if everyN == 0; return false; end
    if everyN == 1; return true; end
    return mod(step, everyN) == 0
end
####################################################################################################
# this trick is extracted from KrylovKit. It allows for the Jacobian to be specified as a matrix (sparse / dense) or as a function.
apply(A::AbstractMatrix, x::AbstractVector) = A * x
apply(f, x) = f(x)

apply!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector) = LA.mul!(y, A, x)
apply!(y, f, x) = f(y, x)

# empty eigenvectors to save memory
# _empty(a::AbstractVector{T}, ::Type{U}=T) where {T,U} = Vector{U}()
# _empty(a::AbstractMatrix{T}, ::Type{U}=T) where {T,U} = similar(a, (0,0))
####################################################################################################
"""
$(TYPEDSIGNATURES)

Function to detect continuation branches which loop on themselves.
"""
function detect_loop(br::ContResult, x, p::T; rtol = T(1e-3), verbose::Bool = true) where T
    if verbose == false
        return false
    end
    N::Int = length(br)
    printstyled(color = :magenta, "\n    ┌─ Entry in detect_loop, rtol = $(convert(T, rtol))\n")
    out::Bool = false
    for bp in br.specialpoint[begin:end-1]
        printstyled(color = :magenta, "    ├─ bp type = ", Symbol(bp.type),
                    ", ||δx|| = ", norminf(minus(bp.x, x))::T, 
                    ", |δp| = ", abs(bp.param - p)::T,
                    " \n")
        if (norminf(minus(bp.x, x)) / norminf(x) < rtol) && isapprox(bp.param, p; rtol)
            out = true
            printstyled(color = :magenta, "    ├─\t Loop detected!, n = $N\n")
            break
        end
    end
    printstyled(color = :magenta, "    └─ Loop detected = $out\n")
    return out
end
detect_loop(br::ContResult, u; rtol = 1e-3, verbose = true) = detect_loop(br, u.x, u.param; rtol, verbose)
detect_loop(br::ContResult, ::Nothing; rtol = 1e-3, verbose = true) = detect_loop(br, br.specialpoint[end].x, br.specialpoint[end].param; rtol = rtol, verbose = verbose)
####################################################################################################
"""
$(TYPEDEF)

Structure to hold a specific finaliser and simplify dispatch on it. 
It is mainly used for periodic orbits computation and adaption of mesh and section.
It is meant to be called like a callable struct.
"""
struct Finaliser{Tp, Tf}
    "Bifurcation problem"
    prob::Tp
    "Finalizer to be called"
    finalise_solution::Tf
    "Section updated every updateSectionEveryStep step"
    updateSectionEveryStep::UInt
end

finalise_default(z, tau, step, contResult; k...) = true