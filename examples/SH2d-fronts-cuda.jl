using Revise
using PseudoArcLengthContinuation, LinearAlgebra, Plots
const Cont = PseudoArcLengthContinuation
TY = Float64
AF = Array{TY}
#################################################################
using CuArrays
CuArrays.allowscalar(false)
import LinearAlgebra: mul!, axpby!
mul!(x::CuArray, y::CuArray, α::T) where {T <: Number} = (x .= α .* y)
mul!(x::CuArray, y::T, α::CuArray) where {T <: Number} = (x .= α .* y)
axpby!(a::T, X::CuArray, b::T, Y::CuArray) where {T <: Number} = (Y .= a .* X .+ b .* Y)

TY = Float64
AF = CuArray{TY}
#################################################################
# to simplify plotting of the solution
heatmapsol(x) = heatmap(reshape(Array(x), Nx, Ny)', color=:viridis)

Nx = 2^8
Ny = 2^8
lx = 4pi * 2
ly = 2*2pi/sqrt(3) * 2

X = -lx .+ 2lx/Nx * collect(0:Nx-1)
Y = -ly .+ 2ly/Ny * collect(0:Ny-1)

sol0 = [(cos(x) .+ cos(x/2) * cos(sqrt(3) * y/2) ) for x in X, y in Y]
		sol0 .= sol0 .- minimum(vec(sol0))
		sol0 ./= maximum(vec(sol0))
		sol0 = sol0 .- 0.25
		sol0 .*= 1.7
		heatmap(sol0, color=:viridis)

using AbstractFFTs, FFTW, KrylovKit
import Base: *, \

# Making the linear operator a subtype of Cont.LinearSolver is handy as we will use it
# in the Newton iterations.
struct SHLinearOp <: Cont.AbstractLinearSolver
	tmp_real         # temporary
	tmp_complex      # temporary
	l1
	fftplan
	ifftplan
end

struct SHEigOp <: Cont.AbstractEigenSolver
	sh::SHLinearOp
end

function SHLinearOp(Nx, lx, Ny, ly; AF = Array{TY})
	# AF is a type, it could be CuArray{TY} to run the following on GPU
	k1 = vcat(collect(0:Nx/2), collect(Nx/2+1:Nx-1) .- Nx)
	k2 = vcat(collect(0:Ny/2), collect(Ny/2+1:Ny-1) .- Ny)
	d2 = [(1-(pi/lx * kx)^2 - (pi/ly * ky)^2)^2 + 1. for kx in k1, ky in k2]
	tmpc = Complex.(AF(zeros(Nx, Ny)))
	return SHLinearOp(AF(zeros(Nx, Ny)), tmpc, AF(d2), plan_fft!(tmpc), plan_ifft!(tmpc))
end

function *(c::SHLinearOp, u)
	c.tmp_complex .= Complex.(u)
	c.fftplan * c.tmp_complex
	c.tmp_complex .= c.l1 .* c.tmp_complex
	c.ifftplan * c.tmp_complex
	c.tmp_real .= real.(c.tmp_complex)
	return copy(c.tmp_real)
end

function \(c::SHLinearOp, u)
	c.tmp_complex .= Complex.(u)
	c.fftplan * c.tmp_complex
	c.tmp_complex .=  c.tmp_complex ./ c.l1
	c.ifftplan * c.tmp_complex
	c.tmp_real .= real.(c.tmp_complex)
	return copy(c.tmp_real)
end

function (sh::SHLinearOp)(J, rhs)
	u, l, ν = J
	udiag = l .+ 1 .+ 2ν .* u .- 3 .* u.^2
	res, info = res, info = KrylovKit.linsolve( u -> -u .+ sh \ (udiag .* u), sh \ rhs, tol = 1e-9, maxiter = 6)
	return res, true, info.numops
end

# function (sheig::SHEigOp)(J, nev::Int)
# 	u, l, ν = J
# 	sh = sheig.sh
# 	udiag = l .+ 1 .+ 2ν .* u .- 3 .* u.^2
# 	N = size(u)
# 	vals, vec, info = KrylovKit.eigsolve( u -> -(sh * u) .+ (udiag .* u), AF(rand(eltype(u), N)), nev, :LR, tol = 1e-9, maxiter = 6, verbosity = 2)
# 	@show vals
# 	return vals, vec, true, info.numops
# end

function F_shfft(u, l = -0.15, ν = 1.3; shlop::SHLinearOp)
	return -(shlop * u) .+ ((l+1) .* u .+ ν .* u.^2 .- u.^3)
end

L = SHLinearOp(Nx, lx, Ny, ly, AF = AF)
# Leig = SHEigOp(L) # for eigenvalues computation

opt_new = Cont.NewtonPar(verbose = true, tol = 1e-8, linsolver = L)
	sol_hexa, hist, flag = @time Cont.newton(
				x -> F_shfft(x, -.1, 1.3, shlop = L),
				u -> (u, -0.1, 1.3),
				AF(sol0),
				opt_new, normN = x -> maximum(abs.(x)))
	println("--> norm(sol) = ", maximum(abs.(sol_hexa)))

#################################################################
deflationOp = DeflationOperator(2.0, (x, y)->dot(x, y), 1.0, [sol_hexa])

opt_new.maxIter = 250
outdef, _, flag, _ = @time Cont.newtonDeflated(
				x -> F_shfft(x, -.1, 1.3, shlop = L),
				u -> (u, -0.1, 1.3),
				0.4 .* sol_hexa .* AF([exp(-1(x+0lx)^2/25) for x in X, y in Y]),
				opt_new, deflationOp, normN = x-> maximum(abs.(x)))
		println("--> norm(sol) = ", norm(outdef))
		heatmapsol(outdef) |> display
		flag && push!(deflationOp, outdef)

#################################################################
opts_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.005, ds= -0.0015, pMax = -0.0, pMin = -1.0, theta = 0.5, plot_every_n_steps = 5, newtonOptions = opt_new, a = 0.5, detect_fold = true, detect_bifurcation = false)
	opts_cont.newtonOptions.tol = 1e-6
	opts_cont.newtonOptions.maxIter = 50
	opts_cont.maxSteps = 100

	opts_cont.computeEigenValues = false
	opts_cont.nev = 20

	br, u1 = @time Cont.continuation(
		(u, p) -> F_shfft(u, p, 1.3, shlop = L),
		(u, p) -> (u, p, 1.3),
		deflationOp.roots[2],
		-0.1,
		opts_cont, plot = true, verbosity = 2,
		plotsolution = (x;kwargs...)->heatmap!(reshape(Array(x), Nx, Ny)', color=:viridis, subplot=4),
		printsolution = x-> norm(x,2), normC = x-> maximum(abs.(x)))
