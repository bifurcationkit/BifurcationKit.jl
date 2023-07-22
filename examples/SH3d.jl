using Revise, Parameters, KrylovKit
using GLMakie
using BifurcationKit
using LinearAlgebra, SparseArrays, DiffEqOperators, LinearMaps
const BK = BifurcationKit

# GLMakie.inline!(true)

norminf(x) = norm(x, Inf)

contour3dMakie(x; k...) = GLMakie.contour(x;  k...)
contour3dMakie(x::AbstractVector; k...) = contour3dMakie(reshape(x,Nx,Ny,Nz); k...)
contour3dMakie(ax, x; k...) = (contour(ax, x;  k...))
contour3dMakie(ax, x::AbstractVector; k...) = contour3dMakie(ax, reshape(x,Nx,Ny,Nz); k...)
contour3dMakie!(ax, x; k...) = (contour!(ax, x;  k...))
contour3dMakie!(ax, x::AbstractVector; k...) = contour3dMakie!(ax, reshape(x,Nx,Ny,Nz); k...)

function Laplacian3D(Nx, Ny, Nz, lx, ly, lz, bc = :Neumann)
	speye(n) = sparse(I, n, n)
	hx = 2lx/Nx; hy = 2ly/Ny; hz = 2lz/Nz
	D2x = CenteredDifference{1}(2, 2, hx, Nx)
	D2y = CenteredDifference{1}(2, 2, hy, Ny)
	D2z = CenteredDifference{1}(2, 2, hz, Nz)
	Qx = Neumann0BC(hx); Qy = Neumann0BC(hy); Qz = Neumann0BC(hz)

	_A = kron(speye(Ny), sparse(D2x * Qx)[1]) + kron(sparse(D2y * Qy)[1], speye(Nx))
	A = kron(speye(Nz), _A) + kron(kron(sparse(D2z * Qz)[1], speye(Ny)), speye(Nx))
	return sparse(A), D2x
end

# main functional
function F_sh(u, p)
	@unpack l, ν, L1 = p
	return -(L1 * u) .+ (l .* u .+ ν .* u.^2 .- u.^3)
end

# differential of the functional
function dF_sh(u, p, du)
	@unpack l, ν, L1 = p
	return -(L1 * du) .+ (l .+ 2 .* ν .* u .- 3 .* u.^2) .* du
end

#Jacobian
function J_sh(u, p)
	@unpack l, ν, L1 = p
	return -L1  .+ spdiagm(0 => l .+ 2 .* ν .* u .- 3 .* u.^2)
end

# various differentials
d2F_sh(u, p, dx1, dx2) = (2 .* p.ν .* dx2 .- 6 .* dx2 .* u) .* dx1
d3F_sh(u, p, dx1, dx2, dx3) = (-6 .* dx2 .* dx3) .* dx1

# these types are useful to switch to GPU
TY = Float64
AF = Array{TY}
####################################################################################################
Nx = Ny = Nz = 22; N = Nx*Ny*Nz
lx = ly = lz = pi

X = -lx .+ 2lx/(Nx) * collect(0:Nx-1)
Y = -ly .+ 2ly/(Ny) * collect(0:Ny-1)
Z = -lz .+ 2lz/(Nz) * collect(0:Nz-1)

# initial guess for newton
sol0 = [(cos(x) .* cos(y )) for x in X, y in Y, z in Z]
	sol0 .= sol0 .- minimum(vec(sol0))
	sol0 ./= maximum(vec(sol0))
	sol0 .*= 1.7

# parameters for PDE
Δ, D2x = Laplacian3D(Nx, Ny, Nz, lx, ly, lz, :Neumann);
L1 = (I + Δ)^2;
par = (l = 0.1, ν = 1.2, L1 = L1);

Pr = cholesky(L1)
using SuiteSparse
LinearAlgebra.ldiv!(o::Vector, P::SuiteSparse.CHOLMOD.Factor{Float64}, v::Vector) = o .= -(P \ v)

# rtol must be small enough to pass the folds and to get precise eigenvalues
ls = GMRESKrylovKit(verbose = 0, rtol = 1e-9, maxiter = 150, ishermitian = true, Pl = Pr)
####################################################################################################
struct SH3dEig{Ts, Tσ} <: BK.AbstractEigenSolver
	ls::Ts
	σ::Tσ
end

BifurcationKit.geteigenvector(eigsolve::SH3dEig, vecs, n::Union{Int, Array{Int64,1}}) = vecs[n]

function (sheig::SH3dEig)(J, nev::Int; verbosity = 0, kwargs...)
	σ = sheig.σ
	nv = 30
	Jshift = du -> J(du) .- σ .* du
	A = du -> sheig.ls(Jshift, du)[1]
	# we adapt the krylov dimension as function of the requested eigenvalue number
	vals, vec, info = KrylovKit.eigsolve(A, AF(rand(Nx*Ny*Nz)), nev, :LM, tol = 1e-12, maxiter = 20, verbosity = verbosity, ishermitian = true, krylovdim = max(nv, nev + nv))
	vals2 = 1 ./vals .+ σ
	Ind = sortperm(vals2, by = real, rev = true)
	return vals2[Ind], vec[Ind], true, info.numops
end

eigSH3d = SH3dEig((@set ls.rtol = 1e-9), 0.1)

prob = BK.BifurcationProblem(F_sh, AF(vec(sol0)), par, (@lens _.l),
	J = (x, p) -> (dx -> dF_sh(x, p, dx)),
	plotSolution = (ax, x, p) -> contour3dMakie(ax, x),
	recordFromSolution = (x, p) -> (n2 = norm(x), n8 = norm(x, 8)),
	issymmetric = true)

optnew = NewtonPar(verbose = true, tol = 1e-8, maxIter = 20, linsolver = @set ls.verbose = 0)
@set! optnew.eigsolver = eigSH3d
@set! optnew.linsolver = DefaultLS()
sol_hexa = @time newton(prob, optnew)
println("--> norm(sol) = ", norm(sol_hexa.u, Inf64))

contour3dMakie(sol0)
contour3dMakie(sol_hexa.u)
###################################################################################################
optcont = ContinuationPar(dsmin = 0.0001, dsmax = 0.005, ds= -0.001, pMax = 0.15, pMin = -.1, newtonOptions = setproperties(optnew; tol = 1e-9, maxIter = 15), maxSteps = 146, detectBifurcation = 3, nev = 15, nInversion = 4, plotEveryStep  = 1)

	br = @time continuation(
		reMake(prob, u0 = 0prob.u0), PALC(tangent = Bordered(), bls = BorderingBLS(solver = optnew.linsolver, checkPrecision = false)), optcont;
		plot = true, verbosity = 1,
		normC = norminf,
		event = BK.FoldDetectEvent,
		# finaliseSolution = (z, tau, step, contResult; k...) -> begin
		# 	if length(contResult) == 1
		# 		pretty_table(contResult.branch)
		# 	else
		# 		pretty_table(contResult.branch, overwrite = true,)
		# 	end
		# 	true
		# end
		)

BK.plotBranch(br)
####################################################################################################
getNormalForm(br, 3; nev = 25)

br1 = @time continuation(br, 3, setproperties(optcont; saveSolEveryStep = 10, detectBifurcation = 3, pMax = 0.1, plotEveryStep = 5, dsmax = 0.01);
	plot = true, verbosity = 3,
	δp = 0.005,
	verbosedeflation = true,
	finaliseSolution = (z, tau, step, contResult; k...) -> begin
		if isnothing(br.eig) == true
			Base.display(contResult.eig[end].eigenvals)
		end
		true
	end,
	# callbackN = cb,
	normC = norminf)

BK.plotBranch(br, br1...)
BK.plotBranch(br1[15])

BK.plotBranch(br1...)

fig = Figure(resolution = (1200, 900))
	for i=1:min(25,length(br1))
		ix = div(i,5)+1; iy = i%5+1
		@show i, ix, iy
		ax = Axis3(fig[ix, iy], title = "$i", aspect = (1, 1, 1))
		hidedecorations!(ax, grid=false)
		contour3dMakie!(ax, br1[i].sol[2].x)
		ax.protrusions = (0, 0, 0, 10)
		# out = AbstractPlotting.contour!(ax, reshape(br1[i].sol[2].x, Nx, Ny, Nz))
		# @show out
		# Colorbar(fig[ix, iy], )
		# Colorbar(ax, hm)
	end
	display(fig)
