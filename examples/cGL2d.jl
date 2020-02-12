using Revise
	using DiffEqOperators, ForwardDiff, IncompleteLU
	using PseudoArcLengthContinuation, LinearAlgebra, Plots, SparseArrays, Parameters, Setfield
	const PALC = PseudoArcLengthContinuation

norminf = x -> norm(x, Inf)

function Laplacian2D(Nx, Ny, lx, ly, bc = :Dirichlet)
	hx = 2lx/Nx
	hy = 2ly/Ny
	D2x = CenteredDifference(2, 2, hx, Nx)
	D2y = CenteredDifference(2, 2, hy, Ny)
	if bc == :Neumann
		Qx = Neumann0BC(hx)
		Qy = Neumann0BC(hy)
	elseif  bc == :Dirichlet
		Qx = Dirichlet0BC(typeof(hx))
		Qy = Dirichlet0BC(typeof(hy))
	else
		Qx = Dirichlet0BC(typeof(hx))
		Qy = Dirichlet0BC(typeof(hy))
	end
	D2xsp = sparse(D2x * Qx)[1]
	D2ysp = sparse(D2y * Qy)[1]
	if bc == :Periodic
		D2xsp[1,end] = D2xsp[1,2]
		D2xsp[end,1] = D2xsp[1,2]

		D2ysp[1,end] = D2ysp[1,2]
		D2ysp[end,1] = D2ysp[1,2]
	end
	A = kron(sparse(I, Ny, Ny), D2xsp) + kron(D2ysp, sparse(I, Nx, Nx))
	return A, D2x
end

function NL(u, p)
	@unpack r, μ, ν, c3, c5 = p
	n = div(length(u), 2)
	u1 = @view u[1:n]
	u2 = @view u[n+1:2n]

	ua = u1.^2 .+ u2.^2

	f = similar(u)
	f1 = @view f[1:n]
	f2 = @view f[n+1:2n]

	@. f1 .= r * u1 - ν * u2 - ua * (c3 * u1 - μ * u2) - c5 * ua^2 * u1
	@. f2 .= r * u2 + ν * u1 - ua * (c3 * u2 + μ * u1) - c5 * ua^2 * u2

	return f
end

function Fcgl!(f, u, p)
	mul!(f, p.Δ, u)
	f .= f .+ NL(u, p)
end

function Fcgl(u, p)
	f = similar(u)
	Fcgl!(f, u, p)
end

# computation of the first derivative
d1Fcgl(x, p, dx) = ForwardDiff.derivative(t -> Fcgl(x .+ t .* dx, p), 0.)

d1NL(x, p, dx) = ForwardDiff.derivative(t -> NL(x .+ t .* dx, p), 0.)

function dFcgl(x, p, dx)
	f = similar(dx)
	mul!(f, p.Δ, dx)
	nl = d1NL(x, p, dx)
	f .= f .+ nl
end


# computation of the second derivative
d2Fcgl(x, p, dx1, dx2) = ForwardDiff.derivative(t2 -> ForwardDiff.derivative( t1 -> Fcgl(x .+ t1 .* dx1 .+ t2 .* dx2, p), 0.), 0.)


# remark: I checked this against finite differences
function Jcgl(u, p)
	@unpack r, μ, ν, c3, c5, Δ = p

	n = div(length(u), 2)
	u1 = @view u[1:n]
	u2 = @view u[n+1:2n]

	ua = u1.^2 .+ u2.^2

	f1u = zero(u1)
	f2u = zero(u1)
	f1v = zero(u1)
	f2v = zero(u1)

	@. f1u =  r - 2 * u1 * (c3 * u1 - μ * u2) - c3 * ua - 4 * c5 * ua * u1^2 - c5 * ua^2
	@. f1v = -ν - 2 * u2 * (c3 * u1 - μ * u2)  + μ * ua - 4 * c5 * ua * u1 * u2
	@. f2u =  ν - 2 * u1 * (c3 * u2 + μ * u1)  - μ * ua - 4 * c5 * ua * u1 * u2
	@. f2v =  r - 2 * u2 * (c3 * u2 + μ * u1) - c3 * ua - 4 * c5 * ua * u2 ^2 - c5 * ua^2

	jacdiag = vcat(f1u, f2v)

	Δ + spdiagm(0 => jacdiag, n => f1v, -n => f2u)
end

####################################################################################################
Nx = 41*1
	Ny = 21*1
	n = Nx*Ny
	lx = pi
	ly = pi/2

	Δ = Laplacian2D(Nx, Ny, lx, ly)[1]
	par_cgl = (r = 0.5, μ = 0.1, ν = 1.0, c3 = -1.0, c5 = 1.0, Δ = blockdiag(Δ, Δ))
	sol0 = zeros(2Nx, Ny)

eigls = EigArpack(1.0, :LM)
	# eigls = eig_MF_KrylovKit(tol = 1e-8, dim = 60, x₀ = rand(ComplexF64, Nx*Ny), verbose = 1)
	opt_newton = PALC.NewtonPar(tol = 1e-9, verbose = true, eigsolver = eigls, maxIter = 20)
	out, hist, flag = @time PALC.newton(
		x ->  Fcgl(x, par_cgl),
		x ->  Jcgl(x, par_cgl),
		vec(sol0), opt_newton, normN = norminf)
####################################################################################################
# test for the Jacobian expression
# sol0 = rand(2Nx*Ny)
# J0 = ForwardDiff.jacobian(x-> Fcgl(x, par_cgl), sol0) |> sparse
# J1 = Jcgl(sol0, par_cgl)
# norm(J0 - J1, Inf)
####################################################################################################
opts_br = ContinuationPar(dsmin = 0.001, dsmax = 0.005, ds = 0.001, pMax = 2.5, detectBifurcation = 1, nev = 5, plotEveryNsteps = 50, newtonOptions = (@set opt_newton.verbose = false), maxSteps = 1060)

	br, u1 = @time PALC.continuation(
		(x, p) -> Fcgl(x, @set par_cgl.r = p),
		(x, p) -> Jcgl(x, @set par_cgl.r = p),
		vec(sol0), par_cgl.r,
		opts_br, verbosity = 0)
####################################################################################################
ind_hopf = 2
# number of time slices
M = 30
r_hopf, Th, orbitguess2, hopfpt, vec_hopf = PALC.guessFromHopf(br, ind_hopf, opt_newton.eigsolver, M, 22*sqrt(0.1); phase = 0.25)

orbitguess_f2 = reduce(hcat, orbitguess2)
orbitguess_f = vcat(vec(orbitguess_f2), Th) |> vec

poTrap = p -> PeriodicOrbitTrapProblem(
	x ->  Fcgl(x, p),
	x ->  Jcgl(x, p),
	real.(vec_hopf),
	hopfpt.u,
	M)

ls0 = GMRESIterativeSolvers(N = 2Nx*Ny, tol = 1e-9)#, Pl = lu(I + par_cgl.Δ))
poTrapMF = p -> PeriodicOrbitTrapProblem(
	x ->  Fcgl(x, p),
	x ->  (dx -> d1Fcgl(x, p, dx)),
	real.(vec_hopf),
	hopfpt.u,
	M, ls0)

poTrap(@set par_cgl.r = r_hopf - 0.1)(orbitguess_f) |> plot
poTrapMF(@set par_cgl.r = r_hopf - 0.1)(orbitguess_f) |> plot


plot();PALC.plotPeriodicPOTrap(orbitguess_f, M, Nx, Ny; ratio = 2);title!("")
deflationOp = DeflationOperator(2.0,(x,y) -> dot(x[1:end-1],y[1:end-1]),1.0,[zero(orbitguess_f)])

####################################################################################################
# circulant pre-conditioner
# Jpo = poTrap(@set par_cgl.r = r_hopf - 0.1)(Val(:JacFullSparse), orbitguess_f)[1:2n*M-2n,1:2n*M-2n]
# kΔ = spdiagm(0 => ones(M-1), -1 => ones(M-2), M-2 => [1])
# 	kI = spdiagm(0 => ones(M-1), -1 => -ones(M-2), M-2 => [-1])
#
# 	#diagonal precond
# 	# kΔ = spdiagm(0 => ones(M-1))
# 	# kI = spdiagm(0 => ones(M-1))
#
# 	h = orbitguess_f[end] / M
# 	Precs2 = kron(kI, spdiagm(0 => ones(2n))) ./1 -  h/2 * kron(kΔ, par_cgl.Δ)
# 	ls = GMRESIterativeSolvers(verbose = false, tol = 1e-4, N = size(Precs2,1), restart = 20, maxiter = 40, Pl = lu(Precs2), log=true)
# 	ls(Jpo, rand(ls.N))
####################################################################################################
# slow version DO NOT RUN!!!
# opt_po = (@set opt_po.eigsolver = eig_MF_KrylovKit(tol = 1e-4, x₀ = rand(2Nx*Ny), verbose = 2, dim = 20))
opt_po = (@set opt_po.eigsolver = DefaultEig())
opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.03, ds= 0.001, pMax = 2.5, maxSteps = 250, plotEveryNsteps = 3, newtonOptions = (@set opt_po.linsolver = DefaultLS()), computeEigenValues = true, nev = 5, precisionStability = 1e-7, detectBifurcation = 0)
	@assert 1==0 "Too much memory used!"
	br_pok2, upo , _= @time PALC.continuationPOTrap(
			p -> poTrap(@set par_cgl.r = p),
			orbitguess_f, r_hopf - 0.01,
			opts_po_cont, :FullLU;
			verbosity = 2,	plot = true,
			plotSolution = (x ;kwargs...) -> plotPeriodicPOTrap(x, M, Nx, Ny; kwargs...),
			printSolution = (u,p) -> PALC.amplitude(u, Nx*Ny, M), normC = norminf)
####################################################################################################
# we use an ILU based preconditioner for the newton method at the level of the full Jacobian of the PO functional
Jpo = poTrap(@set par_cgl.r = r_hopf - 0.01)(Val(:JacFullSparse), orbitguess_f)
Precilu = @time ilu(Jpo, τ = 0.003)
ls = GMRESIterativeSolvers(verbose = false, tol = 1e-3, N = size(Jpo,1), restart = 40, maxiter = 50, Pl = Precilu, log=true)
	ls(Jpo, rand(ls.N))

# ls = GMRESKrylovKit(verbose = 0, Pl = Precilu, rtol = 1e-3)
	ls(Jpo, rand(size(Jpo,1)))

opt_po = @set opt_newton.verbose = true
	outpo_f, _, flag = @time PALC.newton(poTrapMF(@set par_cgl.r = r_hopf - 0.01),
			orbitguess_f,
			(@set opt_po.linsolver = ls),
			:FullMatrixFree;
			normN = norminf,
			# callback = (x, f, J, res, iteration, options) -> (println("--> amplitude = ", PALC.amplitude(x, Nx*Ny, M; ratio = 2));true)
			)
	flag && printstyled(color=:red, "--> T = ", outpo_f[end], ", amplitude = ", PALC.amplitude(outpo_f, Nx*Ny, M; ratio = 2),"\n")
plot();PALC.plotPeriodicPOTrap(outpo_f, M, Nx, Ny; ratio = 2);title!("")

opt_po = @set opt_po.eigsolver = EigKrylovKit(tol = 1e-3, x₀ = rand(2n), verbose = 2, dim = 25)
# opt_po = @set opt_po.eigsolver = DefaultEig()
opt_po = @set opt_po.eigsolver = EigArpack(; tol = 1e-3, v0 = rand(2n))
opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.03, ds = 0.001, pMax = 2.2, maxSteps = 250, plotEveryNsteps = 3, newtonOptions = (@set opt_po.linsolver = ls), nev = 5, precisionStability = 1e-5, detectBifurcation = 2, dsminBisection =1e-7)
	br_po, _ , _= @time PALC.continuationPOTrap(
			p -> poTrapMF(@set par_cgl.r = p),
			outpo_f, r_hopf - 0.01,
			opts_po_cont, :FullMatrixFree;
			verbosity = 3,	plot = true,
			plotSolution = (x ;kwargs...) -> PALC.plotPeriodicPOTrap(x, M, Nx, Ny; ratio = 2, kwargs...),
			printSolution = (u, p) -> PALC.amplitude(u, Nx*Ny, M; ratio = 2), normC = norminf)

# branches = Any[br_po]
# push!(branches, br_po)
plotBranch([branches[1], br_po]; putbifptlegend = false, label="", xlabel="r",ylabel="Amplitude", legend = :bottomright);title!("")

###################################################################################################
# preconditioner taking into account the constraint
# Jpo = poTrap(@set par_cgl.r = r_hopf - 0.1)(Val(:JacFullSparse), orbitguess_f)[1:2n*M-2n,1:2n*M-2n]
# 	Jpo = blockdiag(Jpo, spdiagm(0 => ones(2Nx*Ny+1)) )
# Precilu = @time ilu(Jpo, τ = 0.004)
# ls = GMRESIterativeSolvers(verbose = false, tol = 1e-3, N = size(Jpo,1), restart = 30, maxiter = 50, Pl = Precilu, log=true)
# 	ls(Jpo, rand(ls.N))
####################################################################################################
# this preconditioner does not work very well here
Jpo = poTrap(@set par_cgl.r = r_hopf - 0.1)(Val(:JacCyclicSparse), orbitguess_f)
Precilu = @time ilu(Jpo, τ = 0.003)
ls = GMRESIterativeSolvers(verbose = false, tol = 1e-3, N = size(Jpo,1), restart = 30, maxiter = 50, Pl = Precilu, log=true)
	ls(Jpo, rand(ls.N))

opt_po = @set opt_newton.verbose = true
	outpo_f, hist, flag = @time PALC.newton(
			poTrapMF(@set par_cgl.r = r_hopf - 0.01),
			orbitguess_f,
			(@set opt_po.linsolver = ls), :BorderedMatrixFree;
			normN = norminf,
			# callback = (x, f, J, res, iteration, options; kwargs...) -> (println("--> amplitude = ", PALC.amplitude(x, Nx*Ny, M));@show kwargs;true)
			)
	plot();plotPeriodic(outpo_f);title!("")

function callbackPO(x, f, J, res, iteration, linsolver = ls, prob = poTrap, p = par_cgl; kwargs...)
	@show kwargs ls.N
	# we update the preconditioner every 10 continuation steps
	if mod(kwargs[:iterationC], 10) == 9 && iteration == 1
		@info "update Preconditioner"
		Jpo = poTrap(@set p.r = kwargs[:p])(Val(:JacCyclicSparse), x)
		Precilu = @time ilu(Jpo, τ = 0.003)
		ls.Pl = Precilu
	end
	true
end

opt_po = (@set opt_po.eigsolver = EigKrylovKit(tol = 1e-4, x₀ = rand(2n), verbose = 2, dim = 20))
opt_po = (@set opt_po.eigsolver = DefaultEig())
opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.03, ds= 0.001, pMax = 2.15, maxSteps = 450, plotEveryNsteps = 3, newtonOptions = (@set opt_po.linsolver = ls), computeEigenValues = true, nev = 5, precisionStability = 1e-7, detectBifurcation = 1)
# opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.01, ds= -0.001, pMax = 1.5, maxSteps = 400, plotEveryNsteps = 3, newtonOptions = (@set opt_po.linsolver = ls))
	br_pok2, upo , _= @time PALC.continuationPOTrap(
			p -> poTrap(@set par_cgl.r = p),
			outpo_f, r_hopf - 0.01,
			opts_po_cont, :BorderedMatrixFree;
			verbosity = 2,	plot = true,
			plotSolution = (x ;kwargs...) -> PALC.plotPeriodicPOTrap(x, M, Nx, Ny; kwargs...),
			callbackN = callbackPO,
			printSolution = (u, p) -> PALC.amplitude(u, Nx*Ny, M), normC = norminf)


# branches = Any[br_pok2]
# push!(branches, br_pok2)
# plotBranch(branches,label="", xlabel="r",ylabel="Amplitude");title!("")
####################################################################################################
# Jpo = poTrap(@set par_cgl.r = r_hopf - 0.1)(Val(:JacFullSparse), orbitguess_f)
# rhs = rand(size(Jpo,1))
# @time Jpo \ rhs
#
# using IterativeSolvers
# 	ls = GMRESIterativeSolvers(verbose = true, tol = 1e-3, N = size(Jpo,1), restart = 10, maxiter = 1000)
# 	ls(Jpo, rand(ls.N))
#
# n = Nx*Ny
# Jpo = @time  poTrap(@set par_cgl.r = r_hopf - 0.1)(Val(:JacFullSparse), orbitguess_f)[1:2n*M-2n,1:2n*M-2n]
# rhs = rand(size(Jpo,1))
# @time Jpo \ rhs
# ls = GMRESIterativeSolvers(verbose = true, tol = 1e-3, N = size(Jpo,1), restart = 10, maxiter = 1000)
# ls(Jpo, rand(ls.N))
#
# kΔ = spdiagm(0 => ones(M-1), -1 => ones(M-2), M-2 => [1])
# 	kI = spdiagm(0 => ones(M-1), -1 => -ones(M-2), M-2 => [-1])
# 	h = orbitguess_f[end] / M
# 	Jcglsp = Jcgl(orbitguess_f[1:2n], par_cgl)
# 	Precs2 = kron(kI, spdiagm(0 => ones(2n))) -  h/2 * kron(kΔ, par_cgl.Δ)
# 	ls = GMRESIterativeSolvers(verbose = true, tol = 1e-3, N = 2n*M-2n, restart = 20, maxiter = 1000, Pl = lu(Precs2), log=true)
#
# ls(Jpo, rand(ls.N))
####################################################################################################
####################################################################################################
# Experimental, full Inplace
function NL!(f, u, p, t = 0.)
	@unpack r, μ, ν, c3, c5 = p
	n = div(length(u), 2)
	u1v = @view u[1:n]
	u2v = @view u[n+1:2n]

	f1 = @view f[1:n]
	f2 = @view f[n+1:2n]

	@inbounds for ii = 1:n
		u1 = u1v[ii]
		u2 = u2v[ii]
		ua = u1^2+u2^2
		f1[ii] = r * u1 - ν * u2 - ua * (c3 * u1 - μ * u2) - c5 * ua^2 * u1
		f2[ii] = r * u2 + ν * u1 - ua * (c3 * u2 + μ * u1) - c5 * ua^2 * u2
	end
	return f
end

function dNL!(f, u, p, du)
	@unpack r, μ, ν, c3, c5 = p
	n = div(length(u), 2)
	u1v = @view u[1:n]
	u2v = @view u[n+1:2n]

	du1v = @view du[1:n]
	du2v = @view du[n+1:2n]

	f1 = @view f[1:n]
	f2 = @view f[n+1:2n]

	@inbounds for ii = 1:n
		u1 = u1v[ii]
		u2 = u2v[ii]
		du1 = du1v[ii]
		du2 = du2v[ii]
		ua = u1^2+u2^2
		f1[ii] = (-5*c5*u1^4 + (-6*c5*u2^2 - 3*c3)*u1^2 + 2*μ*u1*u2 - c5*u2^4 - c3*u2^2 + r) * du1 +
		(-4*c5*u2*u1^3 + μ*u1^2 + (-4*c5*u2^3 - 2*c3*u2)*u1 + 3*u2^2*μ - ν) * du2

		f2[ii] = (-4*c5*u2*u1^3 - 3*μ*u1^2 + (-4*c5*u2^3 - 2*c3*u2)*u1 - u2^2*μ + ν) * du1 + (-c5*u1^4 + (-6*c5*u2^2 - c3)*u1^2 - 2*μ*u1*u2 - 5*c5*u2^4 - 3*c3*u2^2 + r) * du2
	end

	return f
end

function Fcgl!(f, u, p, t = 0.)
	NL!(f, u, p)
	mul!(f, p.Δ, u, 1., 1.)
end

function dFcgl!(f, x, p, dx)
	# 19.869 μs (0 allocations: 0 bytes)
	dNL!(f, x, p, dx)
	mul!(f, p.Δ, dx, 1., 1.)
end

sol0f = vec(sol0)
out_ = similar(sol0f)
@time Fcgl!(out_, sol0f, par_cgl)
@time dFcgl!(out_, sol0f, par_cgl, sol0f)

ls0 = GMRESIterativeSolvers(N = 2Nx*Ny, tol = 1e-9)#, Pl = lu(I + par_cgl.Δ))
poTrapMFi = p -> PeriodicOrbitTrapProblem(
	(o, x) ->  Fcgl!(o, x, p),
	(o, x, dx) -> dFcgl!(o, x, p, dx),
	real.(vec_hopf),
	hopfpt.u,
	M, ls0; isinplace = true)

pb = poTrapMFi(@set par_cgl.r = r_hopf - 0.01)
outpo_f, _, flag = @time PALC.newton(pb,
	orbitguess_f, (@set opt_po.linsolver = ls),
	:FullMatrixFree; normN = norminf)

@assert 1==0 "tester map inplace dans gmresIS et voir si allocate"
@assert 1==0 "tester code_warntype dans POTrapFunctionalJac! st POTrapFunctional!"

lsi = GMRESIterativeSolvers!(verbose = false, N = length(orbitguess_f), tol = 1e-3, restart = 40, maxiter = 50, Pl = Precilu, log=true)

outpo_f, _, flag = @time PALC.newton(pb,
	orbitguess_f, (@set opt_po.linsolver = lsi),
	:FullMatrixFreeInplace; normN = norminf)



res = copy(orbitguess_f)
@code_warntype PALC.POTrapFunctional!(pb, res, orbitguess_f)
@code_warntype PALC.POTrapFunctionalJac!(pb, res, orbitguess_f, orbitguess_f)
using LinearMaps
Jmap! = LinearMap{Float64}((o, v) -> PALC.POTrapFunctionalJac!(pb, o, orbitguess_f, v), lsi.N, lsi.N ; ismutating = true)

@time mul!(res, Jmap!, orbitguess_f)

using IterativeSolvers

@time IterativeSolvers.gmres(Jmap!, orbitguess_f; tol = 1e-3, log = true, verbose = true, Pl = Precilu)

Jmap = LinearMap{Float64}((v) -> pb(orbitguess_f, v), lsi.N, lsi.N ; ismutating = false)
@time mul!(res, Jmap, orbitguess_f)
@time IterativeSolvers.gmres(Jmap, orbitguess_f; tol = 1e-3, log = true, verbose = true, Pl = Precilu)

@time IterativeSolvers.gmres(Jpo, orbitguess_f; tol = 1e-3, log = true, verbose = false, Pl = Precilu)

####################################################################################################
# Computation of Fold of limit cycle
function d2Fcglpb(f, x, dx1, dx2)
	return ForwardDiff.derivative(t2 -> ForwardDiff.derivative( t1-> f(x .+ t1 .* dx1 .+ t2 .* dx2,), 0.), 0.)
end

function JacT(F, x, v)
	# ForwardDiff.gradient(x -> dot(v, F(x)), x)
	ReverseDiff.gradient(x -> dot(v, F(x)), x)
end

function JacT2(F, x, v)
	# ForwardDiff.gradient(x -> dot(v, F(x)), x)
	Tracker.gradient(x -> dot(v, F(x)), x)
end


# dsol0 = rand(2n)
# 	dsol1 = rand(2n)
# 	sol0v = vec(sol0)
# 	@time d2Fcgl(sol0v, par_cgl, dsol0, dsol1)

# we look at the second fold point
indfold = 2
foldpt = PALC.FoldPoint(br_po, indfold)

Jpo = poTrap(@set par_cgl.r = r_hopf - 0.1)(Val(:JacFullSparse), orbitguess_f)
Precilu = @time ilu(Jpo, τ = 0.005)
ls = GMRESIterativeSolvers(verbose = false, tol = 1e-5, N = size(Jpo, 1), restart = 40, maxiter = 60, Pl = Precilu, log=true)
	ls(Jpo, rand(ls.N))

outfold, hist, flag = @time PALC.newtonFold(
		(x, p) -> poTrap(@set par_cgl.r = p)(x),
		(x, p) -> poTrap(@set par_cgl.r = p)(Val(:JacFullSparse), x),
		(x, p) -> transpose(poTrap(@set par_cgl.r = p)(Val(:JacFullSparse), x)),
		(x, p, dx1, dx2) -> d2Fcglpb(poTrap(@set par_cgl.r = p), x, dx1, dx2),
		br_pok2, indfold, #index of the fold point
		@set opt_po.linsolver = ls)
		# opt_po)
	flag && printstyled(color=:red, "--> We found a Fold Point at α = ", outfold.p," from ", br_pok2.bifpoint[indfold][3],"\n")

outfold, hist, flag = @time PALC.newtonFold(
		(x, p) -> poTrapMF(@set par_cgl.r = p)(x),
		(x, p) -> (dx -> poTrapMF(@set par_cgl.r = p)(x, dx)),
		(x, p) -> (dx -> JacT(x -> poTrapMF(@set par_cgl.r = p)(x), x, dx)),
		(x, p, dx1, dx2) -> d2Fcglpb(poTrap(@set par_cgl.r = p), x, dx1, dx2),
		br_pok2, indfold, #index of the fold point
		@set opt_po.linsolver = ls)
		# opt_po)
	flag && printstyled(color=:red, "--> We found a Fold Point at α = ", outfold.p," from ", br_pok2.bifpoint[indfold][3],"\n")



@time JacT( poTrapMF(@set par_cgl.r = 0.95), orbitguess_f, orbitguess_f)

@time JacT2( poTrapMF(@set par_cgl.r = 0.95), orbitguess_f, orbitguess_f)


optcontfold = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, pMax = 40.1, pMin = -10., newtonOptions = (@set opt_po.linsolver = ls), maxSteps = 20)

# optcontfold = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, pMax = 40.1, pMin = -10., newtonOptions = opt_po, maxSteps = 10)

outfoldco, hist, flag = @time PALC.continuationFold(
	(x, r, p) -> poTrap(setproperties(par_cgl, (r=r, c5=p)))(x),
	(x, r, p) -> poTrap(setproperties(par_cgl, (r=r, c5=p)))(Val(:JacFullSparse), x),
	(x, r, p) -> transpose(poTrap(setproperties(par_cgl, (r=r, c5=p)))(Val(:JacFullSparse), x)),
	p -> ((x, r, dx1, dx2) -> d2Fcglpb(poTrap(setproperties(par_cgl, (r=r, c5=p))), x, dx1, dx2)),
	br_pok2, indfold,
	par_cgl.c5, plot = true, verbosity = 2,
	optcontfold)

plotBranch(outfoldco, label="", xlabel="c5", ylabel="r");title!("")


####################################################################################################
# Continuation of periodic orbits on the GPU
using CuArrays
CuArrays.allowscalar(false)
import LinearAlgebra: mul!, axpby!
mul!(x::CuArray, y::CuArray, α::T) where {T <: Number} = (x .= α .* y)
mul!(x::CuArray, α::T, y::CuArray) where {T <: Number} = (x .= α .* y)
axpby!(a::T, X::CuArray, b::T, Y::CuArray) where {T <: Number} = (Y .= a .* X .+ b .* Y)

par_cgl_gpu = @set par_cgl.Δ = CuArrays.CUSPARSE.CuSparseMatrixCSC(par_cgl.Δ);
Jpo = poTrap(@set par_cgl.r = r_hopf - 0.01)(Val(:JacFullSparse), orbitguess_f)
Precilu = @time ilu(Jpo, τ = 0.003)

struct LUperso
	L
	Ut	# transpose of U in LU decomposition
end

import Base: ldiv!
# https://github.com/JuliaDiffEq/DiffEqBase.jl/blob/master/src/init.jl#L146-L150
function LinearAlgebra.ldiv!(_lu::LUperso, rhs::Array)
	@show "bla"
	_x = (_lu.Ut) \ ((_lu.L) \ rhs)
	rhs .= vec(_x)
	# CuArrays.unsafe_free!(_x)
	rhs
end

function LinearAlgebra.ldiv!(_lu::LUperso, rhs::CuArrays.CuArray)
	_x = UpperTriangular(_lu.Ut) \ (LowerTriangular(_lu.L) \ rhs)
	rhs .= vec(_x)
	CuArrays.unsafe_free!(_x)
	rhs
end

import PseudoArcLengthContinuation: extractPeriodFDTrap
extractPeriodFDTrap(x::CuArray) = x[end:end]

sol0_f = vec(sol0)
	sol0gpu = CuArray(sol0_f)
	_dxh = rand(length(sol0_f))
	_dxd = CuArray(_dxh)

	outh = Fcgl(sol0_f, par_cgl);
	outd = Fcgl(sol0gpu, par_cgl_gpu);
	@assert norm(outh-Array(outd), Inf) < 1e-12

	outh = dFcgl(sol0_f, par_cgl, _dxh);
	outd = dFcgl(sol0gpu, par_cgl_gpu, _dxd);
	@assert norm(outh-Array(outd), Inf) < 1e-12


orbitguess_cu = CuArray(orbitguess_f)
norm(orbitguess_f - Array(orbitguess_cu), Inf)


Precilu_gpu = LUperso(LowerTriangular(CuArrays.CUSPARSE.CuSparseMatrixCSR(I+Precilu.L)), UpperTriangular(CuArrays.CUSPARSE.CuSparseMatrixCSR(sparse(Precilu.U'))));

Precilu_host = LUperso((I+Precilu.L), (sparse(Precilu.U')));

rhs = rand(size(Jpo,1))
	sol_0 = Precilu \ rhs
	sol_1 = UpperTriangular(sparse(Precilu.U')) \ (LowerTriangular(I+Precilu.L)  \ (rhs))
	# sol_2 = LowerTriangular(Precilu.U') \ (LowerTriangular(sparse(I+Precilu.L))  \ (rhs))
	norm(sol_1-sol_0, Inf64)
	# norm(sol_2-sol_0, Inf64)

sol_0 = (I+Precilu.L) \ rhs
	sol_1 = LowerTriangular(CuArrays.CUSPARSE.CuSparseMatrixCSR(I+Precilu.L)) \ CuArray(rhs)
	@assert norm(sol_0-Array(sol_1), Inf64) < 1e-10

sol_0 = (Precilu.U)' \ rhs
	sol_1 = UpperTriangular(CuArrays.CUSPARSE.CuSparseMatrixCSR(sparse(Precilu.U'))) \ CuArray(rhs)
	norm(sol_0-Array(sol_1), Inf64)
	@assert norm(sol_0-Array(sol_1), Inf64) < 1e-10


sol_0 = Precilu \ rhs
	sol_1 = ldiv!(Precilu_host, copy(rhs))
	@assert norm(sol_1-sol_0, Inf64) < 1e-10

sol_0 = ldiv!(Precilu_gpu, copy(CuArray(rhs)));
	sol_1 = ldiv!(Precilu_host, copy(rhs))
	norm(sol_1-Array(sol_0), Inf64)
	@assert norm(sol_1-Array(sol_0), Inf64) < 1e-10


# matrix-free problem on the gpu
ls0gpu = GMRESKrylovKit(rtol = 1e-9)
poTrapMFGPU = p -> PeriodicOrbitTrapProblem(
	x ->  Fcgl(x, p),
	x ->  (dx -> dFcgl(x, p, dx)),
	CuArray(real.(vec_hopf)),
	CuArray(hopfpt.u),
	M, ls0gpu)

pb = poTrapMF(@set par_cgl.r = r_hopf - 0.1);
pbgpu = poTrapMFGPU(@set par_cgl_gpu.r = r_hopf - 0.1);

pbgpu(orbitguess_cu);
pbgpu(orbitguess_cu, orbitguess_cu);

ls = GMRESKrylovKit(verbose = 2, Pl = Precilu, rtol = 1e-3, dim  = 20)
	outh, _, _ = @time ls((Jpo), orbitguess_f) #0.4s

lsgpu = GMRESKrylovKit(verbose = 2, Pl = Precilu_gpu, rtol = 1e-3, dim  = 20)
	Jpo_gpu = CuArrays.CUSPARSE.CuSparseMatrixCSR(Jpo);
	outd, _, _ = @time lsgpu(Jpo_gpu, orbitguess_cu)

@assert norm(outh-Array(outd), Inf) < 1e-12


outh = @time pb(orbitguess_f);
	outd = @time pbgpu(orbitguess_cu);
	norm(outh-Array(outd), Inf)

_dxh = rand(length(orbitguess_f));
	_dxd = CuArray(_dxh);
	outh = @time pb(orbitguess_f, _dxh);
	outd = @time pbgpu(orbitguess_cu, _dxd);
	norm(outh-Array(outd), Inf)


outpo_f, hist, flag = @time PALC.newton(
		poTrapMF(@set par_cgl.r = r_hopf - 0.01),
		orbitguess_f,
		(@set opt_po.linsolver = ls), :FullMatrixFree;
		normN = x -> maximum(abs.(x)),
		# callback = (x, f, J, res, iteration, options) -> (println("--> amplitude = ", amplitude(x));true)
		) #14s
flag && printstyled(color=:red, "--> T = ", outpo_f[end], ", amplitude = ", amplitude(outpo_f, Nx*Ny, M),"\n")


opt_po = @set opt_newton.verbose = true
	outpo_f, hist, flag = @time PALC.newton(
			poTrapMFGPU(@set par_cgl_gpu.r = r_hopf - 0.01),
			orbitguess_cu,
			(@set opt_po.linsolver = lsgpu), :FullMatrixFree;
			normN = x -> maximum(abs.(x)),
			# callback = (x, f, J, res, iteration, options) -> (println("--> amplitude = ", PALC.amplitude(x, Nx*Ny, M));true)
			) #7s
	flag && printstyled(color=:red, "--> T = ", outpo_f[end:end], ", amplitude = ", amplitude(outpo_f, Nx*Ny, M),"\n")


opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.03, ds= 0.001, pMax = 2.2, maxSteps = 35, plotEveryNsteps = 3, newtonOptions = (@set opt_po.linsolver = lsgpu))
	br_pok2, upo , _= @time PALC.continuationPOTrap(
		p -> poTrapMFGPU(@set par_cgl_gpu.r = p),
		orbitguess_cu, r_hopf - 0.01,
		opts_po_cont, :FullMatrixFree;
		verbosity = 2,
		printSolution = (u,p) -> PALC.amplitude(u, Nx*Ny, M), normC = x -> maximum(abs.(x)))
