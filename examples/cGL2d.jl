using Revise
	using DiffEqOperators, ForwardDiff, IncompleteLU
	using BifurcationKit, LinearAlgebra, Plots, SparseArrays, Parameters, Setfield
	const BK = BifurcationKit

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
	@unpack r, μ, ν, c3, c5, γ = p
	n = div(length(u), 2)
	u1 = @view u[1:n]
	u2 = @view u[n+1:2n]

	ua = u1.^2 .+ u2.^2

	f = similar(u)
	f1 = @view f[1:n]
	f2 = @view f[n+1:2n]

	@. f1 .= r * u1 - ν * u2 - ua * (c3 * u1 - μ * u2) - c5 * ua^2 * u1
	@. f2 .= r * u2 + ν * u1 - ua * (c3 * u2 + μ * u1) - c5 * ua^2 * u2 + γ

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
	par_cgl = (r = 0.5, μ = 0.1, ν = 1.0, c3 = -1.0, c5 = 1.0, Δ = blockdiag(Δ, Δ), γ = 0.)
	sol0 = zeros(2Nx, Ny)

eigls = EigArpack(1.0, :LM)
	# eigls = eig_MF_KrylovKit(tol = 1e-8, dim = 60, x₀ = rand(ComplexF64, Nx*Ny), verbose = 1)
	opt_newton = NewtonPar(tol = 1e-9, verbose = true, eigsolver = eigls, maxIter = 20)
	out, hist, flag = @time newton(Fcgl, Jcgl, vec(sol0), par_cgl, opt_newton, normN = norminf)
####################################################################################################
# test for the Jacobian expression
# sol0 = rand(2Nx*Ny)
# J0 = ForwardDiff.jacobian(x-> Fcgl(x, par_cgl), sol0) |> sparse
# J1 = Jcgl(sol0, par_cgl)
# norm(J0 - J1, Inf)
####################################################################################################
opts_br = ContinuationPar(dsmin = 0.001, dsmax = 0.15, ds = 0.001, pMax = 2.5, detectBifurcation = 3, nev = 9, plotEveryStep = 50, newtonOptions = (@set opt_newton.verbose = false), maxSteps = 1060, nInversion = 4)
	br, = @time continuation(Fcgl, Jcgl, vec(sol0), par_cgl, (@lens _.r), opts_br, verbosity = 2)
####################################################################################################
# normal form computation
using ForwardDiff

function D(f, x, p, dx)
	return ForwardDiff.derivative(t->f(x .+ t .* dx, p), 0.)
end

d1Fcgl(x,p,dx) = D(Fcgl, x, p, dx)
d2Fcgl(x,p,dx1,dx2) = D((z, p0) -> d1Fcgl(z, p0, dx1), x, p, dx2)
d3Fcgl(x,p,dx1,dx2,dx3) = D((z, p0) -> d2Fcgl(z, p0, dx1, dx2), x, p, dx3)
jet = (Fcgl, Jcgl, d2Fcgl, d3Fcgl)

hopfpt = computeNormalForm(jet..., br, 2)
####################################################################################################
ind_hopf = 1
# number of time slices
M = 30
r_hopf, Th, orbitguess2, hopfpt, vec_hopf = BK.guessFromHopf(br, ind_hopf, opt_newton.eigsolver, M, 22*sqrt(0.1); phase = 0.25)

orbitguess_f2 = reduce(hcat, orbitguess2)
orbitguess_f = vcat(vec(orbitguess_f2), Th) |> vec

poTrap = PeriodicOrbitTrapProblem(Fcgl, Jcgl, real.(vec_hopf), hopfpt.u, M, 2n)

ls0 = GMRESIterativeSolvers(N = 2n, reltol = 1e-9)#, Pl = lu(I + par_cgl.Δ))
poTrapMF = setproperties(poTrap; J = (x, p) ->  (dx -> d1Fcgl(x, p, dx)), linsolver = ls0)

poTrap(orbitguess_f, @set par_cgl.r = r_hopf - 0.1) |> plot
poTrapMF(orbitguess_f, @set par_cgl.r = r_hopf - 0.1) |> plot


plot();BK.plotPeriodicPOTrap(orbitguess_f, M, Nx, Ny; ratio = 2);title!("")
deflationOp = DeflationOperator(2.0, (x,y) -> dot(x[1:end-1],y[1:end-1]), 1.0, [zero(orbitguess_f)])

####################################################################################################
# circulant pre-conditioner
# Jpo = poTrap(Val(:JacCyclicSparse), orbitguess_f, (@set par_cgl.r = r_hopf - 0.1))
# kΔ = spdiagm(0 => ones(M-1), -1 => ones(M-2), M-2 => [1])
# 	kI = spdiagm(0 => ones(M-1), -1 => -ones(M-2), M-2 => [-1])
#
# 	#diagonal precond
# 	kΔ = spdiagm(0 => ones(M-1))
# 	kI = spdiagm(0 => ones(M-1))
#
# 	h = orbitguess_f[end] / M
# 	Precs2 = kron(kI, spdiagm(0 => ones(2n))) ./1 -  h/2 * kron(kΔ, par_cgl.Δ)
# 	ls = GMRESIterativeSolvers(verbose = true, reltol = 1e-4, N = size(Precs2,1), restart = 20, maxiter = 40, Pl = lu(Precs2), log=true)
# 	ls(Jpo, rand(ls.N))
####################################################################################################
#
# 									slow version DO NOT RUN!!!
#
####################################################################################################
# opt_po = (@set opt_po.eigsolver = eig_MF_KrylovKit(tol = 1e-4, x₀ = rand(2Nx*Ny), verbose = 2, dim = 20))
opt_po = (@set opt_po.eigsolver = DefaultEig())
opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.03, ds= 0.001, pMax = 2.5, maxSteps = 250, plotEveryStep = 3, newtonOptions = (@set opt_po.linsolver = DefaultLS()), nev = 5, precisionStability = 1e-7, detectBifurcation = 0)
	@assert 1==0 "Too much memory will be used!"
	br_pok2, upo , _= @time BK.continuation(
			poTrap, orbitguess_f, (@set par_cgl.r = p), (@lens _.r),
			opts_po_cont; linearPO = :FullLU
			verbosity = 2,	plot = true,
			plotSolution = (x ;kwargs...) -> plotPeriodicPOTrap(x, M, Nx, Ny; kwargs...),
			printSolution = (u,p) -> BK.amplitude(u, Nx*Ny, M), normC = norminf)
####################################################################################################
# we use an ILU based preconditioner for the newton method at the level of the full Jacobian of the PO functional
Jpo = @time poTrap(Val(:JacFullSparse), orbitguess_f, @set par_cgl.r = r_hopf - 0.01) # 0.5sec

Precilu = @time ilu(Jpo, τ = 0.005) # 2 sec
# P = @time lu(Jpo) # 97 sec

# @time Jpo \ rand(ls.N) # 97 sec

ls = GMRESIterativeSolvers(verbose = false, reltol = 1e-3, N = size(Jpo,1), restart = 40, maxiter = 50, Pl = Precilu, log=true)
	ls(Jpo, rand(ls.N))

# ls = GMRESKrylovKit(verbose = 0, Pl = Precilu, rtol = 1e-3)
	# @time ls(Jpo, rand(size(Jpo,1)))

opt_po = @set opt_newton.verbose = true
	outpo_f, _, flag = @time newton(poTrapMF,
			orbitguess_f, (@set par_cgl.r = r_hopf - 0.01),
			(@set opt_po.linsolver = ls); linearPO = :FullMatrixFree,
			normN = norminf,
			# callback = (x, f, J, res, iteration, options) -> (println("--> amplitude = ", BK.amplitude(x, Nx*Ny, M; ratio = 2));true)
			)
	flag && printstyled(color=:red, "--> T = ", outpo_f[end], ", amplitude = ", BK.amplitude(outpo_f, Nx*Ny, M; ratio = 2),"\n")
plot();BK.plotPeriodicPOTrap(outpo_f, M, Nx, Ny; ratio = 2);title!("")

opt_po = @set opt_po.eigsolver = EigKrylovKit(tol = 1e-3, x₀ = rand(2n), verbose = 2, dim = 25)
# opt_po = @set opt_po.eigsolver = DefaultEig()
opt_po = @set opt_po.eigsolver = EigArpack(; tol = 1e-3, v0 = rand(2n))
opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.03, ds = 0.001, pMax = 2.2, maxSteps = 250, plotEveryStep = 3, newtonOptions = (@set opt_po.linsolver = ls), nev = 5, precisionStability = 1e-5, detectBifurcation = 0 , dsminBisection =1e-7)

br_po, = @time continuation(
		poTrapMF, outpo_f, (@set par_cgl.r = r_hopf - 0.01), (@lens _.r),
		opts_po_cont; linearPO = :FullMatrixFree,
		verbosity = 3,	plot = true,
		plotSolution = (x, p;kwargs...) -> BK.plotPeriodicPOTrap(x, M, Nx, Ny; ratio = 2, kwargs...),
		printSolution = (u, p) -> BK.amplitude(u, Nx*Ny, M; ratio = 2), normC = norminf)

branches = Any[br_pok2]
# push!(branches, br_po)
plot(branches[1]; putbifptlegend = false, label="", xlabel="r", ylabel="Amplitude", legend = :bottomright)
###################################################################################################
# automatic branch switching from Hopf point
br_po, _ = continuation(
	# arguments for branch switching
	jet..., br, 1,
	# arguments for continuation
	opts_po_cont, poTrapMF;
	ampfactor = 3., linearPO = :FullMatrixFree,
	verbosity = 3,	plot = true,
	# callbackN = (x, f, J, res, iteration, itl, options; kwargs...) -> (println("--> amplitude = ", BK.amplitude(x, n, M; ratio = 2));true),
	finaliseSolution = (z, tau, step, contResult; k...) ->
	(Base.display(contResult.eig[end].eigenvals) ;true),
	plotSolution = (x, p; kwargs...) -> BK.plotPeriodicPOTrap(x, M, Nx, Ny; ratio = 2, kwargs...),
	printSolution = (u, p) -> BK.amplitude(u, Nx*Ny, M; ratio = 2), normC = norminf)

###################################################################################################
# preconditioner not taking into account the constraint
# Jpo = @time poTrap(Val(:JacCyclicSparse), orbitguess_f, @set par_cgl.r = r_hopf - 0.01) # 0.5sec
# Precilu = poTrap(Val(:BlockDiagSparse), orbitguess_f, @set par_cgl.r = r_hopf - 0.01)[1:end-2n,1:end-2n] |> lu
# Precilu = lu(blockdiag([par_cgl.Δ .+ par_cgl.r for  _=1:M-1]...))
# ls = GMRESIterativeSolvers(verbose = true, reltol = 1e-3, N = size(Jpo,1), restart = 30, maxiter = 50, Pl = Precilu, log=true)
# ls(Jpo, rand(ls.N))
###################################################################################################
# this preconditioner does not work very well here
Jpo = poTrap(Val(:JacCyclicSparse), orbitguess_f, (@set par_cgl.r = r_hopf - 0.1))
Precilu = @time ilu(Jpo, τ = 0.003)
ls = GMRESIterativeSolvers(verbose = false, reltol = 1e-3, N = size(Jpo,1), restart = 30, maxiter = 50, Pl = Precilu, log=true)
	ls(Jpo, rand(ls.N))

opt_po = @set opt_newton.verbose = true
	outpo_f, hist, flag = @time newton(poTrapMF,
			orbitguess_f, (@set par_cgl.r = r_hopf - 0.01),
			(@set opt_po.linsolver = ls); linearPO = :BorderedMatrixFree,
			normN = norminf)

function callbackPO(x, f, J, res, iteration, linsolver = ls, prob = poTrap, p = par_cgl; kwargs...)
	@show ls.N keys(kwargs)
	# we update the preconditioner every 10 continuation steps
	if mod(kwargs[:iterationC], 10) == 9 && iteration == 1
		@info "update Preconditioner"
		Jpo = poTrap(Val(:JacCyclicSparse), x, (@set p.r = kwargs[:p]))
		Precilu = @time ilu(Jpo, τ = 0.003)
		ls.Pl = Precilu
	end
	true
end

opt_po = (@set opt_po.eigsolver = EigKrylovKit(tol = 1e-4, x₀ = rand(2n), verbose = 2, dim = 20))
opt_po = (@set opt_po.eigsolver = DefaultEig())
opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.03, ds= 0.001, pMax = 2.15, maxSteps = 450, plotEveryStep = 3, newtonOptions = (@set opt_po.linsolver = ls), nev = 5, precisionStability = 1e-7, detectBifurcation = 1)
# opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.01, ds= -0.001, pMax = 1.5, maxSteps = 400, plotEveryStep = 3, newtonOptions = (@set opt_po.linsolver = ls))
	br_pok2, upo , _= @time continuation(
			poTrap, outpo_f, (@set par_cgl.r = r_hopf - 0.01), (@lens _.r),
			opts_po_cont; linearPO = :BorderedMatrixFree,
			verbosity = 2,	plot = true,
			plotSolution = (x, p;kwargs...) -> BK.plotPeriodicPOTrap(x, M, Nx, Ny; kwargs...),
			callbackN = callbackPO,
			printSolution = (u, p; kwargs...) -> BK.amplitude(u, Nx*Ny, M; ratio = 2 ),
			normC = norminf)


# branches = Any[br_pok2]
# push!(branches, br_pok2)
# plotBranch(branches,label="", xlabel="r",ylabel="Amplitude");title!("")
####################################################################################################
# Jpo = poTrap(Val(:JacFullSparse), orbitguess_f, (@set par_cgl.r = r_hopf - 0.1))
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


ls = GMRESIterativeSolvers(verbose = false, reltol = 1e-3, N = size(Jpo,1), restart = 40, maxiter = 50, Pl = Precilu, log=true)
	ls(Jpo, rand(ls.N))

ls0 = GMRESIterativeSolvers(N = 2Nx*Ny, reltol = 1e-9)#, Pl = lu(I + par_cgl.Δ))
poTrapMFi = PeriodicOrbitTrapProblem(
	Fcgl!, dFcgl!,
	real.(vec_hopf), hopfpt.u,
	M, 2n, ls0; isinplace = true)


@time poTrapMFi(orbitguess_f, par_cgl, orbitguess_f)



outpo_f, _, flag = @time newton(poTrapMFi,
	orbitguess_f, (@set par_cgl.r = r_hopf - 0.01), (@set opt_po.linsolver = ls); normN = norminf, linearPO = :FullMatrixFree)

@assert 1==0 "tester map inplace dans gmresIS et voir si allocate"
@assert 1==0 "tester code_warntype dans POTrapFunctionalJac! st POTrapFunctional!"

lsi = GMRESIterativeSolvers!(verbose = false, N = length(orbitguess_f), reltol = 1e-3, restart = 40, maxiter = 50, Pl = Precilu, log=true)

outpo_f, _, flag = @time newton(poTrapMFi,
	orbitguess_f, (@set par_cgl.r = r_hopf - 0.01), (@set opt_po.linsolver = lsi); normN = norminf, linearPO = :FullMatrixFree)

####################################################################################################
# Computation of Fold of limit cycle
function d2Fcglpb(f, x, dx1, dx2)
	return ForwardDiff.derivative(t2 -> ForwardDiff.derivative( t1 -> f(x .+ t1 .* dx1 .+ t2 .* dx2), 0.), 0.)
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
foldpt = BK.FoldPoint(br_po, indfold)

Jpo = poTrap(Val(:JacFullSparse), orbitguess_f, (@set par_cgl.r = r_hopf - 0.1))
Precilu = @time ilu(Jpo, τ = 0.005)
ls = GMRESIterativeSolvers(verbose = false, reltol = 1e-5, N = size(Jpo, 1), restart = 40, maxiter = 60, Pl = Precilu, log=true)
	ls(Jpo, rand(ls.N))

outfold, hist, flag = @time BK.newtonFold(
		(x, p) -> poTrap(x, p),
		(x, p) -> poTrap(Val(:JacFullSparse), x, p),
		br_po , indfold; #index of the fold point
		options = (@set opt_po.linsolver = ls),
		d2F = (x, p, dx1, dx2) -> d2Fcglpb(z -> poTrap(z, p), x, dx1, dx2))
	flag && printstyled(color=:red, "--> We found a Fold Point at α = ", outfold.p," from ", br_po.foldpoint[indfold].param,"\n")

optcontfold = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, pMax = 40.1, pMin = -10., newtonOptions = (@set opt_po.linsolver = ls), maxSteps = 20)

# optcontfold = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, pMax = 40.1, pMin = -10., newtonOptions = opt_po, maxSteps = 10)

outfoldco, hist, flag = @time BK.continuationFold(
	(x, p) -> poTrap(x, p),
	(x, p) -> poTrap(Val(:JacFullSparse), x, p),
	br_po, indfold, (@lens _.c5),
	optcontfold;
	d2F = (x, p, dx1, dx2) -> d2Fcglpb(z->poTrap(z,p), x, dx1, dx2),
	plot = true, verbosity = 2)

plot(outfoldco, label="", xlabel="c5", ylabel="r")


####################################################################################################
# Continuation of periodic orbits on the GPU
using CUDA
CUDA.allowscalar(false)
import LinearAlgebra: mul!, axpby!
mul!(x::CuArray, y::CuArray, α::T) where {T <: Number} = (x .= α .* y)
mul!(x::CuArray, α::T, y::CuArray) where {T <: Number} = (x .= α .* y)
axpby!(a::T, X::CuArray, b::T, Y::CuArray) where {T <: Number} = (Y .= a .* X .+ b .* Y)

par_cgl_gpu = @set par_cgl.Δ = CUDA.CUSPARSE.CuSparseMatrixCSC(par_cgl.Δ);
Jpo = poTrap(Val(:JacFullSparse), orbitguess_f, (@set par_cgl.r = r_hopf - 0.01))
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

function LinearAlgebra.ldiv!(_lu::LUperso, rhs::CuArray)
	_x = UpperTriangular(_lu.Ut) \ (LowerTriangular(_lu.L) \ rhs)
	rhs .= vec(_x)
	CUDA.unsafe_free!(_x)
	rhs
end

import BifurcationKit: extractPeriodFDTrap
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


Precilu_gpu = LUperso(LowerTriangular(CUDA.CUSPARSE.CuSparseMatrixCSR(I+Precilu.L)), UpperTriangular(CUDA.CUSPARSE.CuSparseMatrixCSR(sparse(Precilu.U'))));

Precilu_host = LUperso((I+Precilu.L), (sparse(Precilu.U')));

rhs = rand(size(Jpo,1))
	sol_0 = Precilu \ rhs
	sol_1 = UpperTriangular(sparse(Precilu.U')) \ (LowerTriangular(I+Precilu.L)  \ (rhs))
	# sol_2 = LowerTriangular(Precilu.U') \ (LowerTriangular(sparse(I+Precilu.L))  \ (rhs))
	norm(sol_1-sol_0, Inf64)
	# norm(sol_2-sol_0, Inf64)

sol_0 = (I+Precilu.L) \ rhs
	sol_1 = LowerTriangular(CUDA.CUSPARSE.CuSparseMatrixCSR(I+Precilu.L)) \ CuArray(rhs)
	@assert norm(sol_0-Array(sol_1), Inf64) < 1e-10

sol_0 = (Precilu.U)' \ rhs
	sol_1 = UpperTriangular(CUDA.CUSPARSE.CuSparseMatrixCSR(sparse(Precilu.U'))) \ CuArray(rhs)
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
poTrapMFGPU = PeriodicOrbitTrapProblem(
	Fcgl, dFcgl,
	CuArray(real.(vec_hopf)),
	CuArray(hopfpt.u),
	M, 2n, ls0gpu)

pb = poTrapMF(@set par_cgl.r = r_hopf - 0.1);
pbgpu = poTrapMFGPU(@set par_cgl_gpu.r = r_hopf - 0.1);

pbgpu(orbitguess_cu);
pbgpu(orbitguess_cu, orbitguess_cu);

ls = GMRESKrylovKit(verbose = 2, Pl = Precilu, rtol = 1e-3, dim  = 20)
	outh, = @time ls((Jpo), orbitguess_f) #0.4s

lsgpu = GMRESKrylovKit(verbose = 2, Pl = Precilu_gpu, rtol = 1e-3, dim  = 20)
	Jpo_gpu = CUDA.CUSPARSE.CuSparseMatrixCSR(Jpo);
	outd, = @time lsgpu(Jpo_gpu, orbitguess_cu)

@assert norm(outh-Array(outd), Inf) < 1e-12


outh = @time pb(orbitguess_f);
	outd = @time pbgpu(orbitguess_cu);
	norm(outh-Array(outd), Inf)

_dxh = rand(length(orbitguess_f));
	_dxd = CuArray(_dxh);
	outh = @time pb(orbitguess_f, _dxh);
	outd = @time pbgpu(orbitguess_cu, _dxd);
	norm(outh-Array(outd), Inf)


outpo_f, hist, flag = @time newton(
		poTrapMF(@set par_cgl.r = r_hopf - 0.01),
		orbitguess_f,
		(@set opt_po.linsolver = ls), :FullMatrixFree;
		normN = x -> maximum(abs.(x)),
		# callback = (x, f, J, res, iteration, options) -> (println("--> amplitude = ", amplitude(x));true)
		) #14s
flag && printstyled(color=:red, "--> T = ", outpo_f[end], ", amplitude = ", amplitude(outpo_f, Nx*Ny, M),"\n")


opt_po = @set opt_newton.verbose = true
	outpo_f, hist, flag = @time newton(
			poTrapMFGPU(@set par_cgl_gpu.r = r_hopf - 0.01),
			orbitguess_cu,
			(@set opt_po.linsolver = lsgpu), :FullMatrixFree;
			normN = x -> maximum(abs.(x)),
			# callback = (x, f, J, res, iteration, options) -> (println("--> amplitude = ", BK.amplitude(x, Nx*Ny, M));true)
			) #7s
	flag && printstyled(color=:red, "--> T = ", outpo_f[end:end], ", amplitude = ", amplitude(outpo_f, Nx*Ny, M),"\n")


opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.03, ds= 0.001, pMax = 2.2, maxSteps = 35, plotEveryStep = 3, newtonOptions = (@set opt_po.linsolver = lsgpu))
	br_pok2, upo , _= @time BK.continuationPOTrap(
		p -> poTrapMFGPU(@set par_cgl_gpu.r = p),
		orbitguess_cu, r_hopf - 0.01,
		opts_po_cont, :FullMatrixFree;
		verbosity = 2,
		printSolution = (u,p) -> BK.amplitude(u, Nx*Ny, M), normC = x -> maximum(abs.(x)))
