using Revise
	using DiffEqOperators, ForwardDiff, DifferentialEquations
	using BifurcationKit, LinearAlgebra, Plots, SparseArrays, Parameters, LoopVectorization
	const BK = BifurcationKit

norminf(x) = norm(x, Inf)

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
	end
	D2xsp = sparse(D2x * Qx)[1]
	D2ysp = sparse(D2y * Qy)[1]

	A = kron(sparse(I, Ny, Ny), D2xsp) + kron(D2ysp, sparse(I, Nx, Nx))
	return A, D2x
end

function NL!(f, u, p, t = 0.)
	@unpack r, μ, ν, c3, c5 = p
	n = div(length(u), 2)
	u1 = @view u[1:n]
	u2 = @view u[n+1:2n]

	ua = u1.^2 .+ u2.^2

	f1 = @view f[1:n]
	f2 = @view f[n+1:2n]

	@. f1 .= r * u1 - ν * u2 - ua * (c3 * u1 - μ * u2) - c5 * ua^2 * u1
	@. f2 .= r * u2 + ν * u1 - ua * (c3 * u2 + μ * u1) - c5 * ua^2 * u2

	return f
end

NL(u, p) = NL!(similar(u), u, p)

function Fcgl!(f, u, p, t = 0.)
	mul!(f, p.Δ, u)
	f .= f .+ NL(u, p)
end

Fcgl(u, p, t = 0.) = Fcgl!(similar(u), u, p, t)

# computation of the first derivative
# d1Fcgl(x, p, dx) = ForwardDiff.derivative(t -> Fcgl(x .+ t .* dx, p), 0.)

d1NL(x, p, dx) = ForwardDiff.derivative(t -> NL(x .+ t .* dx, p), 0.)

function dFcgl(x, p, dx)
	f = similar(dx)
	mul!(f, p.Δ, dx)

	nl = d1NL(x, p, dx)
	f .= f .+ nl
end

function Jcgl(u, p, t = 0.)
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
	sol0 = 0.1rand(2Nx, Ny)
	sol0_f = vec(sol0)

prob = BK.BifurcationProblem(Fcgl, sol0_f, par_cgl, (@lens _.r); J = Jcgl)
####################################################################################################
eigls = EigArpack(1.0, :LM)
# eigls = eig_MF_KrylovKit(tol = 1e-8, dim = 60, x₀ = rand(ComplexF64, Nx*Ny), verbose = 1)
opt_newton = NewtonPar(tol = 1e-9, verbose = true, eigsolver = eigls, maxIter = 20)
opts_br = ContinuationPar(dsmax = 0.02, ds = 0.01, pMax = 2., detectBifurcation = 3, nev = 15, newtonOptions = (@set opt_newton.verbose = false), nInversion = 6)

	br = @time continuation(prob, PALC(), opts_br, verbosity = 0)

plot(br)
####################################################################################################
# Look for periodic orbits
f1 = DiffEqArrayOperator(par_cgl.Δ)
f2 = NL!
prob_sp = SplitODEProblem(f1, f2, sol0_f, (0.0, 120.0), @set par_cgl.r = 1.2; reltol = 1e-8, dt = 0.1)
prob = ODEProblem(Fcgl, sol0_f, (0.0, 120.0), (@set par_cgl.r = 1.2))#, jac = Jcgl, jac_prototype = Jcgl(sol0_f, par_cgl))
####################################################################################################
# sol = @time solve(prob, Vern9(); abstol=1e-14, reltol=1e-14)
sol = @time solve(prob_sp, ETDRK2(krylov=true); abstol=1e-14, reltol=1e-14, dt = 0.1) #1.78s
# sol = @time solve(prob, LawsonEuler(krylov=true, m=50); abstol=1e-14, reltol=1e-14, dt = 0.1)
# sol = @time solve(prob_sp, CNAB2(linsolve=LinSolveGMRES()); abstol=1e-14, reltol=1e-14, dt = 0.03)

plot(sol.t, [norm(v[1:Nx*Ny], Inf) for v in sol.u], xlims=(105, 120))

# plotting the solution as a movie
for ii = 1:20:length(sol.t)
	# heatmap(reshape(sol[1:Nx*Ny,ii],Nx,Ny),title="$(sol.t[ii])") |> display
end

####################################################################################################
# this encodes the functional for the Shooting problem
probSh = ShootingProblem(
	# we pass the ODEProblem encoding the flow and the time stepper
	prob_sp, ETDRK2(krylov = true),
	[sol[:, end]], abstol = 1e-10, reltol = 1e-8,
	lens = (@lens _.r),
	jacobian = :FiniteDifferences)

@assert BK.getParams(probSh) == @set par_cgl.r = 1.2

initpo = vcat(sol[end], 6.3) |> vec
	probSh(initpo, @set par_cgl.r = 1.2) |> norminf

ls = GMRESIterativeSolvers(reltol = 1e-4, N = 2n + 1, maxiter = 50, verbose = false)
	optn = NewtonPar(verbose = true, tol = 1e-9,  maxIter = 25, linsolver = ls)
outpo = @time newton(probSh, initpo, optn; normN = norminf)
BK.getPeriod(probSh, outpo.u, BK.getParams(probSh))

heatmap(reshape(outpo.u[1:Nx*Ny], Nx, Ny), color = :viridis)

eig = EigKrylovKit(tol = 1e-7, x₀ = rand(2Nx*Ny), verbose = 2, dim = 40)
	opts_po_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.03, ds= -0.01, pMax = 2.5, maxSteps = 32, newtonOptions = (@set optn.eigsolver = eig), nev = 15, tolStability = 1e-3, detectBifurcation = 0, plotEveryStep = 1)
br_po = @time continuation(probSh, outpo.u, PALC(),
		opts_po_cont;
		verbosity = 3,
		plot = true,
		linearAlgo = MatrixFreeBLS(@set ls.N = probSh.M*2n+2),
		plotSolution = (x, p; kwargs...) -> heatmap!(reshape(x[1:Nx*Ny], Nx, Ny); color=:viridis, kwargs...),
		recordFromSolution = (u, p; k...) -> (amp = BK.getAmplitude(p.prob, u, (@set par_cgl.r = p.p); ratio = 2), period = u[end]),
		normC = norminf)

####################################################################################################
# automatic branch switching
ls = GMRESIterativeSolvers(reltol = 1e-4, maxiter = 50, verbose = false)
	optn = NewtonPar(verbose = true, tol = 1e-9,  maxIter = 25, linsolver = ls)
eig = EigKrylovKit(tol = 1e-7, x₀ = rand(2Nx*Ny), verbose = 2, dim = 40)
	opts_po_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.02, ds= 0.01, pMax = 2.5, maxSteps = 32, newtonOptions = (@set optn.eigsolver = eig), nev = 15, tolStability = 1e-3, detectBifurcation = 0, plotEveryStep = 1)

Mt = 1 # number of time sections
br_po = continuation(
	br, 2,
	# arguments for continuation
	opts_po_cont,
	ShootingProblem(Mt, prob_sp, ETDRK2(krylov = true); abstol = 1e-10, reltol = 1e-8, jacobian = :FiniteDifferences) ;
	verbosity = 3, plot = true, ampfactor = 1.5, δp = 0.01,
	# callbackN = (x, f, J, res, iteration, itl, options; kwargs...) -> (println("--> amplitude = ", BK.amplitude(x, n, M; ratio = 2));true),
	linearAlgo = MatrixFreeBLS(@set ls.N = Mt*2n+2),
	finaliseSolution = (z, tau, step, contResult; k...) ->begin
		BK.haseigenvalues(contResult) && Base.display(contResult.eig[end].eigenvals)
		return true
	end,
	recordFromSolution = (u, p; k...) -> (amp = BK.getAmplitude(p.prob, u, (@set par_cgl.r = p.p); ratio = 2), period = u[end]),
	normC = norminf)
