using Revise
	using DiffEqOperators, ForwardDiff, DifferentialEquations
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

function NL(u, p)
	out = similar(u)
	NL!(out, u, p)
end

function Fcgl!(f, u, p, t = 0.)
	mul!(f, p.Δ, u)
	f .= f .+ NL(u, p)
end

function Fcgl(u, p, t = 0.)
	f = similar(u)
	Fcgl!(f, u, p, t)
end


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

####################################################################################################
# opts_br0 = ContinuationPar(dsmin = 0.001, dsmax = 0.005, ds = 0.001, pMax = 2.5, detectBifurcation = 1, nev = 5, plotEveryNsteps = 50, newtonOptions = opt_newton)
# 	opts_br0.newtonOptions.verbose = false
# 	opts_br0.maxSteps = 1060
#
# 	br, u1 = @time PALC.continuation(
# 		(x, p) -> Fcgl(x, @set par_cgl.r = p),
# 		(x, p) -> Jcgl(x, @set par_cgl.r = p),
# 		vec(sol0), par_cgl.r,
# 		opts_br0, verbosity = 0,
# 		plot = false)
####################################################################################################
# Look for periodic orbits
f1 = DiffEqArrayOperator(par_cgl.Δ)
f2 = NL!
prob_sp = SplitODEProblem(f1, f2, sol0_f, (0.0, 120.0), @set par_cgl.r = 1.2)
prob = ODEProblem(Fcgl, sol0_f, (0.0, 120.0), @set par_cgl.r = 1.2)#; jac = Jbr, jac_prototype = Jbr(sol0_f, par_cgl))
####################################################################################################
# sol = @time solve(prob, Vern9(); abstol=1e-14, reltol=1e-14)
sol = @time solve(prob_sp, ETDRK2(krylov=true); abstol=1e-14, reltol=1e-14, dt = 0.1)
# sol = @time solve(prob, LawsonEuler(krylov=true, m=50); abstol=1e-14, reltol=1e-14, dt = 0.1)
# sol = @time solve(prob_sp, CNAB2(linsolve=LinSolveGMRES()); abstol=1e-14, reltol=1e-14, dt = 0.03)
# sol = @time solve(prob, KenCarp4(); abstol=1e-12, reltol=1e-10)


plot(sol.t, [norm(v[1:Nx*Ny], Inf) for v in sol.u],xlims=(115,120))

# plotting the solution as a movie
for ii = 1:20:length(sol.t)
	# heatmap(reshape(sol[1:Nx*Ny,ii],Nx,Ny),title="$(sol.t[ii])") |> display
end

####################################################################################################
# this encodes the functional for the Shooting problem
probSh = p -> PALC.ShootingProblem(
	# pass the vector field and parameter (to be passed to the vector field)
	u -> Fcgl(u, p), p,

	# we pass the ODEProblem encoding the flow and the time stepper
	prob_sp, ETDRK2(krylov = true),

	# we pass M_{sh}
	1,

	# this is the phase condition
	x -> PALC.sectionShooting(x, Array(sol[:, end:end]), p, Fcgl);

	# these are options passed to the ODE time stepper
	atol = 1e-14, rtol = 1e-14, dt = 0.1)

initpo = vcat(sol(116.), 6.9) |> vec
	probSh(@set par_cgl.r = 1.2)(initpo) |> norminf

ls = GMRESIterativeSolvers(tol = 1e-4, N = 2Nx * Ny + 1, maxiter = 50, verbose = false)
	optn = NewtonPar(verbose = true, tol = 1e-9,  maxIter = 20, linsolver = ls)
outpo, _ = @time PALC.newton(
		x -> probSh(@set par_cgl.r = 1.2)(x),
		x -> (dx -> probSh(@set par_cgl.r = 1.2)(x, dx)),
		initpo, optn; normN = norminf,
		# callback = (x, f, J, res, iteration, options; kwargs...) -> (println("--> T = ",x[end]);x[end] = max(0.1,x[end]);x[end] = min(30.1,x[end]);true)
		)

heatmap(reshape(outpo[1:Nx*Ny], Nx, Ny), color = :viridis)

eig = EigKrylovKit(tol = 1e-7, x₀ = rand(2Nx*Ny), verbose = 2, dim = 40)
	opts_po_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.02, ds= -0.01, pMax = 1.5, maxSteps = 32, newtonOptions = (@set optn.eigsolver = eig), nev = 5, precisionStability = 1e-3, detectBifurcation = 2)
	br_po, upo , _= @time PALC.continuationPOShooting(
		p -> probSh(@set par_cgl.r = p),
		# (x, p) -> (dx -> probSh(@set par_cgl.r = p)(x, dx)),
		outpo, 1.2, opts_po_cont;
		verbosity = 3,
		plot = true,
		# callbackN = cb_ss,
		plotSolution = (x, p; kwargs...) -> heatmap!(reshape(x[1:Nx*Ny], Nx, Ny); color=:viridis, kwargs...),
		printSolution = (u, p) -> PALC.getAmplitude(probSh(@set par_cgl.r = p), u; ratio = 2), normC = norminf)
