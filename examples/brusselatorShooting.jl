using Revise
	using BifurcationKit, LinearAlgebra, Plots, SparseArrays, Setfield, Parameters
	const BK = BifurcationKit

f1(u, v) = u^2 * v
norminf = x -> norm(x, Inf)

function plotsol(x; kwargs...)
	N = div(length(x), 2)
	plot!(x[1:N], label="u"; kwargs...)
	plot!(x[N+1:2N], label="v"; kwargs...)
end

function Fbru!(f, x, p)
	@unpack α, β, D1, D2, l = p
	n = div(length(x), 2)
	h = 1.0 / (n+0); h2 = h*h
	c1 = D1 / l^2 / h2
	c2 = D2 / l^2 / h2

	u = @view x[1:n]
	v = @view x[n+1:2n]

	# Dirichlet boundary conditions
	f[1]   = c1 * (α	  - 2u[1] + u[2] ) + α - (β + 1) * u[1] + f1(u[1], v[1])
	f[end] = c2 * (v[n-1] - 2v[n] + β / α)			 + β * u[n] - f1(u[n], v[n])

	f[n]   = c1 * (u[n-1] - 2u[n] +  α   ) + α - (β + 1) * u[n] + f1(u[n], v[n])
	f[n+1] = c2 * (β / α  - 2v[1] + v[2])			 + β * u[1] - f1(u[1], v[1])

	for i=2:n-1
		  f[i] = c1 * (u[i-1] - 2u[i] + u[i+1]) + α - (β + 1) * u[i] + f1(u[i], v[i])
		f[n+i] = c2 * (v[i-1] - 2v[i] + v[i+1])			  + β * u[i] - f1(u[i], v[i])
	end
	return f
end

Fbru(x, p)= Fbru!(similar(x), x, p)

function Jbru_sp(x, p)
	@unpack α, β, D1, D2, l = p
	# compute the Jacobian using a sparse representation
	n = div(length(x), 2)
	h = 1.0 / n; h2 = h*h

	c1 = D1 / p.l^2 / h2
	c2 = D2 / p.l^2 / h2

	u = @view x[1:n]
	v = @view x[n+1:2n]

	diag   = zeros(eltype(x), 2n)
	diagp1 = zeros(eltype(x), 2n-1)
	diagm1 = zeros(eltype(x), 2n-1)

	diagpn = zeros(eltype(x), n)
	diagmn = zeros(eltype(x), n)

	@. diagmn = β - 2 * u * v
	@. diagm1[1:n-1] = c1
	@. diagm1[n+1:end] = c2

	@. diag[1:n]    = -2c1 - (β + 1) + 2 * u * v
	@. diag[n+1:2n] = -2c2 - u * u

	@. diagp1[1:n-1]   = c1
	@. diagp1[n+1:end] = c2

	@. diagpn = u * u
	J = spdiagm(0 => diag, 1 => diagp1, -1 => diagm1, n => diagpn, -n => diagmn)
	return J
end

function finalise_solution(z, tau, step, contResult)
	n = div(length(z.u), 2)
	printstyled(color=:red, "--> Solution constant = ", norm(diff(z.u[1:n])), " - ", norm(diff(z.u[n+1:2n])), "\n")
	return true
end

n = 100
####################################################################################################
# test for the Jacobian expression
# using ForwardDiff
# sol0 = rand(2n)
# J0 = ForwardDiff.jacobian(x-> Fbru(x, par_bru), sol0) |> sparse
# J1 = Jbru_sp(sol0, par_bru)
# J0 - J1
####################################################################################################
# different parameters to define the Brusselator model and guess for the stationary solution
par_bru = (α = 2., β = 5.45, D1 = 0.008, D2 = 0.004, l = 0.3)
	sol0 = vcat(par_bru.α * ones(n), par_bru.β/par_bru.α * ones(n))

# par_bru = (α = 2., β = 4.6, D1 = 0.0016, D2 = 0.008, l = 0.061)
# 	xspace = LinRange(0, par_bru.l, n)
# 	sol0 = vcat(		par_bru.α .+ 2 .* sin.(pi*xspace/par_bru.l),
# 			par_bru.β/par_bru.α .- 0.5 .* sin.(pi*xspace/par_bru.l))

# eigls = EigArpack(1.1, :LM)
# 	opt_newton = BK.NewtonPar(eigsolver = eigls)
# 	out, hist, flag = @time BK.newton(
# 		x ->  Fbru(x, par_bru),
# 		x -> Jbru_sp(x, par_bru),
# 		sol0, opt_newton, normN = norminf)
#
# 		plot();plotsol(out);plotsol(sol0, label = "sol0",line=:dash)
####################################################################################################
eigls = EigArpack(1.1, :LM)
opts_br_eq = ContinuationPar(dsmin = 0.001, dsmax = 0.02, ds = 0.005, pMax = 1.7, detectBifurcation = 3, nev = 21, plotEveryStep = 50, newtonOptions = NewtonPar(eigsolver = eigls, tol = 1e-9), nInversion = 4)

	br, = @time BK.continuation(
		Fbru, Jbru_sp, sol0, par_bru, (@lens _.l),
		opts_br_eq, verbosity = 0,
		plot = false,
		printSolution = (x, p) -> x[n÷2], normC = norminf)
#################################################################################################### Continuation of the Hopf Point using Jacobian expression
ind_hopf = 1
	# hopfpt = BK.HopfPoint(br, ind_hopf)

	hopfpoint, _, flag = @time newton(
		Fbru, Jbru_sp,
		br, ind_hopf, par_bru, (@lens _.l);
		normN = norminf)
	flag && printstyled(color=:red, "--> We found a Hopf Point at l = ", hopfpoint.p[1], ", ω = ", hopfpoint.p[2], ", from l = ", br.bifpoint[ind_hopf].param, "\n")
####################################################################################################Continuation of Periodic Orbit
M = 10
l_hopf, Th, orbitguess2, hopfpt, vec_hopf = BK.guessFromHopf(br, ind_hopf, opts_br_eq.newtonOptions.eigsolver, M, 22*0.075)
#
orbitguess_f2 = reduce(hcat, orbitguess2)
orbitguess_f = vcat(vec(orbitguess_f2), Th) |> vec
####################################################################################################
# essai Shooting
using DifferentialEquations, DiffEqOperators, ForwardDiff, Sundials

FOde(f, x, p, t) = Fbru!(f, x, p)
Jbru(x, dx, p) = ForwardDiff.derivative(t -> Fbru(x .+ t .* dx, p), 0.)
JacOde(J, x, p, t) = copyto!(J, Jbru_sp(x, p))

u0 = sol0 .+ 0.01 .* rand(2n)
par_hopf = (@set par_bru.l = br.bifpoint[1].param + 0.01)
# probsundials = ODEProblem(FOde, u0, (0., 520.), par_hopf) # gives 0.68s

jac_prototype = Jbru_sp(ones(2n), @set par_bru.β = 0)
	jac_prototype.nzval .= ones(length(jac_prototype.nzval))

# vf = ODEFunction(FOde; jac = (J,u,p,t) -> J .= Jbru_sp(u,p), jac_prototype = jac_prototype)
	# probsundials = ODEProblem(vf,  u0, (0.0, 520.), @set par_bru.l = br.bifpoint[1].param) # gives .37s

using SparseDiffTools, SparseArrays, DiffEqDiffTools
_colors = matrix_colors(jac_prototype)
# JlgvfColorsAD(J, u, p, colors = _colors) =  SparseDiffTools.forwarddiff_color_jacobian!(J, (out, x) -> Fbru!(out,x,p), u, colorvec = colors)
vf = ODEFunction(FOde; jac_prototype = jac_prototype, colorvec = _colors)
probsundials = ODEProblem(vf,  sol0, (0.0, 520.), par_bru) # gives 0.22s


sol = @time solve(probsundials, Rodas4P(); abstol = 1e-10, retol = 1e-8, tspan = (0., 520.), save_everystep = false)
####################################################################################################
M = 10
dM = 3
orbitsection = Array(orbitguess_f2[:, 1:dM:M])

initpo = vcat(vec(orbitsection), 3.0)

BK.plotPeriodicShooting(initpo[1:end-1], length(1:dM:M));title!("")

probSh = ShootingProblem(Fbru, par_hopf,
	probsundials, Rodas4P(),
	[orbitguess_f2[:,ii] for ii=1:dM:M]; abstol = 1e-10, retol = 1e-8, parallel = true)

res = @time probSh(initpo, par_hopf)
norminf(res)
res = probSh(initpo, par_hopf, initpo)
norminf(res)

ls = GMRESIterativeSolvers(tol = 1e-7, N = length(initpo), maxiter = 100, verbose = false)
	optn_po = NewtonPar(verbose = true, tol = 1e-9,  maxIter = 25, linsolver = ls)
	# deflationOp = BK.DeflationOperator(2.0, (x,y) -> dot(x[1:end-1], y[1:end-1]),1.0, [outpo])
	outpo, = @time newton(probSh,
			initpo, par_hopf, optn_po;
			normN = norminf)
	plot(initpo[1:end-1], label = "Initial guess")
	plot!(outpo[1:end-1], label = "solution") |> display
	println("--> amplitude = ", BK.amplitude(outpo, n, length(1:dM:M); ratio = 2))

opts_po_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, pMax = 1.5, maxSteps = 500, newtonOptions = (@set optn_po.tol = 1e-7), nev = 25, precisionStability = 1e-8, detectBifurcation = 0)

# simplified call
eig = EigKrylovKit(tol= 1e-12, x₀ = rand(2n), verbose = 0, dim = 40)
# eig = DefaultEig()
opts_po_cont_floquet = @set opts_po_cont.newtonOptions = NewtonPar(linsolver = ls, eigsolver = eig, tol = 1e-7, verbose = true)
opts_po_cont_floquet = setproperties(opts_po_cont_floquet; nev = 10, precisionStability = 3e-3, detectBifurcation = 0, maxSteps = 40, ds = 0.03, dsmax = 0.03, pMax = 2.0, tolBisectionEigenvalue = 0.)

br_po, = @time continuation(probSh,
		outpo, par_hopf, (@lens _.l),
		opts_po_cont_floquet,
		MatrixFreeBLS(@set ls.N = ls.N+1);
		updateSectionEveryStep = 2,
		verbosity = 3,
		plot = true,
		finaliseSolution = (z, tau, step, contResult; k...) ->
			(Base.display(contResult.eig[end].eigenvals) ;true),
		plotSolution = (x, p; kwargs...) -> BK.plotPeriodicShooting!(x[1:end-1], length(1:dM:M); kwargs...),
		printSolution = (u, p) -> u[end], normC = norminf)

####################################################################################################
# automatic branch switching with Shooting
using ForwardDiff
function D(f, x, p, dx)
	return ForwardDiff.derivative(t->f(x .+ t .* dx, p), 0.)
end
d1Fbru(x,p,dx1) = D((z, p0) -> Fbru(z, p0), x, p, dx1)
	d2Fbru(x,p,dx1,dx2) = D((z, p0) -> d1Fbru(z, p0, dx1), x, p, dx2)
	d3Fbru(x,p,dx1,dx2,dx3) = D((z, p0) -> d2Fbru(z, p0, dx1, dx2), x, p, dx3)

jet  = (Fbru, Jbru_sp, d2Fbru, d3Fbru)

# linear solvers
ls = GMRESIterativeSolvers(tol = 1e-7, maxiter = 100, verbose = false)
eig = EigKrylovKit(tol= 1e-12, x₀ = rand(2n), verbose = 0, dim = 40)
# newton parameters
optn_po = NewtonPar(verbose = true, tol = 1e-9,  maxIter = 25, linsolver = ls, eigsolver = eig)
# continuation parameters
opts_po_cont = ContinuationPar(dsmax = 0.03, ds= 0.01, pMax = 2.5, maxSteps = 100, newtonOptions = (@set optn_po.tol = 1e-7), nev = 15, precisionStability = 1e-3, detectBifurcation = 0, plotEveryStep = 2, saveSolEveryStep = 2)

Mt = 2
br_po, = continuation(
	jet...,	br, 2,
	# arguments for continuation
	opts_po_cont, ShootingProblem(Mt, par_bru, probsundials, Rodas4P(); abstol = 1e-10, retol = 1e-8, parallel = false);
	ampfactor = 1., δp = 0.005,
	verbosity = 3,	plot = true,
	updateSectionEveryStep = 1,
	linearAlgo = MatrixFreeBLS(@set ls.N = 2+2n*Mt),
	finaliseSolution = (z, tau, step, contResult; k...) ->
		(Base.display(contResult.eig[end].eigenvals) ;true),
	plotSolution = (x, p; kwargs...) -> plot!(x[1:end-1]; kwargs...),
	normC = norminf)

####################################################################################################
# Multiple Poincare Shooting with Hyperplane parametrization
dM = 5
normals = [Fbru(orbitguess_f2[:,ii], par_hopf)/(norm(Fbru(orbitguess_f2[:,ii], par_hopf))) for ii = 1:dM:M]
centers = [orbitguess_f2[:,ii] for ii = 1:dM:M]

probHPsh = PoincareShootingProblem(Fbru, par_hopf, probsundials, Rodas4P(), normals, centers; abstol = 1e-10, retol = 1e-8, parallel = true, δ = 1e-8)

initpo_bar = reduce(vcat, BK.projection(probHPsh, centers))

# P = @time PrecPartialSchurKrylovKit(dx -> probHPsh(vec(outpo_psh), par_hopf, dx), rand(length(vec(initpo_bar))), 25, :LM; verbosity = 2, krylovdim = 50)
# 	scatter(real.(P.eigenvalues), imag.(P.eigenvalues))
# 		plot!(1 .+ cos.(LinRange(0,2pi,100)), sin.(LinRange(0,2pi,100)))

ls = GMRESIterativeSolvers(tol = 1e-7, N = length(vec(initpo_bar)), maxiter = 500, verbose = false)#, Pr = P)
	optn = NewtonPar(verbose = true, tol = 1e-9,  maxIter = 30, linsolver = ls)
	outpo_psh, = @time newton(probHPsh,
		vec(initpo_bar), par_hopf, optn;
		normN = norminf)

plot(outpo_psh, label = "Solution")
	plot!(initpo_bar |> vec, label = "Initial guess")

BK.getPeriod(probHPsh, outpo_psh, par_hopf)

# simplified call
eig = EigKrylovKit(tol= 1e-12, x₀ = rand(2n-1), verbose = 0, dim = 40)
	opts_po_cont_floquet = ContinuationPar(dsmin = 0.0001, dsmax = 0.15, ds= 0.001, pMax = 2.5, maxSteps = 500, nev = 10, precisionStability = 1e-5, detectBifurcation = 3, plotEveryStep = 1)
	opts_po_cont_floquet = @set opts_po_cont_floquet.newtonOptions = NewtonPar(linsolver = ls, eigsolver = eig, tol = 1e-7, verbose = true, maxIter = 15)

br_po, = @time continuation(
	probHPsh,
	outpo_psh, par_hopf, (@lens _.l),
	opts_po_cont_floquet;
	linearAlgo = MatrixFreeBLS(@set ls.N = ls.N+1),
	verbosity = 3,
	plot = true,
	plotSolution = (x, p; kwargs...) -> BK.plot!(x; label="", kwargs...),
	updateSectionEveryStep = 2,
	finaliseSolution = (z, tau, step, contResult) ->
		(Base.display(contResult.eig[end].eigenvals) ;true),
	normC = norminf)

####################################################################################################
# automatic branch switching from Hopf point with Poincare Shooting
# linear solver
ls = GMRESIterativeSolvers(tol = 1e-7, maxiter = 100, verbose = false)
# newton parameters
optn_po = NewtonPar(verbose = true, tol = 1e-7,  maxIter = 25, linsolver = ls, eigsolver = eig = EigKrylovKit(tol= 1e-12, x₀ = rand(2n-1), verbose = 0, dim = 50))
# continuation parameters
opts_po_cont = ContinuationPar(dsmax = 0.03, ds= 0.005, pMax = 2.5, maxSteps = 100, newtonOptions = optn_po, nev = 10, precisionStability = 1e-5, detectBifurcation = 0, plotEveryStep = 2)

Mt = 1
	br_po, = continuation(
	jet...,	br, 1,
	# arguments for continuation
	opts_po_cont, PoincareShootingProblem(Mt, par_bru, probsundials, Rodas4P(); abstol = 1e-10, retol = 1e-8, parallel = true);
	linearAlgo = MatrixFreeBLS(@set ls.N = (2n-1)*Mt+1),
	ampfactor = 1.0, δp = 0.005,
	verbosity = 3,	plot = true, printPeriod = true,
	finaliseSolution = (z, tau, step, contResult; k...) ->
		(Base.display(contResult.eig[end].eigenvals) ;true),
	updateSectionEveryStep = 1,
	plotSolution = (x, p; kwargs...) -> BK.plotPeriodicShooting!(x[1:end-1], Mt; kwargs...),
	normC = norminf)
