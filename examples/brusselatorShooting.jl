using Revise
	using PseudoArcLengthContinuation, LinearAlgebra, Plots, SparseArrays, Setfield, Parameters
	const PALC = PseudoArcLengthContinuation

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

function Fbru(x, p)
	f = similar(x)
	Fbru!(f, x, p)
end

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
# 	opt_newton = PALC.NewtonPar(eigsolver = eigls)
# 	out, hist, flag = @time PALC.newton(
# 		x ->  Fbru(x, par_bru),
# 		x -> Jbru_sp(x, par_bru),
# 		sol0, opt_newton, normN = norminf)
#
# 		plot();plotsol(out);plotsol(sol0, label = "sol0",line=:dash)
####################################################################################################
eigls = EigArpack(1.1, :LM)
opts_br_eq = ContinuationPar(dsmin = 0.001, dsmax = 0.02, ds = 0.01, pMax = 1.9, detectBifurcation = 2, nev = 21, plotEveryNsteps = 50, newtonOptions = NewtonPar(eigsolver = eigls, tol = 1e-9), maxSteps = 200, dsminBisection = 1e-7)

	br, _ = @time PALC.continuation(
		(x, p) ->    Fbru(x, @set par_bru.l = p),
		(x, p) -> Jbru_sp(x, @set par_bru.l = p),
		sol0, par_bru.l,
		opts_br_eq, verbosity = 0,
		plot = false,
		printSolution = (x, p) -> x[div(n,2)], normC = norminf)
#################################################################################################### Continuation of the Hopf Point using Jacobian expression
ind_hopf = 1
	# hopfpt = PALC.HopfPoint(br, ind_hopf)

	hopfpoint, _, flag = @time PALC.newtonHopf(
		(x, p) ->  Fbru(x, @set par_bru.l = p),
		(x, p) -> Jbru_sp(x, @set par_bru.l = p),
		br, ind_hopf,
		opts_br_eq.newtonOptions, normN = norminf)
	flag && printstyled(color=:red, "--> We found a Hopf Point at l = ", hopfpoint.p[1], ", ω = ", hopfpoint.p[2], ", from l = ", br.bifpoint[ind_hopf].param, "\n")
####################################################################################################Continuation of Periodic Orbit
M = 10

l_hopf, Th, orbitguess2, hopfpt, vec_hopf = PALC.guessFromHopf(br, ind_hopf, opts_br_eq.newtonOptions.eigsolver, M, 22*0.075)
#
orbitguess_f2 = reduce(hcat, orbitguess2)
orbitguess_f = vcat(vec(orbitguess_f2), Th) |> vec
####################################################################################################
# essai Shooting
using DifferentialEquations, DiffEqOperators, ForwardDiff, Sundials

FOde(f, x, p, t) = Fbru!(f, x, p)
Jbru(x, dx, p) = ForwardDiff.derivative(t -> Fbru(x .+ t .* dx, p), 0.)
JacOde(J, x, p, t) = copyto!(J, Jbru_sp(x, p))

u0 = vec(sol0) .+ 0.01 .* rand(2n)
par_hopf = (@set par_bru.l = l_hopf + 0.01)

jac_prototype = Jbru_sp(ones(sol0 |> length), @set par_bru.β = 0)
	jac_prototype.nzval .= ones(length(jac_prototype.nzval))

ff = ODEFunction(FOde, jac_prototype = JacVecOperator{Float64}(FOde, u0, par_hopf))
probsundials = ODEProblem(FOde, u0, (0., 5200.), par_hopf)

# vf = ODEFunction((u,p,t)->Fbru(u,p); jac = (u,p,t) -> Jbru_sp(u,p))
	# prob = ODEProblem(vf,  u0, (0.0, .1), par_hopf)


# heatmap(sol[:,:], color = :viridis)
####################################################################################################
M = 10
dM = 10
orbitsection = Array(orbitguess_f2[:, 1:dM:M])

initpo = vcat(vec(orbitsection), 3.0)

PALC.plotPeriodicShooting(initpo[1:end-1], length(1:dM:M));title!("")

# PALC.sectionShooting(initpo, Array(orbitguess_f2[:,1:dM:M]), par_hopf)

probSh = p -> PALC.ShootingProblem(u -> Fbru(u, p), p, probsundials, Rodas4P(),
		length(1:dM:M), x -> PALC.sectionShooting(x, Array(orbitguess_f2[:,1:dM:M]), p, Fbru); atol = 1e-10, rtol = 1e-8)

res = @time probSh(par_hopf)(initpo)
norminf(res)
res = probSh(par_hopf)(initpo, initpo)
norminf(res)


ls = GMRESIterativeSolvers(tol = 1e-7, N = length(initpo), maxiter = 100, verbose = false)
	# ls = GMRESKrylovKit{Float64}(verbose = 0, dim = 200, atol = 1e-9, rtol = 1e-5)
	optn_po = NewtonPar(verbose = true, tol = 1e-9,  maxIter = 25, linsolver = ls)
	# deflationOp = PALC.DeflationOperator(2.0, (x,y) -> dot(x[1:end-1], y[1:end-1]),1.0, [outpo])
	outpo ,_ = @time PALC.newton(probSh(par_hopf),
			initpo, optn_po;
			normN = norminf)
	plot(initpo[1:end-1], label = "Init guess")
	plot!(outpo[1:end-1], label = "sol")

opts_po_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, pMax = 1.5, maxSteps = 500, newtonOptions = (@set optn_po.tol = 1e-7), nev = 25, precisionStability = 1e-8, detectBifurcation = 0)
# 	br_po, _, _= @time PALC.continuationPOShooting(
# 		p -> probSh(@set par_hopf.l = p),
# 		outpo, par_hopf.l,
# 		opts_po_cont; verbosity = 2,
# 		plot = true,
# 		plotSolution = (x; kwargs...) -> PALC.plotPeriodicShooting!(x[1:end-1], length(1:dM:M); kwargs...),
# 		printSolution = (u, p) -> u[end], normC = norminf)

# simplified call
eig = EigKrylovKit(tol= 1e-12, x₀ = rand(2n), verbose = 0, dim = 40)
# eig = DefaultEig()
opts_po_cont_floquet = @set opts_po_cont.newtonOptions = NewtonPar(linsolver = ls, eigsolver = eig, tol = 1e-7, verbose = true)
opts_po_cont_floquet = setproperties(opts_po_cont_floquet; nev = 10, precisionStability = 1e-2, detectBifurcation = 2, maxSteps = 5000, ds = 0.03, dsmax = 0.03, pMax = 2.5)

br_po, _ , _ = @time PALC.continuationPOShooting(
		p -> probSh(@set par_hopf.l = p),
		outpo, par_hopf.l,
		opts_po_cont_floquet; verbosity = 3,
		plot = true,
		# callbackN = cb_ss,
		finaliseSolution = (z, tau, step, contResult) ->
			(Base.display(contResult.eig[end].eigenvals) ;true),
		plotSolution = (x, p; kwargs...) -> PALC.plotPeriodicShooting!(x[1:end-1], length(1:dM:M); kwargs...),
		printSolution = (u, p) -> u[end], normC = norminf)
####################################################################################################
# Multiple Poincare Shooting with Hyperplane parametrization
using ForwardDiff

function dprobHPsh(pb,x,dx)
	ForwardDiff.derivative(t -> pb(x .+ t .* dx), 0.)
end

dM = 10
normals = [Fbru(orbitguess_f2[:,ii], par_hopf)/(norm(Fbru(orbitguess_f2[:,ii], par_hopf))) for ii = 1:dM:M]
	centers = [orbitguess_f2[:,ii] for ii = 1:dM:M]

probHPsh = p -> PALC.PoincareShootingProblem(u -> Fbru(u, p), p, probsundials, Rodas4P(), normals, centers; atol = 1e-10, rtol = 1e-8)

hyper = probHPsh(par_hopf).psh.section
initpo_bar = zeros(size(orbitguess_f2,1)-1, length(normals))
	for ii=1:length(normals)
		initpo_bar[:, ii] .= PALC.R(hyper, centers[ii], ii)
	end

probHPsh(par_hopf)(vec(initpo_bar))
probHPsh(par_hopf)(vec(initpo_bar)) |> norminf


ls = GMRESIterativeSolvers(tol = 1e-11, N = length(vec(initpo_bar)), maxiter = 500, verbose = false)
	optn = NewtonPar(verbose = true, tol = 1e-9,  maxIter = 20, linsolver = ls)
	outpo_psh, _ = @time PALC.newton(probHPsh(par_hopf),
			vec(initpo_bar), optn,
			; normN = norminf)

plot(outpo_psh, label = "Solution")
	plot!(initpo_bar |> vec, label = "Init Cont")

PALC.getPeriod(probHPsh(par_hopf), outpo_psh)

# simplified call
eig = EigKrylovKit(tol= 1e-12, x₀ = rand(2n-1), verbose = 0, dim = 40)
	opts_po_cont_floquet = ContinuationPar(dsmin = 0.0001, dsmax = 0.05, ds= 0.001, pMax = 2.5, maxSteps = 500, nev = 10, precisionStability = 1e-5, detectBifurcation = 2, plotEveryNsteps = 1)

	opts_po_cont_floquet = @set opts_po_cont_floquet.newtonOptions = NewtonPar(linsolver = ls, eigsolver = eig, tol = 1e-9, verbose = true)


br_po, _ , _ = @time PALC.continuationPOShooting(
	p -> probHPsh(@set par_hopf.l = p),
	outpo_psh, 0.6,
	opts_po_cont_floquet; verbosity = 3,
	plot = true,
	plotSolution = (x, p; kwargs...) -> PALC.plot!(x; label="", kwargs...),
	# printSolution = (x, p) -> norminf(x),
	finaliseSolution = (z, tau, step, contResult) ->
		(Base.display(contResult.eig[end].eigenvals) ;true),
	normC = norminf)
