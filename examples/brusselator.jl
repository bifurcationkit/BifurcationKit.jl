using Revise
	using PseudoArcLengthContinuation, LinearAlgebra, Plots, SparseArrays, Setfield, Parameters
	const PALC = PseudoArcLengthContinuation

f1(u, v) = u * u * v
norminf = x -> norm(x, Inf)

function plotsol(x; kwargs...)
	N = div(length(x), 2)
	plot!(x[1:N], label="u"; kwargs...)
	plot!(x[N+1:2N], label="v"; kwargs...)
end

function Fbru!(f, x, p)
	@unpack α, β, D1, D2, l = p
	n = div(length(x), 2)
	h = 1.0 / n; h2 = h*h
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
	f
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

n = 500
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

# # parameters for an isola of stationary solutions
# par_bru = (α = 2., β = 4.6, D1 = 0.0016, D2 = 0.008, l = 0.061)
# 	xspace = LinRange(0, par_bru.l, n)
# 	sol0 = vcat(		par_bru.α .+ 2 .* sin.(pi*xspace/par_bru.l),
# 			par_bru.β/par_bru.α .- 0.5 .* sin.(pi*xspace/par_bru.l))

# eigls = EigArpack(1.1, :LM)
# 	opt_newton = PALC.NewtonPar(eigsolver = eigls)
# 	out, hist, flag = @time PALC.newton(
# 		x ->    Fbru(x, par_bru),
# 		x -> Jbru_sp(x, par_bru),
# 		sol0, opt_newton, normN = norminf)
#
# 	plot();plotsol(out);plotsol(sol0, label = "sol0",line=:dash)
####################################################################################################
eigls = EigArpack(1.1, :LM)
opts_br_eq = ContinuationPar(dsmin = 0.001, dsmax = 0.01, ds = 0.001, pMax = 1.9, detectBifurcation = 2, nev = 21, plotEveryNsteps = 50, newtonOptions = NewtonPar(eigsolver = eigls, tol = 1e-9), maxSteps = 1060, nInversion = 6, tolBisectionEigenvalue = 1e-4)

	br, _ = @time continuation(
		Fbru, Jbru_sp, sol0, par_bru, (@lens _.l),
		opts_br_eq, verbosity = 0,
		plot = true,
		plotSolution = (x, p; kwargs...) -> (plotsol(x; label="", kwargs... )),
		printSolution = (x, p) -> x[div(n,2)], normC = norminf)
#################################################################################################### Continuation of the Hopf Point using Jacobian expression
ind_hopf = 1
	# hopfpt = PALC.HopfPoint(br, ind_hopf)
	optnew = opts_br_eq.newtonOptions
	hopfpoint, _, flag = @time newton(
		Fbru, Jbru_sp,
		br, ind_hopf, par_bru, (@lens _.l);
		options = (@set optnew.verbose=true), normN = norminf)
	flag && printstyled(color=:red, "--> We found a Hopf Point at l = ", hopfpoint.p[1], ", ω = ", hopfpoint.p[2], ", from l = ", br.bifpoint[ind_hopf].param, "\n")

if 1==0
	br_hopf, u1_hopf = @time PALC.continuation(
		Fbru, Jbru_sp,
		br, ind_hopf, par_bru, (@lens _.l), (@lens _.β),
		ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, pMax = 6.5, pMin = 0.0, newtonOptions = optnew); verbosity = 2, normC = norminf)

	plot(br_hopf, xlabel="beta", ylabel = "l", label="")
end
# test with analytical Hessian but with dummy expression ;)
d2Fbru(x, p, dx1, dx2) = dx1 .* dx2

hopfpoint, hist, flag = @time PALC.newton(
	Fbru, Jbru_sp,
	br, ind_hopf, par_bru, (@lens _.l);
	options = (@set optnew.verbose = true),
	d2F = d2Fbru, normN = norminf)

if 1==0
	br_hopf, u1_hopf = @time PALC.continuation(
		Fbru, Jbru_sp,
		br, ind_hopf, par_bru, (@lens _.l), (@lens _.β),
		ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, pMax = 6.5, pMin = 0.0, newtonOptions = optnew); plot = true,
		d2F = d2Fbru, verbosity = 2, normC = norminf)
end
####################################################################################################
using ForwardDiff
function D(f, x, p, dx)
	return ForwardDiff.derivative(t->f(x .+ t .* dx, p), 0.)
end
d1Fbru(x,p,dx1) = D((z, p0) -> Fbru(z, p0), x, p, dx1)
	d2Fbru(x,p,dx1,dx2) = D((z, p0) -> d1Fbru(z, p0, dx1), x, p, dx2)
	d3Fbru(x,p,dx1,dx2,dx3) = D((z, p0) -> d2Fbru(z, p0, dx1, dx2), x, p, dx3)

jet  = (Fbru, Jbru_sp, d2Fbru, d3Fbru)

hopfpt = PALC.computeNormalForm(jet..., br, 1; verbose = true)
####################################################################################################Continuation of Periodic Orbit
# number of time slices
M = 51
l_hopf, Th, orbitguess2, hopfpt, vec_hopf = PALC.guessFromHopf(br, ind_hopf, opts_br_eq.newtonOptions.eigsolver, M, 2.7; phase = 0.25)
#
# orbitguess_f2 = orbitguess2[1]; for ii=2:M; global orbitguess_f2 = hcat(orbitguess_f2, orbitguess2[ii]);end
orbitguess_f2 = reduce(vcat, orbitguess2)
orbitguess_f = vcat(vec(orbitguess_f2), Th) |> vec

poTrap = PeriodicOrbitTrapProblem(Fbru, Jbru_sp, real.(vec_hopf), hopfpt.u, M)

poTrap(orbitguess_f,  @set par_bru.l = l_hopf + 0.01) |> plot
PALC.plotPeriodicPOTrap(orbitguess_f, n, M; ratio = 2)


using ForwardDiff
d1Fbru(x, p, dx) = ForwardDiff.derivative(t -> Fbru(x .+ t .* dx, p), 0.)

ls0 = GMRESIterativeSolvers(N = 2n, tol = 1e-9)#, Pl = lu(I + par_cgl.Δ))

poTrapMF = PeriodicOrbitTrapProblem(
			Fbru, (x, p) -> (dx -> d1Fbru(x, p, dx)),
			real.(vec_hopf),
			hopfpt.u,
			M, ls0)

deflationOp = DeflationOperator(2.0, (x,y) -> dot(x[1:end-1], y[1:end-1]),1.0, [zero(orbitguess_f)])
# deflationOp = DeflationOperator(2.0, (x,y) -> dot(x[1:end-1], y[1:end-1]),1.0, [outpo_f])
####################################################################################################
opt_po = PALC.NewtonPar(tol = 1e-10, verbose = true, maxIter = 14)
	outpo_f, _, flag = @time PALC.newton(poTrap,
			orbitguess_f, (@set par_bru.l = l_hopf + 0.01),
			opt_po,
			# deflationOp,
			# :FullLU;
			normN = norminf,
			callback = (x, f, J, res, iteration, itl, options; kwargs...) -> (println("--> amplitude = ", PALC.amplitude(x, n, M; ratio = 2));true)
			)
	flag && printstyled(color=:red, "--> T = ", outpo_f[end], ", amplitude = ", PALC.amplitude(outpo_f, n, M; ratio = 2),"\n")
	PALC.plotPeriodicPOTrap(outpo_f, n, M; ratio = 2)


opt_po = @set opt_po.eigsolver = EigKrylovKit(tol = 1e-5, x₀ = rand(2n), verbose = 2, dim = 40)
opt_po = @set opt_po.eigsolver = DefaultEig()
# opt_po = @set opt_po.eigsolver = EigArpack(; tol = 1e-5, v0 = rand(2n))
opts_po_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.01, ds= 0.01, pMax = 3.0, maxSteps = 2, newtonOptions = opt_po, saveSolEveryNsteps = 2,
	plotEveryNsteps = 5,
	nev = 11, precisionStability = 1e-6,
	detectBifurcation = 2, dsminBisection = 1e-6, maxBisectionSteps = 15)
	br_po, _ , _= @time PALC.continuation(poTrap,
			outpo_f, (@set par_bru.l = l_hopf + 0.01), (@lens _.l),
			opts_po_cont;
			linearPO = :FullLU,
			# linearPO = :BorderedLU,
			# tangentAlgo = BorderedPred(),
			verbosity = 3,	plot = true,
			# callbackN = (x, f, J, res, iteration, options; kwargs...) -> (println("--> amplitude = ", PALC.amplitude(x, n, M));true),
			finaliseSolution = (z, tau, step, contResult) ->
				(Base.display(contResult.eig[end].eigenvals) ;true),
			plotSolution = (x, p;kwargs...) -> heatmap!(reshape(x[1:end-1], 2*n, M)'; ylabel="time", color=:viridis, kwargs...),
			# printSolution = (x, p;kwargs...) -> PALC.amplitude(x, n, M; ratio = 2),
			normC = norminf)

####################################################################################################
using IncompleteLU
Jpo = @time poTrap(l_hopf + 0.01)(Val(:JacFullSparse), orbitguess_f)
@time lu(Jpo)
Jpo = @time poTrap(l_hopf + 0.01)(Val(:JacCyclicSparse), orbitguess_f)
@time lu(Jpo)

Precilu = @time ilu(Jpo, τ = 0.005)
ls = GMRESIterativeSolvers(verbose = false, tol = 1e-4, N = size(Jpo,1), restart = 30, maxiter = 150, log=true, Pl = Precilu)
	@time ls(Jpo, rand(ls.N))

opt_po = PALC.NewtonPar(tol = 1e-10, verbose = true, maxIter = 20)
	outpo_f, _, flag = @time PALC.newton(poTrap(l_hopf + 0.01),
			orbitguess_f,
			(@set opt_po.linsolver = ls), :BorderedMatrixFree;
			normN = norminf,
			# callback = (x, f, J, res, iteration, options; kwargs...) -> (println("--> amplitude = ", amplitude(x), " T = ", x[end], ", T0 = ",orbitguess_f[end]);true)
			)
	printstyled(color=:red, "--> T = ", outpo_f[end], ", amplitude = ", PALC.amplitude(outpo_f, n, M; ratio = 2),"\n")
	PALC.plotPeriodicPOTrap(outpo_f, n, M; ratio = 2)

opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.05, ds= 0.01, pMax = 2.2, maxSteps = 3000, newtonOptions = (@set opt_po.linsolver = ls))
	br_pok, _ , _= @time PALC.continuationPOTrap(poTrap,
			outpo_f, l_hopf + 0.01,
			opts_po_cont; linearPO = :FullMatrixFree,
			verbosity = 2,
			plot = true,
			# callbackN = (x, f, J, res, iteration, options; kwargs...) -> (println("--> amplitude = ", PALC.amplitude(x, n, M));true),
			plotSolution = (x, p;kwargs...) -> heatmap!(reshape(x[1:end-1], 2*n, M)'; ylabel="time", color=:viridis, kwargs...), normC = norminf)
####################################################################################################
# automatic branch switching from Hopf point
opt_po = NewtonPar(tol = 1e-10, verbose = true, maxIter = 14)
opts_po_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.04, ds = 0.01, pMax = 2.2, maxSteps = 200, newtonOptions = opt_po, saveSolEveryNsteps = 2,
	plotEveryNsteps = 1, nev = 11, precisionStability = 1e-6,
	detectBifurcation = 0, dsminBisection = 1e-6, maxBisectionSteps = 15, tolBisectionEigenvalue = 0.)

M = 51
br_po, _ = continuation(
	# arguments for branch switching
	jet..., br, 1,
	# arguments for continuation
	opts_po_cont, PeriodicOrbitTrapProblem(M = M);
	#
	ampfactor = 1, δp = 0.01,
	verbosity = 3,	plot = true, linearPO = :FullLU,
	# callbackN = (x, f, J, res, iteration, itl, options; kwargs...) -> (println("--> amplitude = ", PALC.amplitude(x, n, M; ratio = 2));true),
	finaliseSolution = (z, tau, step, contResult) ->
		(Base.display(contResult.eig[end].eigenvals) ;true),
	plotSolution = (x, p; kwargs...) -> heatmap!(reshape(x[1:end-1], 2*n, M)'; ylabel="time", color=:viridis, kwargs...),
	# printSolution = (x, p;kwargs...) -> PALC.amplitude(x, n, M; ratio = 2),
	normC = norminf)

# branches = [br_po]
push!(branches, br_po)
plot(branches, legend = :bottomright, plotfold = false)

####################################################################################################
# semi-automatic branch switching from bifurcation BP-PO
br_po2, _ = PALC.continuationPOTrapBPFromPO(
	# arguments for branch switching
	br_po, 1,
	# arguments for continuation
	opts_po_cont;
	ampfactor = 1., δp = 0.01,
	verbosity = 3,	plot = true, linearPO = :FullLU,
	# callbackN = (x, f, J, res, iteration, itl, options; kwargs...) -> (println("--> amplitude = ", PALC.amplitude(x, n, M; ratio = 2));true),
	finaliseSolution = (z, tau, step, contResult) ->
		(Base.display(contResult.eig[end].eigenvals) ;true),
	plotSolution = (x, p; kwargs...) -> (heatmap!(reshape(x[1:end-1], 2*n, M)'; ylabel="time", color=:viridis, kwargs...);plot!(branches[2],legend = :bottomright, subplot=1)),
	# printSolution = (x, p;kwargs...) -> PALC.amplitude(x, n, M; ratio = 2),
	normC = norminf)
push!(branches, br_po2)

plot();for _b in branches; plot!(_b; label = ""); end; title!("")

plot(branches[2])
