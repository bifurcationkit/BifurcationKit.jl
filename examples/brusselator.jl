using Revise
using PseudoArcLengthContinuation, LinearAlgebra, Plots, SparseArrays
const Cont = PseudoArcLengthContinuation

f1(u, v) = u^2*v

function F_bru(x, α, β; D1 = 0.008, D2 = 0.004, l = 1.0)
	n = div(length(x), 2)
	h = 1.0 / (n+1); h2 = h*h

	u = @view x[1:n]
	v = @view x[n+1:2n]

	# output
	f = similar(x)

	f[1] = u[1] - α
	f[n] = u[n] - α
	for i=2:n-1
		f[i] = D1/l^2 * (u[i-1] - 2u[i] + u[i+1]) / h2 - (β + 1) * u[i] + α + f1(u[i], v[i])
	end


	f[n+1] = v[1] - β / α
	f[end] = v[n] - β / α;
	for i=2:n-1
		f[n+i] = D2/l^2 * (v[i-1] - 2v[i] + v[i+1]) / h2 + β * u[i] - f1(u[i], v[i])
	end

	return f
end

function Jac_mat(x, α, β; D1 = 0.008, D2 = 0.004, l = 1.0)
	n = div(length(x), 2)
	h = 1.0 / (n+1); hh = h*h

	J = zeros(2n, 2n)

	J[1, 1] = 1.0
	for i=2:n-1
		J[i, i-1] = D1 / hh/l^2
		J[i, i]   = -2D1 / hh/l^2 - (β + 1) + 2x[i] * x[i+n]
		J[i, i+1] = D1 / hh/l^2
		J[i, i+n] = x[i]^2
	end
	J[n, n] = 1.0

	J[n+1, n+1] = 1.0
	for i=n+2:2n-1
		J[i, i-n] = β - 2x[i-n] * x[i]
		J[i, i-1] = D2 / hh/l^2
		J[i, i]   = -2D2 / hh/l^2 - x[i-n]^2
		J[i, i+1] = D2 / hh/l^2
	end
	J[2n, 2n] = 1.0
	return J
end

function Jac_sp(x, α, β; D1 = 0.008, D2 = 0.004, l = 1.0)
	# compute the Jacobian using a sparse representation
	n = div(length(x), 2)
	h = 1.0 / (n+1); hh = h*h

	diag  = zeros(2n)
	diagp1 = zeros(2n-1)
	diagm1 = zeros(2n-1)

	diagpn = zeros(n)
	diagmn = zeros(n)

	diag[1] = 1.0
	diag[n] = 1.0
	diag[n + 1] = 1.0
	diag[end] = 1.0

	for i=2:n-1
		diagm1[i-1] = D1 / hh/l^2
		diag[i]   = -2D1 / hh/l^2 - (β + 1) + 2x[i] * x[i+n]
		diagp1[i] = D1 / hh/l^2
		diagpn[i] = x[i]^2
	end

	for i=n+2:2n-1
		diagmn[i-n] = β - 2x[i-n] * x[i]
		diagm1[i-1] = D2 / hh/l^2
		diag[i]   = -2D2 / hh/l^2 - x[i-n]^2
		diagp1[i] = D2 / hh/l^2
	end
	return spdiagm(0 => diag, 1 => diagp1, -1 => diagm1, n => diagpn, -n => diagmn)
end



function finalise_solution(z, tau, step, contResult)
	n = div(length(z), 2)
	printstyled(color=:red, "--> Solution constant = ", norm(diff(z[1:n])), " - ", norm(diff(z[n+1:2n])), "\n")
end

n = 101
# const Δ = spdiagm(0=>2ones(N), -1=>0ones(N-1), 1=>-ones(N-1))
Jac_fd(u0, α, β, l = l) = Cont.finiteDifferences(u->F_bru(u, α, β, l=l), u0)

a = 2.
b = 5.45

sol0 = vcat(a * ones(n), b/a * ones(n))

opt_newton = Cont.NewtonPar(tol = 1e-11, verbose = true)
	# ca fait dans les 60.2k Allocations
	out, hist, flag = @time Cont.newton(
		x -> F_bru(x, a, b),
		x -> Jac_sp(x, a, b),
		sol0,
		opt_newton)

opts_br0 = ContinuationPar(dsmin = 0.001, dsmax = 0.0061, ds= 0.0051, pMax = 1.8, save = false, theta = 0.01, detect_fold = true, detect_bifurcation = true, nev = 16, plot_every_n_steps = 50)
	opts_br0.newtonOptions.maxIter = 20
	opts_br0.newtonOptions.tol = 1e-8
	opts_br0.maxSteps = 280

	br, u1 = @time Cont.continuation(
		(x, p) ->   F_bru(x, a, b, l = p),
		(x, p) -> Jac_sp(x, a, b, l = p),
		out,
		0.3,
		opts_br0,
		plot = true,
		plotsolution = (x;kwargs...)->(N = div(length(x), 2);plot!(x[1:N], subplot=4, label="");plot!(x[N+1:2N], subplot=4, label="")),
		finaliseSolution = finalise_solution,
		printsolution = x->norm(x, Inf64))

# J0 = Jac_mat(sol0, a, 16) |> sparse
# using Arpack, ArnoldiMethod
# @time eigs(J0, nev = 10, which = :SM, sigma = 0.01, maxiter=10000)
# @time sort(eigen(Array(J0)).values, by = x -> abs(x), rev = false)[1:15]
#################################################################################################### Continuation of the Hopf Point using Dense method
# ind_hopf = 1
# hopfpt = Cont.HopfPoint(br, ind_hopf)
# 	hopfvariable = β -> HopfProblemMinimallyAugmented(
# 					(x, p) ->   F_bru(x, a, β, l = p),
# 					(x, p) -> Jac_mat(x, a, β, l = p),
# 					(x, p) -> transpose(Jac_mat(x, a, β, l = p)),
# 					br.eig[br.bifpoint[ind_hopf][2]][2][:, br.bifpoint[ind_hopf][end]-1],
# 					br.eig[br.bifpoint[ind_hopf][2]][2][:, br.bifpoint[ind_hopf][end]],
# 					opts_br0.newtonOptions.linsolve)
# 	hopfPb = (u, p)->hopfvariable(p)(u)
#
# Jac_hopf_fdMA(u0, α) = Cont.finiteDifferences( u-> hopfPb(u, α), u0)
# # cas des differences finies ~ 190s
# opt_hopf = Cont.NewtonPar(tol = 1e-10, verbose = true, maxIter = 20)
# 	outhopf, hist, flag = @time Cont.newton(
# 						x ->        hopfPb(x, b),
# 						x -> Jac_hopf_fdMA(x, b),
# 						hopfpt,
# 						opt_hopf)
# 	flag && printstyled(color=:red, "--> We found a Hopf Point at l = ", outhopf[end-1], ", ω = ", outhopf[end], "\n")
#
# opt_hopf_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, pMax = 14.1, pMin = 0.0, a = 2., theta = 0.4)
# 	opt_hopf_cont.maxSteps = 70
# 	opt_hopf_cont.newtonOptions.verbose = true
#
# 	br_hopf, u1_hopf = @time Cont.continuation(
# 					(x, β) ->        hopfPb(x, β),
# 					(x, β) -> Jac_hopf_fdMA(x, β),
# 					outhopf, b,
# 					opt_hopf_cont, plot = true,
# 					printsolution = u -> u[end-1])
#################################################################################################### Continuation of the Hopf Point using Jacobian expression
ind_hopf = 1
	hopfpt = Cont.HopfPoint(br, ind_hopf)

	outhopf, hist, flag = @time Cont.newtonHopf((x, p) ->  F_bru(x, a, b, l = p),
                (x, p) -> Jac_mat(x, a, b, l = p),
				br, ind_hopf,
				NewtonPar(verbose = true))
	flag && printstyled(color=:red, "--> We found a Hopf Point at l = ", outhopf[end-1], ", ω = ", outhopf[end], "from l = ",hopfpt[end-1],"\n")

br_hopf, u1_hopf = @time Cont.continuationHopf(
			(x, p, β) ->   F_bru(x, a, β, l = p),
			(x, p, β) -> Jac_mat(x, a, β, l = p),
			br, ind_hopf,
			b,
			ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, pMax = 6.5, pMin = 0.0, a = 2., theta = 0.4, newtonOptions = NewtonPar(verbose=true)))
Cont.plotBranch(br_hopf, xlabel="beta", ylabel = "l", label="")
#################################################################################################### Continuation of Periodic Orbit
function plotPeriodic(outpof,n,M)
	outpo = reshape(outpof[1:end-1], 2n, M)
	plot(heatmap(outpo[1:n,:], ylabel="Time"),
			heatmap(outpo[n+2:end,:]))
end

ind_hopf = 1
hopfpt = Cont.HopfPoint(br, ind_hopf)

l_hopf = hopfpt[end-1]
ωH     = hopfpt[end] |> abs
M = 50


orbitguess = zeros(2n, M)
plot([0, 1], [0, 0])
	phase = []; scalphase = []
	vec_hopf = br.eig[br.bifpoint[ind_hopf][2]][2][:, br.bifpoint[ind_hopf][end]-1]
	for ii=1:M
	t = (ii-1)/(M-1)
	orbitguess[:, ii] .= real.(hopfpt[1:2n] +
						26*0.1 * vec_hopf * exp(2pi * complex(0, 1) * (t - 0.279))) #k=1
	push!(phase, t);push!(scalphase, dot(orbitguess[:, ii]- hopfpt[1:2n], real.(vec_hopf)))
end
	plot!(phase, scalphase)

orbitguess_f = vcat(vec(orbitguess), 2pi/ωH) |> vec

poTrap = l-> PeriodicOrbitTrap(
			x-> F_bru(x, a, b, l = l),
			x-> Jac_sp(x, a, b, l = l),
			real.(vec_hopf),
			hopfpt[1:2n],
			M,
			opt_newton.linsolve,
			opt_newton)

poTrap(l_hopf + 0.01)(orbitguess_f) |> plot

plot(heatmap(orbitguess[1:n,:], ylabel="Time"),heatmap(orbitguess[n+2:end,:]))

opt_po = Cont.NewtonPar(tol = 1e-8, verbose = true, maxIter = 50)
	outpo_f, hist, flag = @time Cont.newton(
						x ->  poTrap(l_hopf + 0.01)(x),
						x ->  poTrap(l_hopf + 0.01)(x, :jacsparse),
						orbitguess_f,
						opt_po)
	println("--> T = ", outpo_f[end])
	plotPeriodic(outpo_f,n,M)

# # stability
# Jfloquet = 	Cont.JacobianPeriodicFD(poTrap(l_hopf + 0.01), outpo_f, 1.0) |> sparse
# # df = ones(2n*M); df[end-2n+1:end].=0;Bfloquet = spdiagm(0 => df);
# using Arpack
# reseig = Arpack.eigs(Jfloquet, Bfloquet, nev = 10, which = :LM, tol = 1e-8)
# # res = KrylovKit.geneigsolve(Jfloquet, Bfloquet)
# eis = eig_KrylovKit{Float64}(which = :LM, tol = 1e-4 )
# eis(Jfloquet, 100)
#
# Jmono = x -> Cont.FloquetPeriodicFD(poTrap(1.3), upo.u, x)
# 	KrylovKit.eigsolve(Jmono,rand(2n),10, :LM)

# opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.05, ds= 0.001, pMax = 2.3, maxSteps = 400, secant = true, theta=0.1, plot_every_n_steps = 3, newtonOptions = NewtonPar(verbose = true, eigsolve = Cont.FloquetFD2(poTrap)), detect_bifurcation = true)
opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.05, ds= 0.001, pMax = 2.3, maxSteps = 400, secant = true, theta=0.1, plot_every_n_steps = 3, newtonOptions = NewtonPar(verbose = true))
	br_pok2, upo , _= @time Cont.continuation(
							(x, p) ->  poTrap(p)(x),
							(x, p) ->  poTrap(p)(x, :jacsparse),
							outpo_f, l_hopf + 0.01,
							opts_po_cont,
							plot = true,
							plotsolution = (x;kwargs...)->heatmap!(reshape(x[1:end-1], 2*n, M)', subplot=4, ylabel="time"),
							printsolution = u -> u[end])

# branches = []
# push!(branches,br_pok1)
# Cont.plotBranch(branches, ylabel="T", xlabel = "l", label="")
#################################################################################################### Example pde2path
a = 1.5
b = 4.1
l0 = 1.4
opt_newton = Cont.NewtonPar(tol = 1e-11, verbose = true)
	# ca fait dans les 60.2k Allocations
	out, hist, flag = @time Cont.newton(
		x -> F_bru(x, a, b; D1 = 0.01, D2 = 0.1, l = l0),
		x -> Jac_mat(x, a, b; D1 = 0.01, D2 = 0.1, l = l0),
		vcat(a*ones(n),b/a*ones(n)),
		opt_newton)
		plot(out)

opts_br0 = ContinuationPar(dsmin = 0.001, dsmax = 0.01, ds= 0.001, pMax = 1.8, save = false, theta = 0.01, detect_fold = true, detect_bifurcation = true, nev = 16, plot_every_n_steps = 50)
	opts_br0.newtonOptions.maxIter = 20
	opts_br0.newtonOptions.tol = 1e-8
	opts_br0.maxSteps = 540

	br2, _ = @time Cont.continuation(
		(x, p) ->   F_bru(x, p, b; D1 = 0.01, D2 = 0.1, l = l0),
		(x, p) -> Jac_mat(x, p, b; D1 = 0.01, D2 = 0.1, l = l0),
		out,
		a,
		opts_br0,
		plot = true,
		plotsolution = (x;kwargs...)->(N = div(length(x), 2);plot!(x[1:N], subplot=4, label="");plot!(x[N+1:2N], subplot=4, label="")),
		finaliseSolution = finalise_solution,
		printsolution = x->norm(x))


outhopf,_ = newtonHopf(
	(x, p) ->   F_bru(x, p, b; D1 = 0.01, D2 = 0.1, l = l0),
	(x, p) -> Jac_mat(x, p, b; D1 = 0.01, D2 = 0.1, l = l0),
	(x, p) -> transpose(Jac_mat(x, p, b; D1 = 0.01, D2 = 0.1, l = l0)),
	br2, 1,
	NewtonPar(verbose = true))
	flag && printstyled(color=:red, "--> We found a Hopf Point at a = ", outhopf[end-1], ", ω = ", outhopf[end], "\n")

#####
ind_hopf = 1
hopfpt = Cont.HopfPoint(br2, ind_hopf)

p_hopf = outhopf[end-1]
ωH     = outhopf[end] |> abs

M = 35

orbitguess = zeros(2n, M)
plot([0, 1], [0, 0])
	tt = []; phase = []
	vec_hopf = br2.eig[br2.bifpoint[ind_hopf][2]][2][:, br2.bifpoint[ind_hopf][end]+1]
	for ii=1:M
		t = (ii-1)/(M-1)
		orbitguess[:, ii] .= real.(hopfpt[1:2n] +
							11*0.1 * vec_hopf * exp(2pi * complex(0, 1) * (t - 0.230)))
		push!(tt, t);push!(phase, dot(orbitguess[:, ii]- hopfpt[1:2n], real.(vec_hopf)))
	end
	plot!(tt, phase) |> display
	orbitguess_f = vcat(vec(orbitguess), 2pi/ωH) |> vec

poTrap = p -> PeriodicOrbitTrap(
			x ->  F_bru(x, p, b; D1 = 0.01, D2 = 0.1, l = l0),
			x -> Jac_sp(x, p, b; D1 = 0.01, D2 = 0.1, l = l0),
			real.(vec_hopf),
			hopfpt[1:2n],
			M,
			opt_newton.linsolve,
			opt_newton)

poTrap(p_hopf + 0.01)(orbitguess_f) |> plot

plotPeriodic(orbitguess_f,n,M)

opt_po = Cont.NewtonPar(tol = 1e-8, verbose = true, maxIter = 50)
	outpo, hist, flag = @time Cont.newton(
						x ->  poTrap(p_hopf + 0.01)(x),
						x -> poTrap(p_hopf + 0.01)(x,:jacsparse),
						orbitguess_f,
						opt_po)
	println("--> T = ", outpo[end])
	plotPeriodic(outpo,n,M)


using DifferentialEquations
tspan = (0.0, 10.)
prob  = ODEProblem((x,p,t)->F_bru(x, p, b; D1 = 0.01, D2 = 0.1, l = l0), hopfpt[1:2n], tspan, p_hopf - 0.0)
sol   = @time solve(prob, Tsit5(), progress=true);
plot(sol.t, map(norm, sol.u))
plot(sol.u[end])

heatmap(sol.u)
