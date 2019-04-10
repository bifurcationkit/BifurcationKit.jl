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
	return true
end

n = 101
# const Δ = spdiagm(0 => 2ones(N), -1 => 0ones(N-1), 1 => -ones(N-1))
Jac_fd(u0, α, β, l = l) = Cont.finiteDifferences(u->F_bru(u, α, β, l=l), u0)

a = 2.
b = 5.45

sol0 = vcat(a * ones(n), b/a * ones(n))

opt_newton = Cont.NewtonPar(tol = 1e-11, verbose = true, eigsolve = eig_KrylovKit(tol=1e-6, dim = 60))
	# ca fait dans les 60.2k Allocations
	out, hist, flag = @time Cont.newton(
		x -> F_bru(x, a, b),
		x -> Jac_sp(x, a, b),
		sol0,
		opt_newton)

opts_br0 = ContinuationPar(dsmin = 0.001, dsmax = 0.0061, ds= 0.0051, pMax = 1.8, save = false, theta = 0.01, detect_fold = true, detect_bifurcation = true, nev = 41, plot_every_n_steps = 50, newtonOptions = opt_newton)
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
		printsolution = x -> norm(x, Inf64))

# J0 = Jac_sp(sol0, a, 16) |> sparse
# using Arpack, ArnoldiMethod, KrylovKit
# @time Arpack.eigs(J0, nev = 10, which = :LR)
# @time sort(eigen(Array(J0)).values, by = x -> abs(x), rev = false)[1:15]
# @time KrylovKit.eigsolve(J0, 10, :LR, tol = 1e-6)
# @time opt_newton.eigsolve(J0,10)
#################################################################################################### Continuation of the Hopf Point using Jacobian expression
ind_hopf = 1
	hopfpt = Cont.HopfPoint(br, ind_hopf)

	outhopf, hist, flag = @time Cont.newtonHopf((x, p) ->  F_bru(x, a, b, l = p),
			(x, p) -> Jac_sp(x, a, b, l = p),
			br, ind_hopf,
			opt_newton)
	flag && printstyled(color=:red, "--> We found a Hopf Point at l = ", outhopf.p[1], ", ω = ", outhopf.p[2], ", from l = ",hopfpt.p[1],"\n")

br_hopf, u1_hopf = @time Cont.continuationHopf(
			(x, p, β) ->  F_bru(x, a, β, l = p),
			(x, p, β) -> Jac_sp(x, a, β, l = p),
			br, ind_hopf,
			b,
			ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, pMax = 6.5, pMin = 0.0, a = 2., theta = 0.4, newtonOptions = opt_newton), verbosity = 2)
Cont.plotBranch(br_hopf, xlabel="beta", ylabel = "l", label="")
#################################################################################################### Continuation of Periodic Orbit
function plotPeriodic(outpof,n,M)
	outpo = reshape(outpof[1:end-1], 2n, M)
	plot(heatmap(outpo[1:n,:]', ylabel="Time"),
			heatmap(outpo[n+2:end,:]'))
end

ind_hopf = 2
hopfpt = Cont.HopfPoint(br, ind_hopf)

l_hopf = hopfpt.p[1]
ωH     = hopfpt.p[end] |> abs
M = 100


orbitguess = zeros(2n, M)
plot([0, 1], [0, 0])
	phase = []; scalphase = []
	vec_hopf = getEigenVector(opt_newton.eigsolve ,br.eig[br.bifpoint[ind_hopf][2]][2] ,br.bifpoint[ind_hopf][end]-1)

	# br.eig[br.bifpoint[ind_hopf][2]][2][:, br.bifpoint[ind_hopf][end]-1]
	for ii=1:M
	t = (ii-1)/(M-1)
	# use phase 0.279 for default_eig()
	orbitguess[:, ii] .= real.(hopfpt.u +
						26*0.1 * vec_hopf * exp(2pi * complex(0, 1) * (t - .235))) #k=1
	push!(phase, t);push!(scalphase, dot(orbitguess[:, ii]- hopfpt.u, real.(vec_hopf)))
end
	phmin = findmin(abs.(scalphase))
	println("--> min phase for ", phase[phmin[2]])
	plot!(phase, scalphase)

orbitguess_f = vcat(vec(orbitguess), 2pi/ωH) |> vec

poTrap = l-> PeriodicOrbitTrap(
			x-> F_bru(x, a, b, l = l),
			x-> Jac_sp(x, a, b, l = l),
			real.(vec_hopf),
			hopfpt.u,
			M,
			opt_newton.linsolve)

poTrap(l_hopf + 0.01)(orbitguess_f) |> plot

plot(heatmap(orbitguess[1:n,:], ylabel="Time"),heatmap(orbitguess[n+2:end,:]))

opt_po = Cont.NewtonPar(tol = 1e-8, verbose = true, maxIter = 50)
	outpo_f, hist, flag = @time Cont.newton(
			x ->  poTrap(l_hopf + 0.01)(x),
			x ->  poTrap(l_hopf + 0.01)(x, :jacsparse),
			orbitguess_f,
			opt_po)
	println("--> T = ", outpo_f[end], ", amplitude = ", maximum(outpo_f[1:n,:])-minimum(outpo_f[1:n,:]))
	plotPeriodic(outpo_f,n,M)

opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.05, ds= 0.001, pMax = 3.3, maxSteps = 400, secant = true, theta=0.1, plot_every_n_steps = 3, newtonOptions = opt_po)
	br_pok2, upo , _= @time Cont.continuation(
			(x, p) ->  poTrap(p)(x),
			(x, p) ->  poTrap(p)(x, :jacsparse),
			outpo_f, l_hopf + 0.01,
			opts_po_cont,
			plot = true,
			plotsolution = (x;kwargs...) -> heatmap!(reshape(x[1:end-1], 2*n, M)', subplot=4, ylabel="time"),
			printsolution = u -> u[end])
##########################################################################################
# Matrix-Free computation, useless without a preconditionner
opt_po = Cont.NewtonPar(tol = 1e-8, verbose = true, maxIter = 50, linsolve = GMRES_KrylovKit{Float64}(dim=30, verbose = 2))
	outpo_f, hist, flag = @time Cont.newton(
			x ->  poTrap(l_hopf + 0.01)(x),
			x -> (dx -> poTrap(l_hopf + 0.01)(x, dx)),
			orbitguess_f,
			opt_po)
	println("--> T = ", outpo_f[end], ", amplitude = ", maximum(outpo_f[1:n,:])-minimum(outpo_f[1:n,:]))
	plotPeriodic(outpo_f,n,M)
