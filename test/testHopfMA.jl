using Test, PseudoArcLengthContinuation, LinearAlgebra, Plots, SparseArrays
const Cont = PseudoArcLengthContinuation

f1(u, v) = u^2*v

n = 101
a = 2.
b = 5.45

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

function Jac_sp(x, α, β; D1 = 0.008, D2 = 0.004, l = 1.0)
	# compute the Jacobian using a sparse representation
	n = div(length(x), 2)
	hh = 1.0 / (n+1)^2

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

sol0 = vcat(a * ones(n), b / a * ones(n))

opt_newton = Cont.NewtonPar(tol = 1e-11, verbose = true)
	out, hist, flag = @time Cont.newton(
		x -> F_bru(x, a, b),
		x -> Jac_sp(x, a, b),
		sol0 .* (1 .+ 0.01rand(2n)),
		opt_newton)

opt_newton = Cont.NewtonPar(tol = 1e-11, verbose = true,linsolve = GMRES_IterativeSolvers(tol=1e-4, N = 2n))
	out, hist, flag = @time Cont.newton(
		x -> F_bru(x, a, b),
		x -> Jac_sp(x, a, b),
		sol0 .* (1 .+ 0.01rand(2n)),
		opt_newton)

opts_br0 = ContinuationPar(dsmin = 0.001, dsmax = 0.01, ds= 0.0051, pMax = 1.8, theta = 0.01, detect_bifurcation = true, nev = 16)

	br, u1 = @time Cont.continuation(
		(x, p) ->  F_bru(x, a, b, l = p),
		(x, p) -> Jac_sp(x, a, b, l = p),
		out,
		0.3,
		opts_br0,
		plot = false,
		printsolution = x -> norm(x, Inf64), verbosity = 0)
#################################################################################################### Continuation of the Hopf Point using Dense method
ind_hopf = 1
av = randn(Complex{Float64},2n); av = av./norm(av)
bv = randn(Complex{Float64},2n); bv = bv./norm(bv)
hopfpt = Cont.HopfPoint(br, ind_hopf)
# hopfpt[1:2n] .= rand(2n)
hopfvariable = β -> HopfProblemMinimallyAugmented(
					(x, p) ->  F_bru(x, a, β, l = p),
					(x, p) -> Jac_sp(x, a, β, l = p),
					(x, p) -> transpose(Jac_sp(x, a, β, l = p)),
					br.eig[br.bifpoint[ind_hopf][2]][2][:, br.bifpoint[ind_hopf][end]],
					conj.(br.eig[br.bifpoint[ind_hopf][2]][2][:, br.bifpoint[ind_hopf][end]]),
					# av,
					# bv,
					opts_br0.newtonOptions.linsolve)
	hopfPb = (u, p) -> hopfvariable(p)(u)

hopfvariable(b)(hopfpt) |> norm

Bd2Vec(x) = vcat(x.u, x.p)
Vec2Bd(x) = BorderedArray(x[1:end-2], x[end-1:end])
hopfpbVec(x, p) = Bd2Vec(hopfvariable(p)(Vec2Bd(x)))

# finite differences Jacobian
Jac_hopf_fdMA(u0, p) = Cont.finiteDifferences( u-> hopfpbVec(u, p), u0)
# ``analytical'' jacobian
Jac_hopf_MA(u0, pb::HopfProblemMinimallyAugmented) = (return (u0, pb, x -> x))

# # on construit la matrice a inverser pour calculer sigma1 et sigma2
# ω = hopfpt[end]
# # av = hopfvariable(b).a
# # bv = hopfvariable(b).b
# Jh = hopfvariable(b).J(hopfpt[1:end-2], hopfpt[end-1])
# A = Jh + Complex(0, ω ) * I
# A = Array(A)
# Atot = [A av ; (bv') 0.]
# res1 = Atot \ [zeros(2n)' Complex(1,0)]'
# res2 = Atot' \ [zeros(2n)' Complex(1,0)]'
# println("--> σ1 = ", res1[end], ",\n--> σ2 = ", res2[end])
#
# dot(res2, Atot*res1)
#
# v, σ1, _ = Cont.linearBorderedSolver(Jh + Complex(0, ω) * I, av, bv, 0., zeros(2n), 1.0, hopfvariable(b).linsolve)
#
# w, σ2, _ = Cont.linearBorderedSolver(Jh' - Complex(0, ω) * I, bv, av, 0., zeros(2n), 1.0, hopfvariable(b).linsolve)
#
# res1[1:end-1]-v |> norm
# res2[1:end-1]-w |> norm
#
# L1 = hcat(Jh, -ω*I, real.(av), -imag.(av))
# L2 = hcat(ω*I, Jh, imag.(av), real.(av))
# L3 = hcat(real.(bv)', imag.(bv)', 0, 0)
# L4 = hcat(-imag.(bv)', real.(bv)', 0, 0)
# Atot2 = vcat(L1, L2, L3, L4)
# res12 = Atot2 \ hcat(zeros(4n)',[1],[0])'
# res22 = Atot2' \ hcat(zeros(4n)',[1],[0])'
#
# @test res12[1:2n] - real(res1[1:end-1]) |> norm < 1e-10
# @test res12[2n+1:4n] - imag(res1[1:end-1]) |> norm < 1e-10
# @test res22[1:2n] - real(res2[1:end-1]) |> norm < 1e-10
# @test res22[2n+1:4n] - imag(res2[1:end-1]) |> norm < 1e-10
#
# @test v - res1[1:end-1] |> norm < 1e-10
# @test w - res2[1:end-1] |> norm < 1e-10
#
# println("--> version reelle σ1 = ", res12[end-1:end])
# println("--> version reelle σ2 = ", res22[end-1:end])
#
# # on calcule les derivees de sigma1 et sigma2
# jac_hopf_fd = Jac_hopf_fdMA(hopfpt, b)
# println("-->\n FD sigma_omega = ", jac_hopf_fd[end-1:end,end-1:end])
# dot(res22[1:2n], res12[1:2n])
# dot(res22[2n+1:4n], res12[2n+1:4n])
#
# newton convergence toward

outhopf, _, flag, _ = @time Cont.newton(u -> hopfpbVec(u, b),
							# u -> Jac_hopf_fdMA(u, b),
							Bd2Vec(hopfpt), NewtonPar(verbose = true))
	flag && printstyled(color=:red, "--> We found a Hopf Point at l = ", outhopf[end-1], ", ω = ", outhopf[end], " from ", hopfpt.p, "\n")

rhs = rand(length(hopfpt))
jac_hopf_fd = Jac_hopf_fdMA(Bd2Vec(hopfpt), b)
sol_fd = jac_hopf_fd \ rhs
# create a linear solver
hopfls = HopfLinearSolveMinAug()
sol_ma, _, _, sigomMA  = hopfls(Jac_hopf_MA(hopfpt, hopfvariable(b)), BorderedArray(rhs[1:end-2],rhs[end-1:end]),debug_ = true)

println("--> test jacobian expression for Hopf Minimally Augmented")
@test Bd2Vec(sol_ma) - sol_fd |> x->norm(x, Inf64) < 1e-3

@test (Bd2Vec(sol_ma) - sol_fd)[1:end-2] |> x->norm(x, Inf64) < 1e-3

# dF = jac_hopf_fd[:,end-1]
# sig_vec_re = jac_hopf_fd[end-1,1:end-2]
# sig_vec_im = jac_hopf_fd[end,1:end-2]
#
# println("-->\n FD sigma_omega = ", jac_hopf_fd[end-1:end,end-1:end]')
#
# norm(sig_vec_re - real.(sigomMA[1]), Inf64)
# norm(sig_vec_im - imag.(sigomMA[1]), Inf64)
# norm(dF[1:end-2] - sigomMA[end], Inf64)
#
# # on essaie de resoudre jac_hopf_fd \ rhs
# Jh = jac_hopf_fd[1:end-2,1:end-2]
# sig1 = jac_hopf_fd[end-1,1:end-2]
# sig2 = jac_hopf_fd[end,1:end-2]
# sig1p = jac_hopf_fd[end-1,end-1]
# sig2p = jac_hopf_fd[end,end-1]
# sig1o = jac_hopf_fd[end-1,end]
# sig2o = jac_hopf_fd[end,end]
#
# sigma = hcat(sig1,sig2)
#
# X1 = Jh \ rhs[1:end-2]
# X2 = Jh \ dF[1:end-2]#jac_hopf_fd[1:end-2,end-2]
# X2m = hcat(X2, 0*X2)
# C = jac_hopf_fd[end-1:end,end-1:end]
# println("--> dp, dom = ",(C - sigma' * X2m) \ (rhs[end-1:end] - sigma' * X1))
# println("--> dp, dom FD = ",sol_fd[end-1:end])
outhopf, hist, flag = @time Cont.newtonHopf(
		(x, p) ->  F_bru(x, a, b, l = p),
		(x, p) -> Jac_sp(x, a, b, l = p),
		(x, p) -> transpose(Jac_sp(x, a, b, l = p)),
		br, 1,
		NewtonPar(verbose = true))
		flag && printstyled(color=:red, "--> We found a Hopf Point at l = ", outhopf.p[1], ", ω = ", outhopf.p[end], ", from l = ",hopfpt.p[1],"\n")

# version with iterative solver
outhopf, hist, flag = @time Cont.newtonHopf(
		(x, p) ->  F_bru(x, a, b, l = p),
		(x, p) -> Jac_sp(x, a, b, l = p),
		br, 1,
		NewtonPar(verbose = true, linsolve = GMRES_IterativeSolvers(tol=1e-4, N = 2n)))
		flag && printstyled(color=:red, "--> We found a Hopf Point at l = ", outhopf.p[1], ", ω = ", outhopf.p[end], ", from l = ",hopfpt.p[1],"\n")

outhopf, _, flag, _ = @time Cont.newton(u -> hopfvariable(b)(u),
							x -> Jac_hopf_MA(x, hopfvariable(b)),
							hopfpt, NewtonPar(verbose = true, linsolve = HopfLinearSolveMinAug(), eigsolve = Default_eig()))
	flag && printstyled(color=:red, "--> We found a Hopf Point at l = ", outhopf.p[1], ", ω = ", outhopf.p[2], " from ", hopfpt.p, "\n")

# version with analytical Hessian = 2 P(du2) P(du1) QU + 2 PU P(du1) Q(du2) + 2PU P(du2) Q(du1)
function d2F(x, p1::T, p2::T, du1, du2) where T
	n = div(length(x),2)
	out = similar(du1)
	@views out[1:n] .= 2 .* x[n+1:end] .* du1[1:n] .* du2[1:n] .+
				2 .* x[1:n] .* du1[1:n] .* du2[n+1:end] .+
				2 .* x[1:n] .* du2[1:n] .* du1[1:n]
	@inbounds for ii=1:n
		out[ii+n] = -out[ii]
	end
	return out
end

outhopf, hist, flag = @time Cont.newtonHopf(
		(x, p) ->  F_bru(x, a, b, l = p),
		(x, p) -> Jac_sp(x, a, b, l = p),
		(x, p) -> transpose(Jac_sp(x, a, b, l = p)),
		(x, p1, v1, v2) -> d2F(x, 0., 0., v1, v2),
		br, 1,
		NewtonPar(verbose = true))
		flag && printstyled(color=:red, "--> We found a Hopf Point at l = ", outhopf.p[1], ", ω = ", outhopf.p[end], ", from l = ",hopfpt.p[1],"\n")

br_hopf, u1_hopf = @time Cont.continuationHopf(
			(x, p, β) ->   F_bru(x, a, β, l = p),
			(x, p, β) ->  Jac_sp(x, a, β, l = p),
			br, ind_hopf,
			b,
			ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, pMax = 6.5, pMin = 0.0, a = 2., theta = 0.4, maxSteps = 3, newtonOptions = NewtonPar(verbose = false)), verbosity = 0, plot = false)

br_hopf, u1_hopf = @time Cont.continuationHopf(
			(x, p, β) ->   F_bru(x, a, β, l = p),
			(x, p, β) ->  Jac_sp(x, a, β, l = p),
			(x, p, β) ->  transpose(Jac_sp(x, a, β, l = p)),
			p2 -> (x, p1, v1, v2) -> d2F(x, 0., 0., v1, v2),
			br, ind_hopf,
			b,
			ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, pMax = 6.5, pMin = 0.0, a = 2., theta = 0.4, maxSteps = 3, newtonOptions = NewtonPar(verbose = false)), verbosity = 0, plot = false)
#################################################################################################### Continuation of Periodic Orbit
ind_hopf = 2
hopfpt = Cont.HopfPoint(br, ind_hopf)

l_hopf  = hopfpt.p[1]
ωH		= hopfpt.p[2] |> abs
M = 30


orbitguess = zeros(2n, M)
	vec_hopf = br.eig[br.bifpoint[ind_hopf][2]][2][:, br.bifpoint[ind_hopf][end]-1]
	for ii=1:M
	t = (ii-1)/(M-1)
	orbitguess[:, ii] .= real.(hopfpt.u +
						26*0.1 * vec_hopf * exp(2pi * complex(0, 1) * (t - 0.2790)))
end

orbitguess_f = vcat(vec(orbitguess), 2pi/ωH) |> vec

poTrap = l-> PeriodicOrbitTrap(
			x-> F_bru(x, a, b, l = l),
			x-> Jac_sp(x, a, b, l = l),
			real.(vec_hopf),
			hopfpt.u,
			M,
			opt_newton.linsolve)


jac_PO_fd = Cont.finiteDifferences(x -> poTrap(l_hopf + 0.01)(x), orbitguess_f)
jac_PO_sp =  poTrap(l_hopf + 0.01)(orbitguess_f, :jacsparse)

# test of the Jacobian for PeriodicOrbit via Finite differences VS the FD associated jacobian
println("--> test jacobian expression for Periodic Orbit solve problem")
@test norm(jac_PO_fd - jac_PO_sp, Inf64) < 1e-5


# newton to find Periodic orbit
opt_po = Cont.NewtonPar(tol = 1e-8, verbose = true, maxIter = 50)
	outpo_f, hist, flag = @time Cont.newton(
			x ->  poTrap(l_hopf + 0.01)(x),
			x ->  poTrap(l_hopf + 0.01)(x, :jacsparse),
			orbitguess_f,
			opt_po)
	println("--> T = ", outpo_f[end])

# calcul des exposants de Floquet
# floquetES = Cont.FloquetFD(poTrap(l_hopf + 0.01))
# floquetES(poTrap(l_hopf + 0.01), outpo_f, 10 )

# continuation of periodic orbits
opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.05, ds= 0.001, pMax = 2.3, maxSteps = 3, secant = true, theta = 0.1, plot_every_n_steps = 4, newtonOptions = NewtonPar(verbose = false))
	br_pok2, upo , _= @time Cont.continuation(
			(x, p) ->  poTrap(p)(x),
			(x, p) ->  poTrap(p)(x, :jacsparse),
			outpo_f, l_hopf + 0.01,
			opts_po_cont,
			plot = false,
			verbosity = 0)
#################################################################################################### Periodic Orbit computation, with Simple Shooting
poShoot = l-> ShootingProblemTrap(
			x-> F_bru(x, a, b, l = l),
			x-> Jac_sp(x, a, b, l = l),
			real.(vec_hopf),
			hopfpt.u,
			M,
			Default(),
			NewtonPar())
orbitguess_f = vcat(orbitguess[:,1], 2pi/ωH)
poShoot(l_hopf + 0.01)(orbitguess_f)
poshoot, _ = newton(u -> poShoot(l_hopf + 0.01)(u), orbitguess_f, NewtonPar(verbose = true))
println("--> T = ", poshoot[end])

poShoot = l-> ShootingProblemBE(
			x-> F_bru(x, a, b, l = l),
			x-> Jac_sp(x, a, b, l = l),
			real.(vec_hopf),
			hopfpt.u,
			M,
			Default(),
			NewtonPar())
orbitguess_f = vcat(orbitguess[:,1], 2pi/ωH)
poShoot(l_hopf + 0.01)(orbitguess_f)
poshoot, _ = newton(u -> poShoot(l_hopf + 0.01)(u), orbitguess_f, NewtonPar(verbose = true))
println("--> T = ", poshoot[end])

poShoot = l-> ShootingProblemMid(
			x-> F_bru(x, a, b, l = l),
			x-> Jac_sp(x, a, b, l = l),
			real.(vec_hopf),
			hopfpt.u,
			M,
			Default(),
			NewtonPar())
orbitguess_f = vcat(orbitguess[:,1], 2pi/ωH)
poShoot(l_hopf + 0.01)(orbitguess_f)
poshoot, _ = newton(
			u -> poShoot(l_hopf + 0.01)(u),
			# u -> (u, poShoot(l_hopf + 0.01)),
			orbitguess_f,
			NewtonPar(verbose = true, maxIter = 1))
println("--> T = ", poshoot[end])
