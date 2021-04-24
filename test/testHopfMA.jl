# using Revise
using Test, BifurcationKit, LinearAlgebra, SparseArrays, Setfield, Parameters
const BK = BifurcationKit

f1(u, v) = u^2 * v
norminf = x -> norm(x, Inf)

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

	f[n]   = c1 * (u[n-1] - 2u[n] +  α  )  + α - (β + 1) * u[n] + f1(u[n], v[n])
	f[n+1] = c2 * (β / α  - 2v[1] + v[2])			 + β * u[1] - f1(u[1], v[1])

	for i=2:n-1
		  f[i] = c1 * (u[i-1] - 2u[i] + u[i+1]) + α - (β + 1) * u[i] + f1(u[i], v[i])
		f[n+i] = c2 * (v[i-1] - 2v[i] + v[i+1])			  + β * u[i] - f1(u[i], v[i])
	end
	return f
end

Fbru(x, p) = Fbru!(similar(x), x, p)

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

	@. diagp1[1:n-1] = c1
	@. diagp1[n+1:end] = c2

	@. diagpn = u * u
	J = spdiagm(0 => diag, 1 => diagp1, -1 => diagm1, n => diagpn, -n => diagmn)
	return J
end

n = 100
par_bru = (α = 2., β = 5.45, D1 = 0.008, D2 = 0.004, l = 0.3)
	sol0 = vcat(par_bru.α * ones(n), par_bru.β/par_bru.α * ones(n))

opt_newton = NewtonPar(tol = 1e-11, verbose = true)
	out, hist, flag = @time newton(Fbru, Jbru_sp, sol0 .* (1 .+ 0.01rand(2n)), par_bru, opt_newton)

# eigls = EigArpack(1.1, :LM)
opt_newton = NewtonPar(tol = 1e-11)
	out, = @time newton(Fbru, Jbru_sp,
		sol0 .* (1 .+ 0.01rand(2n)), par_bru,
		opt_newton)

opts_br0 = ContinuationPar(dsmin = 0.001, dsmax = 0.1, ds= 0.01, pMax = 1.8, detectBifurcation = 3, nev = 16, nInversion = 4)
	br, = @time continuation(Fbru, Jbru_sp,out, (@set par_bru.l = 0.3), (@lens _.l), opts_br0, printSolution = (x, p) -> norm(x, Inf64))
###################################################################################################
# Hopf continuation with automatic procedure
outhopf, = newtonHopf(Fbru, Jbru_sp, br, 1; startWithEigen = true)
optconthopf = ContinuationPar(dsmin = 0.001, dsmax = 0.15, ds= 0.01, pMax = 6.8, pMin = 0., newtonOptions = opt_newton, maxSteps = 5)
outhopfco, = continuationHopf(Fbru, Jbru_sp, br, 1, (@lens _.β), optconthopf; startWithEigen = true)

# Continuation of the Hopf Point using Dense method
ind_hopf = 1
# av = randn(Complex{Float64},2n); av = av./norm(av)
# bv = randn(Complex{Float64},2n); bv = bv./norm(bv)
hopfpt = BK.HopfPoint(br, ind_hopf)
bifpt = br.bifpoint[ind_hopf]
hopfvariable = HopfProblemMinimallyAugmented(
					Fbru, Jbru_sp, (x, p) -> transpose(Jbru_sp(x, p)), nothing,
					(@lens _.l),
					conj.(br.eig[bifpt.idx].eigenvec[:, bifpt.ind_ev]),
					(br.eig[bifpt.idx].eigenvec[:, bifpt.ind_ev]),
					# av,
					# bv,
					opts_br0.newtonOptions.linsolver)

hopfvariable(hopfpt, par_bru) |> norm

Bd2Vec(x) = vcat(x.u, x.p)
Vec2Bd(x) = BorderedArray(x[1:end-2], x[end-1:end])
hopfpbVec(x, p) = Bd2Vec(hopfvariable(Vec2Bd(x),p))

# finite differences Jacobian
Jac_hopf_fdMA(u0, p) = BK.finiteDifferences( u-> hopfpbVec(u, p), u0)
# ``analytical'' jacobian
Jac_hopf_MA(u0, p, pb::HopfProblemMinimallyAugmented) = (return (x=u0,params=p ,hopfpb=pb))

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
# v, σ1, _ = BK.linearBorderedSolver(Jh + Complex(0, ω) * I, av, bv, 0., zeros(2n), 1.0, hopfvariable(b).linsolve)
#
# w, σ2, _ = BK.linearBorderedSolver(Jh' - Complex(0, ω) * I, bv, av, 0., zeros(2n), 1.0, hopfvariable(b).linsolve)
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

outhopf, _, flag, _ = @time newton((u, p) -> hopfpbVec(u, p),
							# u -> Jac_hopf_fdMA(u, par_bru.β),
							Bd2Vec(hopfpt), par_bru, NewtonPar(verbose = true, tol = 1e-8, maxIter = 10))
	flag && printstyled(color=:red, "--> We found a Hopf Point at l = ", outhopf[end-1], ", ω = ", outhopf[end], " from ", hopfpt.p, "\n")

rhs = rand(length(hopfpt))
jac_hopf_fd = Jac_hopf_fdMA(Bd2Vec(hopfpt), par_bru)
sol_fd = jac_hopf_fd \ rhs
# create a linear solver
hopfls = BK.HopfLinearSolverMinAug()
sol_ma, _, _, sigomMA  = hopfls(Jac_hopf_MA(hopfpt, par_bru, hopfvariable), BorderedArray(rhs[1:end-2],rhs[end-1:end]), debug_ = true)

# TODO TODO fix these two lines
println("--> test jacobian expression for Hopf Minimally Augmented")
@test Bd2Vec(sol_ma) - sol_fd |> x-> norm(x, Inf64) < 1e3

@test (Bd2Vec(sol_ma) - sol_fd)[1:end-2] |> x->norm(x, Inf64) < 1e-1

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
outhopf, hist, flag = @time newton(
		Fbru, Jbru_sp, br, 1, Jᵗ = (x, p) -> transpose(Jbru_sp(x, p)))
		flag && printstyled(color=:red, "--> We found a Hopf Point at l = ", outhopf.p[1], ", ω = ", outhopf.p[end], ", from l = ",hopfpt.p[1],"\n")

outhopf, _, flag, _ = @time newton((u, p) -> hopfvariable(u, p),
							(x, p) -> Jac_hopf_MA(x, p, hopfvariable),
							hopfpt, par_bru, NewtonPar(verbose = true, linsolver = BK.HopfLinearSolverMinAug()))
	flag && printstyled(color=:red, "--> We found a Hopf Point at l = ", outhopf.p[1], ", ω = ", outhopf.p[2], " from ", hopfpt.p, "\n")

# version with analytical Hessian = 2 P(du2) P(du1) QU + 2 PU P(du1) Q(du2) + 2PU P(du2) Q(du1)
function d2F(x, p1, du1, du2)
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

outhopf, hist, flag = @time newton(Fbru, Jbru_sp, br, 1,
		Jᵗ = (x, p) -> transpose(Jbru_sp(x, p)),
		d2F = (x, p1, v1, v2) -> d2F(x, 0., v1, v2))
		flag && printstyled(color=:red, "--> We found a Hopf Point at l = ", outhopf.p[1], ", ω = ", outhopf.p[end], ", from l = ",hopfpt.p[1],"\n")

br_hopf, u1_hopf = @time continuation(
			Fbru, Jbru_sp, br, ind_hopf, (@lens _.β),
			ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, pMax = 6.5, pMin = 0.0, a = 2., theta = 0.4, maxSteps = 3, newtonOptions = NewtonPar(verbose = false)), verbosity = 1, plot = false)

br_hopf, u1_hopf = @time continuation(Fbru, Jbru_sp, br, ind_hopf, (@lens _.β), ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, pMax = 6.5, pMin = 0.0, a = 2., theta = 0.4, maxSteps = 3, newtonOptions = NewtonPar(verbose = false)), Jᵗ = (x, p) ->  transpose(Jbru_sp(x, p)), d2F = (x, p1, v1, v2) -> d2F(x, 0., v1, v2), verbosity = 0, plot = false)
#################################################################################################### Continuation of Periodic Orbit
ind_hopf = 1
hopfpt = BK.HopfPoint(br, ind_hopf)

l_hopf = hopfpt.p[1]
ωH	   = hopfpt.p[2] |> abs
M = 20


orbitguess = zeros(2n, M)
phase = []; scalphase = []
vec_hopf = geteigenvector(opt_newton.eigsolver, br.eig[br.bifpoint[ind_hopf].idx][2], br.bifpoint[ind_hopf].ind_ev-1)
for ii=1:M
	t = (ii-1)/(M-1)
	orbitguess[:, ii] .= real.(hopfpt.u + 26*0.1 * vec_hopf * exp(-2pi * complex(0, 1) * (t - .252)))
end

orbitguess_f = vcat(vec(orbitguess), 2pi/ωH) |> vec

# test guess using function
l_hopf, Th, orbitguess2, hopfpt, vec_hopf = BK.guessFromHopf(br, ind_hopf, opt_newton.eigsolver, M, 2.6; phase = 0.252)

poTrap = PeriodicOrbitTrapProblem(Fbru, Jbru_sp, real.(vec_hopf), hopfpt.u, M, 2n	)

jac_PO_fd = BK.finiteDifferences(x -> poTrap(x, (@set par_bru.l = l_hopf + 0.01)), orbitguess_f)
jac_PO_sp = poTrap(Val(:JacFullSparse), orbitguess_f, (@set par_bru.l = l_hopf + 0.01))

# test of the Jacobian for PeriodicOrbit via Finite differences VS the FD associated jacobian
println("--> test jacobian expression for Periodic Orbit solve problem")
@test norm(jac_PO_fd - jac_PO_sp, Inf64) < 1e-4

# test various jacobians and methods
jac_PO_sp =  poTrap(Val(:BlockDiagSparse), orbitguess_f, (@set par_bru.l = l_hopf + 0.01))
BK.getTimeDiff(poTrap, orbitguess_f)
# BK.Jc(poTrap, reshape(orbitguess_f[1:end-1], 2n, M), par_bru, reshape(orbitguess_f[1:end-1], 2n, M))
# BK.Jc(poTrap, orbitguess_f, par_bru, orbitguess_f)

# newton to find Periodic orbit
opt_po = NewtonPar(tol = 1e-8, verbose = true, maxIter = 150)
	outpo_f, _, flag = @time newton(
		(x, p) ->  poTrap(x, p),
		(x, p) ->  poTrap(Val(:JacFullSparse),x,p),
		copy(orbitguess_f), (@set par_bru.l = l_hopf + 0.01), opt_po)
	println("--> T = ", outpo_f[end])
flag && printstyled(color=:red, "--> T = ", outpo_f[end], ", amplitude = ", BK.amplitude(outpo_f, n, M; ratio = 2),"\n")

newton(poTrap, orbitguess_f, (@set par_bru.l = l_hopf + 0.01), opt_po; linearPO = :FullLU)

# jacobian of the functional
Jpo2 = poTrap(Val(:JacCyclicSparse), orbitguess_f, (@set par_bru.l = l_hopf + 0.01))

# calcul des exposants de Floquet, extract full vector
BK.MonodromyQaD(Val(:ExtractEigenVector), poTrap, orbitguess_f, par_bru, orbitguess_f[1:2n])

# calcul des exposants de Floquet
floquetES = FloquetQaD(DefaultEig())

# continuation of periodic orbits using :BorderedLU linear algorithm
opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.05, ds= 0.001, pMax = 2.3, maxSteps = 3, theta = 0.1, newtonOptions = NewtonPar(verbose = false), detectBifurcation = 1)
	br_pok2, = @time continuation(
		poTrap, orbitguess_f, (@set par_bru.l = l_hopf + 0.01), (@lens _.l), opts_po_cont; linearPO = :BorderedLU,
		plot = false, verbosity = 0)

# test of simple calls to newton / continuation
deflationOp = DeflationOperator(2.0, (x,y) -> dot(x[1:end-1], y[1:end-1]),1.0, [zero(orbitguess_f)])
# opt_po = NewtonPar(tol = 1e-8, verbose = true, maxIter = 15)
opts_po_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.03, ds= 0.01, pMax = 3.0, maxSteps = 3, newtonOptions = (@set opt_po.verbose = false), nev = 2, precisionStability = 1e-8, detectBifurcation = 1)
for linalgo in [:FullLU, :BorderedLU, :FullSparseInplace]
	@show linalgo
	# with deflation
	@time newton(poTrap,
			copy(orbitguess_f), (@set par_bru.l = l_hopf + 0.01), opt_po, deflationOp; linalgo = linalgo, normN = norminf)
	# classic Newton-Krylov
	outpo_f, hist, flag = @time newton(poTrap,
			copy(orbitguess_f), (@set par_bru.l = l_hopf + 0.01), opt_po; linalgo = linalgo, normN = norminf)
	# continuation
	br_pok2, = @time continuation(poTrap,
			copy(orbitguess_f), (@set par_bru.l = l_hopf + 0.01), (@lens _.l),
			opts_po_cont; linearPO = linalgo, verbosity = 0,
			plot = false, normC = norminf)
end

# test of Matrix free computation of Floquet exponents
eil = EigKrylovKit(x₀ = rand(2n))
ls = GMRESKrylovKit()
ls = DefaultLS()
opt_po = NewtonPar(tol = 1e-8, verbose = true, maxIter = 10, linsolver = ls, eigsolver = eil)
opts_po_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.03, ds= 0.01, pMax = 3.0, maxSteps = 3, newtonOptions = (@set opt_po.verbose = false), nev = 2, precisionStability = 1e-8, detectBifurcation = 2)
br_pok2, upo, = continuation(poTrap, outpo_f, (@set par_bru.l = l_hopf + 0.01), (@lens _.l), opts_po_cont; linearPO = :FullLU, normC = norminf, verbosity = 0)
