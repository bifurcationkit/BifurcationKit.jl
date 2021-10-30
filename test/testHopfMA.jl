# using Revise
using Test, BifurcationKit, LinearAlgebra, SparseArrays, Setfield, Parameters, ForwardDiff
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

Jbru_ana(x, p) = ForwardDiff.jacobian(z->Fbru(z,p),x)
jet = BK.getJet(Fbru, Jbru_ana)

n = 100
par_bru = (α = 2., β = 5.45, D1 = 0.008, D2 = 0.004, l = 0.3)
	sol0 = vcat(par_bru.α * ones(n), par_bru.β/par_bru.α * ones(n))

# test that the jacobian is well computed
@test Jbru_sp(sol0, par_bru) - Jbru_ana(sol0, par_bru) |> sparse |> nnz == 0

opt_newton = NewtonPar(tol = 1e-11, verbose = false)
	out, hist, flag = newton(jet[1], jet[2], sol0 .* (1 .+ 0.01rand(2n)), par_bru, opt_newton)

opts_br0 = ContinuationPar(dsmin = 0.001, dsmax = 0.1, ds= 0.01, pMax = 1.8, newtonOptions = opt_newton, detectBifurcation = 3, nev = 16, nInversion = 4)
	br, = continuation(jet[1], jet[2], out, (@set par_bru.l = 0.3), (@lens _.l), opts_br0, recordFromSolution = (x, p) -> norm(x, Inf64))
###################################################################################################
# Hopf continuation with automatic procedure
outhopf, = newton(jet[1], jet[2], br, 1; startWithEigen = true, d2F = jet[3])
outhopf, = newtonHopf(jet[1], jet[2], br, 1; startWithEigen = true)
optconthopf = ContinuationPar(dsmin = 0.001, dsmax = 0.15, ds= 0.01, pMax = 6.8, pMin = 0., newtonOptions = opt_newton, maxSteps = 50, detectBifurcation = 2)
outhopfco, = continuationHopf(jet[1], jet[2], br, 1, (@lens _.β), optconthopf; startWithEigen = true, updateMinAugEveryStep = 1, plot = false)

# Continuation of the Hopf Point using Dense method
ind_hopf = 1
# av = randn(Complex{Float64},2n); av = av./norm(av)
# bv = randn(Complex{Float64},2n); bv = bv./norm(bv)
hopfpt = BK.HopfPoint(br, ind_hopf)
bifpt = br.specialpoint[ind_hopf]
hopfvariable = HopfProblemMinimallyAugmented(
					jet[1], jet[2], nothing, nothing,
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
Jac_hopf_fdMA(u0, p) = ForwardDiff.jacobian( u-> hopfpbVec(u, p), u0)
# ``analytical'' jacobian
Jac_hopf_MA(u0, p, pb::HopfProblemMinimallyAugmented) = (return (x=u0,params=p ,hopfpb=pb))

rhs = rand(length(hopfpt))
jac_hopf_fd = Jac_hopf_fdMA(Bd2Vec(hopfpt), par_bru)
sol_fd = jac_hopf_fd \ rhs

# create a linear solver
hopfls = BK.HopfLinearSolverMinAug()
tmpVecforσ = zeros(ComplexF64, 2+2n)
sol_ma,  = hopfls(Jac_hopf_MA(hopfpt, par_bru, hopfvariable), BorderedArray(rhs[1:end-2],rhs[end-1:end]), debugArray = tmpVecforσ)

# we test the expression for σp
σp_fd = Complex(jac_hopf_fd[end-1,end-1], jac_hopf_fd[end,end-1])
σp_fd_ana = tmpVecforσ[1]
@test σp_fd ≈ σp_fd_ana rtol = 1e-5

# we test the expression for σω
σω_fd = Complex(jac_hopf_fd[end-1,end], jac_hopf_fd[end,end])
σω_fd_ana = tmpVecforσ[2]
@test σω_fd ≈ σω_fd_ana rtol = 1e-4

# we test the expression for σx
σx_fd = jac_hopf_fd[end-1, 1:end-2] + Complex(0,1) * jac_hopf_fd[end, 1:end-2]
σx_fd_ana = tmpVecforσ[3:end]
@test σx_fd ≈ σx_fd_ana rtol = 1e-3

outhopf, hist, flag = newton(
		jet[1], jet[2], br, 1, Jᵗ = (x, p) -> transpose(Jbru_sp(x, p)))
		# flag && printstyled(color=:red, "--> We found a Hopf Point at l = ", outhopf.p[1], ", ω = ", outhopf.p[end], ", from l = ",hopfpt.p[1],"\n")

outhopf, _, flag, _ = newton((u, p) -> hopfvariable(u, p),
							(x, p) -> Jac_hopf_MA(x, p, hopfvariable),
							hopfpt, par_bru, NewtonPar(verbose = false, linsolver = BK.HopfLinearSolverMinAug()))
	# flag && printstyled(color=:red, "--> We found a Hopf Point at l = ", outhopf.p[1], ", ω = ", outhopf.p[2], " from ", hopfpt.p, "\n")

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

outhopf, hist, flag = newton(jet[1], jet[2], br, 1,
		Jᵗ = (x, p) -> transpose(Jbru_sp(x, p)),
		d2F = (x, p1, v1, v2) -> d2F(x, 0., v1, v2))
		# flag && printstyled(color=:red, "--> We found a Hopf Point at l = ", outhopf.p[1], ", ω = ", outhopf.p[end], ", from l = ",hopfpt.p[1],"\n")

br_hopf, u1_hopf = continuation(
			jet[1], jet[2], br, ind_hopf, (@lens _.β),
			ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, pMax = 6.5, pMin = 0.0, a = 2., theta = 0.4, maxSteps = 3, newtonOptions = NewtonPar(verbose = false)), verbosity = 0, plot = false)

br_hopf, u1_hopf = continuation(jet[1], jet[2], br, ind_hopf, (@lens _.β), ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, pMax = 6.5, pMin = 0.0, a = 2., theta = 0.4, maxSteps = 3, newtonOptions = NewtonPar(verbose = false)), Jᵗ = (x, p) ->  transpose(Jbru_sp(x, p)), d2F = (x, p1, v1, v2) -> d2F(x, 0., v1, v2), verbosity = 0, plot = false)
#################################################################################################### Continuation of Periodic Orbit
ind_hopf = 1
hopfpt = BK.HopfPoint(br, ind_hopf)

l_hopf = hopfpt.p[1]
ωH	   = hopfpt.p[2] |> abs
M = 20


orbitguess = zeros(2n, M)
phase = []; scalphase = []
vec_hopf = geteigenvector(opt_newton.eigsolver, br.eig[br.specialpoint[ind_hopf].idx][2], br.specialpoint[ind_hopf].ind_ev-1)
for ii=1:M
	t = (ii-1)/(M-1)
	orbitguess[:, ii] .= real.(hopfpt.u + 26*0.1 * vec_hopf * exp(-2pi * complex(0, 1) * (t - .252)))
end

orbitguess_f = vcat(vec(orbitguess), 2pi/ωH) |> vec

# test guess using function
l_hopf, Th, orbitguess2, hopfpt, vec_hopf = BK.guessFromHopf(br, ind_hopf, opt_newton.eigsolver, M, 2.6; phase = 0.252)

poTrap = PeriodicOrbitTrapProblem(jet[1], Jbru_sp, real.(vec_hopf), hopfpt.u, M, 2n	)

jac_PO_fd = BK.finiteDifferences(x -> poTrap(x, (@set par_bru.l = l_hopf + 0.01)), orbitguess_f)
jac_PO_sp = poTrap(Val(:JacFullSparse), orbitguess_f, (@set par_bru.l = l_hopf + 0.01))

# test of the Jacobian for PeriodicOrbit via Finite differences VS the FD associated jacobian
# test jacobian expression for Periodic Orbit solve problem
@test norm(jac_PO_fd - jac_PO_sp, Inf64) < 1e-4

# test various jacobians and methods
jac_PO_sp =  poTrap(Val(:BlockDiagSparse), orbitguess_f, (@set par_bru.l = l_hopf + 0.01))
BK.getTimeDiff(poTrap, orbitguess_f)
# BK.Jc(poTrap, reshape(orbitguess_f[1:end-1], 2n, M), par_bru, reshape(orbitguess_f[1:end-1], 2n, M))
# BK.Jc(poTrap, orbitguess_f, par_bru, orbitguess_f)

# newton to find Periodic orbit
opt_po = NewtonPar(tol = 1e-8, verbose = false, maxIter = 150)
	outpo_f, _, flag = @time newton(
		(x, p) ->  poTrap(x, p),
		(x, p) ->  poTrap(Val(:JacFullSparse),x,p),
		copy(orbitguess_f), (@set par_bru.l = l_hopf + 0.01), opt_po)
	# println("--> T = ", outpo_f[end])
# flag && printstyled(color=:red, "--> T = ", outpo_f[end], ", amplitude = ", BK.amplitude(outpo_f, n, M; ratio = 2),"\n")

newton(poTrap, orbitguess_f, (@set par_bru.l = l_hopf + 0.01), opt_po; linearPO = :FullLU)

# jacobian of the functional
Jpo2 = poTrap(Val(:JacCyclicSparse), orbitguess_f, (@set par_bru.l = l_hopf + 0.01))

# calcul des exposants de Floquet, extract full vector
BK.MonodromyQaD(Val(:ExtractEigenVector), poTrap, orbitguess_f, par_bru, orbitguess_f[1:2n])

# calcul des exposants de Floquet
floquetES = FloquetQaD(DefaultEig())

# continuation of periodic orbits using :BorderedLU linear algorithm
opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.05, ds= 0.001, pMax = 2.3, maxSteps = 3, theta = 0.1, newtonOptions = NewtonPar(verbose = false), detectBifurcation = 1)
	br_pok2, = continuation(
		poTrap, orbitguess_f, (@set par_bru.l = l_hopf + 0.01), (@lens _.l), opts_po_cont; linearPO = :BorderedLU,
		plot = false, verbosity = 0)

# test of simple calls to newton / continuation
deflationOp = DeflationOperator(2.0, (x,y) -> dot(x[1:end-1], y[1:end-1]),1.0, [zero(orbitguess_f)])
# opt_po = NewtonPar(tol = 1e-8, verbose = false, maxIter = 15)
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
opt_po = NewtonPar(tol = 1e-8, verbose = false, maxIter = 10, linsolver = ls, eigsolver = eil)
opts_po_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.03, ds= 0.01, pMax = 3.0, maxSteps = 3, newtonOptions = (@set opt_po.verbose = false), nev = 2, precisionStability = 1e-8, detectBifurcation = 2)
br_pok2, upo, = continuation(poTrap, outpo_f, (@set par_bru.l = l_hopf + 0.01), (@lens _.l), opts_po_cont; linearPO = :FullLU, normC = norminf, verbosity = 0)
