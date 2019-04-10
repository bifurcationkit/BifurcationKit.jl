using Test, PseudoArcLengthContinuation, LinearAlgebra, SparseArrays, Arpack
const Cont = PseudoArcLengthContinuation
####################################################################################################
# test the type BorderedVector and the different methods associated to it
z_pred = PseudoArcLengthContinuation.BorderedVector(rand(10),1.0)
tau_pred = PseudoArcLengthContinuation.BorderedVector(rand(10),2.0)
Cont.minus!(z_pred, tau_pred)
Cont.eltype(z_pred)

axpy!(2. /3, tau_pred, z_pred)
axpby!(2. /3, tau_pred, 1.0, z_pred)
dot(z_pred, tau_pred)
Cont.dottheta(z_pred, tau_pred, 0.1)
Cont.dottheta(z_pred.u, tau_pred.u, 1.0, 1.0, 0.1)

z = BorderedVector(z_pred, rand(10))
z2 = BorderedVector(z_pred, rand(10))
zero(z2);zero(z_pred)
@test length(z_pred) == 11

copyto!(z,z2)
Cont.minus(z.u,z2.u);Cont.minus!(z.u,z2.u)
Cont.minus(1.,2.);Cont.minus!(1.,2.)
####################################################################################################
# test the linear solver LinearBorderSolver
println("--> Test linear Bordered solver")
J0 = rand(10,10) * 0.1 + I
rhs = rand(10)
sol_explicit = J0 \ rhs
sol_bd1, sol_bd2, _ = PseudoArcLengthContinuation.linearBorderedSolver(J0[1:end-1,1:end-1], J0[1:end-1,end], J0[end,1:end-1], J0[end,end], rhs[1:end-1], rhs[end], Default())

sol_bd1, sol_bd2, _ = PseudoArcLengthContinuation.linearBorderedSolver(J0[1:end-1,1:end-1], J0[1:end-1,end], J0[end,1:end-1], J0[end,end], rhs[1:end-1], rhs[end], Default())

@test norm(sol_explicit[1:end-1] - sol_bd1, Inf64) < 1e-12
@test norm(sol_explicit[end] - sol_bd2, Inf64) < 1e-12
####################################################################################################
# test the linear solvers for matrix free formulations
J0 = I + sprand(100,100,0.1)
Jmf = x -> J0*x
x0 = rand(100)
ls = Default()
out = ls(J0, x0)

ls = GMRES_KrylovKit{Float64}(rtol = 1e-9, dim = 100)
outkk = ls(J0, x0)
@test norm(out[1] - outkk[1], Inf64) < 1e-7
outkk = ls(Jmf, x0)
@test norm(out[1] - outkk[1], Inf64) < 1e-7


ls = GMRES_IterativeSolvers{Float64}(N = 100, tol = 1e-9)
outit = ls(J0, x0)
@test norm(out[1] - outit[1], Inf64) < 1e-7
####################################################################################################
# test the eigen solvers for matrix free formulations
out = Arpack.eigs(J0, nev = 20, which = :LR)

eil = PseudoArcLengthContinuation.eig_KrylovKit(tol = 1e-9)
outkk = eil(J0, 20)
eil = PseudoArcLengthContinuation.eig_MF_KrylovKit(tol = 1e-9, x₀ = x0)
outkkmf = eil(Jmf, 20)

eil = PseudoArcLengthContinuation.Default_eig_sp()
outdefault = eil(J0, 20)
@test norm(out[1] - outdefault[1], Inf64) < 1e-7
