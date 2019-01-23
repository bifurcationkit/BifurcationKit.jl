using Test, PseudoArcLengthContinuation, LinearAlgebra

# test the type BorderedVector
z_pred = PseudoArcLengthContinuation.BorderedVector(rand(10),1.0)
tau_pred = PseudoArcLengthContinuation.BorderedVector(rand(10),2.0)
z_pred = z_pred + 2tau_pred

# test the linear solver LinearBorderSolver
J0 = rand(10,10) * 0.1 + I
rhs = rand(10)
sol_explicit = J0 \ rhs
sol_bd1, sol_bd2, _ = PseudoArcLengthContinuation.linearBorderedSolver(J0[1:end-1,1:end-1], J0[1:end-1,end], J0[end,1:end-1], J0[end,end], rhs[1:end-1], rhs[end], Default())

@test norm(sol_explicit[1:end-1] - sol_bd1, Inf64) < 1e-12
@test norm(sol_explicit[end] - sol_bd2, Inf64) < 1e-12
