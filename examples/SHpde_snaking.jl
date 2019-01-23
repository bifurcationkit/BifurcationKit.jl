using Revise
using JLD2, FileIO
using SparseArrays, LinearAlgebra, DiffEqOperators
using PseudoArcLengthContinuation
using Plots
const Cont = PseudoArcLengthContinuation
################################################################################
# case of the SH equation

nx = 400; Lx = 50; hx = 2*Lx/nx
BC = :periodic
Dxx = sparse(DerivativeOperator{Float64}(2, 2, hx, nx, BC, BC))
Lsh = -(I+Dxx)^2

function R_SH(u, p, b, L1)
	out = similar(u)
	out .= L1 * u .- p .* u .+ b .* u.^3 - u.^5
end

SHJac = (u, v, p, b, L1) -> L1*v .+ (-p .+ 3*b .* u.^2 .- 5*u.^4) .* v
Jac_sp = (u, p, b, L1) -> L1 + spdiagm(0 => -p .+ 3*b .* u.^2 .- 5*u.^4)

Fpde = (u, p)-> R_SH(u, p, 2., Lsh)
# jacobian version with full matrix, waste of ressources!!
Jac_fd(u0, alpha) = Cont.finiteDifferences(u->Fpde(u, alpha), u0)

solfile = load("snakingPDE.jld2")
u0 = solfile["u0"]

sol_ = copy(u0)
	opt_new = Cont.NewtonPar(verbose = true, tol = 1e-10)
	# allocations 26.47k, 0.038s, tol = 1e-10
	sol_, hist, flag = @time Cont.newton(
				u ->   R_SH(u, 0.7, 2., Lsh),
				u -> Jac_sp(u, 0.7, 2., Lsh),
				u0, opt_new)
	Plots.plot(sol_)

if 1==0
	# this is for testing
	# use of Sparse Jacobian
	sol_, hist, flag = @time Cont.newton(
				x->  R_SH(x, 0.7, 2., Lsh),
				u->Jac_sp(u, 0.7, 2., Lsh),
				u0, opt_new)

	# use of FD Jacobian
	sol_, hist, flag = @time Cont.newton(
				x->R_SH(x, 0.7, 2., Lsh),
				u->Jac_fd(u, 0.7),
				u0, opt_new)
end

opts = Cont.ContinuationPar(dsmin = 0.0001,
			dsmax = 0.0035,
			ds = -0.001,
			doArcLengthScaling = true,
			a = 0.1,
			NewtonOptions = opt_new,
			detect_fold = true,
			maxSteps = 1200,
			theta = .5)
	@assert opts.a<=1.5 "sinon ca peut changer le sens du time step"
	opts.NewtonOptions.maxIter = 100
	opts.NewtonOptions.tol = 1e-9
	opts.NewtonOptions.damped = false
	# opts.detect_fold = true

	# opts.maxSteps = 180
	# opts.theta = .5
	br, u1 = @time Cont.continuation(
					(x, p)->  R_SH(x, p, 2., Lsh),
					(x, p)->Jac_sp(x, p, 2., Lsh),
					sol_, 0.7, opts,
					plot = true,
					plotsolution = (x;kwargs...)->(plot!(x, subplot=4, ylabel="solution", label="")))
#####################################################
# case with computation of eigenvalues
# opt_new = Cont.NewtonPar(linsolve = Default(),	eigsolve = eig_KrylovKit{Float64}())
