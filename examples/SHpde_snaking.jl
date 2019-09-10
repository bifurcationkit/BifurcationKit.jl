using Revise
using SparseArrays, LinearAlgebra, DiffEqOperators
using PseudoArcLengthContinuation
using Plots
const Cont = PseudoArcLengthContinuation
################################################################################
# case of the SH equation

nx = 400; Lx = 50; hx = 2*Lx/nx
const X = -Lx .+ 2Lx/nx*(1:nx) |> collect
BC = :periodic
Dxx = sparse(DerivativeOperator{Float64}(2, 2, hx, nx, BC, BC))
Lsh = -(I+Dxx)^2

function R_SH(u, p, b, L1)
	out = similar(u)
	out .= L1 * u .- p .* u .+ b .* u.^3 - u.^5
end

Jac_sp = (u, p, b, L1) -> L1 + spdiagm(0 => -p .+ 3*b .* u.^2 .- 5 .* u.^4)

Fpde = (u, p)-> R_SH(u, p, 2., Lsh)
# jacobian version with full matrix, waste of ressources!!
Jac_fd(u0, alpha) = Cont.finiteDifferences(u->Fpde(u, alpha), u0)

sol0 = 1.65cos.(X) .* exp.(-X.^2/(2*5^2))
	opt_new = Cont.NewtonPar(verbose = true, tol = 1e-11)
	# allocations 26.47k, 0.038s, tol = 1e-10
	sol1, hist, flag = @time Cont.newton(
				u ->   R_SH(u, 0.7, 2., Lsh),
				u -> Jac_sp(u, 0.7, 2., Lsh),
				sol0, opt_new)
	Plots.plot(X,sol1)

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
			doArcLengthScaling = false,
			a = 0.2,
			newtonOptions = opt_new,
			detect_fold = true, detect_bifurcation = false,
			maxSteps = 2080,
			theta = .4, plot_every_n_steps = 200)
	@assert opts.a<=1.5 "sinon ca peut changer le sens du time step"
	opts.newtonOptions.maxIter = 100
	opts.newtonOptions.tol = 1e-9
	opts.newtonOptions.linesearch = false
	br, u1 = @time Cont.continuation(
					(x, p)->  R_SH(x, p, 2., Lsh),
					(x, p)->Jac_sp(x, p, 2., Lsh),
					sol1, 0.7, opts,
					plot = true,
					plotsolution = (x;kwargs...)->(plot!(X, x, subplot=4, ylabel="solution", label="")))
#####################################################
# case with computation of eigenvalues
# opt_new = Cont.NewtonPar(linsolver = Default(),	eigsolver = eig_KrylovKit{Float64}())
