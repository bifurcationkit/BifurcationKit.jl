using Revise
	using SparseArrays, LinearAlgebra, DiffEqOperators
	using PseudoArcLengthContinuation
	using Plots
	const Cont = PseudoArcLengthContinuation
################################################################################
# case of the SH equation
norminf(x) = norm(x, Inf64)
Nx = 200; Lx = 30.;
X = -Lx .+ 2Lx/Nx*(0:Nx-1) |> collect
hx = X[2]-X[1]

Q = Neumann0BC(hx)
# Q = Dirichlet0BC(hx |> typeof)
Dxx = sparse(CenteredDifference(2, 2, hx, Nx) * Q)[1]
Lsh = -(I + Dxx)^2

function R_SH(u, p, b, L1)
	out = similar(u)
	out .= L1 * u .- p .* u .+ b .* u.^3 - u.^5
end

Jac_sp = (u, p, b, L1) -> L1 + spdiagm(0 => -p .+ 3*b .* u.^2 .- 5 .* u.^4)

sol0 = 1.65cos.(X) .* exp.(-X.^2/(2*5^2))
	opt_new = Cont.NewtonPar(verbose = true, tol = 1e-12)
	# allocations 26.47k, 0.038s, tol = 1e-10
	sol1, hist, flag = @time Cont.newton(
				u ->   R_SH(u, 0.7, 2., Lsh),
				u -> Jac_sp(u, 0.7, 2., Lsh),
				sol0, opt_new, normN = norminf)
	Plots.plot(X, sol1)

opts = Cont.ContinuationPar(dsmin = 0.0005, dsmax = 0.0055, ds = -0.001,
			newtonOptions = opt_new,
			detect_fold = true, maxSteps = 1200,
			theta = .6, plot_every_n_steps = 200)
	opts.newtonOptions.tol = 1e-12
	opts.computeEigenValues = true
	br, u1 = @time Cont.continuation(
					(x, p)->  R_SH(x, p, 2., Lsh),
					(x, p)->Jac_sp(x, p, 2., Lsh),
					sol1, 0.7, opts,
					verbosity = 2,
					plot = true,
					# tangentalgo = BorderedPred(),
					linearalgo  = MatrixBLS(),
					plotsolution = (x;kwargs...)->(plot!(X, x, subplot = 4, ylabel="solution", label="")), normC = norminf)
					brs = [br]
#####################################################
# case with computation of eigenvalues
# opt_new = Cont.NewtonPar(linsolver = Default(),	eigsolver = eig_KrylovKit{Float64}())
plotBranch(brs, label = "")

####################################################################################################
sol0 = 1.1cos.(X) .* exp.(-0X.^2/(2*5^2))
	opt_new = Cont.NewtonPar(verbose = true, tol = 1e-12)
	# allocations 26.47k, 0.038s, tol = 1e-10
	sol1, hist, flag = @time Cont.newton(
				u ->   R_SH(u, -1.95, 2., Lsh),
				u -> Jac_sp(u, -1.95, 2., Lsh),
				sol0, opt_new)
	Plots.plot(X, sol1)


opts = Cont.ContinuationPar(dsmin = 0.001, dsmax = 0.005, ds = 0.001,
			newtonOptions = opt_new, pMin = -2.,
			detect_fold = true, maxSteps = 1000,
			theta = .4, plot_every_n_steps = 200)
	@assert opts.a<=1.5 "sinon ca peut changer le sens du time step"
	opts.newtonOptions.maxIter = 30
	opts.newtonOptions.tol = 1e-11
	opts.computeEigenValues = true
	br, u1 = @time Cont.continuation(
					(x, p)->  R_SH(x, p, 2., Lsh),
					(x, p)->Jac_sp(x, p, 2., Lsh),
					sol1, -1., opts,
					verbosity = 2,
					plot = true,
					# tangentalgo = BorderedPred(),
					# linearalgo  = MatrixBLS(),
					plotsolution = (x;kwargs...)->(plot!(X, x, subplot = 4, ylabel="solution", label="")))

push!(brs, br)
