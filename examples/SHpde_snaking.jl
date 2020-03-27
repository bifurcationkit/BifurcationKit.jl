using Revise
	using SparseArrays, LinearAlgebra, DiffEqOperators, Setfield
	using PseudoArcLengthContinuation
	using Plots
	const PALC = PseudoArcLengthContinuation
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
	optnew = PALC.NewtonPar(verbose = true, tol = 1e-12)
	# allocations 26.47k, 0.038s, tol = 1e-10
	sol1, hist, flag = @time PALC.newton(
				u ->   R_SH(u, 0.7, 2., Lsh),
				u -> Jac_sp(u, 0.7, 2., Lsh),
				sol0, optnew, normN = norminf)
	Plots.plot(X, sol1)

opts = PALC.ContinuationPar(dsmin = 0.0005, dsmax = 0.0055, ds = -0.001,
			newtonOptions = optnew,
			maxSteps = 1200,
			theta = .6, plotEveryNsteps = 200, computeEigenValues = true)
	br, u1 = @time PALC.continuation(
					(x, p)->  R_SH(x, p, 2., Lsh),
					(x, p)->Jac_sp(x, p, 2., Lsh),
					sol1, 0.7, opts,
					verbosity = 2,
					plot = true,
					# tangentAlgo = BorderedPred(),
					linearAlgo  = MatrixBLS(),
					plotSolution = (x, p;kwargs...)->(plot!(X, x; ylabel="solution", label="", kwargs...)), normC = norminf)
	brs = [br]
#####################################################
# case with computation of eigenvalues
# optnew = PALC.NewtonPar(linsolver = Default(),	eigsolver = eig_KrylovKit{Float64}())
plotBranch(brs, label = "");title!("")

####################################################################################################
sol0 = 1.1cos.(X) .* exp.(-0X.^2/(2*5^2))
	optnew = PALC.NewtonPar(verbose = true, tol = 1e-12)
	# allocations 26.47k, 0.038s, tol = 1e-10
	sol1, hist, flag = @time PALC.newton(
				u ->   R_SH(u, -1.95, 2., Lsh),
				u -> Jac_sp(u, -1.95, 2., Lsh),
				sol0, optnew)
	Plots.plot(X, sol1)


opts = PALC.ContinuationPar(dsmin = 0.001, dsmax = 0.005, ds = 0.001,
			newtonOptions = setproperties(optnew; maxIter = 30, tol = 1e-11), pMin = -2.,
			maxSteps = 1000, theta = .4, plotEveryNsteps = 200, computeEigenValues = true)
	@assert opts.a<=1.5 "sinon ca peut changer le sens du time step"

	br, u1 = @time PALC.continuation(
					(x, p)->  R_SH(x, p, 2., Lsh),
					(x, p)->Jac_sp(x, p, 2., Lsh),
					sol1, -1., opts,
					verbosity = 2,
					plot = true,
					# tangentAlgo = BorderedPred(),
					# linearAlgo  = MatrixBLS(),
					plotSolution = (x, p;kwargs...)->(plot!(X, x; ylabel="solution", label="", kwargs...)))

push!(brs, br)
