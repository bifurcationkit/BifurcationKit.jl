using Revise
	using SparseArrays, LinearAlgebra, DiffEqOperators, Setfield, Parameters
	using BifurcationKit
	using Plots
	const BK = BifurcationKit
################################################################################
# case of the SH equation
norminf(x) = norm(x, Inf64)
Nx = 200; Lx = 10.;
X = -Lx .+ 2Lx/Nx*(0:Nx-1) |> collect
hx = X[2]-X[1]

Q = Neumann0BC(hx)
# Q = Dirichlet0BC(hx |> typeof)
Dxx = sparse(CenteredDifference(2, 2, hx, Nx) * Q)[1]
Lsh = -(I + Dxx)^2

function R_SH(u, par)
	@unpack p, b, L1 = par
	out = similar(u)
	out .= L1 * u .- p .* u .+ b .* u.^3 - u.^5
end

Jac_sp = (u, par) -> par.L1 + spdiagm(0 => -par.p .+ 3*par.b .* u.^2 .- 5 .* u.^4)
parSH = (p = 0.7, b = 2., L1 = Lsh)
####################################################################################################
sol0 = 1.1cos.(X) .* exp.(-0X.^2/(2*5^2))
	optnew = BK.NewtonPar(verbose = true, tol = 1e-12)
	# allocations 26.47k, 0.038s, tol = 1e-10
	sol1, hist, flag = @time BK.newton(
	R_SH, Jac_sp,
	sol0, (@set parSH.p = -1.95), optnew)
	Plots.plot(X, sol1)


opts = BK.ContinuationPar(dsmin = 0.0001, dsmax = 0.01, ds = -0.005,
		newtonOptions = setproperties(optnew; maxIter = 30, tol = 1e-10), pMin = -1.,
		maxSteps = 300, plotEveryNsteps = 40, detectBifurcation = 2, nInversion = 4, tolBisectionEigenvalue = 1e-17, dsminBisection = 1e-7)

	plot = true,
		# tangentAlgo = BorderedPred(),
	linearAlgo  = MatrixBLS(),
		plotSolution = (x, p;kwargs...)->(plot!(X, x; ylabel="solution", label="", kwargs...)), normC = norminf)
	brs = [br]
#####################################################
# case with computation of eigenvalues
# optnew = PALC.NewtonPar(linsolver = Default(),	eigsolver = eig_KrylovKit{Float64}())
plot(brs, label = "")


####################################################################################################
sol0 = 1.1cos.(X) .* exp.(-0X.^2/(2*5^2))
	optnew = PALC.NewtonPar(verbose = true, tol = 1e-12)
	# allocations 26.47k, 0.038s, tol = 1e-10
	sol1, hist, flag = @time PALC.newton(
	R_SH, Jac_sp,
	sol0, (@set parSH.p = -1.95), optnew)
	Plots.plot(X, sol1)


opts = PALC.ContinuationPar(dsmin = 0.001, dsmax = 0.005, ds = 0.001,
		newtonOptions = setproperties(optnew; maxIter = 30, tol = 1e-11), pMin = -2.,
		maxSteps = 1000, theta = .4, plotEveryNsteps = 200, computeEigenValues = true)
	@assert opts.a<=1.5 "sinon ca peut changer le sens du time step"

	br, u1 = @time PALC.continuation(
		R_SH, Jac_sp, sol1, (@set parSH.p = -1.), (@lens _.p), opts,
		verbosity = 2,
		plot = true,
		# tangentAlgo = BorderedPred(),
		# linearAlgo  = MatrixBLS(),
		plotSolution = (x, p;kwargs...)->(plot!(X, x; ylabel="solution", label="", kwargs...)))

push!(brs, br)

plot(brs)
