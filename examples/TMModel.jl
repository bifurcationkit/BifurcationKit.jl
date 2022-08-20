using Revise, Test, ForwardDiff, Parameters, Setfield, Plots, LinearAlgebra
using BifurcationKit
const BK = BifurcationKit

norminf(x) = norm(x, Inf)
####################################################################################################
function TMvf!(dz, z, p, t)
	@unpack J, α, E0, τ, τD, τF, U0 = p
	E, x, u = z
	SS0 = J * u * x * E + E0
	SS1 = α * log(1 + exp(SS0 / α))
	dz[1] = (-E + SS1) / τ
	dz[2] =	(1.0 - x) / τD - u * x * E
	dz[3] = (U0 - u) / τF +  U0 * (1.0 - u) * E
	dz
end

TMvf(z, p) = TMvf!(similar(z), z, p, 0)

par_tm = (α = 1.5, τ = 0.013, J = 3.07, E0 = -2.0, τD = 0.200, U0 = 0.3, τF = 1.5, τS = 0.007) #2.87
z0 = [0.238616, 0.982747, 0.367876 ]
prob = BK.BifurcationProblem(TMvf, z0, par_tm, (@lens _.E0); recordFromSolution = (x, p) -> (E = x[1], x = x[2], u = x[3]),)

opts_br = ContinuationPar(pMin = -10.0, pMax = -0.9, ds = 0.04, dsmax = 0.125, nInversion = 8, detectBifurcation = 3, maxBisectionSteps = 25, nev = 3)
	opts_br = @set opts_br.newtonOptions.verbose = false
	br = continuation(prob, PALC(tangent=Bordered()), opts_br;
	plot = true, normC = norminf)

plot(br, plotfold=false, markersize=4, legend=:topleft)
####################################################################################################
hopfpt = getNormalForm(br, 4)

# newton parameters
optn_po = NewtonPar(verbose = true, tol = 1e-8,  maxIter = 8)

# continuation parameters
opts_po_cont = ContinuationPar(dsmax = 0.1, ds= 0.0001, dsmin = 1e-4, pMax = 0., pMin=-5., maxSteps = 110, newtonOptions = (@set optn_po.tol = 1e-7), nev = 3, tolStability = 1e-8, detectBifurcation = 0, plotEveryStep = 20, saveSolEveryStep=1)

# arguments for periodic orbits
args_po = (	recordFromSolution = (x, p) -> begin
		xtt = BK.getPeriodicOrbit(p.prob, x, @set par_tm.E0 = p.p)
		return (max = maximum(xtt[1,:]),
				min = minimum(xtt[1,:]),
				period = getPeriod(p.prob, x, @set par_tm.E0 = p.p))
	end,
	plotSolution = (x, p; k...) -> begin
		xtt = BK.getPeriodicOrbit(p.prob, x, @set par_tm.E0 = p.p)
		@show size(xtt[:,:]) maximum(xtt[1,:])
		plot!(xtt.t, xtt[1,:]; label = "E", k...)
		plot!(xtt.t, xtt[2,:]; label = "x", k...)
		plot!(xtt.t, xtt[3,:]; label = "u", k...)
		plot!(br; subplot = 1, putspecialptlegend = false)
		end,
	normC = norminf)

Mt = 200 # number of sections
	br_potrap = continuation(br, 4, opts_po_cont,
					PeriodicOrbitTrapProblem(M = Mt, jacobian = :Dense, updateSectionEveryStep = 0);
					verbosity = 2,	plot = true,
					args_po...,
					callbackN = BK.cbMaxNorm(1000.),
					)

plot(br, br_potrap, markersize = 3)
	plot!(br_potrap.param, br_potrap.min, label = "")
####################################################################################################
# based on collocation
hopfpt = getNormalForm(br, 4)

# newton parameters
optn_po = NewtonPar(verbose = true, tol = 1e-8,  maxIter = 10)

# continuation parameters
opts_po_cont = ContinuationPar(dsmax = 0.1, ds= -0.001, dsmin = 1e-4, pMax = 0., pMin=-5., maxSteps = 150, newtonOptions = (@set optn_po.tol = 1e-7), nev = 3, tolStability = 1e-5, detectBifurcation = 0, plotEveryStep = 40, saveSolEveryStep=1)

br_pocoll = @time continuation(
	br, 4, opts_po_cont,
	PeriodicOrbitOCollProblem(20, 5, updateSectionEveryStep = 0);
	verbosity = 2,	plot = true,
	args_po...,
	plotSolution = (x, p; k...) -> begin
		xtt = BK.getPeriodicOrbit(p.prob, x, p.p)
		plot!(xtt.t, xtt[1,:]; label = "", marker =:d, markersize = 1.5, k...)
		plot!(br; subplot = 1, putspecialptlegend = false)

	end,
	callbackN = BK.cbMaxNorm(1000.),
	)

plot(br, br_pocoll, markersize = 3)
	# plot!(br_pocoll.param, br_pocoll.min, label = "")
	# plot!(br, br_potrap, markersize = 3)
	# plot!(br_potrap.param, br_potrap.min, label = "", marker = :d)

####################################################################################################
# idem with Standard shooting
using DifferentialEquations#, TaylorIntegration

# this is the ODEProblem used with `DiffEqBase.solve`
probsh = ODEProblem(TMvf!, copy(z0), (0., 1000.), par_tm; abstol = 1e-10, reltol = 1e-9)

opts_po_cont = ContinuationPar(dsmax = 0.09, ds= -0.0001, dsmin = 1e-4, pMax = 0., pMin=-5., maxSteps = 120, newtonOptions = NewtonPar(optn_po; tol = 1e-6, maxIter = 7), nev = 3, tolStability = 1e-8, detectBifurcation = 0, plotEveryStep = 10, saveSolEveryStep=1)

br_posh = @time continuation(
	br, 4,
	# arguments for continuation
	opts_po_cont,
	# this is where we tell that we want Standard Shooting
	ShootingProblem(15, probsh, Rodas4P(), parallel = true, updateSectionEveryStep = 1, jacobian = :autodiffDense,);
	# ShootingProblem(15, probsh, TaylorMethod(15), parallel = false);
	ampfactor = 1.0, δp = 0.0005,
	usedeflation = true,
	# OPTION UTILE?
	linearAlgo = MatrixBLS(),
	verbosity = 2,	plot = true,
	args_po...,
	)

plot(br_posh, br, markersize=3)
	# plot(br, br_potrap, br_posh, markersize=3)
####################################################################################################
# idem with Poincaré shooting
@assert 1==0 "There is a PB with DE!!! I am getting NaN :("

function TMvfExtended!(dz, z, p, t)
	# we write the first part
	TMvf!(dz, z, p, t)
	dz[4:end] .= @views ForwardDiff.derivative(t -> TMvf(z[1:3] .+ t .* z[4:end], p), 0)
end

probmono = ODEProblem(TMvfExtended!, vcat(z0, z0), (0., 1000.), par_tm; abstol = 1e-10, reltol = 1e-9)

opts_po_cont = ContinuationPar(dsmax = 0.02, ds= 0.001, dsmin = 1e-6, pMax = 0., pMin=-5., maxSteps = 200, newtonOptions = NewtonPar(optn_po;tol = 1e-6, maxIter=35), nev = 3, tolStability = 1e-8, detectBifurcation = 0, plotEveryStep = 1, saveSolEveryStep = 1)

br_popsh = @time continuation(
	br, 4,
	# arguments for continuation
	opts_po_cont,
	# this is where we tell that we want Poincaré Shooting
	# PoincareShootingProblem(Mt, probsh, Rodas5(), probmono, Rodas4P(), parallel = false);
	PoincareShootingProblem(1, probsh, Rodas5(); parallel = false, updateSectionEveryStep = 2);
	# PoincareShootingProblem(Mt, probsh, RadauIIA3(); parallel = false, abstol = 1e-10, reltol = 1e-9);
	ampfactor = 1.0, δp = 0.005,
	# usedeflation = true,
	linearAlgo = MatrixBLS(),
	verbosity = 2,	plot = true,
	args_po...,
	callbackN = BK.cbMaxNorm(1e2),
	recordFromSolution = (x, p) -> (return (max = getMaximum(p.prob, x, @set par_tm.E0 = p.p), period = getPeriod(p.prob, x, @set par_tm.E0 = p.p))),
	normC = norminf)

plot(br, br_popsh, markersize=3)
