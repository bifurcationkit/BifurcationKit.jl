# using Revise
# using Plots, Test
using BifurcationKit, LinearAlgebra, ForwardDiff, Parameters
const BK = BifurcationKit
####################################################################################################
struct EigMaps{T} <: BK.AbstractEigenSolver
    solver::T
end

function (eig::EigMaps)(J, nev; kwargs...)
    λs, evs, cv, it = eig.solver(J + I, nev; kwargs)
    return log.(Complex.(λs)), evs, cv, it
end
####################################################################################################
opt_newton = NewtonPar(tol = 1e-9, max_iterations = 20, verbose = false)
opts_br = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds = 0.01, p_max = 0.4, p_min = -0.5, detect_bifurcation = 3, nev = 2, newton_options = opt_newton, max_steps = 100, n_inversion = 4, tol_bisection_eigenvalue = 1e-8, dsmin_bisection = 1e-9)
####################################################################################################
# case of the period doubling
Fpd(u, p) = @. (-1+p.μ * p.a) * u + p.c * u^3
pars_pd = (μ = -0.2, a = 0.456, c = -1.234)

probMap = BK.BifurcationProblem((x, p) -> Fpd(x, p) .- x, [0.0], pars_pd, (@lens _.μ))

@set! opts_br.newton_options.eigsolver = EigMaps(DefaultEig())
br = continuation(probMap, PALC(), opts_br; normC = norminf, verbosity = 0)

prob = BK.BifurcationProblem(Fpd, [0.0], pars_pd, (@lens _.μ))

pd = BK.PeriodDoubling(br.specialpoint[1].x, br.specialpoint[1].τ, br.specialpoint[1].param, (@set pars_pd.μ = br.specialpoint[1].param), BK.getlens(br), [1.], [1.], nothing, :none)

nf = BK.period_doubling_normal_form(prob, pd, DefaultLS(), verbose = false)
@test nf.nf.a ≈ pars_pd.a
@test nf.nf.b3 ≈ pars_pd.c
show(nf)

# test of the predictor
pred = predictor(nf, 0.1)
@test pred.x0 ≈ 0
@test pred.x1 ≈ sqrt(-pars_pd.c*((pars_pd.a*0.1)^3 - 3*(pars_pd.a*0.1)^2 + 4*(pars_pd.a*0.1) - 2)*(pars_pd.a*0.1)*((pars_pd.a*0.1) - 2))/(pars_pd.c*((pars_pd.a*0.1)^3 - 3*(pars_pd.a*0.1)^2 + 4*(pars_pd.a*0.1) - 2))
####################################################################################################
# case of the Neimark-Sacker
function Fns!(f, u, p, t)
    @unpack θ, μ, c3, a = p
    z = complex(u[1], u[2])
    dz = z * cis(θ) * (1 + a * μ + c3 * abs2(z))

    f[1] = real(dz)
    f[2] = imag(dz)

    return f
end
Fns(x, p) = Fns!(similar(x), x, p, 0.)
pars_ns = (a = 1.123, μ = -0.1, θ = 0.1, c3 = -1.123 - 0.456im)

prob_ns = BK.BifurcationProblem((x, p) -> Fns(x, p) .- x, 0.01rand(2), pars_ns, (@lens _.μ))
br = BK.continuation(prob_ns, PALC(), opts_br; normC = norminf, verbosity = 0)

prob = BK.BifurcationProblem(Fns, [0.0], pars_pd, (@lens _.μ))
ns = BK.NeimarkSacker(br.specialpoint[1].x, br.specialpoint[1].τ, br.specialpoint[1].param, (abs∘imag)(eigenvals(br, br.specialpoint[1].idx)[1]), (@set pars_ns.μ = br.specialpoint[1].param), BK.getlens(br), [1.], [1.], nothing, :none)

nf = BK.neimark_sacker_normal_form(prob, br, 1; nev = 2, verbose = true)
@test nf.nf.a ≈ pars_ns.a
@test nf.nf.b ≈ pars_ns.c3
show(nf)
