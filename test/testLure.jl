# using Revise, Plots
using Parameters, LinearAlgebra, Test
using BifurcationKit, Test
const BK = BifurcationKit

recordFromSolution(x, p) = (u1 = x[1], u2 = x[2])
####################################################################################################
function lur!(dz, u, p, t)
    @unpack α, β = p
    x, y, z = u
    dz[1] = y
    dz[2] = z
    dz[3] = -α * z - β * y - x + x^2
    dz
end

lur(z, p) = lur!(similar(z), z, p, 0)
par_lur = (α = 1.0, β = 0.)
z0 = zeros(3)
prob = BifurcationProblem(lur, z0, par_lur, (@lens _.β); record_from_solution = recordFromSolution)

opts_br = ContinuationPar(p_min = -0.4, p_max = 1.8, ds = -0.01, dsmax = 0.01, n_inversion = 8, detect_bifurcation = 3, max_bisection_steps = 25, nev = 3, plot_every_step = 20, max_steps = 1000)
    opts_br = @set opts_br.newton_options.verbose = false
    br = continuation(prob, PALC(tangent = Bordered(), θ = 0.3), opts_br;
    bothside = true, normC = norminf)

# plot(br)
####################################################################################################
# newton parameters
optn_po = NewtonPar(tol = 1e-8,  max_iterations = 25)

# continuation parameters
opts_po_cont = ContinuationPar(dsmax = 0.03, ds= 0.0001, dsmin = 1e-4, p_max = 1.8, p_min=-5., max_steps = 122, newton_options = (@set optn_po.tol = 1e-8), nev = 3, tol_stability = 1e-4, detect_bifurcation = 3, plot_every_step = 20, save_sol_every_step=1, n_inversion = 6)

Mt = 90 # number of time sections
    br_po = continuation(
    br, 2, opts_po_cont,
    PeriodicOrbitTrapProblem(M = Mt; update_section_every_step = 1,
    jacobian = :Dense);
    ampfactor = 1., δp = 0.01,
    verbosity = 0, plot = false,
    record_from_solution = (x, p) -> (xtt=reshape(x[1:end-1],3,Mt); return (max = maximum(xtt[1,:]), min = minimum(xtt[1,:]), period = x[end])),
    finalise_solution = (z, tau, step, contResult; prob = nothing, kwargs...) -> begin
            return z.u[end] < 40
            true
        end,
    normC = norminf)

# plot(br, br_po)
# plot(br_po, vars=(:param, :period))

show(br_po)

# test normal forms
for _ind in (1,3,16)
    if _ind <= length(br_po.specialpoint) &&
        length(br_po.specialpoint) >=3 &&
        br_po.specialpoint[_ind].type ∈ (:bp, :pd, :ns)
            println("")
            pt = get_normal_form(br_po, _ind; verbose = true)
            predictor(pt, 0.1, 1.)
            show(pt)
    end
end

# aBS from PD
br_po_pd = continuation(br_po, 1, setproperties(br_po.contparams, detect_bifurcation = 3, max_steps = 5, ds = 0.01, dsmax = 0.01, plot_every_step = 10);
    verbosity = 0, plot = false,
    ampfactor = .2, δp = -0.005,
    usedeflation = true,
    record_from_solution = (x, p) -> (xtt=reshape(x[1:end-1],3,Mt); return (max = maximum(xtt[1,:]), min = minimum(xtt[1,:]), period = x[end])),
    normC = norminf
    )

# plot(br, br_po, br_po_pd, xlims=(0.5,0.65))
####################################################################################################
# continuation parameters
opts_po_cont = ContinuationPar(dsmax = 0.03, ds= -0.0001, dsmin = 1e-4, p_max = 1.8, p_min=-5., max_steps = 120, newton_options = (@set optn_po.tol = 1e-8), nev = 3, tol_stability = 1e-4, detect_bifurcation = 3, plot_every_step = 20, save_sol_every_step = 1, n_inversion = 6)

br_po = continuation(
    br, 2, opts_po_cont,
    BK.PeriodicOrbitOCollProblem(20, 4);
    alg = PALC(tangent = Bordered()),
    ampfactor = 1., δp = 0.01,
    # usedeflation = true,
    # verbosity = 2,    plot = true,
    normC = norminf)

# test normal forms
for _ind in (1,)
    if length(br_po.specialpoint) >=1 && br_po.specialpoint[_ind].type ∈ (:bp, :pd, :ns)
        println("")
        pt = get_normal_form(br_po, _ind; verbose = true)
        # predictor(pt, 0.1, 1.)
        show(pt)
    end
end
####################################################################################################
using OrdinaryDiffEq

probsh = ODEProblem(lur!, copy(z0), (0., 1000.), par_lur; abstol = 1e-10, reltol = 1e-8)

optn_po = NewtonPar(tol = 1e-12, max_iterations = 25)

# continuation parameters
opts_po_cont = ContinuationPar(dsmax = 0.02, ds= -0.001, dsmin = 1e-4, max_steps = 122, newton_options = (@set optn_po.tol = 1e-12), tol_stability = 1e-5, detect_bifurcation = 3, plot_every_step = 10, n_inversion = 6, nev = 3)

br_po = continuation(
    br, 2, opts_po_cont,
    ShootingProblem(15, probsh, Rodas5P(); parallel = false, update_section_every_step = 1);
    ampfactor = 1., δp = 0.0051,
    # verbosity = 3,    plot = true,
    record_from_solution = (x, p) -> (return (max = getmaximum(p.prob, x, @set par_lur.β = p.p), period = getperiod(p.prob, x, @set par_lur.β = p.p))),
    # plot_solution = plotSH,
    # finalise_solution = (z, tau, step, contResult; prob = nothing, kwargs...) -> begin
    #         BK.haseigenvalues(contResult) && Base.display(contResult.eig[end].eigenvals)
    #         return z.u[end] < 30 && length(contResult.specialpoint) < 3
    #         true
    #     end,
    callback_newton = BK.cbMaxNorm(10),
    normC = norminf)

show(br_po)

# plot(br, br_po)
# plot(br_po, vars=(:param, :period))

@test br_po.specialpoint[1].param ≈ 0.6273246 rtol = 1e-4
@test br_po.specialpoint[2].param ≈ 0.5417461 rtol = 1e-4

# test showing normal form
for _ind in (1,3)
    if length(br_po.specialpoint) >=3 && br_po.specialpoint[_ind].type ∈ (:bp, :pd, :ns)
        println("")
        pt = get_normal_form(br_po, _ind; verbose = true, δ = 1e-5)
        show(pt)
        predictor(pt, 0.1, 1.)
        show(pt)
    end
end

# aBS from PD
br_po_pd = continuation(br_po, 1, setproperties(br_po.contparams, detect_bifurcation = 3, max_steps = 5, ds = 0.01, plot_every_step = 1, save_sol_every_step = 1);
    verbosity = 0, plot = false,
    usedeflation = false,
    ampfactor = .3, δp = -0.005,
    record_from_solution = (x, p) -> (return (max = BK.getmaximum(p.prob, x, @set par_lur.β = p.p), period = getperiod(p.prob, x, @set par_lur.β = p.p))),
    normC = norminf,
    callback_newton = BK.cbMaxNorm(10),
    )
@test br_po_pd.sol[1].x[end] ≈ 16.956 rtol = 1e-4

# plot(br_po, br_po_pd)
#######################################
@info "testLure Poincare"
opts_po_cont_ps = @set opts_po_cont.newton_options.tol = 1e-7
@set opts_po_cont_ps.dsmax = 0.0025
br_po = continuation(br, 2, opts_po_cont_ps,
    PoincareShootingProblem(2, probsh, Rodas4P(); parallel = false, reltol = 1e-6, update_section_every_step = 1, jacobian = BK.AutoDiffDenseAnalytical());
    ampfactor = 1., δp = 0.0051, #verbosity = 3,plot=true,
    callback_newton = BK.cbMaxNorm(10),
    record_from_solution = (x, p) -> (return (max = getmaximum(p.prob, x, @set par_lur.β = p.p), period = getperiod(p.prob, x, @set par_lur.β = p.p))),
    normC = norminf)

# plot(br_po, br)

show(br_po)
# test showing normal form
for _ind in (1,)
    if length(br_po.specialpoint) >=1 && br_po.specialpoint[_ind].type ∈ (:bp, :pd, :ns)
        println("")
        pt = get_normal_form(br_po, _ind; verbose = true)
        predictor(pt, 0.1, 1.)
        show(pt)
    end
end

# aBS from PD
br_po_pd = BK.continuation(br_po, 1, setproperties(br_po.contparams, detect_bifurcation = 3, max_steps = 2, ds = 0.01, plot_every_step = 1);
    # verbosity = 3, plot = true,
    ampfactor = .3, δp = -0.005,
    normC = norminf,
    callback_newton = BK.cbMaxNorm(10),
    record_from_solution = (x, p) -> (return (max = getmaximum(p.prob, x, @set par_lur.β = p.p), period = getperiod(p.prob, x, @set par_lur.β = p.p))),
    )

# plot(br_po_pd, br_po)
