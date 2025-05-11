using SciMLBase: ODEProblem, DAEProblem, EnsembleProblem, terminate!, VectorContinuousCallback, ContinuousCallback
const ODEType = Union{ODEProblem, DAEProblem}

function get_vector_field(odeprob::Union{ODEProblem, DAEProblem})
    if isinplace_sciml(odeprob)
        return (x, p) -> (out = similar(x); odeprob.f(out, x, p, odeprob.tspan[1]); return out)
    else
        return (x, p) -> odeprob.f(x, p, odeprob.tspan[1])
    end
end
get_vector_field(pb::EnsembleProblem) = get_vector_field(pb.prob)
####################################################################################################
###                                     STANDARD SHOOTING
####################################################################################################
_sync_jacobian!(sh) = @reset sh.flow.jacobian = sh.jacobian

# this constructor takes into account a parameter passed to the vector field
# if M = 1, we disable parallel processing
function ShootingProblem(prob::ODEType, alg, ds, section; parallel = false, par = prob.p, kwargs...)
    _M = length(ds)
    parallel = _M == 1 ? false : parallel
    _pb = parallel ? EnsembleProblem(prob) : prob
    kwargsSh = [k for k in kwargs if first(k) ∈ fieldnames(ShootingProblem)]
    kwargsDE = setdiff(kwargs, kwargsSh)
    sh = ShootingProblem(;M = _M, flow = Flow(_pb, alg; kwargsDE...), kwargsSh..., ds = ds, section = section, parallel = parallel, par = par)
    # set jacobian for the flow too
    _sync_jacobian!(sh)
end

ShootingProblem(prob::ODEType, alg, M::Int, section; kwargs...) = ShootingProblem(prob, alg, diff(LinRange(0, 1, M + 1)), section; kwargs...)

function ShootingProblem(prob::ODEType, alg, centers::AbstractVector; parallel = false, par = prob.p, kwargs...)
    F = get_vector_field(prob)
    sh = ShootingProblem(prob, alg, diff(LinRange(0, 1, length(centers) + 1)), SectionSS(F(centers[1], par) ./ norm(F(centers[1], par)), centers[1]); parallel, par = par, kwargs...)
    # set jacobian for the flow too
    _sync_jacobian!(sh)
end

# this is the "simplest" constructor to use in automatic branching from Hopf
ShootingProblem(M::Int, prob::ODEType, alg; kwargs...) = ShootingProblem(prob, alg, M, nothing; kwargs...)

# idem but with an ODEProblem to define the derivative of the flow
function ShootingProblem(prob1::ODEType, alg1, 
                         prob2::ODEType, alg2, 
                         ds, section; 
                         parallel = false, 
                         par = prob1.p, 
                         kwargs...)
    _M = length(ds)
    parallel = _M == 1 ? false : parallel
    _pb1 = parallel ? EnsembleProblem(prob1) : prob1
    _pb2 = parallel ? EnsembleProblem(prob2) : prob2
    kwargsSh = [k for k in kwargs if first(k) ∈ fieldnames(ShootingProblem)]
    kwargsDE = setdiff(kwargs, kwargsSh)
    sh = ShootingProblem(;M = _M, flow = Flow(_pb1, alg1, _pb2, alg2; kwargsDE...), kwargsSh..., ds, section, parallel, par = par)
    # set jacobian for the flow too
    _sync_jacobian!(sh)
end

ShootingProblem(M::Int, prob1::ODEType, alg1, prob2::ODEType, alg2; kwargs...) = ShootingProblem(prob1, alg1, prob2, alg2, M, nothing; kwargs...)

ShootingProblem(prob1::ODEType, alg1, prob2::ODEType, alg2, M::Int, section; kwargs...) = ShootingProblem(prob1, alg1, prob2, alg2, diff(LinRange(0, 1, M + 1)), section; kwargs...)

function ShootingProblem(prob1::ODEType, alg1, prob2::ODEType, alg2, centers::AbstractVector; kwargs...)
    F = get_vector_field(prob1)
    p = prob1.p # parameters
    sh = ShootingProblem(prob1, alg1, prob2, alg2, diff(LinRange(0, 1, length(centers) + 1)), SectionSS(F(centers[1], p)./ norm(F(centers[1], p)), centers[1]); kwargs...)
    # set jacobian for the flow too
    _sync_jacobian!(sh)
end
####################################################################################################
###                                     POINCARE SHOOTING
####################################################################################################
function PoincareShootingProblem(prob::ODEProblem,
                                 alg,
                                 hyp::SectionPS;
                                 δ = 1e-8,
                                 interp_points = 50,
                                 parallel = false,
                                 par = prob.p,
                                 kwargs...)
    pSection(out, u, t, integrator) = (hyp(out, u); out .*= integrator.iter > 1)
    affect!(integrator, idx) = terminate!(integrator)
    # we put nothing option to have an upcrossing
    cb = VectorContinuousCallback(pSection, affect!, hyp.M; 
                                    interp_points = interp_points, 
                                    affect_neg! = nothing)
    # change ODEProblem -> EnsembleProblem in the parallel case
    _M = hyp.M
    parallel = _M == 1 ? false : parallel
    _pb = parallel ? EnsembleProblem(prob) : prob

    kwargsSh = [k for k in kwargs if first(k) ∈ fieldnames(PoincareShootingProblem)]
    kwargsDE = setdiff(kwargs, kwargsSh)

    psh = PoincareShootingProblem(;
                flow = Flow(_pb, alg; callback = cb, kwargsDE...),
                kwargsSh...,
                M = hyp.M,
                section = hyp,
                parallel = parallel,
                par = par)
    # set jacobian for the flow too
    _sync_jacobian!(psh)
end

# this is the "simplest" constructor to use in automatic branching from Hopf
# this is a Hack to pass the arguments to construct a Flow. Indeed, we need to provide the
# appropriate callback for Poincare Shooting to work
function PoincareShootingProblem(M::Int,
                            prob::ODEProblem,
                            alg;
                            parallel = false,
                            section = SectionPS(M),
                            par = prob.p,
                            kwargs...)
    kwargsSh = [k for k in kwargs if first(k) ∈ fieldnames(PoincareShootingProblem)]
    kwargsDE = setdiff(kwargs, kwargsSh)
    psh = PoincareShootingProblem(;
                M = M,
                flow = (par = par, prob = prob, alg = alg, kwargs = kwargsDE),
                kwargsSh...,
                parallel = (M == 1 ? false : parallel),
                section = section,
                par = par)
end

function PoincareShootingProblem(M::Int,
                    prob1::ODEProblem, alg1,
                    prob2::ODEProblem, alg2;
                    parallel = false,
                    section = SectionPS(M),
                    lens = nothing,
                    updateSectionEveryStep = 0,
                    jacobian = :autodiffDenseAnalytical,
                    par = prob1.p,
                    kwargs...)
    kwargsSh = [k for k in kwargs if first(k) ∈ fieldnames(PoincareShootingProblem)]
    kwargsDE = setdiff(kwargs, kwargsSh)

    psh = PoincareShootingProblem(M = M, flow = (par = prob1.p, prob1 = prob1, alg1 = alg1, prob2 = prob2, alg2 = alg2, kwargs = kwargsDE), kwargsSh..., parallel = parallel, section = section, par = par)
end

function PoincareShootingProblem(prob::ODEProblem,
                                alg,
                                normals::AbstractVector,
                                centers::AbstractVector;
                                δ = 1e-8,
                                interp_points = 50,
                                parallel = false,
                                radius = Inf,
                                par = prob.p,
                                kwargs...)

    psh = PoincareShootingProblem(prob, alg,
                    SectionPS(normals, centers; radius = radius);
                    δ = δ, interp_points = interp_points, parallel = parallel, par = par, kwargs...)
    # set jacobian for the flow too
    _sync_jacobian!(psh)
end

function PoincareShootingProblem(prob1::ODEProblem, alg1,
                                prob2::ODEProblem, alg2,
                                hyp::SectionPS;
                                δ = 1e-8,
                                interp_points = 50,
                                parallel = false,
                                par = prob1.p,
                                kwargs...)
    p = prob1.p # parameters
    pSection(out, u, t, integrator) = (hyp(out, u); out .*= integrator.iter > 1)
    affect!(integrator, idx) = terminate!(integrator)
    # we put nothing option to have an upcrossing
    cb = VectorContinuousCallback(pSection, affect!, hyp.M; interp_points = interp_points, affect_neg! = nothing)
    # change the ODEProblem -> EnsembleProblem for the parallel case
    _M = hyp.M
    parallel = _M == 1 ? false : parallel
    _pb1 = parallel ? EnsembleProblem(prob1) : prob1
    _pb2 = parallel ? EnsembleProblem(prob2) : prob2

    kwargsSh = [k for k in kwargs if first(k) ∈ fieldnames(PoincareShootingProblem)]
    kwargsDE = setdiff(kwargs, kwargsSh)

    psh = PoincareShootingProblem(;
        M = hyp.M,
        flow = Flow(_pb1, alg1, _pb2, alg2; callback = cb, kwargsDE...), kwargsSh...,
        section = hyp,
        δ = δ,
        parallel = parallel,
        par = par)
    # set jacobian for the flow too
    _sync_jacobian!(psh)
end

function PoincareShootingProblem(prob1::ODEProblem, alg1,
                                prob2::ODEProblem, alg2,
                                normals::AbstractVector, centers::AbstractVector;
                                δ = 1e-8,
                                interp_points = 50,
                                parallel = false,
                                radius = Inf,
                                kwargs...)
    psh = PoincareShootingProblem(prob1, alg2, prob2, alg2,
                    SectionPS(normals, centers; radius = radius);
                    δ = δ, interp_points = interp_points, parallel = parallel, kwargs...)
    # set jacobian for the flow too
    _sync_jacobian!(psh)
end
####################################################################################################
using SciMLBase: AbstractTimeseriesSolution
"""
$(SIGNATURES)

Generate a periodic orbit problem from a solution.

## Arguments
- `pb` a `ShootingProblem` which provides basic information, like the number of time slices `M`
- `bifprob` a bifurcation problem to provide the vector field
- `prob_de::ODEProblem` associated to `sol`
- `sol` basically an `ODEProblem` or a function `t -> sol(t)`
- `tspan::Tuple` estimate of the period of the periodic orbit
- `alg` algorithm for solving the Cauchy problem
- `prob_mono` problem for monodromy
- `alg_mono` algorithm for solving the monodromy Cauchy problem
- `k` kwargs arguments passed to the constructor of `ShootingProblem`

## Output
- returns a `ShootingProblem` and an initial guess.
"""
function generate_ci_problem(shooting::ShootingProblem, 
                            prob_bif::AbstractBifurcationProblem, 
                            prob_de, 
                            sol::AbstractTimeseriesSolution, 
                            tspan::Tuple;
                            prob_mono = nothing,
                            alg = sol.alg,
                            alg_mono = sol.alg,
                            use_bordered_array = false, 
                            ksh...)
    t0 = sol.t[begin]
    u0 = sol(t0)
    M = shooting.M
    
    # points for the sections
    centers = [copy(sol(t)) for t in LinRange(tspan[1], tspan[2], M+1)[1:end-1]]
    
    # shooting kwargs
    sh_kw = (lens = getlens(prob_bif), 
    jacobian = shooting.jacobian,
    parallel = shooting.parallel,
    update_section_every_step = shooting.update_section_every_step,
    )
    
    # do we provide an ODE alg for computing the monodromy?
    if isnothing(prob_mono)
        probsh = ShootingProblem(prob_de, alg, centers; 
        sh_kw..., 
        ksh...)
    else
        probsh = ShootingProblem(prob_de, alg, prob_mono, alg, centers; 
        sh_kw...,
        ksh...)
        @info has_mono_DE(probsh.flow)
    end
    
    if ~use_bordered_array
        @assert u0 isa AbstractVector
        cish = reduce(vcat, centers)
        cish = vcat(cish, tspan[2]-tspan[1])
    else
        cish = BorderedArray(VectorOfArray(deepcopy(centers)), tspan[2]-tspan[1])
    end
    
    return probsh, cish
end

generate_ci_problem(pb::ShootingProblem, prob_bif::AbstractBifurcationProblem, prob_de, sol::AbstractTimeseriesSolution, period::Real; ksh...) = generate_ci_problem(pb, prob_bif, prob_de, sol, (zero(period), period); ksh...)