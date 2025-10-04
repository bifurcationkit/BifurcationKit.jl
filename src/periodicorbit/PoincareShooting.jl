####################################################################################################
# Poincare shooting based on Sánchez, J., M. Net, B. Garcı́a-Archilla, and C. Simó. “Newton–Krylov Continuation of Periodic Orbits for Navier–Stokes Flows.” Journal of Computational Physics 201, no. 1 (November 20, 2004): 13–33. https://doi.org/10.1016/j.jcp.2004.04.018.

"""

$(TYPEDEF)

This composite type implements the Poincaré Shooting method to locate periodic orbits by relying on Poincaré return maps. More details (maths, notations, linear systems) can be found [here](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/periodicOrbitShooting/). The arguments are as described below.

# Fields
$(TYPEDFIELDS)

## Jacobian
$DocStringJacobianPOSh

## Simplified constructors
- The first important constructor is the following which is used for branching to periodic orbits from Hopf bifurcation points
    pb = PoincareShootingProblem(M::Int, prob::Union{ODEProblem, EnsembleProblem}, alg; kwargs...)

- A convenient way is to create a functional is

`pb = PoincareShootingProblem(prob::ODEProblem, alg, section; kwargs...)`

for simple shooting or

`pb = PoincareShootingProblem(prob::Union{ODEProblem, EnsembleProblem}, alg, M::Int, section; kwargs...)`

for multiple shooting . Here `prob` is an `Union{ODEProblem, EnsembleProblem}` which is used to create a flow using the ODE solver `alg` (for example `Tsit5()`). Finally, the arguments `kwargs` are passed to the ODE solver defining the flow. We refer to `DifferentialEquations.jl` for more information.

- Another convenient call is

`pb = PoincareShootingProblem(prob::Union{ODEProblem, EnsembleProblem}, alg, normals::AbstractVector, centers::AbstractVector; δ = 1e-8, kwargs...)`

where `normals` (resp. `centers`) is a list of normals (resp. centers) which defines a list of hyperplanes ``\\Sigma_i``. These hyperplanes are used to define partial Poincaré return maps.

## Computing the functionals
A functional, hereby called `G` encodes this shooting problem. You can then call `pb(orbitguess, par)` to apply the functional to a guess. Note that `orbitguess::AbstractVector` must be of size M * N where N is the number of unknowns in the state space and `M` is the number of Poincaré maps. Another accepted `guess` is such that `guess[i]` is the state of the orbit on the `i`th section. This last form allows for non-vector state space which can be convenient for 2d problems for example.

Note that you can generate this guess from a function solution using `generate_solution`.

- `pb(orbitguess, par)` evaluates the functional G on `orbitguess`
- `pb(orbitguess, par, du)` evaluates the jacobian `dG(orbitguess).du` functional at `orbitguess` on `du`
- `pb`(Val(:JacobianMatrixInplace), J, x, par)` compute the jacobian of the functional analytically. This is based on ForwardDiff.jl. Useful mainly for ODEs.
- `pb(Val(:JacobianMatrix), x, par)` same as above but out-of-place.


!!! tip "Tip"
    You can use the function `getperiod(pb, sol, par)` to get the period of the solution `sol` for the problem with parameters `par`.
"""
@with_kw_noshow struct PoincareShootingProblem{Tf, Tjac <: AbstractJacobianType, Tsection <: SectionPS, Tpar, Tlens} <: AbstractPoincareShootingProblem
    "`M`: the number of Poincaré sections. If `M == 1`, then the simple shooting is implemented and the multiple one otherwise."
    M::Int64 = 0                     # number of Poincaré sections
    "`flow::Flow`: implements the flow of the Cauchy problem though the structure [`Flow`](@ref)."
    flow::Tf = Flow()                # should be a Flow
    "`sections`: function or callable struct which implements a Poincaré section condition. The evaluation `sections(x)` must return a scalar number when `M == 1`. Otherwise, one must implement a function `section(out, x)` which populates `out` with the `M` sections. See [`SectionPS`](@ref) for type of section defined as a hyperplane."
    section::Tsection = SectionPS(M) # Poincaré sections
    "`δ = 1e-8` used to compute the jacobian of the functional by finite differences. If set to `0`, an analytical expression of the jacobian is used instead."
    δ::Float64 = 0e-8                # Numerical value used for the Matrix-Free Jacobian by finite differences.
    "`parallel = false` whether the shooting are computed in parallel (threading). Only available through the use of Flows defined by `EnsembleProblem`."
    parallel::Bool = false           # whether we use DE in Ensemble mode for multiple shooting
    "`par` parameters of the model"
    par::Tpar = nothing
    "`lens` parameter axis"
    lens::Tlens = nothing
    "`update_section_every_step` updates the section every `update_section_every_step` step during continuation"
    update_section_every_step::UInt = 1
    "Describes the type of jacobian used in Newton iterations (see below)."
    jacobian::Tjac = AutoDiffDenseAnalytical()
    @assert jacobian in [AutoDiffMF(), MatrixFree(), AutoDiffDense(), AutoDiffDenseAnalytical(), FiniteDifferences(), FiniteDifferencesMF()] "This jacobian is not defined. Please chose another one."
end

@inline isparallel(psh::PoincareShootingProblem) = psh.parallel
@inline getlens(psh::PoincareShootingProblem) = psh.lens
getparams(prob::PoincareShootingProblem) = prob.par
setparam(prob::PoincareShootingProblem, p) = set(getparams(prob), getlens(prob), p)

function Base.show(io::IO, psh::PoincareShootingProblem)
    println(io, "┌─ Poincaré shooting functional for periodic orbits")
    println(io, "├─ time slices     : ", get_mesh_size(psh))
    println(io, "├─ lens            : ", get_lens_symbol(psh.lens))
    println(io, "├─ jacobian        : ", psh.jacobian)
    println(io, "├─ update section  : ", psh.update_section_every_step)
    if psh.flow isa FlowDE
        println(io, "├─ integrator  : ", typeof(psh.flow.alg).name.name)
    end
    println(io, "└─ parallel        : ", isparallel(psh))
end

R(pb::PoincareShootingProblem, x::AbstractVector, k::Int) = R(pb.section, x, k)
E(pb::PoincareShootingProblem, xbar::AbstractVector, k::Int) = E(pb.section, xbar, k)

"""
$(SIGNATURES)

This function updates the normals and centers of the hyperplanes defining the Poincaré sections.
"""
@views function updatesection!(pb::PoincareShootingProblem, centers_bar, par; _norm = norm)
    M = get_mesh_size(pb); Nm1 = div(length(centers_bar), M)
    centers_barc = reshape(centers_bar, Nm1, M)
    centers = [E(pb.section, centers_barc[:, ii], ii) for ii = 1:M]
    normals = [vf(pb.flow, c, par) for c in centers]
    for ii in eachindex(normals)
        normals[ii] ./= _norm(normals[ii])
    end
    update!(pb.section, normals, centers)
end

"""
$(SIGNATURES)

Compute the period of the periodic orbit associated to `x_bar`.
"""
function getperiod(psh::PoincareShootingProblem, x_bar, par)
    M = get_mesh_size(psh); Nm1 = div(length(x_bar), M)

    # reshape the period orbit guess
    x_barc = reshape(x_bar, Nm1, M)

    # variable to hold the computed result
    xc = similar(x_bar, Nm1 + 1, M)
    outc = similar(xc)

    period = zero(eltype(x_barc))

    # we extend the state space to be able to call the flow, so we fill xc
    if ~isparallel(psh)
        for ii in 1:M
            @views E!(psh.section, xc[:, ii], x_barc[:, ii], ii)
            # We need the callback to be active here!!!
            period += @views evolve(psh.flow, xc[:, ii], par, Inf).t
        end
    else
        for ii in 1:M
            @views E!(psh.section, xc[:, ii], x_barc[:, ii], ii)
        end
        solOde =  evolve(psh.flow, xc, par, repeat([Inf], M))
        period = sum(x->x.t, solOde)
    end
    return period
end
getperiod(psh::PoincareShootingProblem, x_bar, p::Real) = getperiod(psh, x_bar, setparam(psh, p))

function get_time_slices(prob::PoincareShootingProblem, x_bar::AbstractVector)
    M = get_mesh_size(prob); Nm1 = length(x_bar) ÷ M
    # reshape the period orbit guess
    x_barc = reshape(x_bar, Nm1, M)
    xc = similar(x_bar, Nm1 + 1, M)
    for ii=1:M
        @views E!(prob.section, xc[:, ii], x_barc[:, ii], ii)
    end
    xc
end

"""
$(SIGNATURES)

Compute the full periodic orbit associated to `x`. Mainly for plotting purposes.
"""
function get_periodic_orbit(prob::PoincareShootingProblem, x_bar::AbstractVector, p)
    # this function extracts the amplitude of the cycle
    M = get_mesh_size(prob); Nm1 = length(x_bar) ÷ M

    # reshape the period orbit guess
    x_barc = reshape(x_bar, Nm1, M)
    xc = similar(x_bar, Nm1 + 1, M)

    T = getperiod(prob, x_bar, p)

    # !!!! we could use @views but then Sundials will complain !!!
    if ~isparallel(prob)
        E!(prob.section, view(xc, :, 1), view(x_barc, :, 1), 1)
        # We need the callback to be active here!!!
        sol1 = @views evolve(prob.flow, Val(:Full), xc[:, 1], p, T; callback = nothing)
        return sol1
    else # threaded version
        E!(prob.section, view(xc, :, 1), view(x_barc, :, 1), 1)
        sol = @views evolve(prob.flow, Val(:Full), xc[:, 1:1], p, [T]; callback = nothing)
        return sol[1]
    end
end
get_periodic_orbit(prob::PoincareShootingProblem, x::AbstractVector, p::Real) = get_periodic_orbit(prob, x, setparam(prob, p))

"""
$(SIGNATURES)

Compute the projection of each vector (`x[i]` is a `Vector`) on the Poincaré section.
"""
function projection(psh::PoincareShootingProblem, x::AbstractVector)
    # create initial guess. We have to pass it through the projection R
    M = get_mesh_size(psh)
    orbitguess_bar = Vector{eltype(x)}(undef, 0)
    @assert M == length(psh.section.normals)
    for ii=1:M
        push!(orbitguess_bar, R(psh, x[ii], ii))
    end
    return orbitguess_bar
end

"""
$(SIGNATURES)

Compute the projection of each vector (`x[i, :]` is a `Vector`) on the Poincaré section.
"""
function projection(psh::PoincareShootingProblem, x::AbstractMatrix)
    # create initial guess. We have to pass it through the projection R
    M = get_mesh_size(psh)
    m, n = size(x)
    orbitguess_bar = Matrix{eltype(x)}(undef, m, n-1)
    @assert M == length(psh.section.normals)
    for ii=1:M
        orbitguess_bar[ii, :] .= @views R(psh, x[ii, :], ii)
    end
    return orbitguess_bar
end
####################################################################################################
# Poincaré (multiple) shooting with hyperplanes parametrization
function (psh::PoincareShootingProblem)(x_bar::AbstractVector, par; verbose = false)
    M = get_mesh_size(psh)
    Nm1 = div(length(x_bar), M)

    # reshape the period orbit guess
    x_barc = reshape(x_bar, Nm1, M)

    # TODO the following declaration of xc allocates. It would be better to make it inplace
    xc = similar(x_bar, Nm1 + 1, M)

    # variable to hold the result of the computations
    outc = similar(xc)

    # we extend the state space to be able to call the flow, so we fill xc
    #TODO create the projections on the fly
    for ii in 1:M
        E!(psh.section, view(xc, :, ii), view(x_barc, :, ii), ii)
    end

    if ~isparallel(psh)
        for ii in 1:M
            im1 = ii == 1 ? M : ii - 1
            # We need the callback to be active here!!!
            outc[:, ii] .= xc[:, ii] .- evolve(psh.flow, xc[:, im1], par, Inf).u
        end
    else
        solOde = evolve(psh.flow, xc, par, repeat([Inf64], M))
        for ii in 1:M
            im1 = ii == 1 ? M : ii - 1
            # We need the callback to be active here!!!
            @views outc[:, ii] .= xc[:, ii] .- solOde[im1][2]
        end
    end

    # build the array to be returned
    out_bar = similar(x_bar)
    out_barc = reshape(out_bar, Nm1, M)
    for i in 1:M
        R!(psh.section, view(out_barc, :, i), view(outc, :, i), i)
    end
    return out_bar
end

function residual!(pb::PoincareShootingProblem, out, x, p)
    copyto!(out, pb(x, p))
    out
end

residual(pb::PoincareShootingProblem, x, p) = pb(x, p)

"""
This function computes the derivative of the Poincare return map Π(x) = ϕ(t(x),x) where t(x) is the return time of x to the section.
"""
function diff_poincare_map(psh::PoincareShootingProblem, x, par, dx, ii::Int)
    normal = psh.section.normals[ii]
    abs(LA.dot(normal, dx)) > 1e-12 && @warn "Vector does not belong to hyperplane!  dot(normal, dx) = $(abs(LA.dot(normal, dx))) > 1e-12 and $(dot(dx, dx))"
    # compute the Poincare map from x
    tΣ, solΣ = evolve(psh.flow, Val(:SerialTimeSol), x, par, Inf)
    z = vf(psh.flow, solΣ, par)
    # solution of the variational equation at time tΣ
    # We need the callback to be INACTIVE here!!!
    y = evolve(psh.flow, Val(:SerialdFlow), x, par, dx, tΣ; callback = nothing).du
    out = y .- (LA.dot(normal, y) / LA.dot(normal, z)) .* z
end

# jacobian of the shooting functional
function (psh::PoincareShootingProblem)(x_bar::AbstractVector, par, dx_bar::AbstractVector)
    δ = psh.δ
    if δ > 0
        # mostly for debugging purposes
        return (psh(x_bar .+  δ .* dx_bar, par) .- psh(x_bar, par)) ./ δ
    end

    # otherwise analytical Jacobian
    M = get_mesh_size(psh)
    Nm1 = div(length(x_bar), M)

    # reshape the period orbit guess
    x_barc  = reshape( x_bar, Nm1, M)
    dx_barc = reshape(dx_bar, Nm1, M)

    # variable to hold the computed result
    xc  = similar( x_bar, Nm1 + 1, M)
    dxc = similar(dx_bar, Nm1 + 1, M)
    outc = similar(xc)

    # we extend the state space to be able to call the flow, so we fill xc
    for ii in 1:M
         E!(psh.section,  view(xc, :, ii),  view(x_barc, :, ii), ii)
        dE!(psh.section, view(dxc, :, ii), view(dx_barc, :, ii), ii)
    end

    if ~isparallel(psh)
        for ii in 1:M
            im1 = (ii == 1 ? M : ii - 1)
            @views outc[:, ii] .= dxc[:, ii] .- diff_poincare_map(psh, xc[:, im1], par, dxc[:, im1], im1)
        end
    else
        @assert false "Analytical Jacobian for parallel Poincare Shooting not implemented yet. Please use the option δ > 0 to use Matrix-Free jacobian or chose `:FiniteDifferencesDense` to compute jacobian based on finite differences."
    end

    # build the array to be returned
    out_bar = similar(x_bar)
    out_barc = reshape(out_bar, Nm1, M)
    for ii in 1:M
        dR!(psh.section, view(out_barc, :, ii), view(outc, :, ii), ii)
    end
    return out_bar
end

# inplace computation of the matrix of the jacobian of the shooting problem, only serial for now
function (psh::PoincareShootingProblem)(::Val{:JacobianMatrixInplace}, J::AbstractMatrix, x_bar::AbstractVector, par)
    M = get_mesh_size(psh)
    Nm1 = div(length(x_bar), M)
    N = Nm1 + 1

    x_barc = reshape(x_bar, Nm1, M)

    # TODO the following declaration of xc allocates. It would be better to make it inplace
    xc = similar(x_bar, N, M)

    # we extend the state space to be able to call the flow, so we fill xc
    #TODO create the projections on the fly
    for ii in 1:M
        E!(psh.section, view(xc, :, ii), view(x_barc, :, ii), ii)
    end

    # jacobian of the flow
    dflow = (_J, _x, _T) -> ForwardDiff.jacobian!(_J, z -> evolve(psh.flow, Val(:SerialTimeSol), z, par, _T; callback = nothing).u, _x)

    # initialize some temporaries
    Jtmp = zeros(N, N)
    normal = copy(psh.section.normals[1])
    F = zeros(N)
    Rm = zeros(Nm1, N)
    Em = zeros(N, Nm1)

    # put the matrices by blocks
    In = LA.I(Nm1)
    for ii=1:M
        im1 = ii == 1 ? M : ii - 1
        # we find the point on the next section
        tΣ, solΣ = evolve(psh.flow, Val(:SerialTimeSol), view(xc,:,im1), par, Inf)
        # computation of the derivative of the return map
        F .= vf(psh.flow, solΣ, par) #vector field
        normal .= psh.section.normals[ii]
        @views dflow(Jtmp, xc[:, im1], tΣ)
        Jtmp .= Jtmp .- F * normal' * Jtmp ./ LA.dot(F, normal)
        # projection with Rm, Em
        ForwardDiff.jacobian!(Rm, x-> R(psh.section, x, ii), zeros(N))
        ForwardDiff.jacobian!(Em, x-> E(psh.section, x, im1), zeros(Nm1))
        J[(ii-1)*Nm1+1:(ii-1)*Nm1+Nm1, (im1-1)*Nm1+1:(im1-1)*Nm1+Nm1] .= -Rm * Jtmp * Em
        if M == 1
            J[(ii-1)*Nm1+1:(ii-1)*Nm1+Nm1, (ii-1)*Nm1+1:(ii-1)*Nm1+Nm1] .+= In
        else
            J[(ii-1)*Nm1+1:(ii-1)*Nm1+Nm1, (ii-1)*Nm1+1:(ii-1)*Nm1+Nm1] .= In
        end
    end
    return J
end

# out of place version
(psh::PoincareShootingProblem)(::Val{:JacobianMatrix}, x::AbstractVector, par) = psh(Val(:JacobianMatrixInplace), zeros(eltype(x), length(x), length(x)), x, par)
####################################################################################################
# functions needed for Branch switching from Hopf bifurcation point
function re_make(prob::PoincareShootingProblem, prob_vf, hopfpt, ζr, centers, period; k...)

    # create the section
    if _isempty(prob.section)
        normals = [residual(prob_vf, u, hopfpt.params) for u in centers]
        for n in normals; n ./= norm(n); end
    else
        normals = prob.section.normals
        centers = prob.section.centers
    end

    @assert ~(prob.flow isa AbstractFlow) "Somehow, this method was not called as it should. `prob.flow` should be a Named Tuple, prob should be constructed with the simple constructor, not yielding a Flow for its flow field."

    # update the problem, hacky way to pass parameters
    if length(prob.flow) == 4
        probPSh = PoincareShootingProblem(prob.flow.prob, prob.flow.alg, deepcopy(normals), deepcopy(centers);
            parallel = prob.parallel,
            lens = getlens(prob_vf),
            par = getparams(prob_vf),
            update_section_every_step = prob.update_section_every_step,
            jacobian = prob.jacobian,
            prob.flow.kwargs...)
    else
        probPSh = PoincareShootingProblem(prob.flow.prob1,
            prob.flow.alg1,
            prob.flow.prob2,
            prob.flow.alg2,
            deepcopy(normals),
            deepcopy(centers);
            parallel = prob.parallel,
            lens = getlens(prob_vf),
            update_section_every_step = prob.update_section_every_step,
            jacobian = prob.jacobian,
            prob.flow.kwargs...)
    end

    # create initial guess. We have to pass it through the projection R
    hyper = probPSh.section
    M = get_mesh_size(probPSh)
    @assert length(normals) == M
    orbitguess_bar = zeros(length(centers[1])-1, M)
    for ii in eachindex(normals)
        orbitguess_bar[:, ii] .= R(hyper, centers[ii], ii)
    end
    # set jacobian for the flow too
    _sync_jacobian!(probPSh)

    return probPSh, vec(orbitguess_bar)
end

using SciMLBase: AbstractTimeseriesSolution

"""
$(TYPEDSIGNATURES)

Generate a periodic orbit problem from a solution.

## Arguments
- `bifprob` a bifurcation problem to provide the vector field
- `prob_de::ODEProblem` associated to `sol`
- `sol` basically, and `ODEProblem
- `period` estimate of the period of the periodic orbit
- `k` kwargs arguments passed to the constructor of `ShootingProblem`

## Output
- returns a `ShootingProblem` and an initial guess.
"""
function generate_ci_problem(pb::PoincareShootingProblem,
                        bifprob::AbstractBifurcationProblem,
                        prob_de,
                        sol::AbstractTimeseriesSolution,
                        tspan::Tuple;
                        alg = sol.alg,
                        ksh...)
    u0 = sol(0)
    @assert u0 isa AbstractVector
    N = length(u0)
    M = pb.M

    ts = LinRange(tspan[1], tspan[2], M+1)[begin:end-1]
    centers = [copy(sol(t)) for t in ts]
    normals = [residual(bifprob, c, sol.prob.p) for c in centers]
    # normals = [sol(t, Val{1}) for t in ts]
    for n in normals; n ./= norm(n); end
    @assert length(normals) == length(centers) == M

    probpsh = PoincareShootingProblem(remake(prob_de, p = sol.prob.p), alg, normals, centers; lens = getlens(bifprob), par = sol.prob.p, ksh...)

    # create initial guess. We have to pass it through the projection R
    cipsh = reduce(vcat, [R(probpsh, centers[k], k) for k in eachindex(centers)])

    return probpsh, cipsh
end

generate_ci_problem(pb::PoincareShootingProblem, bifprob::AbstractBifurcationProblem, prob_de, sol::AbstractTimeseriesSolution, period::Real; alg = sol.alg, ksh...) = generate_ci_problem(pb, bifprob, prob_de, sol, (zero(period), period); alg = alg, ksh...)
