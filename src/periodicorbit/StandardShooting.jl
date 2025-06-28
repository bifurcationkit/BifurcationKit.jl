"""
$(TYPEDEF)

Create a problem to implement the Simple / Parallel Multiple Standard Shooting method to locate periodic orbits. More details (maths, notations, linear systems) can be found [here](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/periodicOrbitShooting/). The arguments are as described below.

A functional, hereby called `G`, encodes the shooting problem. For example, the following methods are available:

- `pb(orbitguess, par)` evaluates the functional G on `orbitguess`
- `pb(orbitguess, par, du)` evaluates the jacobian `dG(orbitguess)⋅du` functional at `orbitguess` on `du`.
- `pb`(Val(:JacobianMatrixInplace), J, x, par)` compute the jacobian of the functional analytically. This is based on ForwardDiff.jl. Useful mainly for ODEs.
- `pb(Val(:JacobianMatrix), x, par)` same as above but out-of-place.

You can then call `pb(orbitguess, par)` to apply the functional to a guess. Note that `orbitguess::AbstractVector` must be of size `M * N + 1` where N is the number of unknowns of the state space and `orbitguess[M * N + 1]` is an estimate of the period `T` of the limit cycle. This form of guess is convenient for the use of the linear solvers in `IterativeSolvers.jl` (for example) which only accept `AbstractVector`s. Another accepted guess is of the form `BorderedArray(guess, T)` where `guess[i]` is the state of the orbit at the `i`th time slice. This last form allows for non-vector state space which can be convenient for 2d problems for example, use `GMRESKrylovKit` for the linear solver in this case.

Note that you can generate this guess from a function solution using `generate_solution` or `generate_ci_problem`.

# Fields
$(TYPEDFIELDS)

## Jacobian
$DocStringJacobianPOSh

## Simplified constructors
- The first important constructor is the following which is used for branching to periodic orbits from Hopf bifurcation points:


    pb = ShootingProblem(M::Int, prob::Union{ODEProblem, EnsembleProblem}, alg; kwargs...)

- A convenient way to build the functional is to use:


    pb = ShootingProblem(prob::Union{ODEProblem, EnsembleProblem}, alg, centers::AbstractVector; kwargs...)

where `prob` is an `ODEProblem` (resp. `EnsembleProblem`) which is used to create a flow using the ODE solver `alg` (for example `Tsit5()`). `centers` is list of `M` points close to the periodic orbit, they will be used to build a constraint for the phase. `parallel = false` is an option to use Parallel simulations (Threading) to simulate the multiple trajectories in the case of multiple shooting. This is efficient when the trajectories are relatively long to compute. Finally, the arguments `kwargs` are passed to the ODE solver defining the flow. Look at `DifferentialEquations.jl` for more information. Note that, in this case, the derivative of the flow is computed internally using Finite Differences.

- Another way to create a Shooting problem with more options is the following where in particular, one can provide its own scalar constraint `section(x)::Number` for the phase:


    pb = ShootingProblem(prob::Union{ODEProblem, EnsembleProblem}, alg, M::Int, section; parallel = false, kwargs...)

or

    pb = ShootingProblem(prob::Union{ODEProblem, EnsembleProblem}, alg, ds, section; parallel = false, kwargs...)

- The next way is an elaboration of the previous one


    pb = ShootingProblem(prob1::Union{ODEProblem, EnsembleProblem}, alg1, prob2::Union{ODEProblem, EnsembleProblem}, alg2, M::Int, section; parallel = false, kwargs...)

or

    pb = ShootingProblem(prob1::Union{ODEProblem, EnsembleProblem}, alg1, prob2::Union{ODEProblem, EnsembleProblem}, alg2, ds, section; parallel = false, kwargs...)

where we supply now two `ODEProblem`s. The first one `prob1`, is used to define the flow associated to `F` while the second one is a problem associated to the derivative of the flow. Hence, `prob2` must implement the following vector field ``\\tilde F(x,y,p) = (F(x,p), dF(x,p)\\cdot y)``.
"""
@with_kw_noshow struct ShootingProblem{Tf <: AbstractFlow, Tjac <: AbstractJacobianType, Ts, Tsection, Tpar, Tlens} <: AbstractShootingProblem
    "`ds`: vector of time differences for each shooting. Its length is written `M`. If `M == 1`, then the simple shooting is implemented and the multiple one otherwise."
    M::Int64 = 0                         # number of sections
    "`flow::Flow`: implements the flow of the Cauchy problem though the structure [`Flow`](@ref)."
    flow::Tf = Flow()                    # should be a Flow
    ds::Ts = diff(LinRange(0, 1, M + 1)) # difference of times for multiple shooting
    "`section`: implements a phase condition. The evaluation `section(x, T)` must return a scalar number where `x` is a guess for **one point** on the periodic orbit and `T` is the period of the guess. Also, the method `section(x, T, dx, dT)` must be available and which returns the differential of `section`. The type of `x` depends on what is passed to the newton solver. See [`SectionSS`](@ref) for a type of section defined as a hyperplane."
    section::Tsection = nothing          # sections for phase condition
    "`parallel` whether the shooting is computed in parallel (threading). Available through the use of Flows defined by `EnsembleProblem` (this is automatically set up for you)."
    parallel::Bool = false               # whether we use DE in Ensemble mode for multiple shooting
    "`par` parameters of the model"
    par::Tpar = nothing
    "`lens` parameter axis."
    lens::Tlens = nothing
    "updates the section every `update_section_every_step` step during continuation"
    update_section_every_step::UInt = 1
    "Describes the type of jacobian used in Newton iterations (see below)."
    jacobian::Tjac = AutoDiffDense()
    @assert jacobian in [AutoDiffMF(), MatrixFree(), AutoDiffDense(), AutoDiffDenseAnalytical(), FiniteDifferences(), FiniteDifferencesMF()] "This jacobian is not defined. Please chose another one."
end

@inline issimple(sh::ShootingProblem) = get_mesh_size(sh) == 1
@inline isparallel(sh::ShootingProblem) = sh.parallel
@inline getlens(sh::ShootingProblem) = sh.lens
getparams(prob::ShootingProblem) = prob.par
setparam(prob::ShootingProblem, p) = set(getparams(prob), getlens(prob), p)

function Base.show(io::IO, sh::ShootingProblem)
    println(io, "┌─ Standard shooting functional for periodic orbits")
    println(io, "├─ time slices    : ", get_mesh_size(sh))
    println(io, "├─ lens           : ", get_lens_symbol(sh.lens))
    println(io, "├─ jacobian       : ", sh.jacobian)
    println(io, "├─ update section : ", sh.update_section_every_step)
    if sh.flow isa FlowDE
        println(io, "├─ integrator     : ", typeof(sh.flow.alg).name.name)
    end
    println(io, "└─ parallel       : ", isparallel(sh))
end

# this function updates the section during the continuation run
function updatesection!(sh::ShootingProblem, x, pars)
    @debug "Update section shooting"
    xt = get_time_slices(sh, x)
    @views update!(sh.section, vf(sh.flow, xt[:, 1], pars), xt[:, 1])
    sh.section.normal ./= norm(sh.section.normal)
    return true
end

@views function get_time_slices(sh::ShootingProblem, x::AbstractVector)
    M = get_mesh_size(sh)
    N = div(length(x) - 1, M)
    return reshape(x[1:end-1], N, M)
end
@inline get_time_slice(::ShootingProblem, x::AbstractMatrix, ii::Int) = @view x[:, ii]
@inline get_time_slice(::ShootingProblem, x::AbstractVector, ii::Int) = xc[ii]
@inline get_time_slice(sh::ShootingProblem, x::BorderedArray, ii::Int) = x.u[ii]
@inline get_time_slices(::ShootingProblem ,x::BorderedArray) = x.u
####################################################################################################
# Standard shooting functional using AbstractVector, convenient for IterativeSolvers.
function (sh::ShootingProblem)(x::AbstractVector, pars)
    # Sundials does not like @views :(
    T = getperiod(sh, x)
    M = get_mesh_size(sh)
    N = div(length(x) - 1, M)

    # extract the orbit guess and reshape it into a matrix as it's more convenient to handle it
    xc = get_time_slices(sh, x)

    # variable to hold the computed result
    out = similar(x)
    outc = get_time_slices(sh, out)

    if ~isparallel(sh)
        for ii in 1:M
            ip1 = (ii == M) ? 1 : ii+1
            # we can use views but Sundials will complain
            outc[:, ii] .= evolve(sh.flow, xc[:, ii], pars, sh.ds[ii] * T).u .- xc[:, ip1]
        end
    else
        solOde = evolve(sh.flow, xc, pars, sh.ds .* T)
        for ii in 1:M
            ip1 = (ii == M) ? 1 : ii+1
            # we can use views but Sundials will complain
            outc[:, ii] .= @views solOde[ii][2] .- xc[:, ip1]
        end
    end

    # add constraint
    out[end] = @views sh.section(get_time_slice(sh, xc, 1), T)
    return out
end

# shooting functional, this allows for AbstractArray state space
function (sh::ShootingProblem)(x::BorderedArray, pars)
    # period of the cycle
    T = getperiod(sh, x)
    M = get_mesh_size(sh)

    # extract the orbit guess
    xc = get_time_slices(sh, x)

    # variable to hold the computed result
    out = similar(x)

    if ~isparallel(sh)
        for ii in 1:M
            # we can use views but Sundials will complain
            ip1 = (ii == M) ? 1 : ii+1
            copyto!(out.u[ii], evolve(sh.flow, xc[ii], pars, sh.ds[ii] * T).u .- xc[ip1])
        end
    else
        @assert false "Not implemented yet. Try to use an AbstractVector instead"
    end

    # add constraint
    out.p = sh.section(get_time_slice(sh, x, 1), T)
    return out
end

# jacobian of the shooting functional
function (sh::ShootingProblem)(x::AbstractVector, pars, dx::AbstractVector; δ = convert(eltype(x), 1e-8))
    # period of the cycle
    # Sundials does not like @views :(
    dT = getperiod(sh, dx)
    T  = getperiod(sh, x)
    M  = get_mesh_size(sh)

    xc = get_time_slices(sh, x)
    dxc = get_time_slices(sh, dx)

    # variable to hold the computed result
    out = similar(x)
    outc = get_time_slices(sh, out)

    if ~isparallel(sh)
        for ii in 1:M
            ip1 = (ii == M) ? 1 : ii+1
            tmp = jvp(sh.flow, xc[:, ii], pars, dxc[:, ii], sh.ds[ii] * T)
            # call jacobian of the flow, jacobian-vector product
            outc[:, ii] .= @views tmp.du .+ vf(sh.flow, tmp.u, pars) .* sh.ds[ii] .* dT .- dxc[:, ip1]
        end
    else
        solOde = jvp(sh.flow, xc, pars, dxc, sh.ds .* T)
        # call jacobian of the flow, jacobian-vector product
        for ii in 1:M
            ip1 = (ii == M) ? 1 : ii+1
            outc[:, ii] .= solOde[ii].du .+ vf(sh.flow, solOde[ii].u, pars) .* sh.ds[ii] .* dT .- dxc[:, ip1]
        end
    end

    # add constraint
    N = div(length(x) - 1, M)
    out[end] = @views sh.section(x[1:N], T, dx[1:N], dT)
    return out
end

# jacobian of the shooting functional, this allows for Array state space
function (sh::ShootingProblem)(x::BorderedArray, pars, dx::BorderedArray; δ = convert(eltype(x.u), 1e-8))
    dT = getperiod(sh, dx)
    T  = getperiod(sh, x)
    M  = get_mesh_size(sh)

    # variable to hold the computed result
    out = BorderedArray{typeof(x.u), typeof(x.p)}(similar(x.u), typeof(x.p)(0))

    if ~isparallel(sh)
        for ii in 1:M
            ip1 = (ii == M) ? 1 : ii+1
            # call jacobian of the flow
            tmp = jvp(sh.flow, x.u[ii], pars, dx.u[ii], sh.ds[ii] * T)
            copyto!(out.u[ii], tmp.du .+ vf(sh.flow, tmp.u, pars) .* sh.ds[ii] .* dT .- dx.u[ip1])
        end
    else
        @assert false "Not implemented yet. Try using AbstractVectors instead"
    end

    # add constraint
    out.p = sh.section(x.u[1], T, dx.u[1], dT)
    return out
end

# inplace computation of the matrix of the jacobian of the shooting problem, only serial for now
function (sh::ShootingProblem)(::Val{:JacobianMatrixInplace}, J::AbstractMatrix, x::AbstractVector, pars)
    T = getperiod(sh, x)
    M = get_mesh_size(sh)
    N = div(length(x) - 1, M)

    # extract the orbit guess and reshape it into a matrix as it's more convenient to handle it
    xc = get_time_slices(sh, x)

    # jacobian of the flow
    dflow = (_J, _x, _T) -> ForwardDiff.jacobian!(_J, z -> evolve(sh.flow, Val(:SerialTimeSol), z, pars, _T).u, _x)

    # put the matrices by blocks
    In = I(N)
    for ii in 1:M
        @views dflow(J[(ii-1)*N+1:(ii-1)*N+N, (ii-1)*N+1:(ii-1)*N+N], xc[:, ii], sh.ds[ii] * T)
        # we put the identity matrices
        ip1 = (ii == M) ? 1 : ii+1
        if M == 1
            J[(ii-1)*N+1:(ii-1)*N+N, (ip1-1)*N+1:(ip1-1)*N+N] .+= (-1) .* In
        else
            J[(ii-1)*N+1:(ii-1)*N+N, (ip1-1)*N+1:(ip1-1)*N+N]  .= (-1) .* In
        end
        # we fill the last column
        tmp = @views evolve(sh.flow, Val(:SerialTimeSol), xc[:, ii], pars, sh.ds[ii] * T).u
        J[(ii-1)*N+1:(ii-1)*N+N, end] .= vf(sh.flow, tmp, pars) .* sh.ds[ii]
    end

    # we fill the last row
    @views ForwardDiff.gradient!(J[end, 1:N], z -> sh.section(z, T), x[1:N])
    J[end, end] = @views ForwardDiff.derivative(z -> sh.section(x[1:N], z), T)

    return J
end

# out of place version
(sh::ShootingProblem)(::Val{:JacobianMatrix}, x::AbstractVector, pars) = sh(Val(:JacobianMatrixInplace), zeros(eltype(x), length(x), length(x)), x, pars)

function residual!(pb::ShootingProblem, out, x, p)
    copyto!(out, pb(x, p))
    out
end
residual(pb::ShootingProblem, x, p) = pb(x, p)
####################################################################################################
"""
$(SIGNATURES)

Compute the full periodic orbit associated to `x`. Mainly for plotting purposes.
"""
function get_periodic_orbit(prob::ShootingProblem, x::AbstractVector, pars; kode...)
    T = getperiod(prob, x)
    M = get_mesh_size(prob)
    N = div(length(x) - 1, M)
    xv = @view x[1:end-1]
    xc = reshape(xv, N, M)
    Th = eltype(x)

    # !!!! we could use @views but then Sundials will complain !!!
    if ~isparallel(prob)
        sol = [evolve(prob.flow, Val(:Full), xc[:, ii], pars, prob.ds[ii] * T; kode...) for ii in 1:M]
        time = sol[1].t; u = VectorOfArray(sol[1].u)
        # we could also use Matrix(sol[1])
        for ii in 2:M
            append!(time, sol[ii].t .+ time[end])
            append!(u.u, sol[ii].u)
        end
        return SolPeriodicOrbit(t = time, u = u)

    else # threaded version
        sol = evolve(prob.flow, Val(:Full), xc, pars, prob.ds .* T; kode...)
        time = sol[1].t; u = VectorOfArray(sol[1].u)
        for ii in 2:M
            append!(time, sol[ii].t .+ time[end])
            append!(u.u, sol[ii].u)
        end
        return SolPeriodicOrbit(t = time, u = u)
    end
end
get_periodic_orbit(prob::ShootingProblem, x::AbstractVector, p::Real; kode...) = get_periodic_orbit(prob, x, setparam(prob, p); kode...)

function get_po_solution(prob::ShootingProblem, x, pars; kode...)
    T = getperiod(prob, x)
    M = get_mesh_size(prob)
    N = div(length(x) - 1, M)
    xv = @view x[1:end-1]
    xc = reshape(xv, N, M)

    # !!!! we could use @views but then Sundials will complain !!!
    if ~isparallel(prob)
        sol_ode = [evolve(prob.flow, Val(:Full), xc[:, ii], pars, prob.ds[ii] * T; kode...) for ii in 1:M]
    else # threaded version
        sol_ode = evolve(prob.flow, Val(:Full), xc, pars, prob.ds .* T; kode...)
    end

    sol = (period = T, sol = sol_ode)

    return POSolution(prob, sol, pars)
end

function (sol::POSolution{ <: ShootingProblem})(t)
    T = sol.x.period
    t = mod(t, T)
    t0 = zero(t)
    M = get_mesh_size(sol.pb)
    ii = 1
    while ii <= M
        tspan = sol.x.sol[ii].prob.tspan
        if t0 + tspan[1] <= t <= t0 + tspan[2]
            break
        else
            t0 += tspan[2]
            ii += 1
        end
    end
    sol.x.sol[ii](t-t0)
end
####################################################################################################
# functions needed for Branch switching from Hopf bifurcation point
function re_make(prob::ShootingProblem, prob_vf, hopfpt, ζr, orbitguess_a, period; k...)
    # append period at the end of the initial guess
    orbitguess_v = reduce(vcat, orbitguess_a)
    orbitguess = vcat(vec(orbitguess_v), period) |> vec
    section = isnothing(prob.section) ? SectionSS(residual(prob_vf, orbitguess_a[1], hopfpt.params), copy(orbitguess_a[1])) : prob.section
    # update the problem but not the section if the user passed one
    probSh = setproperties(prob, 
                            section = section, 
                            par = getparams(prob_vf), 
                            lens = getlens(prob_vf))
    probSh.section.normal ./= norm(probSh.section.normal)

    # be sure that the vector field is correctly inplace in the Flow structure
    # @set! probSh.flow.F = prob_vf.VF.F

    return probSh, orbitguess
end
