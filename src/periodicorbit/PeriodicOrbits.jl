abstract type AbstractBoundaryValueDiscretization end
abstract type AbstractPeriodicOrbitDiscretization <: AbstractBoundaryValueDiscretization end
# You must implement getparams(::AbstractPeriodicOrbitDiscretization)

# Periodic orbit computations by discretizing the time derivative (collocation, trapezoid)
abstract type AbstractPODifferentialDiscretization <: AbstractPeriodicOrbitDiscretization end
# finite differences is a sub-case of Differential discretization
abstract type AbstractPOFiniteDifferencesDiscretization <: AbstractPODifferentialDiscretization end

# Periodic orbit computations by shooting method
abstract type AbstractPOShootingDiscretization <: AbstractPeriodicOrbitDiscretization end
abstract type AbstractPoincareShootingDiscretization <: AbstractPOShootingDiscretization end
################################################################################
function re_make(prob::AbstractPODifferentialDiscretization;
                params = getparams(prob)
                )
    @set prob.prob_vf = re_make(prob.prob_vf; params)
end

# ShootingProblem and PoincareShootingProblem store params directly in the corresponding struct
function re_make(prob::AbstractPOShootingDiscretization;
                params = getparams(prob)
                )
    @set prob.par = params
end
################################################################################
# get the number of time slices
@inline get_mesh_size(pb::AbstractPeriodicOrbitDiscretization) = pb.M
isinplace(::AbstractPOShootingDiscretization) = false

residual!(pb::PeriodicOrbit, out, x, pars) = po_residual!(get_discretization(pb), out, x, pars)
residual(pb::PeriodicOrbit, x, pars) = po_residual!(get_discretization(pb), similar(x), x, pars)
get_discretization(disc::AbstractPeriodicOrbitDiscretization) = disc

"""
$(TYPEDSIGNATURES)

Compute the period of the periodic orbit associated to `x`.
"""
@inline getperiod(::AbstractPeriodicOrbitDiscretization, x, par = nothing) = _extract_period(x)
@inline getperiod(prob::AbstractWrapperPeriodicOrbitProblem, u, p) = getperiod(get_discretization(prob), u, p)

@inline _extract_period(x::AbstractVector) = x[end]
@inline _extract_period(x::BorderedArray)  = x.p

# The next method only used just in the current file. Allows to set the parameters, like during aBS
_set_params_in_po(pb::AbstractPODifferentialDiscretization, pars) = (@set pb.prob_vf = re_make(pb.prob_vf; params = pars))
_set_params_in_po(pb::AbstractPOShootingDiscretization, pars) = (@set pb.par = pars)

# function to extract trajectories from branch
get_periodic_orbit(prob::AbstractWrapperPeriodicOrbitProblem, u, p) = get_periodic_orbit(get_discretization(prob), u, p)
get_periodic_orbit(br::AbstractBranchResult, ind::Int) = get_periodic_orbit(getprob(br), br.sol[ind].x, setparam(br, br.sol[ind].p))

@inline has_hessian(::PeriodicOrbitFunctionalSh) = true

Base.size(pb::AbstractPOFiniteDifferencesDiscretization) = (pb.M, pb.N)
on_gpu(pb::AbstractPOFiniteDifferencesDiscretization) = pb.ongpu
has_hessian(pb::AbstractPOFiniteDifferencesDiscretization) = pb.d2F == nothing
isinplace(pb::AbstractPOFiniteDifferencesDiscretization) = isinplace(pb.prob_vf)

function applyJ(pb, dest, x, p, dx) #TODO REMOVE?
    if isinplace(pb)
        pb.prob_vf.VF.J(dest, x, p, dx)
    else
        dest .= apply(pb.prob_vf.VF.J(x, p), dx)
    end
    dest
end

"""
$(TYPEDSIGNATURES)

This function generates an initial guess for the solution of the problem `pb` based on the orbit `t -> orbit(t)` for t ∈ [0, 2π] and the period `period`. Used also in `generate_ci_problem`.
"""
function generate_solution(pb::AbstractPeriodicOrbitDiscretization, orbit, period)
    M = get_mesh_size(pb)
    orbitguess_a = [orbit(t) for t in LinRange(0, 2pi, M + 1)[1:M]]
    # append period at the end of the initial guess
    orbitguess_v = reduce(vcat, orbitguess_a)
    if pb  isa PoincareShootingProblem
        return vec(orbitguess_v)
    else
        return vcat(vec(orbitguess_v), period) |> vec
    end
end

"""
$(TYPEDEF)

Structure to encode the solution associated to a functional like `::PeriodicOrbitOCollProblem` or `::ShootingProblem`. In the particular case of `::PeriodicOrbitOCollProblem`, this allows to use the collocation polynomials to interpolate the solution. Hence, if `sol::POSolution`, then one can call

    sol = BifurcationKit.POSolution(prob_coll, x)
    sol(t)

on any time `t`.

## Fields
$(TYPEDFIELDS)
"""
struct POSolution{Tpb, Tx, Tp}
    pb::Tpb
    x::Tx
    pars::Tp
end
POSolution(prob::AbstractPeriodicOrbitDiscretization, x) = POSolution(prob, x, nothing)
####################################################################################################
# method to save solution on the branch
save_solution(::PeriodicOrbitFunctionalSh, x, p) = x

"""
$(TYPEDEF)

Structure to save a solution from a PO functional on the branch. This is useful for branching in case mesh adaptation is used or when the phase condition is adapted. This is for example returned by `save_solution(::PeriodicOrbitFunctionalColl, ...)`

# Internal fields
$(TYPEDFIELDS)
"""
struct POSolutionAndState{T1, T2, T3, T4}
    "Initial mesh."
    mesh::T1
    "Solution on time mesh."
    sol::T2
    "Adapted mesh."
    _mesh::T3
    "Phase condition."
    ϕ::T4
end
@inline _getsolution(x) = x
@inline _getsolution(pb::POSolutionAndState) = pb.sol
minus(x::POSolutionAndState, y::POSolutionAndState) = minus(_getsolution(x), _getsolution(y))
####################################################################################################
"""
$(TYPEDEF)

This struct allows to have a unified interface for periodic orbits methods to record solutions, useful for plotting for example. This is returned by `get_periodic_orbit`.

# Internal fields
$(TYPEDFIELDS)
"""
@with_kw_noshow struct SolPeriodicOrbit{𝒯s, 𝒯u}
    "Time mesh."
    t::𝒯s
    "Solution discretized on time mesh."
    u::𝒯u
end
Base.getindex(sol::SolPeriodicOrbit, i...) = getindex(sol.u, i...)
Base.axes(sol::SolPeriodicOrbit, i) = axes(sol.u, i)
####################################################################################################
function update!(wrap::Union{PeriodicOrbitFunctionalSh, PeriodicOrbitFunctionalTrap}, iter, state)
    prob = get_discretization(wrap)
    success = converged(state)
    bisection = in_bisection(state)
    update_section_every_step = prob.update_section_every_step
    step = state.step
    z = getsolution(state)
    if success && mod_counter(step, update_section_every_step) == 1 && bisection == false
        @debug "[Periodic orbit] update section"
        # Trapezoid and Shooting need the parameters for section update:
        updatesection!(prob, z.u, setparam(wrap, z.p))
    end
    return true
end
####################################################################################################
const _po_sh_jacobian_types = (AutoDiffMF(),
                                MatrixFree(),
                                AutoDiffDense(),
                                AutoDiffDenseAnalytical(),
                                FiniteDifferences(),
                                FiniteDifferencesMF())

const DocStringJacobianPOSh = """
`jacobian` Specify the choice of the linear algorithm, which must belong to `$_po_sh_jacobian_types`. This is used to select a way of inverting the jacobian dG\n
    1. For `MatrixFree()`, matrix free jacobian, the jacobian is specified by the user in `prob`. This is to be used with an iterative solver (e.g. GMRES) to solve the linear system
    2. For `AutoDiffMF()`, we use Automatic Differentiation (AD) to compute the (matrix-free) derivative of `x -> prob(x, p)` using a directional derivative, also called JVP product. This is to be used with an iterative solver (e.g. GMRES) to solve the linear system
    3. For `AutodiffDense()`. Same as for `AutoDiffMF` but the jacobian is formed as a dense Matrix. You can use a direct solver or an iterative one.
    4. For `FiniteDifferences()`, same as for `AutoDiffDense` but we use Finite Differences to compute the jacobian of `x -> prob(x, p)` using the `δ = 1e-8` which can be passed as an argument.
    5. For `AutoDiffDenseAnalytical()`. Same as for `AutoDiffDense` but the jacobian is formed using a mix of AD and analytical formula.
    6. For `FiniteDifferencesMF()`, use Finite Differences to compute the matrix-free jacobian of `x -> prob(x, p)` using the `δ = 1e-8` which can be passed as an argument.
"""
##########################
@inline is_symmetric(prob::PeriodicOrbitFunctionalSh) = false
jacobian(prob::AbstractWrapperPeriodicOrbitProblem, x, p) = _jacobian_po(prob, prob.jacobian, x, p)

########
# useful getters for FloquetColl
get_wrap_po(iter::ContIterable) = get_wrap_po(getprob(iter))
get_wrap_po(pb::AbstractWrapperPeriodicOrbitProblem) = pb

_generate_jacobian(::AbstractPeriodicOrbitDiscretization, J::Union{AutoDiffDense,
                                                                FiniteDifferences,
                                                                AutoDiffMF,
                                                                MatrixFree,
                                                                FullLU,
                                                                FullMatrixFree,
                                                                FullSparse,
                                                                DenseAnalytical}, o, pars; k...) = J

_generate_jacobian(::AbstractPeriodicOrbitDiscretization, ::FiniteDifferencesMF, orbitguess, pars; δ = convert(VI.scalartype(orbitguess), 1e-8)) = (FiniteDifferencesMF(), δ)

function _generate_jacobian(disc::AbstractPOShootingDiscretization, ::AutoDiffDenseAnalytical, orbitguess, pars; k...)
    _J = po_jacobian(disc, orbitguess, pars)
    return (AutoDiffDenseAnalytical(), _J)
end
########
function _jacobian_po(wrap_po::AbstractWrapperPeriodicOrbitProblem, ::AutoDiffDense, x, p)
    ForwardDiff.jacobian(z -> residual(wrap_po, z, p), x)
end

function _jacobian_po(wrap_po::AbstractWrapperPeriodicOrbitProblem, ::FiniteDifferences, x, p)
    return finite_differences(z -> residual(wrap_po, z, p), x)
end

function _jacobian_po(wrap::AbstractWrapperPOShootingProblem, J::Tuple{AutoDiffDenseAnalytical, Tj}, x, p) where {Tj}
    sh = get_discretization(wrap)
    po_jacobian!(sh, J[2], x, p)
    return J[2]
end

function _jacobian_po(wrap_po::AbstractWrapperPeriodicOrbitProblem, J::Tuple{FiniteDifferencesMF, Tj}, x, p) where {Tj}
    δ = J[2]
    return dx -> (residual(wrap_po, x .+ δ .* dx, p) .- 
                  residual(wrap_po, x .- δ .* dx, p)) ./ (2δ)
end

function _jacobian_po(wrap_po::AbstractWrapperPeriodicOrbitProblem, ::AutoDiffMF, x, p)
    return dx -> ForwardDiff.derivative(z -> residual(wrap_po, x .+ z .* dx, p), 0)
end

_jacobian_po(wrap_po::AbstractWrapperPeriodicOrbitProblem, ::Union{MatrixFree, FullMatrixFree}, x, p) = dx -> po_jvp(get_discretization(wrap_po), x, p, dx)


"""
$(TYPEDSIGNATURES)

This is the Newton-Krylov Solver for computing a periodic orbit using the (Standard / Poincaré) Shooting method.
Note that the linear solver has to be appropriately set up in `options`.

# Arguments

Similar to [`newton`](@ref) except that `prob` is either a [`ShootingProblem`](@ref) or a [`PoincareShootingProblem`](@ref). These two problems have specific options to be tuned, we refer to their link for more information and to the tutorials.

- `prob` a problem of type `<: AbstractPOShootingDiscretization` encoding the shooting functional G.
- `orbitguess` a guess for the periodic orbit. See [`ShootingProblem`](@ref) and See [`PoincareShootingProblem`](@ref) for information regarding the shape of `orbitguess`.
- `par` parameters to be passed to the functional
- `options` same as for the regular [`newton`](@ref) method.

# Optional argument
$DocStringJacobianPOSh
""" # TODO This is a bit of a hack. It should be a Functional not a discretization like Collocation
function newton(disc::AbstractPOShootingDiscretization,
                orbitguess,
                options::NewtonPar;
                lens::OpticType = nothing,
                δ = getdelta(disc),
                kwargs...)
    jac = _generate_jacobian(disc, disc.jacobian, orbitguess, getparams(disc); δ)
    probw = PeriodicOrbitFunctionalSh(disc, jac, orbitguess, getparams(disc), lens, nothing, nothing)
    return solve(probw, Newton(), options; kwargs...)
end

"""
$(TYPEDSIGNATURES)

This is the deflated Newton-Krylov Solver for computing a periodic orbit using a (Standard / Poincaré) Shooting method.

# Arguments

Similar to [`newton`](@ref) except that `prob` is either a [`ShootingProblem`](@ref) or a [`PoincareShootingProblem`](@ref).

# Optional argument
$DocStringJacobianPOSh

# Output:
- solution::NonLinearSolution, see [`NonLinearSolution`](@ref)
"""
function newton(disc::AbstractPOShootingDiscretization,
                orbitguess::vectype,
                defOp::DeflationOperator{Tp, Tdot, T, vectype},
                options::NewtonPar{T, S, E};
                lens::OpticType = nothing,
                kwargs...,
            ) where {T, Tp, Tdot, vectype, S, E}
    jac = _generate_jacobian(disc, disc.jacobian, orbitguess, getparams(disc))
    probw = PeriodicOrbitFunctionalSh(disc, jac, orbitguess, getparams(disc), lens, nothing, nothing)
    return solve(probw, defOp, options; kwargs...)
end

####################################################################################################
# Continuation for shooting problems
"""
$(TYPEDSIGNATURES)

This is the continuation method for computing a periodic orbit using a (Standard / Poincaré) Shooting method.

# Arguments

Similar to [`continuation`](@ref) except that `probPO` is either a [`ShootingProblem`](@ref) or a [`PoincareShootingProblem`](@ref). By default, it prints the period of the periodic orbit.

# Optional arguments
- `eigsolver` specify an eigen solver for the computation of the Floquet exponents, defaults to `FloquetQaD`
$DocStringJacobianPOSh
""" # TODO This is a bit of a hack. It should be a Functional not a discretization like Collocation
function continuation(discPO::AbstractPOShootingDiscretization,
                        orbitguess,
                        alg::AbstractContinuationAlgorithm,
                        contParams::ContinuationPar,
                        linear_algo::AbstractBorderedLinearSolver;
                        δ = convert(VI.scalartype(orbitguess), getdelta(discPO)),
                        eigsolver = FloquetQaD(contParams.newton_options.eigsolver),
                        record_from_solution = nothing,
                        plot_solution = nothing,
                        kwargs...)
    if isnothing(getlens(discPO))
        error("You need to provide a lens for your periodic orbit problem.")
    end
    jacobianPO = discPO.jacobian
    jac = _generate_jacobian(discPO, jacobianPO, orbitguess, getparams(discPO); δ)
    if compute_eigenelements(contParams)
        contParams = @set contParams.newton_options.eigsolver = eigsolver
    end

    _plotsol   = modify_po_plot(discPO, getparams(discPO), getlens(discPO); plot_solution)
    record_po = RecordForPeriodicOrbits(record_from_solution, nothing)
    wrap = PeriodicOrbitFunctionalSh(discPO, jac, orbitguess, getparams(discPO), getlens(discPO), _plotsol, record_po)

    br = continuation(
        wrap, alg,
        contParams;
        kwargs...,
        linear_algo,
        kind = PeriodicOrbitCont(),
        )
    return br
end

"""
$(TYPEDSIGNATURES)

This is the continuation routine for computing a periodic orbit.

# Arguments

Similar to [`continuation`](@ref) except that `prob::AbstractPeriodicOrbitDiscretization`.

# Optional argument
- `linear_algo::AbstractBorderedLinearSolver`
$DocStringJacobianPOSh

"""
function continuation(disc::AbstractPeriodicOrbitDiscretization,
                    orbitguess,
                    alg::AbstractContinuationAlgorithm,
                    _contParams::ContinuationPar;
                    linear_algo::Ty = nothing,
                    kwargs...) where {Ty}
    _linear_algo = (Ty == Nothing) ?  MatrixBLS() : linear_algo
    return continuation(disc, orbitguess, alg, _contParams, _linear_algo; kwargs...)
end
####################################################################################################
"""
$(TYPEDSIGNATURES)

Perform automatic branch switching from a Hopf bifurcation point labelled `ind_bif` in the list of the bifurcated points of a previously computed branch `br::ContResult`. It first computes a Hopf normal form.

# Arguments

- `br` branch result from a call to `continuation`
- `ind_hopf` index of the bifurcation point in `br`
- `contParams` parameters for the call to `continuation`
- `disc` discretization used to specify the way toc compute the periodic orbit. It can be [`PeriodicOrbitTrapProblem`](@ref), [`PeriodicOrbitOCollProblem`](@ref), [`ShootingProblem`](@ref) or [`PoincareShootingProblem`](@ref) .

# Optional arguments

- `alg = getalg(br)` continuation algorithm
- `δp` used to specify the guess for the parameter on the bifurcated branch which otherwise defaults to `contParams.ds`. This allows to use an initial step larger than `contParams.dsmax`.
- `ampfactor = 1` multiplicative factor to alter the amplitude of the bifurcated solution. Useful to magnify the bifurcated solution when the bifurcated branch is very steep.
- `use_normal_form = true` whether to use the normal form in order to compute the predictor. When `false`, `ampfactor` and `δp` are used to make a predictor based on the bifurcating eigenvector. Setting `use_normal_form = false` can be useful when computing the normal form is not possible for example when higher order derivatives are not available.
- `usedeflation = true` whether to use nonlinear deflation (see [Deflated problems](@ref)) to help finding the guess on the bifurcated branch
- `nev` number of eigenvalues to be computed to get the right eigenvector
- `autodiff_nf = true` whether to use `autodiff` in `get_normal_form`. This can be used in case automatic differentiation is not working as intended.
- all `kwargs` from [`continuation`](@ref)

A modified version of `prob` is passed to `plot_solution` and `finalise_solution`.

!!! note "Linear solver"
    You have to be careful about the options `contParams.newton_options.linsolver`. In the case of Matrix-Free solver, you have to pass the right number of unknowns `N * M + 1`. Note that the options for the preconditioner are not accessible yet.
"""
function continuation(br::AbstractBranchResult, 
                      ind_bif::Int,
                      _contParams::ContinuationPar,
                      disc::AbstractPeriodicOrbitDiscretization ;
                      bif_prob = getprob(br),
                      detailed::Val{detailed_type} = Val(true),
                      use_normal_form = true,
                      autodiff_nf = true,
                      nev = length(eigenvalsfrombif(br, ind_bif)),
                      kwargs...) where {detailed_type}
    # compute the normal form of the branch point
    verbose = get(kwargs, :verbosity, 0) > 1
    verbose && (println("──▶ Considering bifurcation point:"); _show(stdout, br.specialpoint[ind_bif], ind_bif))

    detailed = Val(detailed_type && use_normal_form) # TODO improve type stability
    hopfpt = hopf_normal_form(bif_prob, br, ind_bif; nev, verbose, detailed, autodiff = autodiff_nf)
    return _continuation(hopfpt, bif_prob, _contParams, disc; verbose, alg = getalg(br), kwargs...)
end

function _continuation(hopfpt::Hopf,
                      bif_prob::AbstractBifurcationProblem,
                      _contParams::ContinuationPar,
                      disc::AbstractPeriodicOrbitDiscretization;
                      verbose = false,
                      alg = PALC(),
                      δp = nothing,
                      ampfactor = 1,
                      usedeflation = false,
                      kwargs...)
    # compute predictor for point on new branch
    ds = isnothing(δp) ? _contParams.ds : δp
    𝒯 = typeof(ds)
    pred = predictor(hopfpt, ds; verbose, ampfactor = 𝒯(ampfactor))

    # we compute a phase so that the constraint equation
    # < u(0) − u_hopf, ψ > = 0 is satisfied.
    ζr = real.(hopfpt.ζ)
    ζi = imag.(hopfpt.ζ)
    # this phase is for POTrap problem constraint to be satisfied
    ϕ = atan(VI.inner(ζr, ζr), VI.inner(ζi, ζr))

    verbose && printstyled(color = :green, "━"^55*
            "\n┌─ Start branching from Hopf bif. point to periodic orbits.",
            "\n├─ Bifurcation type = ", hopfpt.type,
            "\n├─── Hopf param  p0 = ", hopfpt.p,
            "\n├─── new param    p = ", pred.p, ", p - p0 = ", pred.p - hopfpt.p,
            "\n├─── amplitude p.o. = ", pred.amp,
            "\n├─── period       T = ", pred.period,
            "\n├─── phase        ϕ = ", ϕ / pi, "⋅π",
            "\n├─ Method = \n", disc, "\n")

    if pred.amp > 0.1
        @debug "The guess for the amplitude of the first periodic orbit on the bifurcated branch obtained by the predictor is not small: $(pred.amp). This may lead to convergence failure of the first newton step or select a branch far from the Hopf point.\nYou can either decrease `ds` or `δp` (which is  how far from the bifurcation point you want the branch of periodic orbits to start). Alternatively, you can specify a multiplicative factor `ampfactor` to be applied to the predictor amplitude."
    end

    # extract the vector field and use it possibly to affect the PO functional
    bif_prob_rm = re_make(bif_prob; params = setparam(bif_prob, pred.p)) # TODO Not good: we cannot change lens

    # build the initial guess
    M = get_mesh_size(disc)
    orbitguess_a = [pred.orbit(t - ϕ) for t in LinRange(0, 2pi, M + 1)[1:M]] # GIVES RUNTIME DISPATCH
    probPO, orbitguess = re_make(disc, bif_prob_rm, hopfpt, ζr, orbitguess_a, abs(2pi/pred.ω); orbit = pred.orbit)

    if _contParams.newton_options.linsolver isa GMRESIterativeSolvers
        _contParams = @set _contParams.newton_options.linsolver.N = length(orbitguess)
    end

    if usedeflation
        verbose &&
            println("\n├─ Attempt branch switching\n──> Compute point on the current branch...")
        probPO isa PoincareShootingProblem &&
            @warn "Poincaré Shooting does not work very well with stationary states."
        optn = _contParams.newton_options

        # we start with the case of zero amplitude
        orbitzeroamp_a = [hopfpt.x0 for _ = 1:M]
        # this factor prevent shooting jacobian from being singular at fixed points
        if probPO isa PoincareShootingProblem
            Tfactor = 0
        elseif probPO isa AbstractPOFiniteDifferencesDiscretization
            Tfactor = 100 / abs(2pi / pred.ω)
        else
            Tfactor = 0.001
        end
        cb = get(kwargs, :callback_newton, cb_default)
        # TODO should only update guess here, cf Poincaré
        probPO0, orbitzeroamp = re_make(probPO, bif_prob, hopfpt, ζr, orbitzeroamp_a, 𝒯(Tfactor * abs(2pi / pred.ω)))
        sol0 = newton(probPO0, orbitzeroamp, optn; callback = cb, kwargs...)

        # find the bifurcated branch using deflation
        if ~(probPO isa PoincareShootingProblem)
            deflationOp = DeflationOperator(2, (x, y) -> VI.inner(x[begin:end-1], y[begin:end-1]), one(𝒯), [sol0.u]; autodiff = true)
        else
            deflationOp = DeflationOperator(2, (x, y) -> VI.inner(x, y) / M, one(𝒯), [sol0.u]; autodiff = true)
        end

        verbose && println("\n──▶ Compute point on bifurcated branch...")
        solbif = newton(probPO, orbitguess, deflationOp, (@set optn.max_iterations = 10 * optn.max_iterations); callback = cb, kwargs...)
        if converged(solbif) == false
            error("Deflated newton did not converge")
        end
        orbitguess .= solbif.u

        branch = continuation(
            probPO, orbitguess, alg,
            _contParams;
            kwargs...,
        )

        return Branch(branch, hopfpt)
    end

    # perform continuation
    branch = continuation(
        probPO, orbitguess, alg,
        _contParams;
        kwargs...
    )
    return Branch(branch, hopfpt)
end
####################################################################################################
"""
$(TYPEDSIGNATURES)

Branch switching from the curve to Hopf bifurcation points to the curve of periodic orbits emanating from it.

# Arguments
- `br_hopf` curve of kind `HopfCont` that is a curve of Hopf bifurcation points.
- `ind_pt::Int` index of Hopf points from `br_hopf`.
- `disc` discretization for computing periodic orbits.

# Keyword arguments
- `lens` parameter axis to be used for the continuation
- `autodiff_nf` whether to use automatic differentiation for the computation of the normal form.
"""
function continuation_from_hopf_point(br_hopf::AbstractResult{HopfCont, Tprob},
                      ind_pt::Int,
                      options_cont::ContinuationPar,
                      disc::AbstractPeriodicOrbitDiscretization;
                      lens = getlens(br_hopf),
                      autodiff_nf = true,
                      nev::Int = length(eigenvals(br_hopf, ind_pt)),
                      kwargs...) where {Tprob <: HopfMAProblem}
    verbose = get(kwargs, :verbosity, 0) > 1 ? true : false
    # extract the problem, formulations and vector field
    _prob = getprob(br_hopf)
    𝐇 = get_formulation(_prob)
    vector_field = 𝐇.prob_vf
    if ~(𝐇 isa HopfMinimallyAugmentedFormulation)
        error("[PO branching from Hopf curve] You need to provide a curve of Hopf points.\nThe underlying problem is not a `HopfProblemMinimallyAugmented`.\nWe found the type: $(typeof(𝐇))")
    end
    # we get the Hopf point
    bifpt = br_hopf.sol[ind_pt]
    ω = get_frequency(bifpt.x, 𝐇)
    λ = Complex(0, ω)
    x0 = get_solution(bifpt.x)
    params = getparams(br_hopf, ind_pt)
    L = jacobian(vector_field, x0, params)

    # newton parameters
    optionsN = br_hopf.contparams.newton_options

    # TODO! Use Minimally Augmented system for this instead of re-computing all eigenvalues
    # compute the right eigenvector
    verbose && @info "Recomputing eigenvector on the fly"
    _λ, _ev, _ = optionsN.eigsolver.eigsolver(L, nev)
    _ind = argmin(abs.(_λ .- λ))
    verbose && @info "The eigenvalue is $(_λ[_ind])"
    abs(_λ[_ind] - λ) > 10br_hopf.contparams.newton_options.tol && @warn "We did not find the correct eigenvalue $λ. We found $(_λ[_ind])"
    ζ = geteigenvector(optionsN.eigsolver, _ev, _ind)
    ζ ./= LA.norm(ζ)

    # left eigen-elements
    _Jt = has_adjoint(vector_field) ? jacobian_adjoint(vector_field, x0, params) : adjoint(L)
    ζ★, λ★ = get_adjoint_basis(_Jt, conj(_λ[_ind]), optionsN.eigsolver.eigsolver; nev, verbose)

    # check that λ★ ≈ conj(λ)
    abs(λ + λ★) > 1e-2 && @warn "We did not find the left eigenvalue for the Hopf point to be very close to the imaginary part, $λ ≈ $(λ★) and $(abs(λ + λ★)) ≈ 0?\nYou can perhaps increase the number of computed eigenvalues, the number is nev = $nev."

    # normalise left eigenvector
    ζ★ ./= VI.inner(ζ, ζ★)
    @assert VI.inner(ζ, ζ★) ≈ 1

    hopfpt = Hopf(x0, nothing, _get(params, lens),
                ω,
                params, lens,
                ζ, ζ★,
                HopfNormalForm(a = missing, 
                               b = missing,
                               Ψ110 = missing,
                               Ψ001 = missing,
                               Ψ200 = missing
                        ),
                Symbol("?")
                )

    # we compute the Hopf normal form
    nf = __hopf_normal_form(vector_field, hopfpt, 𝐇.linsolver ; verbose, L, autodiff = autodiff_nf)
    @debug "[PO from Hopf curve]" nf params nf.nf.b/br_hopf[ind_pt].l1
    if ~(nf.nf.b ≈ br_hopf[ind_pt].l1)
        @warn("The computation of the Lyapunov exponent for the Hopf normal form differs from the one recorded in the Hopf curve. If you used a a different norm or automatic differentiation, nevermind this warning.")
    end
    bifprob = re_make(vector_field; lens, params)
    return _continuation(nf, bifprob, options_cont, disc; verbose, kwargs...)
end
####################################################################################################
# Branch switching from bifurcations of periodic orbits
"""
$(TYPEDSIGNATURES)

Branch switching at a bifurcation point on a branch of periodic orbits (PO) specified by a `br::AbstractBranchResult`. The functional for computing the PO is `getprob(br)`. A deflated Newton-Krylov solver can be used to improve the branch switching capabilities.

# Arguments
- `br` branch of periodic orbits computed with a [`PeriodicOrbitTrapProblem`](@ref)
- `ind_bif` index of the branch point
- `_contParams` continuation parameters, see [`continuation`](@ref)

# Optional arguments
- `δp = _contParams.ds` used to specify a particular guess for the parameter in the branch which is otherwise determined by `contParams.ds`. This allows to use a step larger than `contParams.dsmax`.
- `ampfactor = 1` factor which alters the amplitude of the bifurcated solution. Useful to magnify the bifurcated solution when the bifurcated branch is very steep.
- `usedeflation = true` whether to use nonlinear deflation (see [Deflated problems](@ref)) to help finding the guess on the bifurcated branch
- `use_normal_form = true` if `false`, the predictor is based on the couple `δp, ampfactor`.

## For normal form
- `detailed = false` whether to fully compute the normal form or a very simplified version.
- `autodiff_nf = true` whether to use `autodiff` in `get_normal_form`. This can be used in case automatic differentiation is not working as intented.

## For continuation
- `linear_algo = BorderingBLS()`, same as for [`continuation`](@ref)
- `kwargs` keywords arguments used for a call to the regular [`continuation`](@ref) and the ones specific to periodic orbits (POs).
"""
function continuation(br::AbstractResult{PeriodicOrbitCont, Tprob},
                      ind_bif::Int,
                      _contParams::ContinuationPar;
                      alg = getalg(br),
                      δp = _contParams.ds, 
                      ampfactor = 1,
                      usedeflation = false,
                      linear_algo = nothing,
                      detailed::Val{detailed_type} = Val(true),
                      prm::Val{prm_type} = Val(getprob(br) isa PeriodicOrbitFunctionalColl ? false : true),
                      use_normal_form = true,
                      autodiff_nf = true,
                      kwargs...) where {Tprob <: AbstractWrapperPeriodicOrbitProblem, detailed_type, prm_type}
    bifpt = br.specialpoint[ind_bif]
    bptype = bifpt.type
    if ~(bptype in (:pd, :bp, :nd))
        error("Branching from $(bptype) not possible yet.")
    end
    if abs(bifpt.δ[1]) != 1 
        error("Only simple bifurcation points are handled properly")
    end

    detailed = Val(detailed_type && use_normal_form)
    nf = get_normal_form(br, ind_bif; detailed, prm, autodiff = autodiff_nf)
    pred = predictor(nf, δp, ampfactor; override = ~use_normal_form)
    orbitguess = pred.orbitguess
    newp = pred.pnew  # new parameter value
    new_disc = get_discretization(pred.prob) # modified discretization

    verbose = get(kwargs, :verbosity, 0) > 0
    verbose && printstyled(color = :green, "━"^55*
            "\n┌─ Start branching from $(bptype) point to periodic orbits.",
            "\n├─ Bifurcation type = ", bifpt.type,
            "\n├─── normal form    = ", use_normal_form ? "based on $(detailed_type ? "Poincaré" : "Iooss") formulation" : "none",
            "\n├─── bif. param  p0 = ", bifpt.param,
            "\n├─── period at bif. = ", getperiod(getprob(br), bifpt.x, setparam(br, bifpt.param)),
            "\n├─── new param    p = ", newp, 
            "\n├─── p - p0         = ", newp - bifpt.param,
            "\n├─── amplitude p.o. = ", pred.ampfactor,
            "\n")

    if pred.ampfactor > 0.1
        @warn "The amplitude of the first periodic orbit on the bifurcated branch\nobtained by the predictor is not small, it is = $(pred.ampfactor).\nYou can either decrease the step size `ds` or specify the distance `δp`\nfrom the bifurcation point where the branch of periodic orbits originates.\nYou can also decrease `ampfactor`"
    end

    # a priori, the following do not overwrite the options in br
    # hence the results / parameters in br are kept intact)
    _contParams = _update_cont_params(_contParams, new_disc, orbitguess)
    new_disc = _set_params_in_po(new_disc, setparam(br, newp))

    if usedeflation
        verbose && println("\n├─ Attempt branch switching\n──> Compute point on the current branch...")
        optn = _contParams.newton_options
        # find point on the first branch
        # TODO! This should not be! Call it po_from_disc_newton ?
        sol0 = newton(new_disc, pred.po, optn; kwargs...)
        if converged(sol0) == false
            error("The first guess did not converge")
        end

        # find the bifurcated branch using deflation
        deflationOp = DeflationOperator(2, (x, y) -> VI.inner(x[begin:end-1], y[begin:end-1]), one(VI.scalartype(orbitguess)), [sol0.u]; autodiff = true)
        verbose && println("\n──> Compute point on the bifurcated branch...")
        solbif = newton(new_disc, orbitguess, deflationOp,
            (@set optn.max_iterations = 10 * optn.max_iterations) ; kwargs...,)
        if converged(solbif) == false
            error("Deflated newton did not converge")
        end
        orbitguess .= solbif.u
    end

    # this should not be
    po_residual(new_disc, orbitguess, setparam(br, newp))[end] |> abs > 1 && @warn "PO constraint not satisfied"
    _linear_algo = isnothing(linear_algo) ? BorderingBLS(_contParams.newton_options.linsolver) : linear_algo

    branch = continuation( new_disc, orbitguess, alg, _contParams;
        kwargs..., # put this first to be overwritten by the following
        linear_algo = _linear_algo,
        kind = br.kind,
        record_from_solution = record_from_solution(getprob(br)),
    )

    return Branch(branch, nf)
end
