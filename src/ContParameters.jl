    """
    options = ContinuationPar(dsmin = 1e-4,...)

Returns a variable containing parameters to affect the `continuation` algorithm used to solve `F(x, p) = 0`.

# Arguments
- `dsmin, dsmax` are the minimum, maximum arclength allowed value. It controls the density of points in the computed branch of solutions.
- `ds = 0.01` is the initial arclength.
- `p_min, p_max` allowed parameter range for `p`
- `max_steps = 100` maximum number of continuation steps
- `newton_options::NewtonPar`: options for the Newton algorithm
- `save_to_file = false`: save to file. A name is automatically generated or can be defined in [`continuation`](@ref). This requires `using JLD2`.
- `save_sol_every_step::Int64 = 0` at which continuation steps do we save the current solution
- `plot_every_step = 10` at which continuation steps do we plot the current solution

## Handling eigen elements, their computation is triggered by the argument `detect_bifurcation` (see below)
- `nev = 3` number of eigenvalues to be computed. It is automatically increased to have at least `nev` unstable eigenvalues. To be set for proper  bifurcation detection. See [Detection of bifurcation points of Equilibria](@ref) for more informations.
- `save_eig_every_step = 1` record eigen vectors every specified steps. **Important** for memory limited resource, *e.g.* GPU.
- `save_eigenvectors = true` **Important** for memory limited resource, *e.g.* GPU.

## Handling bifurcation detection
- `tol_stability = 1e-10` lower bound on the real part of the eigenvalues to test for stability of equilibria and periodic orbits
- `detect_fold = true` detect Fold bifurcations? It is a useful option although the detection of Fold is cheap. Indeed, it may happen that there is a lot of Fold points and this can saturate the memory in memory limited devices (e.g. on GPU)
- `detect_bifurcation::Int` ∈ {0, 1, 2, 3} If set to 0, nothing is done. If set to 1, the eigen-elements are computed. If set to 2, the bifurcations points are detected during the continuation run, but not located precisely. If set to 3, a bisection algorithm is used to locate the bifurcations points (slower). The possibility to switch off detection is a useful option. Indeed, it may happen that there are a lot of bifurcation points and this can saturate the memory of memory limited devices (e.g. on GPU)
- `dsmin_bisection = 1e-16` dsmin for the bisection algorithm for locating bifurcation points
- `n_inversion = 2` number of sign inversions in bisection algorithm
- `max_bisection_steps = 15` maximum number of bisection steps
- `tol_bisection_eigenvalue = 1e-16` tolerance on real part of eigenvalue to detect bifurcation points in the bisection steps

## Handling `ds` adaptation (see [`continuation`](@ref) for more information)
- `a  = 0.5` aggressiveness factor. It is used to adapt `ds` in order to have a number of newton iterations per continuation step roughly constant. The higher `a` is, the larger the step size `ds` is changed at each continuation step.

## Handling event detection
- `detect_event::Int` ∈ {0, 1, 2} If set to 0, nothing is done. If set to 1, the event locations are sought during the continuation run, but not located precisely. If set to 2, a bisection algorithm is used to locate the event (slower).
- `tol_param_bisection_event = 1e-16` tolerance on parameter to locate event

## Misc
- `η = 150.` parameter to estimate tangent at first point with parameter  p₀ + ds / η
- `detect_loop` [WORK IN PROGRESS] detect loops in the branch and stop the continuation

!!! tip "Mutating"
    For performance reasons, we decided to use an immutable structure to hold the parameters. One can use the package `Accessors.jl` to drastically simplify the mutation of different fields. See tutorials for more examples.
"""
@with_kw struct ContinuationPar{T, S <: AbstractLinearSolver, E <: AbstractEigenSolver}
    # tangent predictor parameters for continuation
    dsmin::T    = 1e-4
    dsmax::T    = 1e-1
    ds::T       = 1e-2

    # parameters for continuation
    a::T    = 0.5 # aggressiveness factor for step size adaptation

    # parameters bound
    p_min::T    = -1.0
    p_max::T    =  1.0

    # maximum number of continuation steps
    max_steps::Int64  = 400

    # Newton solver parameters
    newton_options::NewtonPar{T, S, E} = NewtonPar()
    η::T = 150.                         # parameter to estimate tangent at first point by finite differences

    save_to_file::Bool = false          # save to file?
    save_sol_every_step::Int64 = 1      # at what steps do we save the current solution

    # parameters for eigenvalues
    nev::Int64 = 3                      # number of eigenvalues
    save_eig_every_step::Int64 = 1      # what steps do we keep the eigenvectors
    save_eigenvectors::Bool    = true   # useful options because if puts a high memory pressure

    plot_every_step::Int64 = 10

    # handling bifurcation points
    tol_stability::T = 1e-10              # lower bound for stability of equilibria and periodic orbits
    detect_fold::Bool = true              # detect fold points?
    detect_bifurcation::Int64 = 3         # detect other bifurcation points?
    dsmin_bisection::T = 1e-16            # dsmin for the bisection algorithm when locating bifurcation points
    n_inversion::Int64 = 2                # number of sign inversions in bisection algorithm
    max_bisection_steps::Int64 = 15       # maximum number of bisection steps
    tol_bisection_eigenvalue::T = 1e-16   # tolerance on real part of eigenvalue to detect bifurcation points in the bisection steps. Must be small otherwise Shooting and friends will fail detecting bifurcations.

    # handling event detection
    detect_event::Int64 = 0               # event location
    tol_param_bisection_event::T = 1e-16  # tolerance on value of parameter
    detect_loop::Bool = false             # detect if the branch loops

    # various tests to ensure everything is right
    @assert tol_stability >= 0 "You must provide a positive tolerance for tol_stability"
    @assert dsmax >= abs(ds) >= dsmin >= 0 "You must provide a valid interval (ordered) for ds. You passed $(dsmax) >= $(abs(ds)) >= $(dsmin) with \ndsmax = $dsmax\nds    = $ds\ndsmin = $dsmin"
    @assert abs(ds) >= dsmin_bisection >= 0 "You must provide a valid interval for `ds` and `dsmin_bisection`"
    @assert p_max >= p_min "You must provide a valid interval [p_min, p_max]"
    @assert iseven(n_inversion) "The option `n_inversion` number must be even"
    @assert 0 <= detect_bifurcation <= 3 "The option `detect_bifurcation` must belong to {0,1,2,3}"
    @assert 0 <= detect_event <= 2 "The option `detect_event` must belong to {0,1,2}"
    @assert (detect_bifurcation > 1 && detect_event == 0) || (detect_bifurcation <= 1 && detect_event >= 0)  "One of these options must be put to zero: detect_bifurcation = $detect_bifurcation and detect_event = $detect_event"
    @assert tol_bisection_eigenvalue >= 0 "The option `tol_bisection_eigenvalue` must be positive"
    @assert plot_every_step > 0 "plot_every_step must be positive. You can turn off plotting by passing plot = false to `continuation`"
    @assert ~(detect_bifurcation > 1 && save_eig_every_step > 1) "We must at least save all eigenvalues for detection of bifurcation points. Please use save_eig_every_step = 1 or detect_bifurcation = 1."
end

@inline compute_eigenelements(cp::ContinuationPar) = cp.detect_bifurcation > 0
@inline compute_eigenvalues(cp::ContinuationPar) = cp.detect_bifurcation > 0
@inline save_eigenvectors(cp::ContinuationPar) = cp.save_eigenvectors

# clamp ds value
clamp_ds(ds, contparams::ContinuationPar) = sign(ds) * clamp(abs(ds), contparams.dsmin, contparams.dsmax)

"""
Allows to alter the continuation parameters based on the bifurcation problem and the continuation algorithm.
"""
function init(contparams::ContinuationPar{T,S,E}, 
               prob::AbstractBifurcationProblem, 
               alg::AbstractContinuationAlgorithm) where {T,S,E}
    if E <: DefaultEig
        n = length(getu0(prob))
        if n <= 50
            @reset contparams.nev = n
        end
    end
    contparams
end
