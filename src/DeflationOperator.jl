abstract type AbstractDeflationFactor end

"""
Structure for defining a custom distance.
You need to pass a function `d(u, v)`.
"""
struct CustomDist{T}
    dist::T
end
@inline (cd::CustomDist)(u, v) = cd.dist(u, v)

"""
$(TYPEDEF)

This operator allows to handle the following situation. Assume you want to solve `F(x)=0` with a Newton algorithm but you want to avoid the process to return some already known solutions ``roots_i``. The deflation operator penalizes these roots. You can create a `DeflationOperator` to define a scalar function `M(u)` used to find, with Newton iterations, the zeros of the following function

``F(u) ⋅ Πᵢ(||u - rootᵢ||⁻²ᵖ + α) := F(u) ⋅ M(u)`` 

where ``||u||² = dot(u, u)``. The fields of the struct `DeflationOperator` are as follows:

## Fields

$(TYPEDFIELDS)

Given `defOp::DeflationOperator`, one can access its roots via `defOp[n]` as a shortcut for `defOp.roots[n]`. Note that you can also use `defOp[end]`.

Also, one can add (resp. remove) a new root by using `push!(defOp, newroot)` (resp. `pop!(defOp)`). Finally `length(defOp)` is a shortcut for `length(defOp.roots)`

## Constructors

- `DeflationOperator(p::Real, α::Real, roots::Vector{vectype}; autodiff = false)`
- `DeflationOperator(p::Real, dt, α::Real, roots::Vector{vectype}; autodiff = false)`
- `DeflationOperator(p::Real, α::Real, roots::Vector{vectype}, v::vectype; autodiff = false)`

The option `autodiff` triggers the use of automatic differentiation for the computation of the gradient of the scalar function `M`. This works only on `AbstractVector` for now.

## Custom distance

You are asked to pass a scalar product like `dot` to build a `DeflationOperator`. However, in some cases, you may want to pass a custom distance `dist(u, v)`. You can do this using

    `DeflationOperator(p, CustomDist(dist), α, roots)`

Note that passing `CustomDist(dist, true)` will trigger the use of automatic differentiation for the gradient of `M`.

## Linear solvers / jacobians

When used with newton, you have access to the following linear solvers

- custom solver `DeflatedProblemCustomLS()` which requires solving two linear systems `J⋅x = rhs`.
- For other linear solvers `<: AbstractLinearSolver`, a matrix free method is used for the deflated functional.
- if passed `Val(:autodiff)`, then `ForwardDiff.jl` is used to compute the jacobian Matrix of the deflated problem.
- if passed `Val(:fullIterative)`, then a full matrix free method is used for the deflated problem.
"""
struct DeflationOperator{Tp <: Real, Tdot, T <: Real, vectype} <: AbstractDeflationFactor
    "power `p`. You can use an `Int` for example."
    power::Tp

    "function, this function has to be bilinear and symmetric for the linear solver to work well."
    dot::Tdot

    "shift."
    α::T

    "roots."
    roots::Vector{vectype}

    "[internal] to reduce allocations during computation."
    tmp::vectype

    "[internal] to reduce allocations during computation."
    autodiff::Bool

    "[internal] for finite differences."
    δ::T
end

# constructors
DeflationOperator(p::Real, α::T, roots::Vector{vectype}; autodiff = false) where {T, vectype} = DeflationOperator(p, VI.inner, α, roots, _copy(roots[1]), autodiff, T(1e-8))
DeflationOperator(p::Real, dt, α::Real, roots::Vector{vectype}; autodiff = false) where vectype = DeflationOperator(p, dt, α, roots, _copy(roots[1]), autodiff, convert(eltype(roots[1]), 1e-8))
DeflationOperator(p::Real, α::T, roots::Vector{vectype}, v::vectype; autodiff = false) where {vectype, T <: Real} = DeflationOperator(p, VI.inner, α, roots, v, autodiff, T(1e-8))

# methods to deal with DeflationOperator
Base.eltype(df::DeflationOperator{Tp, Tdot, T, vectype}) where {Tp, Tdot, T, vectype} = T
Base.push!(df::DeflationOperator{Tp, Tdot, T, vectype}, v::vectype) where {Tp, Tdot, T, vectype} = push!(df.roots, v)
Base.pop!(df::DeflationOperator) = pop!(df.roots)
Base.getindex(df::DeflationOperator, inds...) = getindex(df.roots, inds...)
Base.length(df::DeflationOperator) = length(df.roots)
Base.isempty(df::DeflationOperator) = isempty(df.roots)
Base.deleteat!(df::DeflationOperator, id) = deleteat!(df.roots, id)
Base.empty!(df::DeflationOperator) = empty!(df.roots)
Base.firstindex(df::DeflationOperator) = 1
Base.lastindex(df::DeflationOperator) = length(df)
Base.copy(df::DeflationOperator) = DeflationOperator(df.power, df.dot, df.α, deepcopy(df.roots), copy(df.tmp), df.autodiff, df.δ)

function Base.show(io::IO, df::DeflationOperator; prefix = "")
    println(io, prefix * "┌─ Deflation operator with ", length(df.roots)," root(s)")
    println(io, prefix * "├─ eltype   = ", eltype(df))
    println(io, prefix * "├─ power    = ", df.power)
    println(io, prefix * "├─ α        = ", df.α)
    println(io, prefix * "├─ dist     = ", df.dot)
    println(io, prefix * "└─ autodiff = ", df.autodiff)
end

# Compute M(u)
# optimized version which does not allocate much.
function (df::DeflationOperator{𝒯p, Tdot, 𝒯})(::Val{:inplace}, u, tmp) where {𝒯p, Tdot, 𝒯}
    length(df.roots) == 0 && return one(𝒯)
    M(u) = one(𝒯) / df.dot(u, u)^df.power + df.α
    # compute u - df.roots[1]
    _copyto!(tmp, u); VI.add!(tmp, df.roots[1], 𝒯(-1))
    out = M(tmp)
    for ii in 2:length(df.roots)
        _copyto!(tmp, u); VI.add!(tmp, df.roots[ii], 𝒯(-1))
        out *= M(tmp)
    end
    return out
end

# Compute M(u), efficient and do not allocate
(df::DeflationOperator{Tp, Tdot, T, vectype})(u::vectype) where {Tp, Tdot, T, vectype} = df(Val(:inplace), u, df.tmp)
(df::DeflationOperator{Tp, Tdot, T, vectype})(u) where {Tp, Tdot, T, vectype} = df(Val(:inplace), u, similar(u))

# version when a custom distance is passed
function (df::DeflationOperator{Tp, Tdot, T, vectype})(::Val{:inplace}, u, tmp) where {Tp, Tdot <: CustomDist, T, vectype}
    length(df.roots) == 0 && return one(T)
    M(u, v) = one(T) / df.dot(u, v)^df.power + df.α
    out = M(u, df.roots[1])
    for ii in 2:length(df.roots)
        out *= M(u, df.roots[ii])
    end
    return out
end

# Compute jvp(M(u),du). We use tmp for storing intermediate values
function (df::DeflationOperator{Tp, Tdot, T, vectype})(::Val{:dMwithTmp}, tmp, u, du) where {Tp, Tdot, T, vectype}
    length(df) == 0 && return zero(T)
    if df.autodiff
        return ForwardDiff.derivative(t -> df(u .+ t .* du), 0)
    else
        _copyto!(tmp, u); VI.add!(tmp, du, df.δ)
        return (df(tmp) - df(u)) / df.δ 
    end
end
(df::DeflationOperator)(u, du) = df(Val(:dMwithTmp), similar(u), u, du)

"""
    pb = DeflatedProblem(prob, M::DeflationOperator, jactype)

Create a `DeflatedProblem`.

This creates a deflated functional (problem) ``M(u) \\cdot F(u) = 0`` where `M` is a `DeflationOperator` which encodes the penalization term. `prob` is an `AbstractBifurcationProblem` which encodes the functional. It is not meant not be used directly albeit by advanced users.

## Arguments
- `jactype` selects the jacobian for the newton solve. Can be `Val(:autodiff)`, `Val(:fullIterative)`, `Val(:Custom)`
"""
struct DeflatedProblem{Tprob <: AbstractBifurcationProblem, Tp, Tdot, T, vectype, Tjac} <: AbstractBifurcationProblem
    prob::Tprob
    M::DeflationOperator{Tp, Tdot, T, vectype}
    jactype::Tjac
end
@inline length(prob::DeflatedProblem) = length(prob.M)
@inline isinplace(prob::DeflatedProblem) = isinplace(prob.prob)
@inline _getvectortype(prob::DeflatedProblem) = _getvectortype(prob.prob)
@inline is_symmetric(::DeflatedProblem) = false

"""
Return the deflated function M(u) * F(u) where M(u) ∈ R
"""
function (dfp::DeflatedProblem{Tprob, Tp, Tdot, T, vectype})(u, par) where {Tprob, Tp, Tdot, T, vectype}
    out = residual(dfp.prob, u, par)
    VI.scale!(out, dfp.M(u))
    return out
end

"""
Return the jacobian of the deflated function M(u) * F(u) where M(u) ∈ R
"""
function (dfp::DeflatedProblem{Tprob, Tp, Tdot, T, vectype})(u::vectype, par, du) where {Tprob, Tp, Tdot, T, vectype}
    # dF(u)⋅du * M(u) + F(u) dM(u)⋅du
    # out = dF(u)⋅du * M(u)
    out = dF(dfp.prob, u, par, du)
    M = dfp.M(u)
    VI.scale!(out, M)
    # we add the remaining part
    if length(dfp) > 0
        F = residual(dfp.prob, u, par)
        # F(u) dM(u)⋅du, out .+= dfp.M(u, du) .* F
        VI.add!(out, F, dfp.M(u, du))
    end
    return out
end

residual(dfp::DeflatedProblem, x, p) = dfp(x, p)

function jacobian(dfp::DeflatedProblem{Tprob, Tp, Tdot, T, vectype, Val{:Custom}}, x, p) where {Tprob <: AbstractBifurcationProblem, Tp, Tdot, T, vectype}
    return (x, p, dfp)
end

jacobian(dfp::DeflatedProblem{Tprob, Tp, Tdot, T, vectype, Val{:fullIterative}}, x, p) where {Tprob <: AbstractBifurcationProblem, Tp, Tdot, T, vectype} = dx -> dfp(x, p, dx)

jacobian(dfp::DeflatedProblem{Tprob, Tp, Tdot, T, vectype, AutoDiff}, x, p) where {Tprob <: AbstractBifurcationProblem, Tp, Tdot, T, vectype} = ForwardDiff.jacobian(z -> dfp(z, p), x)

# getters
getu0(dfp::DeflatedProblem) = getu0(dfp.prob)
getparams(dfp::DeflatedProblem) = getparams(dfp.prob)
@inline getlens(dfp::DeflatedProblem) = getlens(dfp.prob)
getparam(dfp::DeflatedProblem) = getparam(dfp.prob)
setparam(dfp::DeflatedProblem, p0) = setparam(dfp.prob, p0)

function Base.show(io::IO, prob::DeflatedProblem; prefix = "")
    print(io, prefix * "┌─ " );    printstyled(io, "Deflated Problem", bold = true)
    print(" with uType ")
    printstyled(io, _getvectortype(prob), color=:cyan, bold = true)
    print(io, "\n" * prefix * "├─ Symmetric: ")
    printstyled(io, is_symmetric(prob), color=:cyan, bold = true)
    print(io, "\n" * prefix * "├─ jacobian: ")
    printstyled(io, prob.jactype, color=:cyan, bold = true)
    print(io, "\n" * prefix * "├─ Parameter ")
    printstyled(io, get_lens_symbol(getlens(prob)), color=:cyan, bold = true)
    println(io, "\n" * prefix * "└─ deflation operator:")
    show(io, prob.M; prefix = "    ")
end

###################################################################################################
# Implement the Jlinear solvers for the deflated problem
abstract type AbstractLinearSolverForDeflation <: AbstractLinearSolver end

# this is used to define a custom linear solver
"""
$(TYPEDEF)

Custom linear solver for deflated problem, very close to the Sherman-Morrison formula.
"""
@with_kw_noshow struct DeflatedProblemCustomLS{T} <: AbstractLinearSolverForDeflation
    solver::T = nothing
end

"""
Implement the custom linear solver for the deflated problem.
"""
function (dfl::DeflatedProblemCustomLS)(J, rhs)
    # the expression of the Functional is now
    # F(u) * Π_i(dot(u - root_i, u - root_i)^{-power} + shift) := F(u) * M(u)
    # the expression of the differential is
    # dF(u)⋅du * M(u) + F(u) dM(u)⋅du

    # the point at which to compute the Jacobian
    u = J[1]
    p = J[2]

    # deflated Problem composite type
    defPb = J[3]
    linsolve = dfl.solver

    Fu = residual(defPb.prob, u, p)
    Mu = defPb.M(u)
    Ju = jacobian(defPb.prob, u, p)

    if length(defPb.M) == 0
        h1, _, it1 = linsolve(Ju, rhs)
        return h1, true, (it1, 0)
    end

    # linear solve for the deflated problem. We note that Mu ∈ R
    # hence dM(u)⋅du is a scalar. We now solve the following linear problem
    # M(u) * dF(u)⋅h + F(u) dM(u)⋅h = rhs
    h1, h2, _, (it1, it2) = linsolve(Ju, rhs, Fu)

    # We look for the expression of dM(u)⋅h
    # the solution is then h = Mu * h1 - z h2 where z has to be determined
    # z1 = dM(h)⋅h1
    tmp = VI.zerovector(u)
    z1 = defPb.M(Val(:dMwithTmp), tmp, u, h1)

    # z2 = dM(h)⋅h2
    z2 = defPb.M(Val(:dMwithTmp), tmp, u, h2)

    z = z1 / (Mu + z2)

    # we extract the type of defPb
    𝒯 = eltype(defPb.M)

    # return (h1 - z * h2) / Mu, true, (it1, it2)
    _copyto!(tmp, h1)
    VI.add!(tmp, h2, -z)
    VI.scale!(tmp, 𝒯(1) / Mu)
    return tmp, true, (it1, it2)
end

"""
$(TYPEDEF)

Full iterative linear solver for deflated problem.
"""
struct DefProbFullIterativeLinearSolver{T} <: AbstractLinearSolverForDeflation
    solver::T
end
####################################################################################################
"""
$(TYPEDSIGNATURES)

This is the deflated version of the Krylov-Newton Solver for `F(x, p0) = 0`.

We refer to the regular [`solve`](@ref) for more information. It penalises the roots saved in `defOp.roots`. The other arguments are as for `solve`. See [`DeflationOperator`](@ref) for more information on `defOp`.

# Arguments
Compared to [`solve`](@ref), the only different arguments are
- `defOp::DeflationOperator` deflation operator
- `linsolver` linear solver used to invert the Jacobian of the deflated functional.
    - custom solver `DeflatedProblemCustomLS()` which requires solving two linear systems `J⋅x = rhs`.
    - For other linear solvers `<: AbstractLinearSolver`, a matrix free method is used for the deflated functional.
    - if passed `Val(:autodiff)`, then `ForwardDiff.jl` is used to compute the jacobian Matrix of the deflated problem
    - if passed `Val(:fullIterative)`, then a full matrix free method is used for the deflated problem.
"""
function solve(prob::AbstractBifurcationProblem,
                defOp::DeflationOperator{Tp, Tdot, T, vectype},
                options::NewtonPar{T, L, E},
                _linsolver::DeflatedProblemCustomLS = DeflatedProblemCustomLS();
                kwargs...) where {T, L, E, Tp, Tdot, vectype}

    # we create the new functional
    deflatedPb = DeflatedProblem(prob, defOp, Val(:Custom))

    # create the linear solver
    linsolver = @set _linsolver.solver = options.linsolver

    # change the linear solver
    opt_def = @set options.linsolver = linsolver

    return solve(deflatedPb, Newton(), opt_def; kwargs...)
end

function solve(prob::AbstractBifurcationProblem,
                defOp::DeflationOperator{Tp, Tdot, T, vectype},
                options::NewtonPar{T, L, E},
                ::Val{:autodiff}; kwargs...) where {Tp, T, Tdot, vectype, L, E}
    # we create the new functional
    deflatedPb = DeflatedProblem(prob, defOp, AutoDiff())
    return solve(deflatedPb, Newton(), options; kwargs...)
end

function solve(prob::AbstractBifurcationProblem,
                defOp::DeflationOperator{Tp, Tdot, T, vectype},
                options::NewtonPar{T, L, E},
                ::Val{:fullIterative}; kwargs...) where {Tp, T, Tdot, vectype, L, E}
    # we create the new functional
    deflatedPb = DeflatedProblem(prob, defOp, Val(:fullIterative))
    return solve(deflatedPb, Newton(), options; kwargs...)
end

"""
$(TYPEDEF)

This specific Newton-Krylov method first tries to converge to a solution `sol0` close the guess `x0`. It then attempts to converge from the guess `x1` while avoiding the previous converged solution close to `sol0`. This is very handy for branch switching. The method is based on a deflated Newton-Krylov solver.

# Arguments
Compared to [`newton`](@ref), the only different arguments are
- `defOp::DeflationOperator` deflation operator
- `linsolver` linear solver used to invert the Jacobian of the deflated functional.
    - custom solver `DeflatedProblemCustomLS()` which requires solving two linear systems `J⋅x = rhs`.
    - For other linear solvers `<: AbstractLinearSolver`, a matrix free method is used for the deflated functional.
    - if passed `Val(:autodiff)`, then `ForwardDiff.jl` is used to compute the jacobian Matrix of the deflated problem
    - if passed `Val(:fullIterative)`, then a full matrix free method is used for the deflated problem.
"""
function newton(prob::AbstractBifurcationProblem,
                x0::vectype,
                x1::vectype, p0,
                options::NewtonPar{T, L, E},
                defOp::DeflationOperator = DeflationOperator(2, one(eltype(x0)), Vector{vectype}(), _copy(x0); autodiff = true),
                linsolver = DeflatedProblemCustomLS();
                kwargs...) where {T, vectype, L, E}
    prob0 = re_make(prob, u0 = x0, params = p0)
    sol0 = solve(prob0, Newton(), options; kwargs...)
    @assert converged(sol0) "Newton did not converge to the trivial solution x0."
    push!(defOp, sol0.u)
    prob1 = re_make(prob0, u0 = x1)
    sol1 = solve(prob1, defOp, (@set options.max_iterations = 10options.max_iterations), linsolver; kwargs...)
    ~converged(sol1) && @error "Deflated Newton did not converge to the non-trivial solution ( i.e. on the bifurcated branch)."
    @debug "deflated Newton" x0 x1 sol0.u sol1.u
    # we test if the two solutions are different. We first get the norm
    normN = get(kwargs, :normN, norm)
    flag = normN(minus(sol0.u, sol0.u)) < options.tol
    @assert flag "Did not find a non trivial solution using deflated newton"
    return sol1, sol0, flag
end
