abstract type AbstractBifurcationFunction end
abstract type AbstractBifurcationProblem end
abstract type AbstractMABifurcationProblem{T} <: AbstractBifurcationProblem end
# this type is based on the type BifFunction, see below
# it provides all derivatives
abstract type AbstractAllJetBifProblem <: AbstractBifurcationProblem end
using SciMLBase: numargs

_getvectortype(::AbstractBifurcationProblem) = Nothing
isinplace(::Union{AbstractBifurcationProblem, Nothing}) = false

save_solution_default(x, p) = x

"""
Determine if the vector field is of the form `f!(out,z,p)`.
"""
function _isinplace(f)
    m = minimum(numargs(f))
    @assert 1 < m < 4 "You have too many/few arguments in your vector field F. It should be of the form `F(x,p)` or `F!(x,p)`."
    return m == 3
end

"""
$(TYPEDEF)

Structure to hold the vector field and its derivatives. It should rarely be called directly. Also, in essence, it is very close to `SciMLBase.ODEFunction`.

## Fields

$(TYPEDFIELDS)

## Methods
- `residual(pb::BifFunction, x, p)` calls `pb.F(x,p)`
- `jacobian(pb::BifFunction, x, p)` calls `pb.J(x, p)`
- `dF(pb::BifFunction, x, p, dx)` calls `pb.dF(x,p,dx)`
- etc
"""
struct BifFunction{Tf, Tdf, Tdfad, Tj, Tjad, Td2f, Td2fc, Td3f, Td3fc, Tsym, Tδ} <: AbstractBifurcationFunction
    "Vector field. Function of type out-of-place `result = f(x, p)` or inplace `f(result, x, p)`. For type stability, the types of `x` and `result` should match"
    F::Tf
    "Differential of `F` with respect to `x`, signature `dF(x,p,dx)`"
    dF::Tdf
    "Adjoint of the Differential of `F` with respect to `x`, signature `dFad(x,p,dx)`"
    dFad::Tdfad
    "Jacobian of `F` at `(x, p)`. It can assume three forms.
        1. Either `J` is a function and `J(x, p)` returns a `::AbstractMatrix`. In this case, the default arguments of `contparams::ContinuationPar` will make `continuation` work.
        2. Or `J` is a function and `J(x, p)` returns a function taking one argument `dx` and returning `dr` of the same type as `dx`. In our notation, `dr = J * dx`. In this case, the default parameters of `contparams::ContinuationPar` will not work and you have to use a Matrix Free linear solver, for example `GMRESIterativeSolvers`,
        3. Or `J` is a function and `J(x, p)` returns a variable `j` which can assume any type. Then, you must implement a linear solver `ls` as a composite type, subtype of `AbstractLinearSolver` which is called like `ls(j, rhs)` and which returns the solution of the jacobian linear system. See for example `examples/SH2d-fronts-cuda.jl`. This linear solver is passed to `NewtonPar(linsolver = ls)` which itself passed to `ContinuationPar`. Similarly, you have to implement an eigensolver `eig` as a composite type, subtype of `AbstractEigenSolver`."
    J::Tj
    "jacobian adjoint, it should be implemented in an efficient manner. For matrix-free methods, `transpose` is not readily available and the user must provide a dedicated method. In the case of sparse based jacobian, `Jᵗ` should not be passed as it is computed internally more efficiently, i.e. it avoids recomputing the jacobian as it would be if you pass `Jᵗ = (x, p) -> transpose(dF(x, p))`."
    Jᵗ::Tjad
    "Second Differential of `F` with respect to `x`, signature `d2F(x,p,dx1,dx2)`"
    d2F::Td2f
    "Third Differential of `F` with respect to `x`, signature `d3F(x,p,dx1,dx2,dx3)`"
    d3F::Td3f
    "[internal] Second Differential of `F` with respect to `x` which accept complex vectors dxi"
    d2Fc::Td2fc
    "[internal] Third Differential of `F` with respect to `x` which accept complex vectors dxi"
    d3Fc::Td3fc
    "Whether the jacobian is auto-adjoint."
    isSymmetric::Tsym
    "used internally to compute derivatives (with finite differences), for example for normal form computation and codim 2 continuation."
    δ::Tδ
    "optionally sets whether the function is inplace or not"
    inplace::Bool
end

# getters
residual(pb::BifFunction, x, p) = pb.F(x, p)
jacobian(pb::BifFunction, x, p) = pb.J(x, p)
jad(pb::BifFunction, x, p) = pb.Jᵗ(x, p)
dF(pb::BifFunction, x, p, dx) = pb.dF(x, p, dx)
dFad(pb::BifFunction, x, p, dx) = pb.dFad(x, p, dx)
d2F(pb::BifFunction, x, p, dx1, dx2) = pb.d2F(x, p, dx1, dx2)
d2Fc(pb::BifFunction, x, p, dx1, dx2) = pb.d2Fc(x, p, dx1, dx2)
d3F(pb::BifFunction, x, p, dx1, dx2, dx3) = pb.d3F(x, p, dx1, dx2, dx3)
d3Fc(pb::BifFunction, x, p, dx1, dx2, dx3) = pb.d3Fc(x, p, dx1, dx2, dx3)
is_symmetric(pb::BifFunction) = pb.isSymmetric
has_hessian(pb::BifFunction) = ~isnothing(pb.d2F)
has_adjoint(pb::BifFunction) = ~isnothing(pb.Jᵗ)
has_adjoint_MF(pb::BifFunction) = ~isnothing(pb.dFad)
isinplace(pb::BifFunction) = pb.inplace
getdelta(pb::BifFunction) = pb.δ

record_sol_default(x, p; kwargs...) = norm(x)
plot_default(x, p; kwargs...) = nothing              # for Plots.jl
plot_default(ax, x, p; kwargs...) = nothing, nothing # for Makie.jl

# create specific problems where pretty much is available
for (op, at) in (
                (:BifurcationProblem, AbstractBifurcationProblem),
                (:ODEBifProblem, AbstractBifurcationProblem),
                (:PDEBifProblem, AbstractBifurcationProblem),
                (:FoldMAProblem, AbstractMABifurcationProblem),
                (:HopfMAProblem, AbstractMABifurcationProblem),
                (:PDMAProblem, AbstractMABifurcationProblem),
                (:NSMAProblem, AbstractMABifurcationProblem),
                (:BTMAProblem, AbstractMABifurcationProblem),
                (:WrapPOTrap, AbstractBifurcationProblem),
                (:WrapPOSh, AbstractBifurcationProblem),
                (:WrapPOColl, AbstractBifurcationProblem),
                (:WrapTW, AbstractBifurcationProblem),
           )
    if op in (:BifurcationProblem, :ODEBifProblem, :PDEBifProblem)
        @eval begin
            """
            $(TYPEDEF)

            Structure to hold the bifurcation problem.

            ## Fields

            $(TYPEDFIELDS)

            ## Methods

            - `re_make(pb; kwargs...)` modify a bifurcation problem
            - `getu0(pb)` calls `pb.u0`
            - `getparams(pb)` calls `pb.params`
            - `getlens(pb)` calls `pb.lens`
            - `getparam(pb)` calls `get(pb.params, pb.lens)`
            - `setparam(pb, p0)` calls `_set_param(pb.params, pb.lens, p0)`
            - `record_from_solution(pb)` calls `pb.recordFromSolution`
            - `plot_solution(pb)` calls `pb.plotSolution`
            - `is_symmetric(pb)` calls `is_symmetric(pb.prob)`

            ## Constructors
            - `BifurcationProblem(F, u0, params, lens)` all derivatives are computed using ForwardDiff.
            - `BifurcationProblem(F, u0, params, lens; J, Jᵗ, d2F, d3F, kwargs...)` and `kwargs` are the fields above. You can pass your own jacobian with `J` (see [`BifFunction`](@ref) for description of the jacobian function) and jacobian adjoint with `Jᵗ`. For example, this can be used to provide finite differences based jacobian using `BifurcationKit.finiteDifferences`.

            """
            struct $op{Tvf, Tu, Tp, Tl <: Lens, Tplot, Trec, Tgets} <: AbstractAllJetBifProblem
                "Vector field, typically a [`BifFunction`](@ref)"
                VF::Tvf
                "Initial guess"
                u0::Tu
                "parameters"
                params::Tp
                "It can be of two types. Either it is an `Int` which specifies which component of `params` is used for continuation. Alternatively, one can use a `Setfield.Lens` which specifies which parameter axis among `params` is used for continuation. For example, if `params = (α = 1.0, β = 1)`, we can perform continuation w.r.t. `α` by using `lens = (@lens _.α)`. Also, if you have an array of parameters `params = [ 1.0, 2.0]` and want to perform continuation w.r.t. the first variable, you can use `lens = (@lens _[1])`. For more information, we refer to `SetField.jl`."
                lens::Tl
                "user function to plot solutions during continuation. Signature: `plot_solution(x, p; kwargs...)` for Plot.jl and `plot_solution(ax, x, p; kwargs...)` for the Makie package(s)."
                plotSolution::Tplot
                "`record_from_solution = (x, p) -> norm(x)` function used record a few indicators about the solution. It could be `norm` or `(x, p) -> x[1]`. This is also useful when saving several huge vectors is not possible for memory reasons (for example on GPU). This function can return pretty much everything but you should keep it small. For example, you can do `(x, p) -> (x1 = x[1], x2 = x[2], nrm = norm(x))` or simply `(x, p) -> (sum(x), 1)`. This will be stored in `contres.branch` where `contres::ContResult` is the continuation curve of the bifurcation problem. Finally, the first component is used for plotting in the continuation curve."
                recordFromSolution::Trec
                "function to save the full solution on the branch. Some problem are mutable (like periodic orbit functional with adaptive mesh) and this function allows to save the state of the problem along with the solution itself. Signature `save_solution(x, p)`"
                save_solution::Tgets
            end

            _getvectortype(::$op{Tvf, Tu, Tp, Tl, Tplot, Trec}) where {Tvf, Tu, Tp, Tl, Tplot, Trec} = Tu
            plot_solution(prob::$op) = prob.plotSolution
            record_from_solution(prob::$op) = prob.recordFromSolution
            save_solution(prob::$op, x, p) = prob.save_solution(x, p)
        end
    elseif op in (:FoldMAProblem, :HopfMAProblem, :PDMAProblem, :NSMAProblem, :BTMAProblem)
        @eval begin
            """
            $(TYPEDEF)

            Problem wrap of a functional. It is not meant to be used directly albeit perhaps by advanced users.

            $(TYPEDFIELDS)
            """
            struct $op{Tprob, Tjac, Tu0, Tp, Tl <: Union{Nothing, Lens}, Tplot, Trecord} <: $at{Tprob}
                prob::Tprob
                jacobian::Tjac
                u0::Tu0
                params::Tp
                lens::Tl
                plotSolution::Tplot
                recordFromSolution::Trecord
            end

            _getvectortype(::$op{Tprob, Tjac, Tu0, Tp, Tl, Tplot, Trecord}) where {Tprob, Tjac, Tu0, Tp, Tl, Tplot, Trecord} = Tu0
            isinplace(pb::$op) = isinplace(pb.prob)
            # dummy constructor
            $op(prob, lens = getlens(prob)) = $op(prob, nothing, nothing, nothing, lens, nothing, nothing)
        end
    else
        @eval begin
            """
            $(TYPEDEF)

            Problem wrap of a functional. It is not meant to be used directly albeit perhaps by advanced users.

            $(TYPEDFIELDS)
            """
            struct $op{Tprob, Tjac, Tu0, Tp, Tl <: Union{Nothing, Lens}, Tplot, Trecord} <: $at
                prob::Tprob
                jacobian::Tjac
                u0::Tu0
                params::Tp
                lens::Tl
                plotSolution::Tplot
                recordFromSolution::Trecord
            end

            _getvectortype(::$op{Tprob, Tjac, Tu0, Tp, Tl, Tplot, Trecord}) where {Tprob, Tjac, Tu0, Tp, Tl, Tplot, Trecord} = Tu0
            isinplace(pb::$op) = isinplace(pb.prob)
            # dummy constructor
            $op(prob, lens = getlens(prob)) = $op(prob, nothing, nothing, nothing, lens, nothing, nothing)
        end
    end

    # forward getters
    if op in (:BifurcationProblem, :ODEBifProblem, :PDEBifProblem)
        @eval begin
            function $op(_F, u0, parms, lens = (@lens _);
                         jvp = nothing,
                         vjp = nothing,
                         J = nothing,
                         J! = nothing,
                         Jᵗ = nothing,
                         d2F = nothing,
                         d3F = nothing,
                         issymmetric::Bool = false,
                         record_from_solution = record_sol_default,
                         plot_solution = plot_default,
                         delta = convert(eltype(u0), 1e-8),
                         save_solution = save_solution_default,
                         inplace = false)
                @assert lens isa Int || Lens <: Lens
                new_lens = lens isa Int ? (@lens _[lens]) : lens
                if inplace
                    F = _F
                else
                    iip = _isinplace(_F)
                    F = iip ? (x, p) -> _F(similar(x), x, p) : _F
                end
                J = isnothing(J) ? (x, p) -> ForwardDiff.jacobian(z -> F(z, p), x) : J
                jvp = isnothing(jvp) ?
                      (x, p, dx) -> ForwardDiff.derivative(t -> F(x .+ t .* dx, p), zero(eltype(dx))) : dF
                d1Fad(x,p,dx1) = ForwardDiff.derivative(t -> F(x .+ t .* dx1, p), zero(eltype(dx1)))
                if isnothing(d2F)
                    d2F = (x, p, dx1, dx2) -> ForwardDiff.derivative(t -> d1Fad(x .+ t .* dx2, p, dx1), zero(eltype(dx1)))
                    d2Fc = (x, p, dx1, dx2) -> BilinearMap((_dx1, _dx2) -> d2F(x, p, _dx1, _dx2))(dx1, dx2)
                else
                    d2Fc = d2F
                end
                if isnothing(d3F)
                    d3F = (x, p, dx1, dx2, dx3) -> ForwardDiff.derivative(t -> d2F(x .+ t .* dx3, p, dx1, dx2), zero(eltype(dx1)))
                    d3Fc = (x, p, dx1, dx2, dx3) -> TrilinearMap((_dx1, _dx2, _dx3) -> d3F(x, p, _dx1, _dx2, _dx3))(dx1, dx2, dx3)
                else
                    d3Fc = d3F
                end

                d3F = isnothing(d3F) ? (x, p, dx1, dx2, dx3) -> ForwardDiff.derivative(t -> d2F(x .+ t .* dx3, p, dx1, dx2), 0.0) : d3F
                VF = BifFunction(F, jvp, vjp, J, Jᵗ, d2F, d3F, d2Fc, d3Fc, issymmetric, delta, inplace)
                return $op(VF, u0, parms, new_lens, plot_solution, record_from_solution, save_solution)
            end
        end
    end
end

# getters for AbstractBifurcationProblem
getu0(pb::AbstractBifurcationProblem) = pb.u0
"""
Return the parameters of the bifurcation problem.
"""
function getparams(pb::AbstractBifurcationProblem) 
    pb.params
end
@inline getlens(pb::AbstractBifurcationProblem) = pb.lens
getparam(pb::AbstractBifurcationProblem) = get(pb.params, pb.lens)
setparam(pb::AbstractBifurcationProblem, p0) = _set_param(pb.params, pb.lens, p0)
record_from_solution(pb::AbstractBifurcationProblem) = pb.recordFromSolution
plot_solution(pb::AbstractBifurcationProblem) = pb.plotSolution

# specific to AbstractAllJetBifProblem
isinplace(pb::AbstractAllJetBifProblem) = isinplace(pb.VF)
is_symmetric(pb::AbstractAllJetBifProblem) = is_symmetric(pb.VF)
residual(pb::AbstractAllJetBifProblem, x, p) = residual(pb.VF, x, p)
jacobian(pb::AbstractAllJetBifProblem, x, p) = jacobian(pb.VF, x, p)
jad(pb::AbstractAllJetBifProblem, x, p) = jad(pb.VF, x, p)
dF(pb::AbstractAllJetBifProblem, x, p, dx) = dF(pb.VF, x, p, dx)
d2F(pb::AbstractAllJetBifProblem, x, p, dx1, dx2) = d2F(pb.VF, x, p, dx1, dx2)
d2Fc(pb::AbstractAllJetBifProblem, x, p, dx1, dx2) = d2Fc(pb.VF, x, p, dx1, dx2)
d3F(pb::AbstractAllJetBifProblem, x, p, dx1, dx2, dx3) = d3F(pb.VF, x, p, dx1, dx2, dx3)
d3Fc(pb::AbstractAllJetBifProblem, x, p, dx1, dx2, dx3) = d3Fc(pb.VF, x, p, dx1, dx2, dx3)
has_hessian(pb::AbstractAllJetBifProblem) = has_hessian(pb.VF)
has_adjoint(pb::AbstractAllJetBifProblem) = has_adjoint(pb.VF)
has_adjoint_MF(pb::AbstractAllJetBifProblem) = has_adjoint_MF(pb.VF)
getdelta(pb::AbstractAllJetBifProblem) = getdelta(pb.VF)

for op in (:WrapPOTrap, :WrapPOSh, :WrapPOColl, :WrapTW)
    @eval begin
        function Base.show(io::IO, pb::$op)
            printstyled(io, "Problem wrap of\n", bold = true)
            show(io, pb.prob)
        end
    end
end

for (op, txt) in ((:NSMAProblem, "NS"), (:PDMAProblem, "PD"))
    @eval begin
        function Base.show(io::IO, pb::$op)
            printstyled(io, "Problem wrap for curve of " * $txt * " of periodic orbits.\n", bold = true)
            println("Based on the formulation:")
            show(io, pb.prob.prob_vf)
        end
    end
end

function Base.show(io::IO, prob::AbstractBifurcationProblem; prefix = "")
    print(io, prefix * "┌─ Bifurcation Problem with uType ")
    printstyled(io, _getvectortype(prob), color = :cyan, bold = true)
    print(io, "\n" * prefix * "├─ Inplace:  ")
    printstyled(io, isinplace(prob), color = :cyan, bold = true)
    print(io, "\n" * prefix * "├─ Symmetric: ")
    printstyled(io, is_symmetric(prob), color = :cyan, bold = true)
    print(io, "\n" * prefix * "└─ Parameter: ")
    printstyled(io, get_lens_symbol(getlens(prob)), color = :cyan, bold = true)
end

function apply_jacobian(pb::AbstractBifurcationProblem, x, par, dx, transpose_jac = false)
    if is_symmetric(pb)
        # return apply(pb.J(x, par), dx)
        return dF(pb, x, par, dx)
    else
        if transpose_jac == false
            # return apply(pb.J(x, par), dx)
            return dF(pb, x, par, dx)
        else
            if has_adjoint(pb)
                return jad(pb, x, par, dx)
            else
                return apply(transpose(jacobian(pb, x, par)), dx)
            end
        end
    end
end

"""
$(SIGNATURES)

This function changes the fields of a problem `::AbstractBifurcationProblem`. For example, you can change the initial condition by doing

```
re_make(prob; u0 = new_u0)
```
"""
function re_make(prob::AbstractBifurcationProblem;
                u0 = prob.u0,
                params = prob.params,
                lens::Lens = prob.lens,
                record_from_solution = prob.recordFromSolution,
                plot_solution = prob.plotSolution,
                J = missing,
                Jᵗ = missing,
                d2F = missing,
                d3F = missing)
    prob2 = setproperties(prob; u0 = u0, params = params, lens = lens, recordFromSolution = record_from_solution, plotSolution = plot_solution)
    if ~ismissing(J)
        @set! prob2.VF.J = J
    end
    if ~ismissing(Jᵗ)
        @set! prob2.VF.Jᵗ = Jᵗ
    end
    if ~ismissing(d2F)
        @set! prob2.VF.d2F = d2F
    end

    if ~ismissing(d3F)
        @set! prob2.VF.d3F = d3F
    end

    return prob2
end
