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

const _type_jet = [Symbol("T",i,j) for i=0:3, j=1:7 if i+i<7] |> vec
const _field_jet = [(Symbol('R',i,j),i,j) for i=0:3, j=1:7 if i+i<7] |> vec 

@eval begin
    """
    $(TYPEDEF)

    Structure to hold the jet of a vector field. It saves the different functions `Rᵢⱼ` which correspond to the following (i+j) linear form 

    Rᵢⱼ(x,p)(dx₁, ⋅⋅⋅, dxᵢ, dp₁, ⋅⋅⋅, dpⱼ)

    More precisely

    Rᵢⱼ(x,p) = 1/i!j! dⁱₓ dʲₚ F(x, p)

    ## Note

    For now, we ask the user to pass an out-of-place formulation of the functions.

    ## Fields

    $(TYPEDFIELDS)
    """
    @with_kw_noshow struct Jet{$(_type_jet...)}
        $(map(i -> :( $(_field_jet[i][1])::$(_type_jet[i]) = nothing ), 1:length(_type_jet))...)
    end
end

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
- `R21(pb::BifFunction, x, p, dx1, dx2, dp1)` calls `pb.jet.R21(x, p, dx1, dx2, dp1)`. Same for the other Jets.
- etc
"""
struct BifFunction{Tf, TFinp, Tdf, Tdfad, Tj, Tjad, TJinp, Td2f, Td2fc, Td3f, Td3fc, Tsym, Tδ, Tjet} <: AbstractBifurcationFunction
    "Vector field. Function of type out-of-place `result = f(x, p)` or inplace `f(result, x, p)`. For type stability, the types of `x` and `result` should match"
    F::Tf
    "Same as F but inplace with signature F!(result, x, p)"
    F!::TFinp
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
    "Inplace jacobian"
    J!::TJinp
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
    "optionally sets whether the function is inplace or not. You can use `in_bisection(state)` to inquire whether the current state is in bisection mode."
    inplace::Bool
    "jet of the vector field"
    jet::Tjet
end

# getters
residual(pb::BifFunction, x, p) = pb.F(x, p)
residual!(pb::BifFunction, o, x, p) = (pb.F!(o, x, p);o)
jacobian(pb::BifFunction, x, p) = pb.J(x, p)
jacobian!(pb::BifFunction, J, x, p) = pb.J!(J, x, p)
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

# getters for the jet
for Rij in _field_jet
    @eval begin
        $(Rij[1])(pb::Jet, args...; kwargs...) = pb.$(Rij[1])(args...; kwargs...)
        @inline $(Rij[1])(pb::BifFunction, args...; kwargs...) = $(Rij[1])(pb.jet, args...; kwargs...)
        @inline $(Rij[1])(pb::AbstractAllJetBifProblem, args...; kwargs...) = $(Rij[1])(pb.VF, args...; kwargs...)
    end
end

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
            - `setparam(pb, p0)` calls `set(pb.params, pb.lens, p0)`
            - `record_from_solution(pb)` calls `pb.recordFromSolution`
            - `plot_solution(pb)` calls `pb.plotSolution`
            - `is_symmetric(pb)` calls `is_symmetric(pb.prob)`

            ## Constructors
            - `BifurcationProblem(F, u0, params, lens)` all derivatives are computed using ForwardDiff.
            - `BifurcationProblem(F, u0, params, lens; J, Jᵗ, d2F, d3F, kwargs...)` and `kwargs` are the fields above. You can pass your own jacobian with `J` (see [`BifFunction`](@ref) for description of the jacobian function) and jacobian adjoint with `Jᵗ`. For example, this can be used to provide finite differences based jacobian using `BifurcationKit.finite_differences`. You can also pass
                - `record_from_solution` see above
                - `plot_solution` see above
                - `issymmetric[=false]` whether the jacobian is symmetric, this remove the need to provide an adjoint
                - `d2F` second Differential of `F` with respect to `x`, signature `d2F(x,p,dx1,dx2)`
                - `d3F` third Differential of `F` with respect to `x`, signature `d3F(x,p,dx1,dx2,dx3)`
                - `save_solution` specify a way to solve solution which will be written in `br.sol`. This can be useful in very particular situations and we recommend using `record_from_solution` instead. For example, it is used internally to record the mesh in the collocation method because this mesh can be modified.

            """
            struct $op{Tvf, Tu, Tp, Tl <: AllOpticTypes, Tplot, Trec, Tgets} <: AbstractAllJetBifProblem
                "Vector field, typically a [`BifFunction`](@ref)"
                VF::Tvf
                "Initial guess"
                u0::Tu
                "parameters"
                params::Tp
                "Typically a `Accessors.PropertyLens`. It specifies which parameter axis among `params` is used for continuation. For example, if `par = (α = 1.0, β = 1)`, we can perform continuation w.r.t. `α` by using `lens = (@optic _.α)`. If you have an array `par = [ 1.0, 2.0]` and want to perform continuation w.r.t. the first variable, you can use `lens = (@optic _[1])`. For more information, we refer to `Accessors.jl`."
                lens::Tl
                "user function to plot solutions during continuation. Signature: `plot_solution(x, p; kwargs...)` for Plot.jl and `plot_solution(ax, x, p; kwargs...)` for the Makie package(s)."
                plotSolution::Tplot
                "`record_from_solution = (x, p; k...) -> norm(x)` function used record a few indicators about the solution. It could be `norm` or `(x, p) -> x[1]`. This is also useful when saving several huge vectors is not possible for memory reasons (for example on GPU). This function can return pretty much everything but you should keep it small. For example, you can do `(x, p; k...) -> (x1 = x[1], x2 = x[2], nrm = norm(x))` or simply `(x, p; k...) -> (sum(x), 1)`. This will be stored in `contres.branch` where `contres::ContResult` is the continuation curve of the bifurcation problem. Finally, the first component is used for plotting in the continuation curve."
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
            struct $op{Tprob, Tjac, Tu0, Tp, Tl <: Union{Nothing, AllOpticTypes}, Tplot, Trecord} <: $at{Tprob}
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
            struct $op{Tprob, Tjac, Tu0, Tp, Tl <: Union{Nothing, AllOpticTypes}, Tplot, Trecord} <: $at
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
            residual!(pb::$op, o, x, p) = residual!(pb.prob, o, x, p)
        end
    end

    # forward getters
    if op in (:BifurcationProblem, :ODEBifProblem, :PDEBifProblem)
        @eval begin
            """
            ($SIGNATURES)

            Constructor for a bifurcation problem.
            
            ## Optional argument
            """
            function $op(_F, u0, parms, lens = (@optic _);
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
                         inplace = false,
                         kwargs_jet...)
                @assert lens isa Int || lens isa AllOpticTypes
                new_lens = lens isa Int ? (@optic _[lens]) : lens
                if _get(parms, new_lens) isa Int
                    @warn "You passed the parameter value $(_get(parms, new_lens)) for the optic `$new_lens` which is an integer. This may error. Please use a float."
                end
                if inplace || _isinplace(_F)
                    F = (x, p) -> _F(similar(x), x, p)
                    Finp = _F
                else
                    F = _F
                    Finp = (o, x, p) -> copyto!(o, _F(x, p))
                end

                backend = DI.AutoForwardDiff()
                prep = DI.prepare_jacobian(F, backend, u0, DI.Constant(parms))
                J = isnothing(J) ? (x, p) -> DI.jacobian(F, prep, backend, x, DI.Constant(p)) : J
                J! = isnothing(J!) ? (out, x, p) -> out .= J(x, p) : J!

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
                d3F = isnothing(d3F) ? (x, p, dx1, dx2, dx3) -> ForwardDiff.derivative(t -> d2F(x .+ t .* dx3, p, dx1, dx2), zero(eltype(dx1))) : d3F

                VF = BifFunction(F, Finp, jvp, vjp, J, Jᵗ, J!, d2F, d3F, d2Fc, d3Fc, issymmetric, delta, inplace, Jet(;kwargs_jet...))
                return $op(VF, u0, parms, lens, plot_solution, record_from_solution, save_solution)
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
getparam(pb::AbstractBifurcationProblem) = _get(pb.params, pb.lens)
setparam(pb::AbstractBifurcationProblem, p0) = set(pb.params, pb.lens, p0)
record_from_solution(pb::AbstractBifurcationProblem) = pb.recordFromSolution
plot_solution(pb::AbstractBifurcationProblem) = pb.plotSolution

# specific to AbstractAllJetBifProblem
isinplace(pb::AbstractAllJetBifProblem) = isinplace(pb.VF)
is_symmetric(pb::AbstractAllJetBifProblem) = is_symmetric(pb.VF)
residual(pb::AbstractAllJetBifProblem, x, p) = residual(pb.VF, x, p)
residual!(pb::AbstractAllJetBifProblem, o, x, p) = residual!(pb.VF, o, x, p)
jacobian(pb::AbstractAllJetBifProblem, x, p) = jacobian(pb.VF, x, p)
jacobian!(pb::AbstractAllJetBifProblem, J, x, p) = jacobian!(pb.VF, J, x, p)
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
                return apply(jad(pb, x, par), dx)
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
                lens::AllOpticTypes = prob.lens,
                record_from_solution = prob.recordFromSolution,
                plot_solution = prob.plotSolution,
                J = missing,
                Jᵗ = missing,
                d2F = missing,
                d3F = missing)
    prob2 = setproperties(prob; u0 = u0, params = params, lens = lens, recordFromSolution = record_from_solution, plotSolution = plot_solution)
    if ~ismissing(J)
        @reset prob2.VF.J = J
    end
    if ~ismissing(Jᵗ)
        @reset prob2.VF.Jᵗ = Jᵗ
    end
    if ~ismissing(d2F)
        @reset prob2.VF.d2F = d2F
    end

    if ~ismissing(d3F)
        @reset prob2.VF.d3F = d3F
    end

    return prob2
end
