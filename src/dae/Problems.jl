"""
Trait to encode the kind of mass matrix of a `DAEMassBifProblem`. See [`ConstantMass`](@ref).
"""
abstract type AbstractDAEMassType end

"""
$(TYPEDEF)

Marker for a **constant** mass matrix `M` (independent of the state `x` and of the parameters `p`). It is used as the default kind / first type parameter of [`DAEMassBifProblem`](@ref).
"""
struct ConstantMass <: AbstractDAEMassType end

struct TrivialMassMatrix <: AbstractDAEMassType end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
$(TYPEDEF)

Wrapper around an [`AbstractBifurcationProblem`](@ref) to encode a Differential Algebraic Equation (DAE) of the form `M(x, p) * dx/dt = F(x, p)`, where `M` is a (possibly state dependent) mass matrix.

# Type parameters

- `Tkind <: AbstractDAEMassType`: kind of mass matrix, `ConstantMass` by default. It is set through the keyword `type` of the constructor or by using `DAEMassBifProblem{ConstantMass}(...)`.

# Fields

$(TYPEDFIELDS)

# Methods

- `getparams(pb)` calls `getparams(pb.prob_vf)`
- `getlens(pb)` calls `getlens(pb.prob_vf)`
- `getparam(pb)` calls `getparam(pb.prob_vf)`
- `setparam(pb, p0)` calls `setparam(pb.prob_vf, p0)`
- `getu0(pb)` calls `getu0(pb.prob_vf)`
- `residual(pb, x, p)` calls `residual(pb.prob_vf, x, p)`
- `jacobian(pb, x, p)` calls `jacobian(pb.prob_vf, x, p)`
- `getmassmatrix(pb, x, p)` returns the mass matrix `M(x, p)` (or `pb.M` if it is a constant matrix)
- `record_from_solution(pb)`, `save_solution(pb, u, pars)`, `update!(pb, iter, state)` are forwarded to `pb.prob_vf`

# Constructors

- `DAEMassBifProblem(prob, M; type = ConstantMass)` wraps the bifurcation problem `prob` with the mass matrix `M`, which can be a matrix or a function `M(x, p)`.
- `DAEMassBifProblem{ConstantMass}(prob, M)` explicitely sets the kind of mass matrix through the type parameter.

# Remark

The eigenvalues along the continuation are the generalized eigenvalues of `(J(x, p), M(x, p))` where `J` is the jacobian of `prob_vf`, so that stability detection accounts for the mass matrix.

!!! warning "Mass matrix"
    A mass matrix which depends on the state `x` (as opposed to a constant one, or one depending only on the parameters) is only partially supported: it is evaluated once at the current solution during continuation, but the higher order derivatives of the vector field `F` do not account for derivatives of `M`.
"""
struct DAEMassBifProblem{Tkind <: AbstractDAEMassType, Tprob, TM} <: AbstractDAEBifProblem
    "vector field, must be `AbstractBifurcationProblem`."
    prob_vf::Tprob
    "Mass matrix/operator."
    M::TM
end
@inline getparams(dae::DAEMassBifProblem) = getparams(dae.prob_vf)
@inline getlens(dae::DAEMassBifProblem) = getlens(dae.prob_vf)
@inline getdelta(dae::DAEMassBifProblem) = getdelta(dae.prob_vf)
@inline _getvectortype(dae::DAEMassBifProblem) = _getvectortype(dae.prob_vf)
@inline getu0(dae::DAEMassBifProblem) = getu0(dae.prob_vf)
@inline has_adjoint(dae::DAEMassBifProblem) = has_adjoint(dae.prob_vf)
@inline isinplace(dae::DAEMassBifProblem) = isinplace(dae.prob_vf)
@inline is_symmetric(dae::DAEMassBifProblem) = is_symmetric(dae.prob_vf)

getparam(dae::DAEMassBifProblem) = getparam(dae.prob_vf)
residual(dae::DAEMassBifProblem, x, p) = residual(dae.prob_vf, x, p)
residual!(dae::DAEMassBifProblem, o, x, p) = residual!(dae.prob_vf, o, x, p)
jacobian(dae::DAEMassBifProblem, x, p) = jacobian(dae.prob_vf, x, p)
# jacobian!(dae::DAEMassBifProblem, J, x, p) = jacobian!(dae.prob_vf, J, x, p)
# jacobian_adjoint(dae::DAEMassBifProblem, x, p) = jacobian_adjoint(dae.prob_vf, x, p)
# constant (matrix like) mass matrices are returned as-is, state dependent ones are evaluated at (x, p)

record_from_solution(dae::DAEMassBifProblem) = record_from_solution(dae.prob_vf)
plot_solution(dae::DAEMassBifProblem) = plot_solution(dae.prob_vf)
save_solution(dae::DAEMassBifProblem, u, pars) = save_solution(dae.prob_vf, u, pars)
update!(dae::DAEMassBifProblem, iter, state) = update!(dae.prob_vf, iter, state)

R01(dae::DAEMassBifProblem, u, pars) = R01(dae.prob_vf, u, pars)
R02(dae::DAEMassBifProblem, u, pars) = R02(dae.prob_vf, u, pars)
R11(dae::DAEMassBifProblem, u, pars, du) = R11(dae.prob_vf, u, pars, du)
dF(dae::DAEMassBifProblem, u, pars, du) = dF(dae.prob_vf, u, pars, du)
d2F(dae::DAEMassBifProblem, u, pars, du1, du2) = d2F(dae.prob_vf, u, pars, du1, du2)
d2Fc(dae::DAEMassBifProblem, u, pars, du1, du2) = d2Fc(dae.prob_vf, u, pars, du1, du2)
d3F(dae::DAEMassBifProblem, u, pars, du1, du2, du3) = d3F(dae.prob_vf, u, pars, du1, du2, du3)
d3Fc(dae::DAEMassBifProblem, u, pars, du1, du2, du3) = d3Fc(dae.prob_vf, u, pars, du1, du2, du3)
has_hessian(dae::DAEMassBifProblem) = has_hessian(dae.prob_vf)
# has_adjoint_MF(dae::DAEMassBifProblem) = has_adjoint_MF(dae.prob_vf)

getmassmatrix(dae::DAEMassBifProblem{ConstantMass, Tprob, TM}, x, p) where {Tprob, TM <: AbstractMatrix} = dae.M
getmassmatrix(::DAEMassBifProblem{ConstantMass, Tprob, TM}, x, p) where {Tprob, TM <: Union{LA.UniformScaling, TrivialMassMatrix}} = LA.Diagonal(ones(length(x)))
getmassmatrix(dae::DAEMassBifProblem, x, p) = dae.M(x, p)
is_mass_matrix_constant(::DAEMassBifProblem{ConstantMass}) = true
is_mass_matrix_constant(::DAEMassBifProblem) = false

# generic constructors, the kind of mass matrix `type` defaults to `ConstantMass`
function DAEMassBifProblem(prob, M; type = ConstantMass)
    @assert type <: AbstractDAEMassType "The provided `type` for the mass matrix must be a subtype of `AbstractDAEMassType`, e.g. `ConstantMass`."
    return DAEMassBifProblem{type}(prob, M)
end
DAEMassBifProblem{Tkind}(prob, M) where {Tkind <: AbstractDAEMassType} =
    DAEMassBifProblem{Tkind, typeof(prob), typeof(M)}(prob, M)

DAEMassBifProblem{Tkind}(prob, ::LA.UniformScaling{Bool}) where {Tkind <: AbstractDAEMassType} = DAEMassBifProblem{ConstantMass, typeof(prob), TrivialMassMatrix}(prob, TrivialMassMatrix())

function re_make(dae::DAEMassBifProblem{Tkind};
                M = nothing,
                kw...
                ) where {Tkind}
    new_prob = re_make(dae.prob_vf; kw...)
    new_dae = if isnothing(M)
        @set dae.prob_vf = new_prob
    else
        DAEMassBifProblem{Tkind}(new_prob, M)
    end
    return new_dae
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# display
_massmatrix_repr(M::AbstractMatrix) = string(typeof(M), " of size ", size(M))
_massmatrix_repr(M::LA.UniformScaling) = M.λ == 1 ? "UniformScaling (identity)" : string("UniformScaling with factor ", M.λ)
_massmatrix_repr(M) = string("function ", M, " (evaluated as M(x, p))")

function Base.show(io::IO, dae::DAEMassBifProblem{Tkind}; prefix = "") where {Tkind}
    print(io, prefix * "┌─ DAEMassBifProblem\n├─ kind of mass matrix = ")
    printstyled(io, nameof(Tkind); color = :cyan, bold = true)
    print(io, "\n")
    print(io, prefix * "├─ mass matrix : ")
    printstyled(io, _massmatrix_repr(dae.M); color = :cyan, bold = true)
    print(io, "\n" * prefix * "└─ Vector field :\n")
    show(io, dae.prob_vf; prefix = prefix * "   ")
end
