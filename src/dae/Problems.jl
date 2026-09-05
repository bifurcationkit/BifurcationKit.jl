"""
Trait to encode the kind of mass matrix of a `DAEMassBifProblem`. See [`ConstantMass`](@ref).
"""
abstract type AbstractDAEMassType end

"""
$(TYPEDEF)

Marker for a **constant** mass matrix `M` (independent of the state `x` and of the parameters `p`). It is used as the default kind / first type parameter of [`DAEMassBifProblem`](@ref).
"""
struct ConstantMass <: AbstractDAEMassType end

struct IdentityOperator <: AbstractDAEMassType end

dot_with_mass(ζ★, ::IdentityOperator, ζ) = VI.inner(ζ★, ζ)
dot_with_mass(ζ★, Mass::AbstractMatrix, ζ) = LA.dot(ζ★, Mass, ζ)
dot_with_mass(ζ★, Mass, ζ) = VI.inner(ζ★, apply(Mass, ζ))
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
$(TYPEDEF)

Structure holding a mass matrix (or mass operator) together with its optional adjoint and user provided derivatives. It is stored in the field `M` of a [`DAEMassBifProblem`](@ref).

# Fields
$(TYPEDFIELDS)

# Constructors

- `MassFunction(M; Mᵗ = nothing, R01 = nothing, ∇x = nothing, applyM = nothing, dMv = nothing)` where `M` is a matrix/operator or a function `M(x, p)`.

# Methods

The derivatives are used by the minimally augmented formulations to account for a mass matrix depending on the state or the parameters:

- `R01(x, p, v, w)` returns the **scalar** `∂_p ⟨w, M(x, p) v⟩` where `p` is differentiated along the continuation lens. If `nothing`, it is approximated by central finite differences.
- `∇x(x, p, v, w)` returns the **vector** `∇_x ⟨w, M(x, p) v⟩`. If `nothing`, it is approximated by `ForwardDiff.gradient`.
"""
struct MassFunction{TM, TMt, TR01, TGx}
    "Mass matrix/operator, or a function `M(x, p)`."
    M::TM
    "Adjoint of the mass matrix (matrix or function `Mᵗ(x, p)`), or `nothing`."
    Mᵗ::TMt
    "Derivative, with respect to the continuation parameter, of `⟨w, M(x, p) v⟩`: `(x, p, v, w) -> scalar`, or `nothing`."
    R01::TR01
    "Gradient, with respect to the state `x`, of `⟨w, M(x, p) v⟩`: `(x, p, v, w) -> vector`, or `nothing`."
    ∇x::TGx
end
MassFunction(M; Mᵗ = nothing, R01 = nothing, ∇x = nothing) = MassFunction(M, Mᵗ, R01, ∇x)

_to_massfunction(mf::MassFunction; kw...) = mf
_to_massfunction(M; Mᵗ = nothing, R01 = nothing, ∇x = nothing) = MassFunction(M; Mᵗ, R01, ∇x)
getmassmatrix(mf::MassFunction, x, p) = _getmassmatrix(mf.M, x, p)
_getmassmatrix(M::AbstractMatrix, x, p) = M
_getmassmatrix(::Union{LA.UniformScaling, IdentityOperator}, x, p) = LA.Diagonal(ones(length(x)))
_getmassmatrix(M, x, p) = M(x, p)
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
- `getmassmatrix(pb, x, p)` returns the mass matrix `M(x, p)`
- `R01_mass_matrix(pb, x, p, v, w)` returns `∂_p ⟨w, M(x, p) v⟩`, see [`MassFunction`](@ref)
- `∇_x_mass_matrix(pb, x, p, v, w)` returns `∇_x ⟨w, M(x, p) v⟩`, see [`MassFunction`](@ref)
- `has_massmatrix_adjoint(pb)` returns whether a dedicated mass adjoint `pb.M.Mᵗ` was provided
- `getmassmatrix_adjoint(pb, x, p)` returns the user provided mass adjoint `pb.M.Mᵗ` (matrix or function)
- `record_from_solution(pb)`, `save_solution(pb, u, pars)`, `update!(pb, iter, state)` are forwarded to `pb.prob_vf`
- `re_make(pb; M = …, Mᵗ = …, R01 = …, ∇x = …, kwargs…)` rebuilds the problem, possibly with another mass matrix (and/or its derivatives)
- the jet methods of the wrapped problem (`R01`, `dF`, `d2F`, `d3F`, …) are forwarded as well

# Constructors

- `DAEMassBifProblem(prob, M; Mᵗ = nothing, R01 = nothing, ∇x = nothing, type = ConstantMass)` wraps the bifurcation problem `prob` with the mass matrix `M`, which can be a matrix or a function `M(x, p)`. An optional adjoint `Mᵗ` (a matrix or a function `Mᵗ(x, p)`) and optional derivatives `R01`, `∇x` can be provided, see [`MassFunction`](@ref). A [`MassFunction`](@ref) can also be passed directly as `M`.
- `DAEMassBifProblem{ConstantMass}(prob, M)` explicitly sets the kind of mass matrix through the type parameter.
- a `UniformScaling` mass matrix (`I`, `α * I`) is also accepted: the identity case is stored with the marker `IdentityOperator` so that no mass matrix solve is required.

# Remark

The eigenvalues along the continuation are the generalized eigenvalues of `(J(x, p), M(x, p))` where `J` is the jacobian of `prob_vf`, so that stability detection accounts for the mass matrix. During `continuation`, the eigen solver is automatically wrapped into `EigenDAE` to compute these generalized eigen-elements.

!!! warning "Mass matrix"
    A mass matrix which depends on the state `x` (as opposed to a constant one, or one depending only on the parameters) is only partially supported: it is evaluated once at the current solution during continuation, but the higher order derivatives of the vector field `F` do not account for derivatives of `M`.
"""
struct DAEMassBifProblem{Tkind <: AbstractDAEMassType, Tprob, TM} <: AbstractDAEBifProblem
    "vector field, must be `AbstractBifurcationProblem`."
    prob_vf::Tprob
    "Mass matrix/operator wrapped in a `MassFunction`."
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
jacobian!(dae::DAEMassBifProblem, J, x, p) = jacobian!(dae.prob_vf, J, x, p)
jacobian_adjoint(dae::DAEMassBifProblem, x, p) = jacobian_adjoint(dae.prob_vf, x, p)

record_from_solution(dae::DAEMassBifProblem) = record_from_solution(dae.prob_vf)
plot_solution(dae::DAEMassBifProblem) = plot_solution(dae.prob_vf)
save_solution(dae::DAEMassBifProblem, u, pars) = save_solution(dae.prob_vf, u, pars)
update!(dae::DAEMassBifProblem, iter, state) = update!(dae.prob_vf, iter, state)

R01(dae::DAEMassBifProblem, u, pars) = R01(dae.prob_vf, u, pars)
R02(dae::DAEMassBifProblem, u, pars) = R02(dae.prob_vf, u, pars)
R11(dae::DAEMassBifProblem, u, pars, du) = R11(dae.prob_vf, u, pars, du)
dF(dae::DAEMassBifProblem,  u, pars, du) = dF(dae.prob_vf, u, pars, du)
d2F(dae::DAEMassBifProblem,  u, pars, du1, du2) = d2F(dae.prob_vf, u, pars, du1, du2)
d2Fc(dae::DAEMassBifProblem, u, pars, du1, du2) = d2Fc(dae.prob_vf, u, pars, du1, du2)
d3F(dae::DAEMassBifProblem,  u, pars, du1, du2, du3) = d3F(dae.prob_vf, u, pars, du1, du2, du3)
d3Fc(dae::DAEMassBifProblem, u, pars, du1, du2, du3) = d3Fc(dae.prob_vf, u, pars, du1, du2, du3)
has_hessian(dae::DAEMassBifProblem) = has_hessian(dae.prob_vf)

# constant (matrix like) mass matrices are returned as-is, state dependent ones are evaluated at (x, p)
getmassmatrix(dae::DAEMassBifProblem, x, p) = getmassmatrix(dae.M, x, p)

# the identity mass matrix is self-adjoint
Base.adjoint(::IdentityOperator) = IdentityOperator()

"""
$(TYPEDSIGNATURES)

Whether the `MassFunction` provides a dedicated adjoint of the mass matrix.
"""
has_massmatrix_adjoint(::MassFunction{TM, Nothing}) where {TM} = false
has_massmatrix_adjoint(::MassFunction{TM, TMt}) where {TM, TMt} = true
has_massmatrix_adjoint(dae::DAEMassBifProblem) = has_massmatrix_adjoint(dae.M)

"""
$(TYPEDSIGNATURES)

Return the user provided adjoint of the mass matrix `M(x, p)`, that is `mf.Mᵗ`. It
can be a matrix or a function `Mᵗ(x, p)`.
"""
getmassmatrix_adjoint(mf::MassFunction, x, p) = _getmassmatrix_adjoint(mf.Mᵗ, x, p)
_getmassmatrix_adjoint(Mᵗ::AbstractMatrix, x, p) = Mᵗ
_getmassmatrix_adjoint(Mᵗ, x, p) = Mᵗ(x, p)
getmassmatrix_adjoint(dae::DAEMassBifProblem, x, p) = getmassmatrix_adjoint(dae.M, x, p)
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
is_mass_matrix_constant(::DAEMassBifProblem{ConstantMass}) = true
is_mass_matrix_constant(::DAEMassBifProblem{IdentityOperator}) = true
is_mass_matrix_constant(::DAEMassBifProblem) = false

has_trivial_mass_mastrix(::DAEMassBifProblem{IdentityOperator}) = true
has_trivial_mass_mastrix(::DAEMassBifProblem) = false

# generic constructors, the kind of mass matrix `type` defaults to `ConstantMass`
function DAEMassBifProblem(prob, M; Mᵗ = nothing, R01 = FiniteDifferences(), ∇xM = AutoDiff(), type = ConstantMass)
    @assert type <: AbstractDAEMassType "The provided `type` for the mass matrix must be a subtype of `AbstractDAEMassType`, e.g. `ConstantMass`."
    return DAEMassBifProblem{type}(prob, M; Mᵗ, R01, ∇xM)
end
function DAEMassBifProblem{Tkind}(prob, M; Mᵗ = nothing, R01 = FiniteDifferences(), ∇xM = AutoDiff()) where {Tkind <: AbstractDAEMassType}
    mf = _to_massfunction(M; Mᵗ, R01, ∇x = ∇xM)
    return DAEMassBifProblem{Tkind, typeof(prob), typeof(mf)}(prob, mf)
end

function DAEMassBifProblem{Tkind}(prob, ::LA.UniformScaling{Bool}; Mᵗ = nothing, R01 = FiniteDifferences(), ∇xM = AutoDiff()) where {Tkind <: AbstractDAEMassType}
    mf = MassFunction(IdentityOperator(); Mᵗ, R01, ∇x = ∇xM)
    return DAEMassBifProblem{ConstantMass, typeof(prob), typeof(mf)}(prob, mf)
end

function re_make(dae::DAEMassBifProblem{Tkind};
                M = nothing,
                Mᵗ = nothing,
                R01 = FiniteDifferences(),
                ∇xM = AutoDiff(),
                kw...
                ) where {Tkind}
    new_prob = re_make(dae.prob_vf; kw...)
    if isnothing(M) && isnothing(Mᵗ) && isnothing(R01) && isnothing(∇xM)
        # preserve the kind of mass matrix (ConstructionBase would reset it)
        return DAEMassBifProblem{Tkind, typeof(new_prob), typeof(dae.M)}(new_prob, dae.M)
    end
    base = isnothing(M) ? dae.M : _to_massfunction(M)
    newMt = isnothing(Mᵗ) ? base.Mᵗ : Mᵗ
    newR01 = R01 isa FiniteDifferences ? base.R01 : R01
    newGx = ∇xM isa AutoDiff ? base.∇x : ∇xM
    mf = MassFunction(base.M; Mᵗ = newMt, R01 = newR01, ∇x = newGx)
    return DAEMassBifProblem{Tkind, typeof(new_prob), typeof(mf)}(new_prob, mf)
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
$(TYPEDSIGNATURES)

Return `∂_p ⟨w, M(x, p) v⟩` where `p` is differentiated along the continuation lens
`getlens(dae)`. This is the derivative of the mass part of the bordered scalar, used
by the minimally augmented Hopf formulation. If the user provided `mf.R01`, it is
used, otherwise a central finite difference is performed.
"""
R01_mass_matrix(dae::DAEMassBifProblem, x, p, v, w) = dae.M.R01(x, p, v, w)

function R01_mass_matrix(dae::DAEMassBifProblem{Tkind, Tprob, MassFunction{TM, TMt, FiniteDifferences, TGx}}, x, p, v, w) where {Tkind <: AbstractDAEMassType, Tprob, TM, TMt, TGx}
    lens = getlens(dae)
    p0 = _get(p, lens)
    ϵ = getdelta(dae)
    M₊ = getmassmatrix(dae, x, set(p, lens, p0 + ϵ))
    M₋ = getmassmatrix(dae, x, set(p, lens, p0 - ϵ))
    return (dot_with_mass(w, M₊, v) - dot_with_mass(w, M₋, v)) / (2ϵ)
end

function R01_mass_matrix(dae::DAEMassBifProblem{Tkind, Tprob, MassFunction{TM, TMt, AutoDiff, TGx}}, x, p, v, w) where {Tkind <: AbstractDAEMassType, Tprob, TM, TMt, TGx}
    lens = getlens(dae)
    p0 = _get(p, lens)
    return ForwardDiff.derivative(z -> dot_with_mass(w, getmassmatrix(dae, x, set(p, lens, z)), v), p0)
end

"""
$(TYPEDSIGNATURES)

Return `∇_x ⟨w, M(x, p) v⟩`. This is the gradient of the mass part of the bordered
scalar, used by the minimally augmented Hopf formulation. If the user provided
`mf.∇x`, it is used, otherwise it is computed with `ForwardDiff.gradient`.
"""
∇_x_mass_matrix(dae::DAEMassBifProblem, x, p, v, w) = dae.M.∇x(x, p, v, w)

function ∇_x_mass_matrix(dae::DAEMassBifProblem{Tkind, Tprob, MassFunction{TM, TMt, TR01, AutoDiff}}, x, p, v, w) where {Tkind <: AbstractDAEMassType, Tprob, TM, TMt, TR01}
    # real / imaginary split to stay within ForwardDiff's real arithmetic
    vr = real(v); vi = imag(v)
    wr = real(w); wi = imag(w)
    quad(a, b, z) = dot_with_mass(b, getmassmatrix(dae, z, p), a)
    gre = ForwardDiff.gradient(z -> quad(vr, wr, z) + quad(vi, wi, z), x)
    gim = ForwardDiff.gradient(z -> quad(vi, wr, z) - quad(vr, wi, z), x)
    return complex.(gre, gim)
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_massmatrix_repr(M::AbstractMatrix) = string(typeof(M), " of size ", size(M))
_massmatrix_repr(M::LA.UniformScaling) = M.λ == 1 ? "UniformScaling (identity)" : string("UniformScaling with factor ", M.λ)
_massmatrix_repr(M::IdentityOperator) = "IdentityOperator (identity)"
_massmatrix_repr(M) = string("function ", M, " (evaluated as M(x, p))")

function Base.show(io::IO, dae::DAEMassBifProblem{Tkind}; prefix = "") where {Tkind}
    print(io, prefix * "┌─ DAEMassBifProblem\n├─ kind of mass matrix = ")
    printstyled(io, nameof(Tkind); color = :cyan, bold = true)
    print(io, "\n")
    print(io, prefix * "├─ mass matrix : ")
    printstyled(io, _massmatrix_repr(dae.M.M); color = :cyan, bold = true)
    print(io, "\n" * prefix * "├─ mass matrix adjoint : ")
    printstyled(io, isnothing(dae.M.Mᵗ) ? "auto (adjoint of M)" : _massmatrix_repr(dae.M.Mᵗ); color = :cyan, bold = true)
    print(io, "\n" * prefix * "└─ Vector field :\n")
    show(io, dae.prob_vf; prefix = prefix * "   ")
end
