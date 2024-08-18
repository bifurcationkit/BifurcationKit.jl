# # the ideas are based on https://github.com/tkf/Bifurcations.jl/blob/master/src/diffeq.jl
# # I would just grab DiffEqDiffTools.uJacobianWrapper
# # https://github.com/JuliaDiffEq/DiffEqBase.jl/blob/1c86896dfcc0c383ef501014c56a3f8dfc8a63bb/src/nlsolve/newton.jl
# using DiffEqBase: AbstractODEProblem
# using DiffEqOperators
# using Accessors
# using OrdinaryDiffEq
#
# const DEP{iip} = AbstractODEProblem{uType, tType, iip} where {uType, tType}
#
# struct DiffEqWrapper{P, L}
#     de_prob::P
#     param_axis::L
# end
#
# mutable struct JacWrap{Tj, Tp, Tx}
#     J::Tj    # jacobian
#     p::Tp    # parameter
#     #TODO: remove this
#     x::Tx    # vecteur where the jacobian is evaluated
# end
#
# # callable function to compute the Jacobian J(x) = dF(x)
# function (Jac::JacWrap)(x, p)
#     if Jac.J isa DiffEqBase.AbstractDiffEqLinearOperator
#         Jac.x .= x
#         update_coefficients(Jac.J, x, p, zero(eltype(x)))
#         # this is a Hack because update_coefficients does not do it
#         Jac.J.u .= x
#         norm(Jac.J.u - x) ≈ 0 ? printstyled(color=:green,"--> Jac Updated\n") : printstyled(color=:red,"--> Jac Not updated!!, $(norm(Jac.J.u - x))\n")
#     else
#         @error "Not a Matrix Free Operator!"
#     end
#     # needs to return the jacobian so we can call linsolve on it
#     Jac.J
# end
#
# # callable function to compute the Jacobian J(x, p)
# function (Jac::JacWrap{Tj, Tp, Tx})(x, p::Tp) where {Tj, Tp, Tx}
#     Jac.p = p
#     Jac(x)
# end
#
# ####################################################################################################
# # we define a linearsolver to use our interface based on the one in DiffEqBase in linear_nonlinear.jl
#
# struct DiffEqWrapLS <: AbstractLinearSolver
#     linsolve
# end
#
# function (l::DiffEqWrapLS)(J, rhs)
#     out = similar(rhs)
#
#     # tol is required for DefaultLinSolve() to work
#     l.linsolve(out, J, rhs; tol = 1e-4, verbose = true)
#
#     return out, true, 1
# end
# ####################################################################################################
# # Nonlinear solvers
#
# function ContinuationProblem(
#         deprob::DEP{iip}, param_axis::Lens;
#         kwargs0...) where iip
#     de_prob = deepcopy(deprob)
#     x0 = copy(de_prob.u0)
#     p0 = de_prob.p
#     @assert !(typeof(x0) <: Number) "We need array like structures for the state space, consider using [u0]"
#     pbwrap = DiffEqWrapper(de_prob, param_axis)
#
#     # we extract the vector field
#     if de_prob isa DEP{false}
#         fnewton = x -> de_prob.f(x, p0, de_prob.tspan[1])
#         f = (x, p) -> de_prob.f(x, set(p0, param_axis, p), de_prob.tspan[1])
#     else
#         fnewton = x -> (out = similar(x);de_prob.f(out, x, p0, de_prob.tspan[1]);out)
#         f =  (x, p) -> (out = similar(x);de_prob.f(out, x, set(p0, param_axis, p), de_prob.tspan[1]);out)
#     end
#
#     # we extract the jacobian (operator)
#     if de_prob.f.jac === nothing
#         @warn "No jacobian is provided by ODEProblem"
#         dfnewton = nothing
#         df = nothing
#     else
#         @warn "jacobian is provided by ODEProblem"
#         J, W = OrdinaryDiffEq.build_J_W(ImplicitEuler(), x0, x0, p0, 1., 0., de_prob.f, eltype(x0), Val{}(iip))
#         @show typeof(J), typeof(W)
#         dfnewton = J
#         df = nothing
#     end
#     return (fnewton = fnewton, dfnewton = dfnewton, f = f, df = df, x0 = x0)
# end
#
#
# function newton(de_prob::DEP, options:: NewtonPar{T}; normN = norm, linsolver = DiffEqBase.DefaultLinSolve()) where {T}
#     @unpack fnewton, dfnewton, x0 = ContinuationProblem(de_prob, @lens _.p)
#     if dfnewton == nothing
#         @warn "newton without Jacobian"
#         return newton(fnewton, x0, options)
#     else
#         @warn "newton with Jacobian"
#         options = @set options.linsolver = DiffEqWrapLS(linsolver)
#         J = JacWrap(dfnewton, de_prob.p, copy(x0))
#         return newton(fnewton, J, x0, options)
#     end
# end
#
# function continuation(de_prob::DEP, param_axis::Lens, contParams::ContinuationPar{T, S, E};kwargs...) where {T, S, E}
#     @unpack f, df, x0 = ContinuationProblem(de_prob, param_axis)
#     p = Accessors.get(de_prob.p, param_axis)
#     continuation(f, x0, p, contParams; kwargs...)
# end
