# """
#     pb = BorderedProblem(;F, dxF, dpF, g, dg, dpg)
#
# This composite type encodes a bordered problem, one by which we add a scalar constraint `g(x, p) = 0` to an equation `F(x, p) = 0`. This composite type thus allows to define the functional `G((x, p)) = [F(x, p) g(x, p)]` and solve `G = 0`.
#
# You can then evaluate the functional using or `pb(z)` where `z = BorderedArray(x, p)` or `z = vcat(x, p)`, i.e. the last component of the vector is the Lagrange Multiplier.
#
# # Arguments
# The arguments correspond to the functions `F, g` and their derivatives.
#
# # Simplified constructor
#
# You can create such functional as `pb = BorderedProblem(F, g)`.
#
# !!! warning "Multidimensional constraint (Really Experimental)"
#     It is in fact possible, using this composite type, to define a bordered problem with constraint of dimension `npar > 1`. One has to pass the dimension to `pb = BorderedProblem(F, g, npar)` and possibly the derivatives as well. The second argument of `F,g` is `npar` dimensional (for now an `AbstractVector`). Finally, the only possible linear (bordered) solver in this case is `::MatrixBLS`.
#
# """
# @with_kw struct BorderedProblem{Tf, Tdf, TdpF, Tg, Tdg, Tdpg, Tlens}
#     F::Tf        # functional
#     dxF::Tdf = (x0, p0) -> finiteDifferences(x -> F(x, p0), x0)    # partial Derivative w.r.t. first variable
#     dpF::TdpF = (x0, p0) -> (F(x0, p0 + 1e-8) .- F(x0, p0)) .* 1e8   # partial Derivative w.r.t. first variable
#     g::Tg        # scalar constraint
#     dg::Tdg = (x0, p0) -> finiteDifferences(x -> g(x, p0), x0)       # partial Derivative w.r.t. first variable
#     dpg::Tdpg = (x0, p0) -> (g(x0, p0 + 1e-8) - g(x0, p0)) .* 1e8    # partial Derivative w.r.t. first variable
#     npar::Int = 1                                                    # number of parameters, equals length(g(x,p))
#     lens::Tlens = @lens _[1]
# end
#
# BorderedProblem(F, g, lens::Lens, npar = 1) = BorderedProblem(F = F, g = g, lens = lens, npar = 1)
#
# getParameterDim(pb::BorderedProblem) = pb.npar
#
# function extractParameter(pb::BorderedProblem, x::AbstractVector)
#     if getParameterDim(pb) == 1
#         return x[end]
#     else
#         return @view x[end-pb.npar+1:end]
#     end
# end
#
# extractVector(pb::BorderedProblem, x::AbstractVector) = @view x[1:end-getParameterDim(pb)]
# setVector!(pb::BorderedProblem, out::AbstractVector, x::AbstractVector) = out[1:end-getParameterDim(pb)] .= x
#
# # setParameter!(pb::BorderedProblem, out::BorderedArray, p::Number) = out[end] = p
# setParameter!(pb::BorderedProblem, out::AbstractVector, p) = out[end-getParameterDim(pb)+1:end] .= p
#
# # extractParameter(pb::BorderedProblem, x::BorderedArray) = x.p
# # extractVector(pb::BorderedProblem, x::BorderedArray) = x.u
# # setVector!(pb::BorderedProblem, out::BorderedArray, x) = copyto!(out.u, x)
# # setParameter!(pb::BorderedProblem, out::BorderedArray, p::Number) = out.p = p
# # setParameter!(pb::BorderedProblem, out::BorderedArray, p) = copyto!(out.p, p)
#
# function (pb::BorderedProblem)(xe, par)
#     out = similar(xe)
#     # extract variables
#     p = extractParameter(pb, xe)
#     x = extractVector(pb, xe)
#     # compute the residual
#     setVector!(pb, out, pb.F(x, set(par, pb.lens, p)))
#     setParameter!(pb, out, pb.g(x, set(par, pb.lens, p)))
#     out
# end
#
# function (pb::BorderedProblem)(xe, par, dxe)
#     out = similar(dxe)
#     # extract variables
#     p = extractParameter(pb, xe)
#     x = extractVector(pb, xe)
#
#     # extract the jacobians
#     J = pb.dxF(x, par)
#
#     @error "WIP"
#     out
# end
#
# # Structure to hold the jacobian of the bordered problem
# mutable struct JacobianBorderedProblem{Tpb, Tj, Tdpf, Tdg, Tdpg}
#     pb::Tpb
#     J::Tj
#     dpF::Tdpf
#     dg::Tdg
#     dpg::Tdpg
# end
#
# # simplified constructor
# JacobianBorderedProblem(pb, x, p) = JacobianBorderedProblem(pb, pb.dxF(x, p), pb.dpF(x, p), pb.dg(x, p), pb.dpg(x, p))
#
# JacobianBorderedProblem(pb, x) = JacobianBorderedProblem(pb, extractVector(pb, x), extractParameter(pb, x))
#
#
# function (Jpb::JacobianBorderedProblem)(x, p)
#     # computation of the jacobian of the Bordered problem
#     pb = Jpb.pb
#     if Jpb.J isa AbstractArray
#         copyto!(Jpb.J,      pb.dxF(x, p))
#     else
#         Jpb.J = pb.dxF(x, p)
#     end
#
#     if Jpb.dpF isa AbstractArray
#         copyto!(Jpb.dpF, pb.dpF(x, p))
#     else
#         Jpb.dpF = pb.dpF(x, p)
#     end
#
#     if Jpb.dg isa AbstractArray
#         copyto!(Jpb.dg,  pb.dg(x, p))
#     else
#         Jpb.dg = pb.dg(x, p)
#     end
#     if getParameterDim(pb) == 1
#         Jpb.dpg = pb.dpg(x, p)
#     else
#         copyto!(Jpb.dpg, pb.dpg(x, p))
#     end
#     return Jpb
# end
#
# (Jbp::JacobianBorderedProblem)(x) = Jbp(extractVector(Jbp.pb, x), extractParameter(Jbp.pb, x))
#
# @with_kw struct LinearSolverBorderedProblem{T, Ts, L <: AbstractBorderedLinearSolver} <: AbstractLinearSolver
#     ls::L
#     # these are used to alter the second linear equation in the bordered system
#     xiu::T = 1.0
#     xip::T = 1.0
#     shift::Ts = nothing
# end
#
# # simplified constructor
# LinearSolverBorderedProblem(ls) = LinearSolverBorderedProblem(ls = ls)
#
# function (lsbdp::LinearSolverBorderedProblem)(J::JacobianBorderedProblem, x)
#     # call the bordered linear solver
#     ou, op, flag, it = lsbdp.ls(J.J, J.dpF, J.dg, J.dpg, extractVector(J.pb, x), extractParameter(J.pb, x), lsbdp.xiu, lsbdp.xip; shift = lsbdp.shift)
#     if x isa BorderedArray
#         return BorderedArray(ou, op), flag, it
#     else
#         return vcat(ou, op), flag, it
#     end
# end
#
# ####################################################################################################
# # newton functions
# """
#     newtonBordered(pb::BorderedProblem, z0, options::NewtonPar{T, L, S}; kwargs...)
#
# This function solves the equation associated with the functional `pb` with initial guess
# """
# function newtonBordered(pb::Tpb, z0, par, options::NewtonPar{T, L, S}; kwargs...) where {T, L <: AbstractBorderedLinearSolver, S, Tpb <: BorderedProblem}
#     @show typeof(z0)
#     Jac   = JacobianBorderedProblem(pb, z0)
#     lsbpb = LinearSolverBorderedProblem(options.linsolver)
#     options2 = @set options.linsolver = lsbpb
#     return newton(pb, Jac, z0, par, options2; kwargs...)
# end
#
# """
# This is the newton solver used to solve `F(x, p) = 0` together
# with the scalar condition `n(x, p) = (x - x0) * xp + (p - p0) * lp - n0`
# """
# function _newtonPALC(F, Jh,
#                     z0::BorderedArray{vectype, T},
#                     tau0::BorderedArray{vectype, T},
#                     z_pred::BorderedArray{vectype, T},
#                     options::ContinuationPar{T},
#                     dottheta::DotTheta;
#                     linearbdalgo = BorderingBLS(),
#                     normN = norm,
#                     callback = cbDefault, kwargs...) where {T, vectype}
#     # Extract parameters
#     @error "WIP - will replace newtonPALC once performances improve"
#     newtonOpts = options.newtonOptions
#     @unpack tol, maxIter, verbose, alpha, almin, linesearch = newtonOpts
#     @unpack theta, ds, finDiffEps = options
#
#     N = (x, p) -> arcLengthEq(dottheta, minus(x, z0.u), p - z0.p, tau0.u, tau0.p, theta, ds)
#     normAC = (resf, resn) -> max(normN(resf), abs(resn))
#
#     pb = BorderedProblem(F = F, dxF = Jh, g = N, dg = (x, p) -> tau0.u, dpg = (x, p) -> tau0.p)
#     Jac   = JacobianBorderedProblem(pb, z0)
#     lsbpb = LinearSolverBorderedProblem(ls = linearbdalgo, xiu = theta / length(z0.u), xip = one(T) - theta)
#     options2 = @set newtonOpts.linsolver = lsbpb
#     return newton(pb, Jac, z0, options2; kwargs...)
# end
# ####################################################################################################
# # continuation function
# function continuationBordered(pb, z0, p0::Real, contParams::ContinuationPar, linear_algo::AbstractBorderedLinearSolver; kwargs...) where {T, L <: AbstractBorderedLinearSolver, S}
#     Jac   = p -> JacobianBorderedProblem(pb(p), z0)
#     lsbpb = LinearSolverBorderedProblem(contParams.newtonOptions.linsolver)
#     contParams2 = @set contParams.newtonOptions.linsolver = lsbpb
#     return continuation((z, p) -> pb(p)(z),
#                         (z, p) -> Jac(p)(z),
#                         z0, p0, contParams2; kwargs...)
# end
#
# """
# continuationBordered(prob, z0, p0::Real, contParams::ContinuationPar; kwargs...)
#
# This is the continuation routine for finding the curve of solutions of a family of Bordered problems `p->prob(p)`.
#
# # Arguments
# - `p -> prob(p)` is a family such that `prob(p)::BorderedProblem` encodes the functional G
# - `z0` a guess for the constrained problem.
# - `p0` initial parameter, must be a real number
# - `contParams` same as for the regular `continuation` method
# """
# function continuationBordered(prob, z0, p0::Real, contParams::ContinuationPar; linear_algo = BorderingBLS(), kwargs...) where {T, L <: AbstractBorderedLinearSolver, S}
#     linear_algo = @set linear_algo.solver = contParams.newtonOptions.linsolver
#     return continuationBordered(prob, z0, p0, contParams, linear_algo; kwargs...)
# end
