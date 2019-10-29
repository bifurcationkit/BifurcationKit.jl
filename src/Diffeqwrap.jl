# the ideas are based on https://github.com/tkf/Bifurcations.jl/blob/master/src/diffeq.jl
# I would just grab DiffEqDiffTools.uJacobianWrapper
# https://github.com/JuliaDiffEq/DiffEqBase.jl/blob/1c86896dfcc0c383ef501014c56a3f8dfc8a63bb/src/nlsolve/newton.jl
using DiffEqBase: AbstractODEProblem
using DiffEqOperators
using Setfield
using OrdinaryDiffEq

const DEP{iip} = AbstractODEProblem{uType, tType, iip} where {uType, tType}

struct DiffEqWrapper{P, L}
	de_prob::P
	param_axis::L
end

mutable struct JacWrap{Tj, Tp}
	J::Tj	#jacobian
	p::Tp	#parameter
end

# callable function to compute the Jacobian J(x)
function (Jac::JacWrap{Tj, Tp})(x) where {Tj, Tp}
	if Jac.J isa DiffEqBase.AbstractDiffEqLinearOperator
		update_coefficients(Jac.J, x, Jac.p, zero(eltype(x)))
	end
end

# callable function to compute the Jacobian J(x, p)
function (Jac::JacWrap{Tj, Tp})(x, p::Tp) where {Tj, Tp}
	Jac.p = p
	Jac(x)
end

####################################################################################################
# we define a linearsolver to use our interface based on the one in DiffEqBase in linear_nonlinear.jl

struct DiffEqWrapLS <: AbstractLinearSolver
	linsolve
	tol
end

function (l::DiffEqWrapLS)(J, rhs)
	out = similar(rhs)
	l.linsolve(out, J, rhs; tol = l.tol )

	res = similar(rhs)
	mul!(res, J, out)
	println("--> CV? ", norm(res - rhs, Inf))
	return out, true, 1
end
####################################################################################################
# Nonlinear solvers

function ContinuationProblem(
		deprob::DEP{iip}, param_axis::Lens;
		kwargs0...) where iip
	de_prob = deepcopy(deprob)
	x0 = de_prob.u0
	p0 = de_prob.p
	@assert !(typeof(x0) <: Number) "We need array like structures for the state space, consider using [u0]"
	pbwrap = DiffEqWrapper(de_prob, param_axis)

	# we extract the vector field
	if de_prob isa DEP{false}
		@warn "Out of place problem"
		fnewton = x -> de_prob.f(x, p0, de_prob.tspan[1])
		f = (x, p) -> de_prob.f(x, set(p0, param_axis, p), de_prob.tspan[1])
	else
		@warn "We don't consider inplace problem very efficiently (for now)"
		fnewton = x -> (out = similar(x);de_prob.f(out, x, p0, de_prob.tspan[1]);out)
		f = (x, p) -> (out = similar(x);de_prob.f(out, x, set(p0, param_axis, p), de_prob.tspan[1]);out)
	end

	# we extract the jacobian (operator)
	if de_prob.f.jac === nothing
		@warn "No jacobian is provided by ODEProblem"
		dfnewton = nothing
		df = nothing
	else
		@warn "jacobian is provided by ODEProblem"
		J, W = OrdinaryDiffEq.build_J_W(ImplicitEuler(), x0, x0, p0, 1., 1., de_prob.f, eltype(x0), Val{}(iip))
		@show J, W
		dfnewton = J
		df = nothing
	end
	return (fnewton = fnewton, dfnewton = dfnewton, f = f, df = df, x0 = x0)
end


function newton(de_prob::DEP, options:: NewtonPar{T}; normN = norm, linsolver = DiffEqBase.DefaultLinSolve()) where {T}
	@unpack fnewton, dfnewton, x0 = ContinuationProblem(de_prob, @lens _.p)
	if dfnewton == nothing
		@warn "newton sans J"
		return newton(fnewton, x0, options)
	else
		@warn "newton avec J"
		options = @set options.linsolver = DiffEqWrapLS(linsolver, 1e-9)
		@show typeof(dfnewton) options
		J = JacWrap(dfnewton, de_prob.p)
		return newton(fnewton, J, x0, options)
	end
end

function continuation(de_prob::DEP, param_axis::Lens, contParams::ContinuationPar{T, S, E};kwargs...) where {T, S, E}
	@unpack f, df, x0 = ContinuationProblem(de_prob, param_axis)
	p = Setfield.get(de_prob.p, param_axis)
	continuation(f, x0, p, contParams; kwargs...)
end
