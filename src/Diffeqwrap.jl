# the ideas are based on https://github.com/tkf/Bifurcations.jl/blob/master/src/diffeq.jl
# I would just grab DiffEqDiffTools.uJacobianWrapper
# https://github.com/JuliaDiffEq/DiffEqBase.jl/blob/1c86896dfcc0c383ef501014c56a3f8dfc8a63bb/src/nlsolve/newton.jl
using DiffEqBase: AbstractODEProblem
using DiffEqOperators
using Setfield

const DEP{iip} = AbstractODEProblem{uType, tType, iip} where {uType, tType}

struct DiffEqWrapper{P, L}
	de_prob::P
	param_axis::L
end

function ContinuationProblem(
		deprob::DEP{iip}, param_axis::Lens;
		kwargs0...) where iip
	de_prob = deepcopy(deprob)
	x0 = de_prob.u0
	@assert !(typeof(x0) <: Number) "We need array like structures for the state space, consider using [u0]"
	pbwrap = DiffEqWrapper(de_prob, param_axis)

	# we extract the vector field
	if de_prob isa DEP{false}
		fnewton = x -> de_prob.f(x, de_prob.p, de_prob.tspan[1])
		f = (x, p) -> de_prob.f(x, set(de_prob.p, param_axis, p), de_prob.tspan[1])
	else
		@warn "We don't consider inplace problem very efficiently (for now)"
		fnewton = x -> (out = similar(x);de_prob.f(out, x, de_prob.p, de_prob.tspan[1]);out)
		f = (x, p) -> (out = similar(x);de_prob.f(out, x, set(de_prob.p, param_axis, p), de_prob.tspan[1]);out)
	end

	# we extract the jacobian (operator)
	if de_prob.f.jac === nothing
		@warn "No jacobian is provided by ODEProblem"
		dfnewton = nothing
		df = nothing
	else
		@warn "No jacobian is provided by ODEProblem"
		dfnewton = nothing
		df = nothing
	end
	return (fnewton = fnewton, dfnewton = dfnewton, f = f, df = df, x0 = x0)
end


function newton(de_prob::DEP, options:: NewtonPar{T}; normN = norm) where {T}
	@unpack fnewton, x0 = ContinuationProblem(de_prob, @lens _.p)
	newton(fnewton, x0, options)
end

function continuation(de_prob::DEP, param_axis::Lens, contParams::ContinuationPar{T, S, E};kwargs...) where {T, S, E}
	@unpack f, df, x0 = ContinuationProblem(de_prob, param_axis)
	p = Setfield.get(de_prob.p, param_axis)
	continuation(f, x0, p, contParams; kwargs...)
end
