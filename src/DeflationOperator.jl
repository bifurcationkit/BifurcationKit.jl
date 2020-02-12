include("Newton.jl")
import Base: push!, pop!, length
import Base: show, getindex

"""
DeflationOperator. It is used to describe the following situation. Assume you want to solve `F(x)=0` with a Newton algorithm but you want to avoid the process to return some already known solutions ``roots_i``. You can use `DeflationOperator` to define a function `M(u)` used to find, with Newton iterations, the zeros of the following function
``F(u) / Π_i(dot(u - roots_i, u - roots_i)^{p} + shift) := F(u) / M(u)``. The fields of the struct `DeflationOperator` are as follows:
- power `p`
- `dot` function, this function has to be bilinear and symmetric for the linear solver
- shift
- roots
The deflation operator is is ``M(u) = \\frac{1}{\\prod_{i=1}^{n_{roots}}(shift + norm(u-roots_i)^p)}``
where ``nrm(u) = dot(u,u)``.

Given `defOp::DeflationOperator`, one can access its roots as `defOp[n]` as a shortcut for `defOp.roots[n]`. Also, one can add (resp.remove) a new root by using `push!(defOp, newroot)` (resp. `pop!(defOp)`). Finally `length(defOp)` is a shortcut for `length(defOp.roots)`
"""
struct DeflationOperator{T <: Real, Tf, vectype}
	power::T
	dot::Tf
	shift::T
	roots::Vector{vectype}
end

push!(df::DeflationOperator{T, Tf, vectype}, v::vectype) where {T, Tf, vectype} = push!(df.roots, v)
pop!(df::DeflationOperator) = pop!(df.roots)
getindex(df::DeflationOperator, inds...) = getindex(df.roots, inds...)
length(df::DeflationOperator) = length(df.roots)

function show(io::IO, df::DeflationOperator)
	println(io, "Deflation operator with ", length(df.roots)," roots")
end

function (df::DeflationOperator{T, Tf, vectype})(u::vectype) where {T, Tf, vectype}
	nrm  = u -> df.dot(u, u)
	@assert length(df.roots) > 0 "You need to specify some roots for deflation to work"
	# compute u - df.roots[1]
	tmp = copyto!(similar(u), u);	axpy!(T(-1), df.roots[1], tmp)
	out = T(1) / nrm(tmp)^df.power + df.shift
	for ii = 2:length(df.roots)
		copyto!(tmp, u); axpy!(T(-1), df.roots[ii], tmp)
		out *= T(1) / nrm(tmp)^df.power + df.shift
	end
	return out
end

"""
	pb = DeflatedProblem(F, J, M::DeflationOperator)

This creates a problem encoded a deflated problem ``M(u) \\cdot F(u) = 0`` where `M` is a `DeflationOperator` which encodes the penalization term. `J` is the jacobian of `J`. Can be used to call `newton` and `continuation`.
"""
struct DeflatedProblem{T, Tf, vectype, TF, TJ, def <: DeflationOperator{T, Tf, vectype}}
	F::TF
	J::TJ
	M::def
end

"""
Return the deflated function M(u) * F(u) where M(u) ∈ R
"""
function (df::DeflatedProblem{T, Tf, vectype, TF, TJ, def})(u::vectype) where {T, Tf, vectype, TF, TJ, def}
	out = df.F(u)
	rmul!(out, df.M(u))
	return out
end

###################################################################################################
# Implement the Jacobian operator of the Deflated problem

struct DeflatedLinearSolver <: AbstractLinearSolver end

"""
Implement the linear solver for the deflated problem
"""
function (dfl::DeflatedLinearSolver)(J, rhs)
	# the expression of the Functional is now
	# F(u) / Π_i(dot(u - root_i, u - root_i)^power + shift) := F(u) * M(u)
	# the expression of the differential is
	# dF(u)⋅du * M(u) + F(u) dM(u)⋅du

	# the point at which to compute the Jacobian
	u = J[1]
	# deflated Problem composite type
	defPb = J[2]
	linsolve = J[3]
	Fu = defPb.F(u)
	Mu = defPb.M(u)
	Ju = defPb.J(u)

	# linear solve for the deflated problem. We note that Mu ∈ R
	# hence dM(u)⋅du is a scalar. We now solve the following linear problem
	# M(u) * dF(u)⋅h + F(u) dM(u)⋅h = rhs
	h1, h2, _, (it1, it2) = linsolve(Ju, rhs, Fu)

	# We look for the expression of dM(u)⋅h
	# the solution is then h = Mu * h1 - z h2 where z has to be determined
	delta = 1e-8
	# z1 = (defPb.M(u + delta * h1) - Mu)/delta
	tmp = copyto!(similar(u), u); axpy!(delta, h1, tmp)
	z1 = (defPb.M(tmp) - Mu) / delta

	# z2 = (defPb.M(u + delta * h2) - Mu)/delta
	copyto!(tmp, u); axpy!(delta, h2, tmp)
	z2 = (defPb.M(tmp) - Mu) / delta

	z = z1 / (Mu + z2)

	# return (h1 - z * h2) / Mu, true, (it1, it2)
	copyto!(tmp, h1)
	axpy!(-z, h2, tmp)
	rmul!(tmp, 1 / Mu)
	return tmp, true, (it1, it2)
end
####################################################################################################
"""
	function newton(Fhandle, Jhandle, x0::vectype, options:: NewtonPar{T}, defOp::DeflationOperator{T, Tf, vectype}; kwargs...) where {T, Tf, vectype}

This is the deflated version of the Newton Solver for `F(x) = 0` with Jacobian `J`. It penalises the roots saved in `defOp.roots`. The other arguments are as for `newton`. See [`DeflationOperator`](@ref) for more informations.

Simplified calls are provided, for example when `J` is not passed. It then computed with finite differences.

# Output:
- solution:
- history of residuals
- flag of convergence
- number of iterations
"""
function newton(Fhandle, Jhandle, x0::vectype, options::NewtonPar{T, S, E}, defOp::DeflationOperator{T, Tf, vectype}; kwargs...) where {T, Tf, vectype, S, E}
	# we create the new functional
	deflatedPb = DeflatedProblem(Fhandle, Jhandle, defOp)

	# and its jacobian
	Jacdf = (u0, pb::DeflatedProblem, ls) -> (return (u0, pb, ls))

	# Rename some parameters
	opt_def = @set options.linsolver = DeflatedLinearSolver()
	return newton(u -> deflatedPb(u),
				u -> Jacdf(u, deflatedPb, options.linsolver),
				x0,
				opt_def; kwargs...)
end

# simplified call when no Jacobian is given
function newton(Fhandle, x0::vectype, options::NewtonPar{T, S, E}, defOp::DeflationOperator{T, Tf, vectype};kwargs...) where {T, Tf, vectype, S, E}
	Jhandle = u -> PseudoArcLengthContinuation.finiteDifferences(Fhandle, u)
	return newton(Fhandle,  Jhandle,  x0, options,  defOp;kwargs...)
end
