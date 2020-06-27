import Base: push!, pop!, length, deleteat!, empty!
import Base: show, getindex

"""
DeflationOperator. It is used to handle the following situation. Assume you want to solve `F(x)=0` with a Newton algorithm but you want to avoid the process to return some already known solutions ``roots_i``. The deflation operator penalizes these roots ; the algorithm works very well despite its simplicity. You can use `DeflationOperator` to define a function `M(u)` used to find, with Newton iterations, the zeros of the following function
``F(u) \\cdot Π_i(dot(u - roots_i, u - roots_i)^{-p} + shift) := F(u) \\cdot M(u)``. The fields of the struct `DeflationOperator` are as follows:
- power `p`
- `dot` function, this function has to be bilinear and symmetric for the linear solver to work well
- shift
- roots
The deflation operator is is ``M(u) = \\prod_{i=1}^{n_{roots}}(shift + norm(u-roots_i)^{-p})``
where ``norm(u) = dot(u,u)``.

Given `defOp::DeflationOperator`, one can access its roots as `defOp[n]` as a shortcut for `defOp.roots[n]`. Also, one can add (resp.remove) a new root by using `push!(defOp, newroot)` (resp. `pop!(defOp)`). Finally `length(defOp)` is a shortcut for `length(defOp.roots)`
"""
struct DeflationOperator{T <: Real, Tdot, vectype}
	power::T
	dot::Tdot
	shift::T
	roots::Vector{vectype}
end

push!(df::DeflationOperator{T, Tdot, vectype}, v::vectype) where {T, Tdot, vectype} = push!(df.roots, v)
pop!(df::DeflationOperator) = pop!(df.roots)
getindex(df::DeflationOperator, inds...) = getindex(df.roots, inds...)
length(df::DeflationOperator) = length(df.roots)
deleteat!(df::DeflationOperator, id) = deleteat!(df.roots, id)
empty!(df::DeflationOperator) = empty!(df.roots)

function show(io::IO, df::DeflationOperator)
	println(io, "Deflation operator with ", length(df.roots)," roots")
end

# Compute M(u)
function (df::DeflationOperator{T, Tdot, vectype})(u::vectype) where {T, Tdot, vectype}
	nrm  = u -> df.dot(u, u)
	if length(df.roots) == 0
		return T(1)
	end
	# compute u - df.roots[1]
	tmp = copyto!(similar(u), u);	axpy!(T(-1), df.roots[1], tmp)
	out = T(1) / nrm(tmp)^df.power + df.shift
	for ii in 2:length(df.roots)
		copyto!(tmp, u); axpy!(T(-1), df.roots[ii], tmp)
		out *= T(1) / nrm(tmp)^df.power + df.shift
	end
	return out
end

# Compute dM(u)⋅du. We use a tmp for storing
function (df::DeflationOperator{T, Tdot, vectype})(::Val{:dMwithTmp}, tmp, u::vectype, du) where {T, Tdot, vectype}
	if length(df) == 0
		return T(0)
	end
	δ = 1e-8
	copyto!(tmp, u); axpy!(δ, du, tmp)
	return (df(tmp) - df(u)) / δ
end

(df::DeflationOperator)(u, du) = df(Val(:dMwithTmp), similar(u), u, du)

"""
	pb = DeflatedProblem(F, J, M::DeflationOperator)

This creates a deflated problem ``M(u) \\cdot F(u) = 0`` where `M` is a `DeflationOperator` which encodes the penalization term. `J` is the jacobian of `F`. Can be used to call `newton` and `continuation`.
"""
struct DeflatedProblem{T, Tdot, vectype, TF, TJ, def <: DeflationOperator{T, Tdot, vectype}}
	F::TF
	J::TJ
	M::def
end
@inline length(prob::DeflatedProblem) = length(prob.M)

"""
Return the deflated function M(u) * F(u) where M(u) ∈ R
"""
function (df::DeflatedProblem{T, Tdot, vectype, TF, TJ, def})(u::vectype, par) where {T, Tdot, vectype, TF, TJ, def}
	out = df.F(u, par)
	rmul!(out, df.M(u))
	return out
end

"""
Return the jacobian of the deflated function M(u) * F(u) where M(u) ∈ R
"""
function (df::DeflatedProblem{T, Tdot, vectype, TF, TJ, def})(u::vectype, par, du) where {T, Tdot, vectype, TF, TJ, def}
	# dF(u)⋅du * M(u) + F(u) dM(u)⋅du
	# out = dF(u)⋅du * M(u)
	out = apply(df.J(u, par), du)
	M = df.M(u)
	rmul!(out, M)
	# we add the remaining part
	if length(df) > 0
		F = df.F(u, par)
		# F(u) dM(u)⋅du
		out .+= df.M(u, du) .* F
	end
	return out
end

###################################################################################################
# Implement the Jacobian operator of the Deflated problem

# this is used to define a custom linear solver
struct DeflatedLinearSolver <: AbstractLinearSolver end

"""
Implement the linear solver for the deflated problem
"""
function (dfl::DeflatedLinearSolver)(J, rhs)
	# the expression of the Functional is now
	# F(u) * Π_i(dot(u - root_i, u - root_i)^{-power} + shift) := F(u) * M(u)
	# the expression of the differential is
	# dF(u)⋅du * M(u) + F(u) dM(u)⋅du

	# the point at which to compute the Jacobian
	u = J[1]
	p = J[2]
	# deflated Problem composite type
	defPb = J[3]
	linsolve = J[4]
	Fu = defPb.F(u, p)
	Mu = defPb.M(u)
	Ju = defPb.J(u, p)

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
	tmp = similar(u)
	z1 = defPb.M(Val(:dMwithTmp), tmp, u, h1)

	# z2 = dM(h)⋅h2
	z2 = defPb.M(Val(:dMwithTmp), tmp, u, h2)

	z = z1 / (Mu + z2)

	# return (h1 - z * h2) / Mu, true, (it1, it2)
	copyto!(tmp, h1)
	axpy!(-z, h2, tmp)
	rmul!(tmp, 1 / Mu)
	return tmp, true, (it1, it2)
end
####################################################################################################
"""
	function newton(F, J, x0::vectype, p0, options:: NewtonPar{T}, defOp::DeflationOperator{T, Tf, vectype}, linsolver = DeflatedLinearSolver(); kwargs...) where {T, Tf, vectype}

This is the deflated version of the Krylov-Newton Solver for `F(x, p0) = 0` with Jacobian `J(x, p0)`. We refer to [`newton`](@ref) for more information. It penalises the roots saved in `defOp.roots`. The other arguments are as for `newton`. See [`DeflationOperator`](@ref) for more information.

# Arguments
Compared to [`newton`](@ref), the only different arguments are
- `defOp` deflation operator
- `linsolver` linear solver used to invert the Jacobian of the deflated functional. We have a custom solver `DeflatedLinearSolver()` with requires solving two linear systems `J⋅x = rhs`. For other linear solvers, a matrix free method is used for the deflated functional.

# Output:
- solution:
- history of residuals
- flag of convergence
- number of iterations

# Simplified call
When `J` is not passed. It then computed with finite differences. The call is as follows:

	newton(F, x0, p0, options, defOp; kwargs...)
"""
function newton(F, J, x0::vectype, p0, options::NewtonPar{T, S, E}, defOp::DeflationOperator{T, Tf, vectype}, linsolver::DeflatedLinearSolver = DeflatedLinearSolver(); kwargs...) where {T, Tf, vectype, S, E}
	# we create the new functional
	deflatedPb = DeflatedProblem(F, J, defOp)

	# and its jacobian
	Jacdf = (u, p) -> (u, p, deflatedPb, options.linsolver)

	# change the linear solver
	opt_def = @set options.linsolver = linsolver

	return newton(deflatedPb, Jacdf, x0, p0, opt_def; kwargs...)
end

function newton(F, J, x0::vectype, p0, options::NewtonPar{T, S, E}, defOp::DeflationOperator{T, Tf, vectype}, linsolver::AbstractLinearSolver; kwargs...) where {T, Tf, vectype, S, E}
	# we create the new functional
	deflatedPb = DeflatedProblem(F, J, defOp)

	# change the linear solver
	opt_def = @set options.linsolver = linsolver

	return newton(deflatedPb, (x,p) -> (dx -> deflatedPb(x,p,dx)), x0, p0, opt_def; kwargs...)
end

# simplified call when no Jacobian is given
function newton(F, x0::vectype, p0, options::NewtonPar{T, S, E}, defOp::DeflationOperator{T, Tf, vectype}, linsolver = DeflatedLinearSolver(); kwargs...) where {T, Tf, vectype, S, E}
	J = (u, p) -> finiteDifferences(z -> F(z,p), u)
	return newton(F, J, x0, p0, options, defOp, linsolver; kwargs...)
end

"""
$(TYPEDEF)

This specific Newton-Kyrlov method first tries to converge to a solution `sol0` close the guess `x0`. It then attempts to converge to the guess `x1` while avoiding the previous solution `sol0`. This is very handy for branch switching. The mnethod is based on a deflated Newton-Krylov solver.
"""
function newton(F, J, x0::vectype, x1::vectype, p0, options::NewtonPar{T, S, E}, defOp::DeflationOperator = DeflationOperator(2.0, dot, 1.0, Vector{vectype}()); kwargs...) where {T, Tf, vectype, S, E}
	res0 = newton(F, J, x0, p0, options; kwargs...)
	@assert res0[3] "Newton did not converge to the trivial solution x0."
	push!(defOp, res0[1])
	res1 = newton(F, J, x1, p0, (@set options.maxIter = 10options.maxIter), defOp; kwargs...)
	@assert res1[3] "Deflated Newton did not converge to the non-trivial solution ( i.e. on the bifurcated branch)."
	return res1, res0
end
