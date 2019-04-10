import Base: push!

"""
DeflationOperator with structure
- power
- dot function
- shift
- roots
"""
struct DeflationOperator{T <: Real, vectype}
	power::T
	# this function has to be bilinear and symmetric for the linear solver
	dot::Function
	shift::T
	roots::Vector{vectype}
end

push!(df::DeflationOperator{T, vectype}, v::vectype) where {T, vectype} = push!(df.roots, v)

function (df::DeflationOperator{T, vectype})(u) where {T, vectype}
	nrm  = u -> df.dot(u, u)
	@assert length(df.roots) >0 "You need to specify some roots for deflation to work"
	out = 1 / nrm(u - df.roots[1])^df.power + df.shift
	for ii = 2:length(df.roots)
		n = length(df.roots[ii])
		out *= 1 / nrm(u - df.roots[ii])^df.power + df.shift
	end
	return out
end

function scalardM(df::DeflationOperator{T, vectype}, u, du) where {T, vectype}
	# the deflation operator is Mu = 1/Π_i(shift + norm(u-ri)^p)
	# its differntial is -alpha(u, du) / Mu^2
	# the goal of this function is to compute alpha(u, du)
	delta = 1e-8
	return (df(u + delta * du) - df(u))/delta

	Mu = df(u)
	p  = df.power
	@assert length(df.roots) >0 "You need to specify some roots for deflation to work"
	out = 0.0
	for ii = 1:length(df.roots)
		αi = 1 / (1 / nrm(u-df.roots[ii])^p + df.shift)
		αi *= p / nrm(u-df.roots[ii])^(p+1)
		out += αi * 2df.dot(u-df.roots[ii], du)
	end
	return out * Mu
end

struct DeflatedProblem{T, vectype, def <: DeflationOperator{T, vectype}}
	F::Function
	J::Function
	M::def
end

"""
Return the deflated function M(u) * F(u) where M(u) ∈ R
"""
function (df::DeflatedProblem{T, vectype, def})(u) where {T, vectype, def <: DeflationOperator{T, vectype}}
	return df.M(u) * df.F(u)
end

###################################################################################################
# Linear Solver
# function jac(df::DeflatedProblem, u)
#	 J = df.J(u)
# end

struct DeflatedLinearSolver <: LinearSolver

end

"""
Implement the solve for the linear solver
"""
function (dfl::DeflatedLinearSolver)(J, rhs)
	# the expression of the Functional is now
	# F(u) / Π_i(dot(u - root_i, u - root_i) + shift)^power := F(u) / M(u)
	# the expression of the differential is
	# dF(u)⋅du * M(u) + F(u) dM(u)⋅du

	# the point at which to compute the Jacobian
	u = J[1]
	# deflated Problem structure
	defPb = J[2]
	linsolve = J[3]
	Fu = defPb.F(u)
	Mu = defPb.M(u)

	# linear solve for the deflated problem. We note that the Mu ∈ R
	# hence dM(u)⋅du is a scalar
	# M(u) * dF(u)⋅h + F(u) dM(u)⋅h = rhs
	h1, _, it1 = linsolve(defPb.J(u), rhs)
	h2, _, it2 = linsolve(defPb.J(u), Fu)

	# the expression of dM(u)⋅h
	# the solution is then h = Mu * h1 - z h2 where z has to be determined
	delta = 1e-8
	z1 = (defPb.M(u + delta * h1) - defPb.M(u))/delta
	z2 = (defPb.M(u + delta * h2) - defPb.M(u))/delta
	z = z1 / (Mu + z2)
	return (h1 - z * h2) / Mu, true, (it1, it2)
end
