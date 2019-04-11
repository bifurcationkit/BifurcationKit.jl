# This file implements some Shooting Methods to locate periodic orbits
# The code is quite unstable so I would favour instead PeriodicOrbitFD

####################################################################################################
# Periodic Orbits via implicit midpoint method, order 2 in time
@with_kw struct ShootingProblemMid{vectype, S <: LinearSolver, N} <: PeriodicOrbit
	# Function F(x, p) = 0
	F::Function

	# Jacobian of F wrt x
	J::Function

	# variables to define a Poincare Section
	ϕ::vectype
	xπ::vectype

	# discretisation of the time interval
	M::Int = 100

	linsolve::S
	options_newton::NewtonPar{N}
end

function (poPb::ShootingProblemMid{vectype, S, N})(u0::vectype) where {vectype, S, N}
	T = u0[end]
	u = @view u0[1:end-1]
	# discretisation of the time interval [0, T]
	h = T / poPb.M

	Fper  = (u, u0) -> (u .- u0) .- h .* poPb.F((u.+u0)./2)
	dFper = (u, u0) -> I - h * poPb.J((u.+u0)./2)/2

	x0 = copy(u)
	xnew  = Flow(Fper, dFper, x0, poPb.M, poPb.options_newton)
	return vcat(xnew - u, dot(poPb.ϕ, u - poPb.xπ))
end

struct PeriodicOrbitLinearSolverMid <: LinearSolver end

# function (ls::PeriodicOrbitLinearSolverMid)(Jper, rhs)
# 	# @assert 1==0 "Very unstable Implementation"
# 	printstyled(color=:green, "*"^50*"\n--> Linsolve periodic orbit\n")
#
# 	# parameters extraction
# 	x = Jper[1][1:end-1]	 #the jacobian is evaluated at
# 	T = Jper[1][end]
# 	# @show N, T
# 	poPb = Jper[2]
# 	M = poPb.M
#
# 	Fper = (u, u0)-> (u .- u0) .- h .* poPb.F((u+u0)./2)
# 	dFper =	(u, u0) ->  I - h * poPb.J((u+u0)/2)/2
# 	dFperInv = (u, u0) -> -I - h * poPb.J((u+u0)/2)/2
#
#
# 	# the Jacobian of the functional for periodic orbits is [J -F((uM+uM-1)/2)/M;dg 0]
# 	dg = poPb.ϕ
#
# 	# if I write xi the sequence of time slice in the Backward Euler scheme, we find
# 	# (I - h⋅J(S)/2)⋅∂xi+1/∂x0 = (I + h⋅J(S)/2)⋅∂xi/∂x0 with S = (xi+1 + xi)/2 hence:
# 	# ∂xi+1/∂x0 = A(S)⋅∂xi/∂x0 where A(xi) = (I - h⋅J(S)/2)^{-1}(I + h⋅J(S)/2)
# 	# Hence Jper x = A(xm) A(xm-1)⋯A(x0)x
# 	# We want to solve the linear system Jper * x = rhs, it follows that
# 	# x = A(x0)\A(x1)\⋯A(xM)\rhs with
# 	# A(xi) \ rhs = (I + h⋅J(S)/2)^{-1}(I - h⋅J(S)/2)⋅rhs = (I - 2(I + h⋅J(S)/2)J(S))⋅rhs
#
# 	# discretisation of the time interval [0, T]
# 	h = T / poPb.M
# 	δ = 1e-9
# 	dTFper = (poPb(vcat(x, T+δ)) - poPb(Jper[1])) / δ
#
# 	# we solve the equations to find F(xM)
# 	x0 = copy(x)
# 	xM = copy(x)
# 	xhist = [x]
# 	for ii=1:poPb.M
# 		xM, _, flag = newton(u -> Fper(u, x0), u -> dFper(u, x0), x0, poPb.options_newton)
# 		@assert flag == true "Newton did not converge at i = $ii"
# 		x0 .= xM
# 		push!(xhist, xM)
# 	end
#
# 	# the Jacobian of the functional for periodic orbits is [J dTFper;dg 0]
# 	# we want to solve J⋅out + dF z = h = rhs[1:N]. We compute x1 = J\h and x2 = J\dF and out = x1 - z*x2
# 	# with (dg, out) = (dg, x1) - z (dg, x2) = rhs[N+1]
# 	x1 = copy(rhs[1:end-1])
# 	x2 = copy(dTFper[1:end-1])
# 	# here we start from xM and compute xM-1 and use this to inverse Jper
#
#
# 	for ii=poPb.M+1:-1:2
# 		PO1 = xhist[ii]
# 		PO0 = xhist[ii-1]
#
# 		S = (PO1 .+ PO0) / 2
#
# 		# jacobian at S
# 		J = h/2 * poPb.J(S)
#
# 		# @show norm(J, Inf64)
# 		# @error "semble instable"
# 		x1 .= x1 .- 2(I+J) \ (J*x1)
# 		x2 .= x2 .- 2(I+J) \ (J*x2)
# 		# @show norm(inv(I+J), Inf64) norm((J), Inf64)
# 		# x1 .= (I + J) \ (x1 - J * x1)
# 		# x2 .= (I + J) \ (x2 - J * x2)
#
# 		# PO1 .= PO0
# 	end
# 	@show norm(x1, Inf64), length(xhist)
#
# 	@show rhs[end] dot(dg, x1) dot(dg, x2)
# 	# we can now solve the bordered system
# 	z = (dot(dg, x1) - rhs[end]) / (dot(dg, x2))
# 	@show z
# 	@show norm(x1, Inf64) norm(x2, Inf64)
# 	@show norm(x1 - z * x2, Inf64)
#
# 	printstyled(color=:green, "----> end\n")
# 	# @assert 1==0
# 	return vcat(x1 - z * x2, z), true, 1
# end

####################################################################################################
####################################################################################################
# other functional for searching for Periodic Orbits with Trapezoidal rule, order 2 in time

@with_kw struct ShootingProblemTrap{vectype, S <: LinearSolver, N} <: PeriodicOrbit
	# Function F(x, p) = 0
	F::Function

	# Jacobian of F wrt x
	J::Function

	# variables to define a Poincare Section
	ϕ::vectype
	xπ::vectype

	#discretisation of the time interval
	M::Int = 100

	linsolve::S
	options_newton::NewtonPar{N}
end

function (poPb::ShootingProblemTrap{vectype, S, N})(u0::vectype) where {vectype, S, N}
	T = u0[end]
	u = @view u0[1:end-1]
	# discretisation of the time interval [0, T]
	h = T / poPb.M
	# we solve M times the Backward Euler scheme
	Fper  = (u, u0) -> (u .- u0) .- h .* (poPb.F(u) .+ poPb.F(u0)) / 2
	dFper = (u, u0) -> I - h * poPb.J(u) / 2

	x0 = copy(u)
	xnew  = Flow(Fper, dFper, x0, poPb.M, poPb.options_newton)
	return vcat(x0 - u, dot(x0 - poPb.xπ, poPb.ϕ))
end

####################################################################################################
####################################################################################################
# other functional for searching for Periodic Orbits based on Backward Euler Scheme

@with_kw struct ShootingProblemBE{vectype, S <: LinearSolver, N} <: PeriodicOrbit
	# Function F(x, p) = 0
	F::Function

	# Jacobian of F wrt x
	J::Function

	# variables to define a Poincare Section
	ϕ::vectype
	xπ::vectype

	#discretisation of the time interval
	M::Int = 100

	linsolve::S
	options_newton::NewtonPar{N}
end

function (poPb::ShootingProblemBE{vectype, S, N})(u0::vectype) where {vectype, S, N}
	T = u0[end]
	u = @view u0[1:end-1]
	# discretisation of the time interval [0, T]
	h = T / poPb.M
	# we solve M times the Backward Euler scheme
	Fper  = (u, u0) -> (u .- u0) .- h .* poPb.F(u)
	dFper = (u, u0) -> I - h * poPb.J(u)

	x0 = copy(u)
	xnew  = Flow(Fper, dFper, x0, poPb.M, poPb.options_newton)
	return vcat(x0 - u, dot(x0 - poPb.xπ, poPb.ϕ))
end
