using BlockArrays, SparseArrays
# This file implements some Finite Differences methods to locate periodic orbits

####################################################################################################
# method using the Trapezoidal rule (Order 2 in time) and discretisation of the periodic orbit. This is not a shooting method!

@with_kw struct PeriodicOrbitTrap{vectype, S <: LinearSolver, N} <: PeriodicOrbit
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

"""
This encodes the functional for finite differences using the Trapezoidal rule
"""
function (poPb::PeriodicOrbitTrap{vectype, S, N})(u0::vectype) where {vectype <: AbstractVector, S, N}
    M = poPb.M
    N = div(length(u0) - 1, M)
    T = u0[end]
    h = T / M

    u0c = reshape(u0[1:end-1], N, M)
    outc = similar(u0c)
    for ii=2:M
        outc[:, ii] .= (u0c[:, ii] .- u0c[:, ii-1]) .- h/2 .* (poPb.F(u0c[:, ii]) .+ poPb.F(u0c[:, ii-1]))
    end

    # closure condition ensuring a periodic orbit
    outc[:, 1] .= u0c[:, M] .- u0c[:, 1]

    return vcat(vec(outc),
			dot(u0c[:, 1] .- poPb.xπ, poPb.ϕ)) # this is the phase condition
end

"""
Matrix free expression of the Jacobian computed at `u0` and applied to `du`.
"""
function (poPb::PeriodicOrbitTrap{vectype, S, N})(u0::vectype, du) where {vectype, S, N}
    M = poPb.M
    N = div(length(u0) - 1, M)
    T = u0[end]
    h = T / M

    u0c = reshape(u0[1:end-1], N, M)
    duc = reshape(du[1:end-1], N, M)
    outc = similar(u0c)

    for ii=2:M
        outc[:, ii] .= (duc[:, ii] .- duc[:, ii-1]) .- h/2 .*( apply(poPb.J(u0c[:, ii]), duc[:, ii]) .+ apply(poPb.J(u0c[:, ii-1]), duc[:, ii-1]) )
    end

    # closure condition
    outc[:, 1] .= duc[:, M] .- duc[:, 1]

    δ = 1e-9
    dTFper = (poPb(vcat(u0[1:end-1], T + δ)) - poPb(u0)) / δ
    return vcat(vec(outc) .+ dTFper[1:end-1] .* du[end],
				dot(duc[:, 1], poPb.ϕ) + dTFper[end] * du[end])
end

"""
Sparse Matrix expression expression of the Jacobian for the periodic problem computed at the space-time guess: `u0`
"""
function JacobianPeriodicFD(poPb::PeriodicOrbitTrap{vectype, S, N}, u0::vectype, γ = 1.0) where {vectype, S, N}
	# extraction of various constants
	M = poPb.M
    N = div(length(u0) - 1, M)
    T = u0[end]
    h = T / M

	J = BlockArray(spzeros(M * N ,M * N), N * ones(Int64,M),  N * ones(Int64,M))

	In = spdiagm( 0 => ones(N))
	On = spzeros(N, N)

	u0c = reshape(u0[1:end-1], N, M)
    outc = similar(u0c)

	for ii=2:M
		Jn = In - h/2 .* poPb.J(u0c[:, ii])
		setblock!(J, Jn, ii, ii)

		Jn = -In - h/2 .* poPb.J(u0c[:, ii-1])
		setblock!(J, Jn, ii,ii-1)
	end
	setblock!(J, -γ * In, 1, 1)
	setblock!(J,  In, 1, M)
	return J
end


function (poPb::PeriodicOrbitTrap{vectype, S, N})(u0::vectype, tp::Symbol = :jacobian) where {vectype, S, N}
	# extraction of various constants
	M = poPb.M
    N = div(length(u0) - 1, M)
    T = u0[end]
    h = T / M

	J = JacobianPeriodicFD(poPb, u0)

	# we now set up the last line / column
	δ = 1e-9
    dTFper = (poPb(vcat(u0[1:end-1], T + δ)) - poPb(u0)) / δ

	# this bad for performance. Get converted to SparseMatrix at the next line
	J = hcat(J, dTFper[1:end-1])
	J = vcat(J, spzeros(1, N*M+1))

	J[N*M+1, 1:N] .=  poPb.ϕ
	J[N*M+1, N*M+1] = dTFper[end]

	return J
end

####################################################################################################
# Computation of Floquet Coefficients
"""
Matrix-Free expression expression of the Monodromy matrix for the periodic problem computed at the space-time guess: `u0`
"""
function FloquetPeriodicFD(poPb::PeriodicOrbitTrap{vectype, S, N}, u0::vectype, du::vectype) where {vectype, S, N}
	# extraction of various constants
	M = poPb.M
    N = div(length(u0)-1, M)
    T = u0[end]
    h = T / M

	out = copy(du)

	u0c = reshape(u0[1:end-1], N, M)

	for ii=2:M
		out .= out./h .+ 1/2 .* apply(poPb.J(u0c[:, ii-1]), out)
		res, _ = poPb.linsolve(I/h - 1/2 * poPb.J(u0c[:, ii]), out)
		res .= out
	end
	return out
end

struct FloquetFD <: EigenSolver
	poPb
end

function (fl::FloquetFD)(J, sol, nev)
	@show sol.p
	@show length(sol.u)
	Jmono = x -> FloquetPeriodicFD(fl.poPb(sol.p), sol.u, x)
	n = div(length(sol.u)-1, 50)
	@show n
	vals, vec, info = KrylovKit.eigsolve(Jmono,rand(n),15, :LM)
	return log.(vals), vec, true, info.numops
end
