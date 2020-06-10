# using Revise, Plots, Test
using OrdinaryDiffEq, ForwardDiff
	# using Zygote, DiffEqSensitivity
	using LinearAlgebra, Parameters

norminf = x -> norm(x, Inf)

function Fsl!(f, u, p, t)
	@unpack r, μ, ν, c3, c5 = p
	u1 = u[1]
	u2 = u[2]

	ua = u1^2 + u2^2

	f[1] = r * u1 - ν * u2 - ua * (c3 * u1 - μ * u2) - c5 * ua^2 * u1
	f[2] = r * u2 + ν * u1 - ua * (c3 * u2 + μ * u1) - c5 * ua^2 * u2

	return f
end

Fsl(x, p) = Fsl!(similar(x), x, p, 0.)
dFsl(x, dx, p) = ForwardDiff.derivative(t -> Fsl(x .+ t .* dx, p), 0.)

# function to compute differentials
function diffAD(f, x, dx)
	# Zygote.pullback(t->f(x .+ t.* dx), 0.)[1]
	ForwardDiff.derivative(t->f(x .+ t.* dx), 0.)
end

####################################################################################################
par_sl = (r = 0.1, μ = 0., ν = 1.0, c3 = 1.0, c5 = 0.0,)
u0 = [.001, .001]
δ = 1e-8
prob = ODEProblem(Fsl!, u0, (0., 100.), par_sl)
algsl = Rodas4P()
####################################################################################################
sol = @time solve(prob, algsl, abstol =1e-9, reltol=1e-6)
# plot(sol[1,:], sol[2,:])

function flowTS(x, t, pb = prob; alg = algsl, kwargs...)
	_pb = remake(pb; u0 = x, tspan = (zero(t), t) )
	sol = DiffEqBase.concrete_solve(_pb, alg; abstol =1e-9, reltol=1e-6, save_everystep = false, kwargs...)
	return sol.t, sol
end
flowDE = (x, t, pb = prob; alg = algsl, kwargs...) -> flowTS(x, t, pb = pb; alg = alg, kwargs...)[2][end]


dflowDE = (x, dx, ts; kwargs...) -> diffAD(z -> flowDE(z, ts; kwargs...), x, dx)
dflowDEfd = (x, dx, ts) -> (flowDE(x .+ δ .* dx, ts) - flowDE(x, ts)) ./ δ

@test dflowDE(u0,u0,1.) .- dflowDEfd(u0,u0,1.) |> norminf < 10δ
####################################################################################################
# defining the Poincare Map
normals = [[-1., 0.]]
centers = [zeros(2)]

sectionH(x, c, n) = dot( x .- c[1], n[1])
pSection(u, t, integrator) = sectionH(u, centers, normals) * (integrator.iter > 1)
affect!(integrator) = terminate!(integrator)
cb = ContinuousCallback(pSection, affect!; affect_neg! = nothing)

Π(x) = flowDE(x, Inf; callback = cb, save_everystep = false)
dΠ(x, dx) = diffAD(Π, x, dx)
dΠFD(x, dx) = (Π(x .+ δ .* dx) .- Π(x)) ./ δ

function DPoincare(x, dx, p, normal, center, _cb; verbose = false)
	verbose && printstyled(color=:blue, "\nEntree dans DPoincare\n")
	abs(dot(normal, dx)) > 1e-12 && @warn "Vector does not belong to hyperplane!  dot(normal, dx) = $(abs(dot(normal, dx))) and $(dot(dx, dx))"
	# compute the Poincare map from x
	_tΣ, _solΣ = flowTS(x, Inf; callback = _cb, save_everystep = false)
	tΣ, solΣ = _tΣ[end], _solΣ[end]

	z = Fsl(solΣ, p)
	verbose && @show z tΣ solΣ
	# solution of the variational equation at time tΣ
	# We need the callback to be INACTIVE here!!!
	y = dflowDE(x, dx, tΣ; callback = nothing)
	verbose && @show y
	out = y .- (dot(normal, y) / dot(normal, z)) .* z
end

# vecteurs for the flow and the variational flow
u0 = [0, 1.]
du0 = [0, -1.]

# check that we cross the sections the way we want
ts,ss = flowTS([0., 1], Inf; callback = cb, save_everystep = true, save_at = LinRange(0,1,20))
# plot(ts,ss[:,:]')
# plot(ss[1,:], ss[2,:], label="flow");scatter!(ss[1,[1]], ss[2,[1]]);plot!(sol[1,:], sol[2,:], label="sol")

# these vectors should be the same
# println("--> dΠ using Forward Diff")
# res = dΠ(u0, du0); show(res)

# println("--> dΠ using Zygote")
# gives completely wrong answer
# Zygote.pullback(t->Π(u0 .+ t.* du0), 0.)[1] |> Base.show

println("--> dΠ using Analytical formula")
resAna = DPoincare(u0, du0, par_sl, normals[1], centers[1], cb);show(resAna)

println("--> dΠ using Finite differences")
resFD = dΠFD(u0, du0);show(resFD)

println("--> Norm of the difference = ", resAna - resFD |> norminf)
@test resAna - resFD |> norminf < 1000δ

####################################################################################################
# comparison with BK
using BifurcationKit
	const BK = BifurcationKit

function FslMono!(f, x, p, t)
	u = x[1:2]
	du = x[3:4]
	Fsl!(view(f,1:2), u, p, t)
	f[3:4] .= dFsl(u, du, p)
end
probMono = ODEProblem(FslMono!, vcat(u0, u0), (0., 100.), par_sl)

probHPsh = BK.PoincareShootingProblem(Fsl, par_sl,
		prob, algsl,
		# probMono, Rodas4P(autodiff=false),
		normals, centers; abstol =1e-10, reltol=1e-10)

@show BK.diffPoincareMap(probHPsh, u0, par_sl, du0, 1)


resDP = DPoincare(u0, du0, par_sl, normals[1], centers[1], cb; verbose = true)
resDPBK = BK.diffPoincareMap(probHPsh, u0, par_sl, du0, 1)
@test norminf(resDP - resDPBK) < 1e-6
