using Test, BifurcationKit, ForwardDiff, RecursiveArrayTools, LinearAlgebra
const BK = BifurcationKit
N = 10
M = 5
δ = 1e-8
####################################################################################################
# test the jacobian of the multiple shooting functional using Linear flow
# TODO do example with A matrix and exp(At)
vf(x, p) = x
flow(x, p, t) = exp(t) .* x
dflow(x, p, dx, t) = (t = t, u = flow(x, p, t), du = exp(t) .* dx)
section(x, T) = dot(x[1:N], ones(N))
section(x::BorderedArray, T) = section(vec(x.u[:,:]), T)
par = nothing

fl = BK.Flow(vf, flow, dflow)

probSh = BK.ShootingProblem(M, fl,
	LinRange(0, 1, M+1) |> diff,
	section, false)

show(probSh)

poguess = VectorOfArray([rand(N) for ii=1:M])
	po = BorderedArray(poguess, 1.)

dpoguess = VectorOfArray([rand(N) for ii=1:M])
	dpo = BorderedArray(dpoguess, 2.)

# use of AbstractArray structure
pov = vcat(vec(po.u), po.p)
dpov = vcat(vec(dpo.u), dpo.p)
resv = probSh(pov, par)

dresv = probSh(pov, par, dpov; δ = δ)
resad = ForwardDiff.derivative(t -> probSh(pov .+ t .* dpov, par), 0.)
@test norm(resad[1:end-1] - dresv[1:end-1], Inf) < 1e-14

# use of BorderedArray structure
res = probSh(po, par)
@test norm(vec(res.u[:,:]) - resv[1:end-1] , Inf) ≈ 0
@test norm(res.p - resv[end], Inf) ≈ 0

dres = probSh(po, par, dpo; δ = δ)
@test norm(dres.p - dresv[end], Inf) ≈ 0
@test norm(vec(dres.u[:,:]) - dresv[1:end-1], Inf) ≈ 0
####################################################################################################
# test the jacobian of the multiple shooting functional using nonLinear flow
vf(x, p) = x.^2
flow(x, p, t) = x ./ (1 .- t .* x)
dflow(x, p, dx, t) = (t = t, u = flow(x, p, t), du = dx ./ (1 .- t .* x).^2)

fl = BK.Flow(vf, flow, dflow)

probSh = BK.ShootingProblem(M, fl,
	LinRange(0,1,M+1) |> diff ,
	section, false)

poguess = VectorOfArray([rand(N) for ii=1:M])
	po = BorderedArray(poguess, 1.)

dpoguess = VectorOfArray([rand(N) for ii=1:M])
	dpo = BorderedArray(dpoguess, 2.)

# use of AbstractArray structure
pov = vcat(vec(po.u), po.p)
dpov = vcat(vec(dpo.u), dpo.p)
resv = probSh(pov, par)

dresv = probSh(pov, par, dpov; δ = δ)
resad = ForwardDiff.derivative(t -> probSh(pov .+ t .* dpov, par), 0.)
@test norm(resad[1:end-1] - dresv[1:end-1], Inf) < 1e-14

# use of BorderedArray structure
res = probSh(po, par)
@test norm(vec(res.u[:,:]) - resv[1:end-1] , Inf) ≈ 0
@test norm(res.p - resv[end], Inf) ≈ 0

dres = probSh(po, par, dpo; δ = δ)
@test norm(dres.p - dresv[end], Inf) ≈ 0
@test norm(vec(dres.u[:,:]) - dresv[1:end-1], Inf) ≈ 0
####################################################################################################
# test the hyperplane projections for Poincare Shooting
M = 1
normals = [rand(2) for ii=1:M]
for ii=1:M
	normals[ii] /= norm(normals[ii])
end
centers = [rand(2) for ii=1:M]

hyper = BK.SectionPS(normals, centers)

x = 1:50 |>collect .|> Float64 |> vec
x = rand(2)
xb = BK.R(hyper, x, 1)
# test
for ii=1:M
	x2 = BK.E(hyper, xb, ii)
	# test that x2 in Sigma2
	@test dot(x2 - centers[ii], normals[ii]) < 1e-14
	# that we have Rk∘Ek = Id and Ek∘Rk = IdΣ
	@test BK.R(hyper, x2, ii) ≈ xb
end

# test of the derivatives of E and R
dx = rand(2)
_out1 = (BK.R(hyper, x .+ δ .* dx, 1) - BK.R(hyper, x, 1)) ./ δ
_out2 = zero(_out1)
_out2 = BK.dR!(hyper, _out2, dx, 1)

@test norm(_out2 - _out1, Inf) < 1e-5

dx = rand(2-1)
_out1 = ForwardDiff.derivative(t -> BK.E(hyper, xb .+ t .* dx,1), 0.)
_out2 = BK.dE(hyper, dx, 1)
@test norm(_out2 - _out1, Inf) < 1e-12

# flow for Poincare return map
# we consider the ODE dr = r⋅(1-r), dθ = 1 [2π]
# the flow is given by Φ(r0,θ0,t) = (r0 e^{t}/(1-r0+r0 e^{t}),θ+t)
vf(x, p) = x
flow(x, p, t) = (exp(t) .* x[1] / (1-x[1]+x[1]*exp(t)),x[2]+t)
Π(x, p, t) = (exp(2pi-x[1]) .* x[1] / (1-x[1]+x[1]*exp(2pi-x[1])), 0) # return map
# dflow(x, p, dx, t) = (t = t, u = flow(x, p, t), du = exp(t) .* dx)
section(x, T) = dot(x[1:2], [1, 0])
# section(x::BorderedArray, T) = section(vec(x.u[:,:]), T)
par = nothing

fl = BK.Flow(F = vf, flow = Π, flowSerial = flow)
sectionps = SectionPS(normals, centers)

probPSh = PoincareShootingProblem(fl, M, sectionps; δ = 1e-8)

show(probPSh)

poguess = VectorOfArray([rand(2) for ii=1:M])
	po = BorderedArray(poguess, 1.)

dpoguess = VectorOfArray([rand(2) for ii=1:M])
	dpo = BorderedArray(dpoguess, 2.)

ci = reduce(vcat, BK.projection(probPSh, poguess.u))
probPSh(ci, par)
probPSh(ci, par, ci)

probPSh = PoincareShootingProblem(fl, M, sectionps)
probPSh(ci, par)
probPSh(ci, par, ci)
