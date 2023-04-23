# using Revise
using Test, BifurcationKit, ForwardDiff, LinearAlgebra, RecursiveArrayTools, Setfield
const BK = BifurcationKit
N = 10
M = 5
δ = 1e-8
####################################################################################################
# test AbstractFlow interface
struct MyFlow <: BK.AbstractFlow end
fl = MyFlow()
x0 = 0
p0 = 0
BK.evolve(fl, x0,p0,0,0)
BK.vf(fl,x0, p0)
BK.evolve(fl, x0, p0, 0)
BK.evolve(fl, Val(:Full), x0, p0, 0)
BK.evolve(fl, Val(:SerialTimeSol), x0, p0, 0)
BK.evolve(fl, Val(:TimeSol), x0, p0, 0)
BK.evolve(fl, Val(:SerialdFlow), x0, p0, 0, 0)
####################################################################################################
# test the jacobian of the multiple shooting functional using Linear flow
# TODO do example with A matrix and exp(At)
vf(x, p) = x
flow(x, p, t) = (u=exp(t) .* x, t=t)
dflow(x, p, dx, t) = (flow(x, p, t)..., du = exp(t) .* dx)
section(x, T) = dot(x[1:N], ones(N))
section(x, T, dx, dT) = dot(dx[1:N], ones(N)) * T
section(x::BorderedArray, T) = section(vec(x.u[:,:]), T)
section(x::BorderedArray, T, dx, dT) = section(vec(x.u[:,:]), T, vec(dx.u[:,:]), dT)
par = nothing

fl = BK.Flow(vf, flow, dflow); @set! fl.flowFull = flow
BK.evolve(fl, Val(:Full), rand(N), par, 0.)

probSh = BK.ShootingProblem(M = M, flow = fl,
	ds = LinRange(0, 1, M+1) |> diff,
	section = section)

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
flow(x, p, t) = (u=x ./ (1 .- t .* x),t=t)
dflow(x, p, dx, t) = (flow(x, p, t)..., du = dx ./ (1 .- t .* x).^2)

fl = BK.Flow(vf, flow, dflow)

probSh = BK.ShootingProblem(M = M, flow = fl,
	ds = LinRange(0,1,M+1) |> diff ,
	section = section)

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
poguess = VectorOfArray([rand(2) for ii=1:M])

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
vf(x, p) = [x[1]*(1-x[1]), 1.]
flow(x, p, t; k...) = (t = t, u = [exp(t) .* x[1] / (1-x[1]+x[1]*exp(t)),x[2]+t])
Π2(x, p, t = 0) = (t = 2pi -x[2], u = flow(x, p, 2pi-x[2]).u)
Π(x, p, t = 0) = flow(x, p, 2pi-x[2])	# return map
dflow(x, p, dx, t; k...) = (t = t, u = flow(x, p, t).u, du = ForwardDiff.derivative( z -> flow(x .+ z .* dx, p, t).u, 0),)
section(x, T) = dot(x[1:2], [1, 0])
# section(x::BorderedArray, T) = section(vec(x.u[:,:]), T)
par = nothing

fl = BK.Flow(F = vf, flow = Π, flowSerial = Π2, dfSerial = dflow)
sectionps = SectionPS(normals, centers)
probPSh = PoincareShootingProblem(flow = fl, M = M, section = sectionps)


ci = reduce(vcat, BK.projection(probPSh, poguess.u))
dci = rand(length(ci))

# we test that we have the analytical version of the flow
z0 = rand(2)
@test ForwardDiff.derivative(z -> flow(z0, par, z).u, 0.) ≈ vf(z0, par)

#test the show method
show(probPSh)

# test functional and its differential
probPSh(ci, par)
probPSh(ci, par, dci)

# test the analytical of the differential of the return Map
z0 = rand(2)
dz0 = rand(2)
_out0 = BK.diffPoincareMap(probPSh, z0, par, dz0, 1)
_out1 = ForwardDiff.derivative(z -> Π(z0 .+ z .* dz0, par).u, 0)
display(_out0)
display(_out1)

# test the analytical version of the functional
_out0 = probPSh(ci, par, dci)
	δ = 1e-6
	_out2 = (probPSh(ci .+ δ .* dci, par) .- probPSh(ci, par)) ./ δ
	_out1 = ForwardDiff.derivative(z -> probPSh(ci .+ z .* dci, par), 0)
	display(_out0)
	display(_out1)
	display(_out2)
	_out0 - _out1 |> display
