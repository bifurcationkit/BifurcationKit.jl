using Test, PseudoArcLengthContinuation, KrylovKit, RecursiveArrayTools, LinearAlgebra
const PALC = PseudoArcLengthContinuation
N = 10
M = 5
δ = 1e-8
####################################################################################################
# test the jacobian of the multiple shooting functional using Linear flow
# TODO do example with A matrix and exp(At)
vf(x) = x
flow(x, t) = exp(t) .* x
dflow(x, dx, t) = (flow(x, t), exp(t) .* dx)
section(x) = dot(x[1], ones(length(x[1])))
section(x::BorderedArray) = section(x.u)

fl = PALC.Flow(vf, flow, dflow)

probSh = PALC.ShootingProblem(fl,
	LinRange(0, 1, M+1) |> diff ,
	x -> section(x))

poguess = VectorOfArray([rand(N) for ii=1:M])
	po = BorderedArray(poguess, 1.)

res = probSh(po)

dpoguess = VectorOfArray([rand(N) for ii=1:M])
	dpo = BorderedArray(dpoguess, 2.)

dres = probSh(po, dpo)

# computation of finite differences
poguess2 = VectorOfArray([po.u[ii] .+ δ .* dpo.u[ii] for ii=1:M])
po2 = BorderedArray(poguess2, po.p + δ * dpo.p)
res2 = probSh(po2)

dresfd = BorderedArray((res2.u - res.u) ./δ, (res2.p - res.p)/δ)

@test dresfd.p - dres.p ≈ 0.0
@test norm(dresfd.u - dres.u, Inf) < 10δ

# use of AbstractArray structure
pov = vcat(vec(po.u), po.p)
dpov = vcat(vec(dpo.u), dpo.p)
resv = probSh(pov)
@test norm(res.u[1] - resv[1:N] , Inf) ≈ 0
dres = probSh(pov, dpov)
resfd = (probSh(pov .+ δ .* dpov) .- resv) ./ δ
@test norm(resfd - dres, Inf) < 10δ
####################################################################################################
# test the jacobian of the multiple shooting functional using nonLinear flow
vf(x) = x.^2
flow(x, t) = x ./ (1 .- t .* x)
dflow(x, dx, t) = (flow(x, t), dx ./ (1 .- t .* x).^2)
section(x) = dot(x[1], ones(length(x[1])))

fl = PALC.Flow(vf, flow, dflow)

probSh = PALC.ShootingProblem(fl,
	LinRange(0,1,M+1) |> diff ,
	x -> section(x))

poguess = VectorOfArray([rand(N) for ii=1:M])
	po = BorderedArray(poguess, 1.)

res = probSh(po)

dpoguess = VectorOfArray([rand(N) for ii=1:M])
	dpo = BorderedArray(dpoguess, 2.)

dres = probSh(po, dpo)

# computation of finite differences
poguess2 = VectorOfArray([po.u[ii] .+ δ .* dpo.u[ii] for ii=1:M])
po2 = BorderedArray(poguess2, po.p + δ * dpo.p)
res2 = probSh(po2)

dresfd = BorderedArray((res2.u - res.u) ./δ, (res2.p - res.p)/δ)

@test dresfd.p - dres.p ≈ 0.0
@test norm(dresfd.u - dres.u, Inf) < 10δ

# use of AbstractArray structure
pov = vcat(vec(po.u), po.p)
dpov = vcat(vec(dpo.u), dpo.p)
resv = probSh(pov)
@test norm(res.u[1] - resv[1:N] , Inf) ≈ 0
dres = probSh(pov, dpov)
resfd = (probSh(pov .+ δ .* dpov) .- resv) ./ δ
@test norm(resfd - dres, Inf) < 10δ
####################################################################################################
# test the hyperplane projections for Poincare Shooting
M = 3
normals = [rand(50) for ii=1:M]
for ii=1:M
	normals[ii] /= norm(normals[ii])
end
centers = [rand(50) for ii=1:M]

hyper = PALC.HyperplaneSections(normals, centers)

x = 1:50 |>collect .|> Float64 |> vec
x = rand(50)
xb = PALC.R(hyper, x, 1)
# test
for ii=1:M
	x2 = PALC.E(hyper, xb, ii)
	# test that x2 in Sigma2
	@test dot(x2 - centers[ii], normals[ii]) < 1e-14
	# that we have Rk∘Ek = Id and Ek∘Rk = IdΣ
	@test PALC.R(hyper, x2, ii) ≈ xb
end

# test of the derivatives
dx = rand(50)
δ = 1e-9
_out1 = (PALC.R(hyper, x .+ δ .* dx, 1) - PALC.R(hyper, x, 1)) ./ δ
_out2 = zero(_out1)
_out2 = PALC.dR!(hyper, _out2, dx, 1)

@test norm(_out2 - _out1, Inf) < 1e-6

dx = rand(49)
_out1 = (PALC.E(hyper, xb .+ δ .* dx, 1) - PALC.E(hyper, xb, 1)) ./ δ
_out2 = PALC.dE(hyper, dx, 1)
@test norm(_out2 - _out1, Inf) < 1e-6
