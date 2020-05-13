using Test, PseudoArcLengthContinuation, ForwardDiff, RecursiveArrayTools, LinearAlgebra
const PALC = PseudoArcLengthContinuation
N = 10
M = 5
δ = 1e-8
####################################################################################################
# test the jacobian of the multiple shooting functional using Linear flow
# TODO do example with A matrix and exp(At)
vf(x, p) = x
flow(x, p, t) = exp(t) .* x
dflow(x, p, dx, t) = (t = t, u = flow(x, p, t), du = exp(t) .* dx)
section(x) = dot(x[1:N], ones(N))
section(x::BorderedArray) = section(vec(x.u[:,:]))
par = nothing

fl = PALC.Flow(vf, flow, dflow)

probSh = PALC.ShootingProblem(M, fl,
	LinRange(0, 1, M+1) |> diff,
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
# test the jacobian of the multiple shooting functional using nonLinear flow
vf(x, p) = x.^2
flow(x, p, t) = x ./ (1 .- t .* x)
dflow(x, p, dx, t) = (t = t, u = flow(x, p, t), du = dx ./ (1 .- t .* x).^2)

fl = PALC.Flow(vf, flow, dflow)

probSh = PALC.ShootingProblem(M, fl,
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

# test of the derivatives of E and R
dx = rand(50)
_out1 = (PALC.R(hyper, x .+ δ .* dx, 1) - PALC.R(hyper, x, 1)) ./ δ
_out2 = zero(_out1)
_out2 = PALC.dR!(hyper, _out2, dx, 1)

@test norm(_out2 - _out1, Inf) < 1e-5

dx = rand(49)
_out1 = ForwardDiff.derivative(t -> PALC.E(hyper, xb .+ t .* dx,1), 0.)
_out2 = PALC.dE(hyper, dx, 1)
@test norm(_out2 - _out1, Inf) < 1e-12
