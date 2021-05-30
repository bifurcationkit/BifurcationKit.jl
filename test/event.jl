# using Revise, Plots
using Test
using BifurcationKit, Setfield, ForwardDiff

const BK = BifurcationKit
####################################################################################################
# test vector field for event detection

function F(X, p)
	p1, p2, k = p
	x, y = X
	out = similar(X)
	out[1] = p1 + x - y - x^k/k
	out[2] = p1 + y + x - 2y^k/k
	out
end

J(X, p) = ForwardDiff.jacobian(z -> F(z,p), X)

par = (p1 = -3., p2=-3., k=3)

opts0 = ContinuationPar(dsmax = 0.1, ds = 0.001, maxSteps = 1000, pMin = -3., pMax = 4.0, saveSolEveryStep = 1, newtonOptions = NewtonPar(tol = 1e-10, verbose = false, maxIter = 5), detectBifurcation = 3, detectEvent = 0, nInversion = 8, dsminBisection = 1e-9, maxBisectionSteps = 15, detectFold=false, tolBisectionEvent = 1e-24, plotEveryStep = 10)
	br0, = continuation(F, J, -2ones(2), par, (@lens _.p1), opts0;
		plot = false, verbosity = 0,
		printSolution = (x, p) -> x[1],
		)

# plot(br0, plotspecialpoints=true)
####################################################################################################
opts = ContinuationPar(opts0; saveSolEveryStep = 1, detectBifurcation = 0, detectEvent = 2, dsminBisection = 1e-9, maxBisectionSteps = 15)
	br, = continuation(F, J, -2ones(2), par, (@lens _.p1), opts;
		plot = false, verbosity = 0,
		printSolution = (x, p) -> x[1],
		)

# using PrettyTables
# pretty_table(br.branch[1:40])

# arguments for continuation
args = (F, J, -2ones(2), par, (@lens _.p1), opts)
kwargs = (plot = false, verbosity = 0, printSolution = (x,p) -> x[1], linearAlgo = MatrixBLS(),)

br, = continuation(args...; kwargs...,
	event = BK.ContinuousEvent(1, (iter, state) -> getp(state)+2),)
@test br.specialpoint[1].type==:userC

br, = continuation(args...; kwargs...,
	event = BK.ContinuousEvent(2, (iter, state) -> (getp(state)+2, getx(state)[1]-1)),)
@test br.specialpoint[1].type==Symbol("userC-1")

br, = continuation(args...; kwargs...,
	event = BK.ContinuousEvent(2, (iter, state) -> (getp(state)+2, getx(state)[1]-1),("User-2.0", "User1.0"))
	)
@test br.specialpoint[1].type==Symbol("User-2.0")

br, = continuation(args...; kwargs...,
	event = BK.SaveAtEvent((-2., 0., 1.))
	)
@test br.specialpoint[1].type==Symbol("save-1")
####################################################################################################
br, = continuation(args...; kwargs...,
	event = BK.DiscreteEvent(1, (iter, state) -> getp(state)>-2),
	)
@test br.specialpoint[1].type==:userD

br, = continuation(args...; kwargs...,
	event = BK.DiscreteEvent(2, (iter, state) -> (getp(state)> -2,getp(state)>0)),
	)
@test br.specialpoint[1].type==Symbol("userD-1")

br, = continuation(args...; kwargs...,
	event = BK.FoldDetectEvent,
	)
@test br.specialpoint[1].type==Symbol("fold")

br, = continuation(args...; kwargs...,
	event = BK.BifDetectEvent, plot=false)

for (bp, bp0) in zip(br.specialpoint, br0.specialpoint)
	@test bp.type == bp0.type
	@test bp.x ≈ bp0.x
end

br, = continuation(args...; kwargs...,
	event = BK.DiscreteEvent(2, (iter, state) -> (getp(state)> -2,getp(state)>0),("userD-2", "UserD0")),
	)
@test br.specialpoint[1].type==Symbol("userD-2")
@test br.specialpoint[4].type==Symbol("UserD0")
####################################################################################################
br, = continuation(args...; kwargs..., verbosity = 0,
	event = BK.PairOfEvents(BK.ContinuousEvent(1, (iter, state) -> getp(state)),
	BK.DiscreteEvent(1, (iter, state) -> getp(state)>-2)),)
@test br.specialpoint[1].type==Symbol("userD-1")
@test br.specialpoint[4].type==Symbol("userC-1")

br, = continuation(args...; kwargs..., verbosity = 0,
	event = BK.PairOfEvents(
		BK.ContinuousEvent(2, (iter, state) -> (getp(state)-2, getp(state)-2.5)),
		BK.DiscreteEvent(2, (iter, state) -> (getp(state)> -2, getp(state)>-1)) )
	)
@test br.specialpoint[1].type==Symbol("userD-1")
@test br.specialpoint[4].type==Symbol("userD-2")
@test br.specialpoint[5].type==Symbol("userC-1")

br, = continuation(args...; kwargs...,
	event = BK.PairOfEvents(BK.ContinuousEvent(1, (iter, state) -> getp(state)+2),
		BK.BifDetectEvent),)
@test br.specialpoint[1].type==Symbol("userC-1")

br, = continuation(args...; kwargs...,
	event = BK.PairOfEvents(BK.FoldDetectEvent, BK.DiscreteEvent(1, (iter, state) -> getp(state)>2)),
	)
@test br.specialpoint[1].type==Symbol("fold")
@test br.specialpoint[3].type==Symbol("userD-1")
####################################################################################################
ev1 = BK.ContinuousEvent(1, (iter, state) -> getp(state)-1)
ev2 = BK.ContinuousEvent(2, (iter, state) -> (getp(state)-2, getp(state)-2.5))
ev3 = BK.BifDetectEvent
eves1 = BK.SetOfEvents(ev1, ev2, ev3)

args2 = @set args[end].detectEvent = 2
@set! kwargs.verbosity = 0
	br, = continuation(args2...; kwargs...,
		event = eves1,
		)
	# plot(br, legend = :bottomright)
@test br.specialpoint[1].type==Symbol("bp")
@test br.specialpoint[5].type==Symbol("userC1")
@test br.specialpoint[6].type==Symbol("userC2-1")

evd1 = BK.DiscreteEvent(1, (iter, state) -> getp(state)>0)
evd2 = BK.DiscreteEvent(2, (iter, state) -> (getp(state)> -2,getp(state)>-1))
eves2 = BK.SetOfEvents(ev1, ev2, evd1, evd2)
br, = continuation(args2...; kwargs...,
	event = eves2,
	)
	# plot(br, legend = :bottomright)
@test br.specialpoint[1].type==Symbol("userD2-1")
@test br.specialpoint[6].type==Symbol("userC1")
@test br.specialpoint[7].type==Symbol("userC2-1")

eves3 = SetOfEvents(eves1, eves2)

@set! kwargs.verbosity = 0
	br, = continuation(args2...; kwargs...,
		event = eves3,
		)
	# plot(br, legend = :bottomright)
@test br.specialpoint[1].type==Symbol("userD3-1")
@test br.specialpoint[2].type==Symbol("bp")
@test br.specialpoint[11].type==Symbol("userC2-1")
