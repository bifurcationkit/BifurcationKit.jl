# using Revise, Plots
using Test
using BifurcationKit, Setfield, ForwardDiff

const BK = BifurcationKit


####################################################################################################
_eve = BK.ContinuousEvent(2, (iter, state) -> (getp(state)+2, getx(state)[1]-1))
_eved = BK.DiscreteEvent(1, (iter, state) -> getp(state)>-2)
@test BK.hasCustomLabels(_eve) == false
@test BK.computeEigenElements(_eve) == false
BK.labels(_eve, 1)

_eve = BK.ContinuousEvent(2, (iter, state) -> (getp(state)+2, getx(state)[1]-1), ("event1", "event2"))
BK.labels(_eve, 1)
BK.labels(_eve, [])

SetOfEvents()
SetOfEvents(nothing)
SetOfEvents(_eve)
SetOfEvents(_eved)
BK.split_events(_eve, _eved, nothing)
####################################################################################################
function testBranch(br)
	# test if stability works
	# test that stability corresponds
	out = true
	for ii in eachindex(br.branch)
		if ~isempty(br.eig)
			@test br.eig[ii].eigenvals == eigenvals(br, br.eig[ii].step, false)
			# compute number of unstable eigenvalues
			isstable, n_u, n_i = BK.isStable(br.contparams, br.eig[ii].eigenvals)
			# test that the stability matches the one in eig
			@test br[ii].n_unstable == n_u
			@test br[ii].stable  == isstable
			# test that the field `step` match in the structure
			@test br.branch[ii][end] == br.eig[ii].step
		end
		# we test that step = ii-1
		@test br.branch[ii][end] == ii-1
	end
	# test about bifurcation points
	for bp in br.specialpoint
		id = bp.idx
		if isempty(br.eig) == false && bp.type ∈ [:fold,:hopf,:bp,:nd, :none, :ns, :pd, :bt, :cusp, :gh, :zh, :hh]
			# test that the states marked as bifurcation points are always after true bifurcation points
			@test abs(br[id].n_unstable - br[id-1].n_unstable) > 0
		end
		# test that the bifurcation point belongs to the interval
		@test bp.interval[1] <= bp.param <= bp.interval[2]
		# test that bp.param = br[id].param
		@test bp.param == br[id].param
	end
end
####################################################################################################
# test vector field for event detection
function Feve(X, p)
	p1, p2, k = p
	x, y = X
	out = similar(X)
	out[1] = p1 + x - y - x^k/k
	out[2] = p1 + y + x - 2y^k/k
	out
end

Jeve(X, p) = ForwardDiff.jacobian(z -> Feve(z,p), X)

par = (p1 = -3., p2=-3., k=3)

opts0 = ContinuationPar(dsmax = 0.1, ds = 0.001, maxSteps = 1000, pMin = -3., pMax = 4.0, saveSolEveryStep = 1, newtonOptions = NewtonPar(tol = 1e-10, verbose = false, maxIter = 5), detectBifurcation = 3, detectEvent = 0, nInversion = 8, dsminBisection = 1e-9, maxBisectionSteps = 15, detectFold=false, plotEveryStep = 10)
	br0, = continuation(Feve, Jeve, -2ones(2), par, (@lens _.p1), opts0;
		plot = false, verbosity = 0,
		recordFromSolution = (x, p) -> x[1],
		)
testBranch(br0)
# plot(br0, plotspecialpoints=true)
####################################################################################################
opts = ContinuationPar(opts0; saveSolEveryStep = 1, detectBifurcation = 0, detectEvent = 2, dsminBisection = 1e-9, maxBisectionSteps = 15)
	br, = continuation(Feve, Jeve, -2ones(2), par, (@lens _.p1), opts;
		plot = false, verbosity = 0,
		recordFromSolution = (x, p) -> x[1],
		)

# using PrettyTables
# pretty_table(br.branch[1:40])

# arguments for continuation
args = (Feve, Jeve, -2ones(2), par, (@lens _.p1), opts)
kwargs = (plot = false, verbosity = 0, recordFromSolution = (x,p) -> x[1], linearAlgo = MatrixBLS(),)

br, = continuation(args...; kwargs...,
	event = BK.ContinuousEvent(1, (iter, state) -> getp(state)+2),)
@test length(br.specialpoint) == 3
@test br.specialpoint[1].type==:userC
testBranch(br)

br, = continuation(args...; kwargs...,
	event = BK.ContinuousEvent(2, (iter, state) -> (getp(state)+2, getx(state)[1]-1)),)
@test length(br.specialpoint) == 6
@test br.specialpoint[1].type==Symbol("userC-1")
testBranch(br)

br, = continuation(args...; kwargs...,
	event = BK.ContinuousEvent(2, (iter, state) -> (getp(state)+2, getx(state)[1]-1),("User-2.0", "User1.0"))
	)
@test length(br.specialpoint) == 6
@test br.specialpoint[1].type==Symbol("User-2.0")
testBranch(br)

br, = continuation(args...; kwargs...,
	event = BK.SaveAtEvent((-2., 0., 1.))
	)
@test length(br.specialpoint) == 5
@test br.specialpoint[1].type==Symbol("save-1")
testBranch(br)
####################################################################################################
br, = continuation(args...; kwargs...,
	event = BK.DiscreteEvent(1, (iter, state) -> getp(state)>-2),
	)
@test length(br.specialpoint) == 3
@test br.specialpoint[1].type==:userD
testBranch(br)

br, = continuation(args...; kwargs...,
	event = BK.DiscreteEvent(2, (iter, state) -> (getp(state)> -2,getp(state)>0)),
	)
@test length(br.specialpoint) == 4
@test br.specialpoint[1].type==Symbol("userD-1")
testBranch(br)


br, = continuation(args...; kwargs...,
	event = BK.FoldDetectEvent,
	)
@test length(br.specialpoint) == 4
@test br.specialpoint[1].type==Symbol("fold")
testBranch(br)

br, = continuation(args...; kwargs...,
	event = BK.DiscreteEvent(2, (iter, state) -> (getp(state)> -2,getp(state)>0),("userD-2", "UserD0")),
	)
@test length(br.specialpoint) == 4
@test br.specialpoint[1].type==Symbol("userD-2")
@test br.specialpoint[4].type==Symbol("UserD0")
testBranch(br)
####################################################################################################
br, = continuation(args...; kwargs...,
	event = BK.BifDetectEvent, plot=false)
@test length(br.specialpoint) == 6

for (bp, bp0) in zip(br.specialpoint, br0.specialpoint)
	@test bp.type == bp0.type
	@test bp.x ≈ bp0.x
	@test bp.idx == bp0.idx
end
testBranch(br)
####################################################################################################
br, = continuation(args...; kwargs..., verbosity = 0,
	event = BK.PairOfEvents(BK.ContinuousEvent(1, (iter, state) -> getp(state)),
	BK.DiscreteEvent(1, (iter, state) -> getp(state)>-2)),)
@test length(br.specialpoint) == 4
@test br.specialpoint[1].type==Symbol("userD-1")
@test br.specialpoint[4].type==Symbol("userC-1")
testBranch(br)

br, = continuation(args...; kwargs..., verbosity = 0,
	event = BK.PairOfEvents(
		BK.ContinuousEvent(2, (iter, state) -> (getp(state)-2, getp(state)-2.5)),
		BK.DiscreteEvent(2, (iter, state) -> (getp(state)> -2, getp(state)>-1)) )
	)
@test length(br.specialpoint) == 8
@test br.specialpoint[1].type==Symbol("userD-1")
@test br.specialpoint[4].type==Symbol("userD-2")
@test br.specialpoint[5].type==Symbol("userC-1")
testBranch(br)

br, = continuation(args...; kwargs...,
	event = BK.PairOfEvents(BK.ContinuousEvent(1, (iter, state) -> getp(state)+2),
		BK.BifDetectEvent),)
@test length(br.specialpoint) == 9
@test br.specialpoint[1].type==Symbol("userC-1")
testBranch(br)

br, = continuation(args...; kwargs...,
	event = BK.PairOfEvents(BK.FoldDetectEvent, BK.DiscreteEvent(1, (iter, state) -> getp(state)>2)),
	)
@test length(br.specialpoint) == 7
@test br.specialpoint[1].type==Symbol("fold")
@test br.specialpoint[3].type==Symbol("userD-1")
testBranch(br)
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
@test length(br.specialpoint) == 11
@test br.specialpoint[1].type==Symbol("bp")
@test br.specialpoint[5].type==Symbol("userC1")
@test br.specialpoint[6].type==Symbol("userC2-1")
testBranch(br)

evd1 = BK.DiscreteEvent(1, (iter, state) -> getp(state)>0)
evd2 = BK.DiscreteEvent(2, (iter, state) -> (getp(state)> -2,getp(state)>-1))
eves2 = BK.SetOfEvents(ev1, ev2, evd1, evd2)
br, = continuation(args2...; kwargs...,
	event = eves2,
	)
	# plot(br, legend = :bottomright)
@test length(br.specialpoint) == 10
@test br.specialpoint[1].type==Symbol("userD2-1")
@test br.specialpoint[6].type==Symbol("userC1")
@test br.specialpoint[7].type==Symbol("userC2-1")
testBranch(br)

eves3 = SetOfEvents(eves1, eves2)

@set! kwargs.verbosity = 0
	br, = continuation(args2...; kwargs...,
		event = eves3,
		)
	# plot(br, legend = :bottomright)
@test length(br.specialpoint) == 16
@test br.specialpoint[1].type==Symbol("userD3-1")
@test br.specialpoint[2].type==Symbol("bp")
@test br.specialpoint[11].type==Symbol("userC2-1")
testBranch(br)
