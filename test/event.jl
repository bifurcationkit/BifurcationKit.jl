# using Revise
# using Plots
using Test
using BifurcationKit

const BK = BifurcationKit
###################################################################################################
# general interface
_eve = BK.ContinuousEvent(2, (iter, state) -> (getp(state)+2, getx(state)[1]-1))
_eved = BK.DiscreteEvent(1, (iter, state) -> getp(state)>-2)
@test BK.has_custom_labels(_eve) == false
@test BK.compute_eigenelements(_eve) == false
BK.labels(_eve, 1)

_eve = BK.ContinuousEvent(2, (iter, state) -> (getp(state)+2, getx(state)[1]-1), ("event1", "event2"))
BK.labels(_eve, 1)
BK.labels(_eve, [])

SetOfEvents()
SetOfEvents(nothing)
SetOfEvents(_eve)
SetOfEvents(_eved)
BK.split_events(_eve, _eved, nothing)
BK.has_custom_labels(BK.BifEvent(1,1))
####################################################################################################
function testBranch(br)
    # test if stability works
    # test that stability corresponds
    out = true
    for ii in eachindex(br.branch)
        if ~isempty(br.eig)
            @test br.eig[ii].eigenvals == eigenvals(br, br.eig[ii].step, false)
            # compute number of unstable eigenvalues
            isstable, n_u, n_i = BK.is_stable(br.contparams, br.eig[ii].eigenvals)
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
        if bp.type!=:endpoint
            id = bp.idx
            if isempty(br.eig) == false && bp.type ∈ [:fold, :hopf, :bp, :nd, :none, :ns, :pd, :bt, :cusp, :gh, :zh, :hh]
                # test that the states marked as bifurcation points are always after true bifurcation points
                @test abs(br[id].n_unstable - br[id-1].n_unstable) > 0
            end
            # test that the bifurcation point belongs to the interval
            @test bp.interval[1] <= bp.param <= bp.interval[2]
            # test that bp.param = br[id].param
            @test bp.param == br[id].param
        end
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

par = (p1 = -3., p2=-3., k=3)

opts0 = ContinuationPar(dsmax = 0.1, ds = 0.001, max_steps = 1000, p_min = -3., p_max = 4.0, save_sol_every_step = 1, newton_options = NewtonPar(tol = 1e-10, verbose = false, max_iterations = 5), detect_bifurcation = 3, detect_event = 0, n_inversion = 8, dsmin_bisection = 1e-9, max_bisection_steps = 15, detect_fold=false, plot_every_step = 10)

prob = BK.BifurcationProblem(Feve, -2ones(2), par, (@optic _.p1);
        record_from_solution = (x, p; k...) -> x[1])

br0 = continuation(prob, PALC(), opts0;
    plot = false, verbosity = 0,
    )
testBranch(br0)
# plot(br0, plotspecialpoints=true)
####################################################################################################
# continuous events
opts = ContinuationPar(opts0; save_sol_every_step = 1, detect_bifurcation = 0, detect_event = 2, dsmin_bisection = 1e-9, max_bisection_steps = 15)
br = continuation(prob, PALC(), opts;
    plot = false, verbosity = 0,
    )

# using PrettyTables
# pretty_table(br.branch[1:40])

# arguments for continuation
args = (BK.re_make(prob; record_from_solution = (x,p; k...) -> x[1]), PALC(), opts)
kwargs = (plot = false, verbosity = 0, linear_algo = MatrixBLS(),)

br = continuation(args...; kwargs...,
    verbosity = 0, # test printing
    event = BK.ContinuousEvent(1, (iter, state) -> getp(state)+2),)
@test length(br.specialpoint) == 4
@test br.specialpoint[1].type == :userC
testBranch(br)

br = continuation(args...; kwargs...,
    event = BK.ContinuousEvent(2, (iter, state) -> (getp(state)+2, getx(state)[1]-1)),)
@test length(br.specialpoint) == 7
@test br.specialpoint[1].type == Symbol("userC-1")
testBranch(br)

br = continuation(args...; kwargs...,
    event = BK.ContinuousEvent(2, (iter, state) -> (getp(state)+2, getx(state)[1]-1),("User-2.0", "User1.0"))
    )
@test length(br.specialpoint) == 7
@test br.specialpoint[2].type == Symbol("User-2.0")
testBranch(br)

# code for testing a single value
br = continuation(args...; kwargs...,
    event = BK.SaveAtEvent((-2.,), use_newton = true)
    )
@test length(br.specialpoint)-1 == 3
@test br.specialpoint[2].type == Symbol("save")
testBranch(br)

br = continuation(args...; kwargs...,
    event = BK.SaveAtEvent((-2., 0., 1.), use_newton = true)
    )
@test length(br.specialpoint) == 6
@test br.specialpoint[2].type == Symbol("save-1")
testBranch(br)

br = continuation(args...; kwargs...,
    event = BK.FoldDetectEvent,
    )
@test length(br.specialpoint) == 5
@test br.specialpoint[1].type == Symbol("fold")
testBranch(br)
####################################################################################################
# discrete events
br = continuation(args...; kwargs...,
    event = BK.DiscreteEvent(1, (iter, state) -> getp(state)>-2),
    )
@test length(br.specialpoint) == 4
@test br.specialpoint[1].type == :userD
testBranch(br)

br = continuation(args...; kwargs...,
    event = BK.DiscreteEvent(2, (iter, state) -> (getp(state)> -2,getp(state)>0)),
    )
@test length(br.specialpoint) == 5
@test br.specialpoint[1].type == Symbol("userD-1")
testBranch(br)

br = continuation(args...; kwargs...,
    event = BK.DiscreteEvent(2, (iter, state) -> (getp(state)> -2,getp(state)>0),("userD-2", "UserD0")),
    )
@test length(br.specialpoint) == 5
@test br.specialpoint[1].type == Symbol("userD-2")
@test br.specialpoint[4].type == Symbol("UserD0")
testBranch(br)
####################################################################################################
br = continuation(args...; kwargs...,
    event = BK.BifDetectEvent, plot=false)
@test length(br.specialpoint) == 7

for (bp, bp0) in zip(br.specialpoint, br0.specialpoint)
    @test bp.type == bp0.type
    @test bp.x ≈ bp0.x
    @test bp.idx == bp0.idx
end
testBranch(br)
####################################################################################################
# test PairOfEvents
br = continuation(args...; kwargs..., verbosity = 0,
    event = BK.PairOfEvents(
        BK.ContinuousEvent(1, (iter, state) -> getp(state)),
        BK.DiscreteEvent(1, (iter, state) -> getp(state)>-2)),
        )
@test length(br.specialpoint) == 5
@test br.specialpoint[1].type == Symbol("userD")
@test br.specialpoint[4].type == Symbol("userC")
testBranch(br)

br = continuation(args...; kwargs..., verbosity = 0,
    event = BK.PairOfEvents(
        BK.ContinuousEvent(2, (iter, state) -> (getp(state)-2, getp(state)-2.5)),
        BK.DiscreteEvent(2, (iter, state) -> (getp(state)> -2, getp(state)>-1)) )
    )
@test length(br.specialpoint) == 9
@test br.specialpoint[1].type == Symbol("userD-1")
@test br.specialpoint[4].type == Symbol("userD-2")
@test br.specialpoint[5].type == Symbol("userC-1")
testBranch(br)

br = continuation(args...; kwargs...,
    event = BK.PairOfEvents(
        BK.ContinuousEvent(1, (iter, state) -> getp(state)+2),
        BK.BifDetectEvent),)
@test length(br.specialpoint) == 10
@test br.specialpoint[1].type == Symbol("userC")
testBranch(br)

br = continuation(args...; kwargs...,
    event = BK.PairOfEvents(
        BK.FoldDetectEvent, 
        BK.DiscreteEvent(1, (iter, state) -> getp(state)>2)),
    )
@test length(br.specialpoint) == 8
@test br.specialpoint[1].type == Symbol("fold")
@test br.specialpoint[3].type == Symbol("userD")
testBranch(br)
####################################################################################################
# test SetOfEvents
ev1 = BK.ContinuousEvent(1, (iter, state) -> getp(state)-1)
ev2 = BK.ContinuousEvent(2, (iter, state) -> (getp(state)-2, getp(state)-2.5))
ev3 = BK.BifDetectEvent
eves1 = BK.SetOfEvents(ev1, ev2, ev3)

args2 = @set args[end].detect_event = 2
@reset kwargs.verbosity = 0
br = continuation(args2...; kwargs...,
    event = eves1,
    )
# plot(br, legend = :bottomright)
@test length(br.specialpoint) == 12
@test br.specialpoint[1].type == Symbol("bp")
@test br.specialpoint[5].type == Symbol("userC1")
@test br.specialpoint[6].type == Symbol("userC2-1")
testBranch(br)

evd1 = BK.DiscreteEvent(1, (iter, state) -> getp(state)>0)
evd2 = BK.DiscreteEvent(2, (iter, state) -> (getp(state)> -2,getp(state)>-1))
eves2 = BK.SetOfEvents(ev1, ev2, evd1, evd2)
br = continuation(args2...; kwargs...,
    event = eves2,
    )
    # plot(br, legend = :bottomright)
@test length(br.specialpoint) == 11
@test br.specialpoint[1].type == Symbol("userD2-1")
@test br.specialpoint[6].type == Symbol("userC1")
@test br.specialpoint[7].type == Symbol("userC2-1")
testBranch(br)

eves3 = SetOfEvents(eves1, eves2)
@reset kwargs.verbosity = 0
br = continuation(args2...; kwargs...,
    event = eves3,
    )
# plot(br, legend = :bottomright)
@test length(br.specialpoint) == 17
@test br.specialpoint[1].type == Symbol("userD3-1")
@test br.specialpoint[2].type == Symbol("bp")
@test br.specialpoint[11].type == Symbol("userC2-1")
testBranch(br)

br = continuation(args2...; kwargs...,
    event = BK.SetOfEvents(
        ContinuousEvent(2, (it, state) -> (getp(state)-1, getp(state)-2), false, ("C1", "C2"), 0),
        DiscreteEvent(1, (it, state) -> getp(state) > 3, false, ("D",)),
        SaveAtEvent((0.001,.01))
    ),
    )