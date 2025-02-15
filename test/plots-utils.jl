using BifurcationKit
const BK = BifurcationKit
BK.set_plot_backend!(BK.BK_NoPlot())
BK.get_plot_vars(nothing, (:a, :b))
BK.get_color(:bp)
BK.get_color(:bp_nimp)
#############################
F0_simple(x, p) = p[1] .* x
opts = ContinuationPar(p_min = -3.)
prob = BK.BifurcationProblem(F0_simple, zeros(1), -1.5)
br = continuation(prob, PALC(), opts)

BK.get_axis_labels(1,1,br)
BK.get_axis_labels(:p,1,br)
BK.get_axis_labels(:p,:p,br)
BK.filter_bifurcations(br.specialpoint)