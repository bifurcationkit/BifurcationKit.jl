module PlotsExt
    using Plots, BifurcationKit
    import BifurcationKit: _plot_backend,
                           plot_branch_cont,
                           plot_periodic_potrap,
                           plot_periodic_shooting!,
                           plot_periodic_shooting,
                           plot_eigenvals,
                           compute_eigenelements,
                           get_lens_symbol,
                           _getfirstusertype,
                           ContResult,
                           DCResult,
                           AbstractBranchResult,
                           get_plot_vars,
                           get_axis_labels,
                           _hasstability,
                           filter_bifurcations,
                           get_color,
                           AbstractResult,
                           get_plot_backend,
                           set_plot_backend!,
                           BK_NoPlot, BK_Plots,
                           plotAllDCBranch,
                           plot_DCont_branch,
                           SolPeriodicOrbit,
                           TwoParamCont

    include("RecipesPlots.jl")
    include("plot.jl")

function __init__()
    set_plot_backend!(BK_Plots())
    return nothing
end
end
