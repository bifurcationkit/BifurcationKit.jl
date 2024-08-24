module PlotsExt
    using Plots, BifurcationKit
    import BifurcationKit: plot_branch_cont,
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
                           BK_NoPlot, BK_Plots,
                           plotAllDCBranch,
                           plot_DCont_branch

    include("plot.jl")
end
