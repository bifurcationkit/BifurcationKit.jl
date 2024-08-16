module GLMakieExt
    using GLMakie, BifurcationKit
    import BifurcationKit: plot, 
                           plot!,
                           hasbranch,
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
                           colorbif,
                           get_plot_backend,
                           BK_Makie

    # TODO block precompilation
    get_plot_backend() = BK_Makie()
    include("plot.jl")
end
