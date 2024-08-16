module JLD2Ext
    using JLD2, BifurcationKit
    import BifurcationKit: save_to_file, AbstractContinuationIterable, ContResult
    include("save.jl")
end
