function BK.get_bifurcation_type(it::BK.ContIterable{ <: BK.BoundaryValueProblemCont}, 
                                state, 
                                status::Symbol, 
                                interval::Tuple{T, T}, 
                                eig::BK.AbstractEigenSolver) where T
    known, specialpoint = BK._get_bifurcation_type(it, state, status, interval, eig)
    if specialpoint.type == :hopf
        BK.@reset specialpoint.type = :none
    end
    return known, specialpoint
end

# resolve method ambiguity with the `FoldEig` specialization in `codim2/MinAugFold.jl`
function BK.get_bifurcation_type(it::BK.ContIterable{ <: BK.BoundaryValueProblemCont},
                                state,
                                status::Symbol,
                                interval::Tuple{T, T},
                                eig::BK.FoldEig) where T
    known, specialpoint = BK._get_bifurcation_type(it, state, status, interval, eig.eigsolver)
    if specialpoint.type == :hopf
        BK.@reset specialpoint.type = :none
    end
    return known, specialpoint
end