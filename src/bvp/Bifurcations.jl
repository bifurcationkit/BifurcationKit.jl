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