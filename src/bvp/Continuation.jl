function BK.continuation(prob::BVPBifProblem,
                      alg::BK.AbstractContinuationAlgorithm,
                      contparams::BK.ContinuationPar;
                      linear_algo = nothing,
                      bothside::Bool = false,
                      kwargs...)
    BK._continuation(prob, alg, contparams; kind = BK.BoundaryValueProblemCont(), linear_algo, bothside, kwargs...)
end