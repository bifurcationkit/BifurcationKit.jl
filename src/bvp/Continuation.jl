function BK.continuation(prob::BVPBifProblem,
                      alg::BK.AbstractContinuationAlgorithm,
                      contparams::BK.ContinuationPar;
                      kwargs...)
    BK._continuation(prob, alg, contparams; kind = BK.BoundaryValueProblemCont(), kwargs...)
end

function BK.continuation(prob::BVPBifProblem,
                      alg::BK.DefCont,
                      contparams::BK.ContinuationPar;
                      kwargs...)
    BK._deflated_continuation(prob, alg, contparams; kind = BK.BoundaryValueProblemCont(), kwargs...)
end