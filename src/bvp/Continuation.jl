function BK.continuation(prob::BVPBifProblem,
                      alg::BK.AbstractContinuationAlgorithm,
                      contparams::BK.ContinuationPar;
                      linear_algo = nothing,
                      bothside::Bool = false,
                      kwargs...)
    BK._continuation(prob, alg, contparams; kind = BK.BoundaryValueProblemCont(), linear_algo, bothside, kwargs...)
end

function BK.continuation(prob::BVPBifProblem,
                      alg::BK.DefCont,
                      contparams::BK.ContinuationPar;
                      kwargs...)
    BK._deflated_continuation(prob, alg, contparams; kind = BK.BoundaryValueProblemCont(), kwargs...)
end