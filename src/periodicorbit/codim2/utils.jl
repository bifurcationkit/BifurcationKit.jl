function modifyPO_2ParamsFinalise(prob, kwargs, probMA)
    updateSectionEveryStep = prob.updateSectionEveryStep
    _finsol = get(kwargs, :finaliseSolution, nothing)
    if isnothing(_finsol)
        return (Z, tau, step, contResult; kF...) ->
            begin
                z = Z.u
                # we first check that the continuation step was successful
                # if not, we do not update the problem with bad information
                success = converged(get(kF, :state, nothing)) && ~get(kF, :bisection, false)
                if success && modCounter(step, updateSectionEveryStep) == 1
                    updateSection!(prob, getVec(z, probMA), setParam(contResult, Z.p))
                end
                return true
            end
    else
        return (Z, tau, step, contResult; kF...) ->
            begin
                # we first check that the continuation step was successful
                # if not, we do not update the problem with bad information!
                success = converged(get(kF, :state, nothing)) && ~get(kF, :bisection, false)
                if success && modCounter(step, updateSectionEveryStep) == 1
                    z = Z.u
                    updateSection!(prob, getVec(z, probMA), setParam(contResult, Z.p))
                end
                return _finsol(Z, tau, step, contResult; prob = prob, kF...)
            end
    end
end

function modifyPO_2ParamsFinalise(prob::PeriodicOrbitOCollProblem, kwargs, probMA)
    updateSectionEveryStep = prob.updateSectionEveryStep
    _finsol = get(kwargs, :finaliseSolution, nothing)
    _finsol2 = (Z, tau, step, contResult; kF...) ->
        begin
            # we first check that the continuation step was successful
            # if not, we do not update the problem with bad information
            success = converged(get(kF, :state, nothing)) && ~get(kF, :bisection, false)
            # mesh adaptation
            z = Z.u
            if success && prob.meshadapt
                oldsol = _copy(z)
                oldmesh = getTimes(prob) .* getPeriod(prob, getVec(z, probMA), nothing)
                adapt = computeError!(prob, getVec(z, probMA);
                        verbosity = prob.verboseMeshAdapt,
                        par = setParam(contResult, z.p),
                        K = prob.K)
                if ~adapt.success
                    return false
                end
            end
            if success && modCounter(step, updateSectionEveryStep) == 1
                updateSection!(prob, getVec(z, probMA), setParam(contResult, Z.p))
            end
            if isnothing(_finsol)
                return true
            else
                return _finsol(Z, tau, step, contResult; prob = prob, kF...)
            end
        end
    return _finsol2
end