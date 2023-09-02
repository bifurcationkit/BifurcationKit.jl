function modify_po_2params_finalise(prob, kwargs, probMA)
    update_section_every_step = prob.update_section_every_step
    _finsol = get(kwargs, :finaliseSolution, nothing)
    if isnothing(_finsol)
        return (Z, tau, step, contResult; kF...) ->
            begin
                z = Z.u
                # we first check that the continuation step was successful
                # if not, we do not update the problem with bad information
                success = converged(get(kF, :state, nothing)) && ~get(kF, :bisection, false)
                if success && mod_counter(step, update_section_every_step) == 1
                    updatesection!(prob, getvec(z, probMA), setparam(contResult, Z.p))
                end
                return true
            end
    else
        return (Z, tau, step, contResult; kF...) ->
            begin
                # we first check that the continuation step was successful
                # if not, we do not update the problem with bad information!
                success = converged(get(kF, :state, nothing)) && ~get(kF, :bisection, false)
                if success && mod_counter(step, update_section_every_step) == 1
                    z = Z.u
                    updatesection!(prob, getvec(z, probMA), setparam(contResult, Z.p))
                end
                return _finsol(Z, tau, step, contResult; prob = prob, kF...)
            end
    end
end

function modify_po_2params_finalise(prob::PeriodicOrbitOCollProblem, kwargs, probMA)
    update_section_every_step = prob.update_section_every_step
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
                oldmesh = getTimes(prob) .* getPeriod(prob, getvec(z, probMA), nothing)
                adapt = computeError!(prob, getvec(z, probMA);
                        verbosity = prob.verboseMeshAdapt,
                        par = setparam(contResult, z.p),
                        K = prob.K)
                if ~adapt.success
                    return false
                end
            end
            if success && mod_counter(step, update_section_every_step) == 1
                updatesection!(prob, getvec(z, probMA), setparam(contResult, Z.p))
            end
            if isnothing(_finsol)
                return true
            else
                return _finsol(Z, tau, step, contResult; prob = prob, kF...)
            end
        end
    return _finsol2
end