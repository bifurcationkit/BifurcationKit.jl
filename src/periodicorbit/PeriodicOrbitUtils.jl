abstract type PeriodicOrbitAlgorithm end

"""
$(SIGNATURES)

This function returns several useful quantities regarding a Hopf bifurcation point. More precisely, it returns:
- the parameter value at which a Hopf bifurcation occurs
- the period of the bifurcated periodic orbit
- a guess for the bifurcated periodic orbit
- the equilibrium at the Hopf bifurcation point
- the eigenvector at the Hopf bifurcation point.

The arguments are
- `br`: the continuation branch which lists the Hopf bifurcation points
- `ind_hopf`: index of the bifurcation branch, as in `br.specialpoint`
- `eigsolver`: the eigen solver used to find the eigenvectors
- `M` number of time slices in the periodic orbit guess
- `amplitude`: amplitude of the periodic orbit guess
"""
function guessFromHopf(br, ind_hopf, eigsolver::AbstractEigenSolver, M, amplitude; phase = 0)
	hopfpoint = HopfPoint(br, ind_hopf)
	specialpoint = br.specialpoint[ind_hopf]

	# parameter value at the Hopf point
	p_hopf = hopfpoint.p[1]

	# frequency at the Hopf point
	ωH  = hopfpoint.p[end] |> abs

	# vec_hopf is the eigenvector for the eigenvalues iω
	vec_hopf = geteigenvector(eigsolver, br.eig[specialpoint.idx][2], specialpoint.ind_ev-1)
	vec_hopf ./=  norm(vec_hopf)

	 orbitguess = [real.(hopfpoint.u .+ amplitude .* vec_hopf .* exp(-2pi * complex(0, 1) .* (ii/(M-1) - phase))) for ii=0:M-1]

	return p_hopf, 2pi/ωH, orbitguess, hopfpoint, vec_hopf
end
####################################################################################################
# Amplitude of the u component of the cycle
amplitude(x::AbstractMatrix, n) =  maximum(x[1:n, :]) - minimum(x[1:n, :])

function amplitude(x::AbstractVector, n, M; ratio = 1)
	xc = reshape(x[1:end-1], ratio * n, M)
	amplitude(xc, n)
end

function maximumPOTrap(x::AbstractVector, n, M; ratio = 1)
	xc = reshape(x[1:end-1], ratio * n, M)
	maximum(x[1:n, :])
end
####################################################################################################
# functions to insert the problem into the user passed parameters
function modifyPOFinalise(prob, kwargs, updateSectionEveryStep)
	_finsol = get(kwargs, :finaliseSolution, nothing)
	_finsol2 = isnothing(_finsol) ? (z, tau, step, contResult; kwargs...) ->
		begin
			modCounter(step, updateSectionEveryStep) == 1 && updateSection!(prob, z.u, setParam(contResult, z.p))
			true
		end :
		(z, tau, step, contResult; prob = prob, kwargs...) ->
		begin
			modCounter(step, updateSectionEveryStep) == 1 && updateSection!(prob, z.u, setParam(contResult, z.p))
			_finsol(z, tau, step, contResult; prob = prob, kwargs...)
		end
	return _finsol2
end

function modifyPORecord(probPO, kwargs, par, lens)
	if :recordFromSolution in keys(kwargs)
		_recordsol0 = get(kwargs, :recordFromSolution, nothing)
		return _recordsol = (x, p; k...) -> _recordsol0(x, (prob = probPO, p = p); k...)
	else
		return _recordsol = (x, p; k...) -> (period = getPeriod(probPO, x, set(par, lens, p)),)
	end
end

function modifyPOPlot(probPO, kwargs)
	_plotsol = get(kwargs, :plotSolution, nothing)
	_plotsol2 = isnothing(_plotsol) ? (x, p; k...) -> nothing : (x, p; k...) -> _plotsol(x, (prob = probPO, p = p); k...)
end
