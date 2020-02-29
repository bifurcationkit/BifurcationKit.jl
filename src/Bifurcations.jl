####################################################################################################
function normalize(x)
	out = copyto!(similar(x), x)
	rmul!(out, norm(x))
	return out
end
####################################################################################################
"""
This function checks whether the solution with eigenvalues `eigvalues` is stable and also compute the number of unstable eigenvalues with nonzero imaginary part
"""
function is_stable(contparams::ContinuationPar, eigvalues)::Tuple{Bool, Int64, Int64}
	# the return type definition above is to remove type instability in continuation
	# numerical precision for deciding if an eigenvalue is above a threshold
	precision = contparams.precisionStability

	# update number of unstable eigenvalues
	n_unstable = mapreduce(x -> real(x) > precision, +, eigvalues)

	# update number of unstable eigenvalues with nonzero imaginary part
	n_imag = mapreduce(x -> (abs(round(imag(x), digits = 15)) > precision) * (round(real(x), digits = 15) > precision), +, eigvalues)

	return n_unstable == 0, n_unstable, n_imag
end
####################################################################################################
interval(a, b) = (min(a, b), max(a, b))

# Test function for Fold bifurcation
@inline detectFold(p1, p2, p3) = (p3 - p2) * (p2 - p1) < 0

function locateFold!(contparams::ContinuationPar, contres::ContResult, z, tau, normC, printsolution, verbosity)
	branch = contres.branch
	# Fold point detection based on continuation parameter monotony
	if contparams.detectFold && size(branch)[2] > 2 && detectFold(branch[1, end-2], branch[1, end-1], branch[1, end])
		(verbosity > 0) && printstyled(color=:red, "!! Fold bifurcation point in $(interval(branch[1, end-1],branch[1, end])) \n")
		push!(contres.foldpoint, (
				type = :fold,
				idx = length(branch)-1,
				param = branch[1, end-1],
				norm = normC(z.u),
				printsol = branch[2, end-1], #printsolution(z.u, z.p),
				u = _copy(z.u), tau = normalize(tau.u),
				ind_bif = 0,
				step = length(branch)-1,
				status = :guess,
				δ = (0, 0)))
		detected = true
	end
end

locateFold!(contres::ContResult, iter::PALCIterable, state::PALCStateVariables) = locateFold!(iter.contParams, contres, solution(state), state.tau_old, iter.normC, iter.printSolution, iter.verbosity)

####################################################################################################
"""
Function for coarse detection of bifurcation points.
"""
function getBifurcationType(contparams::ContinuationPar{T,S,E}, state::PALCStateVariables, normC, printsolution, verbosity, status::Symbol) where {T,S,E}

	detected = false

	# get current number of unstable eigenvalues and
	# unstable eigenvalues with nonzero imaginary part
	n_unstable = state.n_unstable[1]
	n_imag = state.n_imag[1]

	#previous values
	n_unstable_prev = state.n_unstable[2]
	n_imag_prev = state.n_imag[2]

	# computation of the index of the bifurcating eigenvalue
	ind_bif = n_unstable
	if n_unstable < n_unstable_prev; ind_bif += 1; end

	tp = :none

	δn_unstable = abs(n_unstable - n_unstable_prev)
	δn_imag		= abs(n_imag - n_imag_prev)

	# codim 1 bifurcation point detection based on eigenvalue distribution

	if δn_unstable == 1
	# In this case, only a single eigenvalue crossed the imaginary axis
	# Either it is a Branch Point
		if δn_imag == 0
			tp = :bp
		elseif δn_imag == 1
	# Either it is a Hopf bifurcation for a Complex valued system
			tp = contparams.newtonOptions.eigsolver isa AbstractFloquetSolver ? :pd : :hopf
		else # I dont know what bifurcation this is
			tp = :bp
		end
		detected = true
	elseif δn_unstable == 2
		if δn_imag == 2
			tp = contparams.newtonOptions.eigsolver isa AbstractFloquetSolver ? :ns : :hopf
		else
			tp = :bp
		end
		detected = true
	elseif δn_unstable > 2
		tp = :nd
		detected = true
	end

	if δn_unstable < δn_imag
		@error "Error in eigenvalues computation. It seems an eigenvalue is missing, probably `conj(λ)` for some already computed eigenvalue λ. This makes the identification of bifurcation points erroneous. You should increase the number of requested eigenvalues."
		tp = :nd
		detected = true
	end

	if detected
		# record information about the bifurcation point
		param_bif = (
			type = tp,
			idx = state.step+1,			# this is the index in br.eig
			param = getp(state),
			norm = normC(getu(state)),
			printsol = printsolution(getu(state), getp(state)),
			u = _copy(getu(state)),
			tau = normalize(state.tau_old.u),
			ind_bif = ind_bif,
			step = state.step,
			status = status,
			δ = (n_unstable - n_unstable_prev, n_imag - n_imag_prev))
		(verbosity>0) && printstyled(color=:red, "!! $(tp) Bifurcation point around p ≈ $(getp(state)), δn_unstable = $δn_unstable, δn_imag = $δn_imag \n")
	end

	return detected, param_bif
end

function closesttozero(ev)
	I = sortperm(abs.(real.(ev)))
	return ev[I]
end

"""
Function to locate precisely bifurcation points using bisection.
"""
function locateBifurcation!(iter::PALCIterable, _state::PALCStateVariables, verbose::Bool = true)
	@assert detectBifucation(_state) "No bifucation detected for the state"
	verbose && println("----> Entering [Loc-Bif], state.n_unstable = ", _state.n_unstable)

	# number of unstable eigenvalues
	(n2, n1) = _state.n_unstable
	if n1 == -1 || n2 == -1 return :none end

	# we create a new state for stepping through the continuation routine
	state = copy(_state)

	# iter = @set iter.contParams.newtonOptions.verbose = false

	verbose && println("----> [Loc-Bif] initial ds = ", _state.ds)

	# the bifurcation point is before the current state
	# so we want to first iterate backward with half step size
	# we turn off stepsizecontrol because it would not make a
	# bisection otherwise
	state.ds /= -1
	state.step = 0
	state.stepsizecontrol = false

	next = (state, state)

	if abs(state.ds) < iter.contParams.dsmin; return :none; end

	# record sequence of unstable eigenvalue number
	nunstbls = [n2]
	nimags   = [state.n_imag[1]]

	verbose && println("----> [Loc-Bif] state.ds = ", state.ds)

	# we put this to be able to reference it at the end of this function
	# we don't know its type yet
	eiginfo = nothing

	# we compute the number of changes in n_unstable
	n_inversion = 0
	status = :guess

	while next !== nothing &&
			abs(state.ds) >= iter.contParams.dsminBisection &&
			state.step < iter.contParams.maxBisectionSteps &&
			n_inversion <= iter.contParams.nInversion

		if state.isconverged == false
			@error "----> Newton failed when locating bifurcation!"
			break
		 end

		# we get the current state
		(i, state) = next

		eiginfo, _, n_unstable, n_imag = computeEigenvalues(iter, state)
		updatestability!(state, n_unstable, n_imag)
		push!(nunstbls, n_unstable)
		push!(nimags, n_imag)

		if nunstbls[end] == nunstbls[end-1]
			# bifurcation point still after current state, keep going
			state.ds /= 2
		else
			# we passed the bifurcation point, reverse continuation
			state.ds /= -2
			n_inversion += 1
		end

		verbose &&	printstyled(color=:blue, "----> $(state.step) - [Loc-Bif] (n1, nc, n2) = ",(n1, nunstbls[end], n2), ", ds = $(state.ds), p = ", getp(state), ", #reverse = ", n_inversion,"\n Eigenvalues:\n")
		verbose && Base.display(closesttozero(eiginfo[1])[1:5])

		# body
		next = iterate(iter, state; _verbosity = 0)
	end
	verbose && printstyled(color=:red, "----> Found at p = ", getp(state), ", δn = ", abs(2nunstbls[end]-n1-n2),", δim = ",abs(2nimags[end]-sum(state.n_imag))," from p = ",getp(_state),"\n")

	# update current state
	if isodd(n_inversion)
		status = :converged
		copyto!(_state.z_pred, state.z_pred)
		copyto!(_state.z_old,  state.z_old)
		copyto!(_state.tau_new, state.tau_new)
		copyto!(_state.tau_old, state.tau_old)

		_state.eigvals = eiginfo[1]
		if iter.contParams.saveEigenvectors
			_state.eigvecs = eiginfo[2]
		end

		# to prevent bifurcation detection, update the following numbers carefully
		_state.n_unstable = (state.n_unstable[2], _state.n_unstable[2])
		_state.n_imag = (state.n_imag[2], _state.n_imag[2])
	else
		@warn "Bisection failed to locate bifurcation point precisely. Fall back to original guess for the bifurcation point. Number of Bisections = $n_inversion"
	end
	verbose && println("----> Leaving [Loc-Bif]")
	return status
end
