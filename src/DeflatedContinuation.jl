"""
$(TYPEDEF)

Structure which holds the parameters specific to Deflated continuation.

# Fields

$(TYPEDFIELDS)

"""
@with_kw_noshow struct DefCont{Tdo, Talg, Tps, Tas, Tud, Tk} <: AbstractContinuationAlgorithm
	"Deflation operator, `::DeflationOperator`"
	deflationOperator::Tdo = nothing
	"Used as a predictor, `::AbstractContinuationAlgorithm`. For example `PALC()`, `Natural()`,..."
	alg::Talg = PALC()
	"maximum number of (active) branches to be computed"
	maxBranches::Int = 100
	"whether to seek new (deflated) solution at every step"
	seekEveryStep::Int = 1
	"maximum number of deflated Newton iterations"
	maxIterDefOp::Int = 5
	"perturb function"
	perturbSolution::Tps = _perturbSolution
	"accept (solution) function"
	acceptSolution::Tas = _acceptSolution
	"function to update the deflation operator"
	updateDeflationOp::Tud = _updateDeflationOp
	"jacobian for deflated newton. Can be `DefProbCustomLinearSolver()`, or `Val(:autodiff)`, `Val(:fullIterative)`"
	jacobian::Tk = DefProbCustomLinearSolver()
end

# iterable which contains the options associated with Deflated Continuation
@with_kw struct DefContIterable{Tit, Talg <: DefCont}
	it::Tit						# replicate continuation iterator
	alg::Talg
end

"""

$(TYPEDEF)

Structure holding the result from deflated continuation.

$(TYPEDFIELDS)
"""
struct DCResult{Tprob, Tbr, Tit, Tsol, Talg} <: AbstractBranchResult
	"Bifurcation problem"
	prob::Tprob
	"Branches of solution"
	branches::Tbr
	"Continuation iterator"
	iter::Tit
	"Solutions"
	sol::Tsol
	"Algorithm"
	alg::Talg
end
Base.lastindex(br::DCResult) = lastindex(br.branches)
Base.getindex(br::DCResult, k::Int) = getindex(br.branches, k)
Base.length(br::DCResult) = length(br.branches)

# state specific to Deflated Continuation, it is updated during the continuation process
mutable struct DCState{T, Tstate}
	tmp::T
	state::Tstate
	isactive::Bool
	DCState(sol::T) where T = new{T, Nothing}(copy(sol), nothing, true)
	DCState(sol::T, state::ContState) where {T} = new{T, typeof(state)}(copy(sol), state, true)
end
# whether the branch is active
isActive(dc::DCState) = dc.isactive
# getters
getx(dc::DCState) = getx(dc.state)
getp(dc::DCState) = getp(dc.state)

function updatebranch!(dcIter::DefContIterable, dcstate::DCState, contResult::ContResult, defOp::DeflationOperator; current_param, step)
	isActive(dcstate) == false &&  return false, 0
	state = dcstate.state 		# continuation state
	it = dcIter.it 				# continuation iterator
	alg = dcIter.alg
	@unpack step, ds, θ = state
	@unpack verbosity = it
	state.z_pred.p = current_param

	getPredictor!(state, it)
	pbnew = reMake(it.prob; u0 = getx(state), params = setParam(it, current_param))
	sol1 = newton(pbnew, defOp, it.contParams.newtonOptions, alg.jacobian; normN = it.normC, callback = it.callbackN, iterationC = step, z0 = state.z)
	if converged(sol1)
		# record previous parameter (cheap) and update current solution
		copyto!(state.z.u, sol1.u); state.z.p = current_param
		state.z_old.p = current_param

		# Get tangent, it only mutates tau
		# getTangent!(state.τ, state.z_pred, state.z, it, ds, θ, it.tangentAlgo, verbosity)
		getPredictor!(state, it)

		# call user function to deal with DeflationOperator, allows to tackle symmetries
		alg.updateDeflationOp(defOp, sol1.u, current_param)

		# compute stability and bifurcation points
		computeEigenElements(it.contParams) && computeEigenvalues!(it, state)
		if it.contParams.detectBifurcation > 1 && detectBifucation(state)
			# we double-ckeck that the previous line, which mutated `state`, did not remove the bifurcation point
			if detectBifucation(state)
				_, bifpt = getBifurcationType(it, state, :guess, getinterval(current_param, current_param-ds))
				if bifpt.type != :none; push!(contResult.specialpoint, bifpt); end
			end
		end
		state.step += 1
		save!(contResult, it, state)
	else
		dcstate.isactive = false
		# save the last solution
		push!(contResult.sol, (x = copy(getx(state)), p = getp(state), step = state.step))
	end
	return converged(sol1), sol1.itnewton
end

# this is a function barrier to make Deflated continuation type stable
# it returns the  set of states and the ContResult
function getStatesContResults(iter::DefContIterable, roots::Vector{Tvec}) where Tvec
	@assert length(roots) > 0 "You must provide roots in the deflation operators. These roots are used as initial conditions of the deflated continuation process."
	contIt = iter.it
	copyto!(contIt.prob.u0, roots[1])
	state = DCState(copy(roots[1]), iterate(contIt)[1])
	states = [state]
	for ii = 2:length(roots)
		push!(states, DCState(copy(roots[ii]), iterate(contIt)[1]))
	end
	# allocate branches to hold the result
	branches = [ContResult(contIt, st.state) for st in states]
	return states, branches
end

# plotting functions
function plotDContBranch(branches, nbrs::Int, nactive::Int, nstep::Int)
	plot(branches..., label = "", title  = "$nbrs branches, actives = $(nactive), step = $nstep")
	for br in branches
		length(br) > 1 && plot!([br.branch[end-1:end].param], [getproperty(br.branch,1)[end-1:end]], label = "", arrow = true, color = :red)
	end
	scatter!([br.branch[1].param for br in branches], [br.branch[1][1] for br in branches], marker = :cross, color=:green, label = "") |> display
end
plotAllDCBranch(branches) = display(plot(branches..., label = ""))

_perturbSolution(x, p, id) = x
_acceptSolution(x, p) = true
_updateDeflationOp(defOp, x, p) = push!(defOp, x)


"""
$(SIGNATURES)

This function computes the set of curves of solutions `γ(s) = (x(s), p(s))` to the equation `F(x,p) = 0` based on the algorithm of **deflated continuation** as described in Farrell, Patrick E., Casper H. L. Beentjes, and Ásgeir Birkisson. “The Computation of Disconnected Bifurcation Diagrams.” ArXiv:1603.00809 [Math], March 2, 2016. http://arxiv.org/abs/1603.00809.

Depending on the options in `contParams`, it can locate the bifurcation points on each branch. Note that you can specify different predictors using `alg`.

# Arguments:
- `prob::AbstractBifurcationProblem` bifurcation problem
- `alg::DefCont`, deflated continuation algorithm, see [`DefCont`](@ref)
- `contParams` parameters for continuation. See [`ContinuationPar`](@ref) for more information about the options

# Optional Arguments:
- `plot = false` whether to plot the solution while computing,
- `callbackN` callback for newton iterations. see docs for `newton`. Can be used to change preconditioners or affect the newton iterations. In the deflation part of the algorithm, when seeking for new branches, the callback is passed the keyword argument `fromDeflatedNewton = true` to tell the user can it is not in the continuation part (regular newton) of the algorithm,
- `verbosity::Int` controls the amount of information printed during the continuation process. Must belong to `{0,⋯,5}`,
- `normN = norm` norm used in the Newton solves,
- `dotPALC = (x, y) -> dot(x, y) / length(x)`, dot product used to define the weighted dot product (resp. norm) ``\\|(x, p)\\|^2_\\theta`` in the constraint ``N(x, p)`` (see online docs on [PALC](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/PALC/)). This argument can be used to remove the factor `1/length(x)` for example in problems where the dimension of the state space changes (mesh adaptation, ...),

# Outputs:
- `contres::DCResult` composite type which contains the computed branches. See [`ContResult`](@ref) for more information,
"""
function continuation(prob::AbstractBifurcationProblem,
	 		algdc::DefCont,
			contParams::ContinuationPar;
			verbosity::Int = 2,
			plot::Bool = true,
			linearAlgo = BorderingBLS(contParams.newtonOptions.linsolver),
			dotPALC = DotTheta(),
			callbackN = (state; kwargs...) -> true,
			filename = "branch-" * string(Dates.now()),
			normN = norm)

	algdc = @set algdc.maxIterDefOp = algdc.maxIterDefOp * contParams.newtonOptions.maxIter
	# allow to remove the corner case and associated specific return variables, type stable
	defOp = algdc.deflationOperator
	@assert length(defOp) > 0 "You must provide at least one guess"

	# we make a copy of the deflation operator
	deflationOp = copy(defOp)

	verbosity > 0 && printstyled(color=:magenta, "#"^51*"\n")
	verbosity > 0 && printstyled(color=:magenta, "--> There are $(length(deflationOp)) branches\n")

	# underlying continuation iterator
	# we "hack" the saveSolEveryStep option because we always want to record the first point on each branch
	contIt = ContIterable(prob, algdc.alg, ContinuationPar(contParams, saveSolEveryStep = contParams.saveSolEveryStep == 0 ? Int(1e14) : contParams.saveSolEveryStep); plot = plot, normC = normN, dotPALC = dotPALC, finaliseSolution = finaliseDefault, callbackN = callbackN, verbosity = verbosity-2, filename = filename)

	iter = DefContIterable(contIt, algdc)

	return deflatedContinuation(iter, deflationOp, contParams, verbosity, plot)
end

function deflatedContinuation(dcIter::DefContIterable,
							deflationOp::DeflationOperator,
							contParams,
							verbosity,
							plot)

	states, branches = getStatesContResults(dcIter, deflationOp.roots)

	contIt = dcIter.it
	alg = dcIter.alg
	par = getParams(contIt.prob)
	lens = getLens(contIt)
	current_param = get(par, lens)

	# we extract the newton options
	optnewton = contParams.newtonOptions

	# function to get new solutions based on Deflated Newton
	function getNewSolution(_st::DCState, _p::Real, _idb)
		prob_df = reMake(contIt.prob; u0 = alg.perturbSolution(getx(_st), _p, _idb), params = set(par, lens, _p))
		newton(prob_df, deflationOp, setproperties(optnewton; maxIter = alg.maxIterDefOp); normN = contIt.normC, callback = contIt.callbackN, fromDeflatedNewton = true)
	end

	nstep = 0
	while ((contParams.pMin < current_param < contParams.pMax) || nstep == 0) &&
		 		nstep < contParams.maxSteps
		# we update the parameter value
		current_param += contParams.ds
		current_param = clampPredp(current_param, contIt)

		verbosity > 0 && println("──"^51)
		nactive = mapreduce(x -> x.isactive, +, states)
		verbosity > 0 && println("--> step = $nstep has $(nactive)/$(length(branches)) active branche(s), p = $current_param")

		# we empty the set of known solutions
		empty!(deflationOp.roots)

		# update the known branches
		for (idb, state) in enumerate(states)
			# this computes the solution for the new parameter value current_param
			# it also updates deflationOp
			flag, itnewton = updatebranch!(dcIter, state, branches[idb], deflationOp;
					current_param = current_param, step = nstep)
			(verbosity>=2 && isActive(state)) && println("----> Continuation of branch $idb in $itnewton Iterations")
			verbosity>=1 && ~flag && itnewton>0 && printstyled(color=:red, "--> Fold for branch $idb ?\n")
		end

		verbosity>1 && printstyled(color = :magenta,"--> looking for new branches\n")
		# number of branches
		nbrs = length(states)
		# number of active branches

		nactive = mapreduce(x -> x.isactive, +, states)
		if plot && mod(nstep, contParams.plotEveryStep) == 0
			plotDContBranch(branches, nbrs, nactive, nstep)
		end

		# only look for new branches if the number of active branches is too small
		if mod(nstep, alg.seekEveryStep) == 0 && nactive < alg.maxBranches
			n_active = 0
			# we restrict to 1:nbrs because we don't want to update the newly found branches
			for (idb, state) in enumerate(states[1:nbrs])
				if isActive(state) && (n_active < alg.maxBranches)
					n_active += 1
					_success = true
					verbosity >= 2 && println("----> Deflating branch $idb")
					while _success
						sol1 = getNewSolution(state, current_param, idb)
						_success = converged(sol1)
						if _success && contIt.normC(sol1.u - getx(state)) < optnewton.tol
							@error "Same solution found for identical parameter value!!"
							_success = false
						end
						if _success
							verbosity>=1 && printstyled(color=:green, "--> new solution for branch $idb \n")
							push!(deflationOp.roots, sol1.u)

							# create a new iterator and iterate it once to set up the ContState
							contitnew = @set contIt.prob = reMake(contIt.prob, u0 = sol1.u, params = set(par,lens,current_param))
							push!(states, DCState(sol1.u, iterate(contitnew)[1]))

							push!(branches, ContResult(contIt, states[end].state))
						end
					end
				end
			end
		end
		nstep += 1
	end
	plot && plotAllDCBranch(branches)
	return DCResult(contIt.prob, branches, contIt, [getx(c.state) for c in states if isActive(c)], dcIter.alg)
end

# function mergeBranches!(brs::Vector{T}, iter::ContIterable; plot = false, iterbrsmax = 2) where {T <: ContResult}
# 	@warn "il faut length(br) > 1"
# 	# get the continuation parameters
# 	@unpack pMin, pMax, ds = brs[1].contparams
# 	# we update the parameter span with provided initial conditions
# 	pMin = ds > 0 ? brs[1].branch[1].param : pMin
# 	pMax = ds < 0 ? brs[1].branch[1].param : pMax
# 	@show pMin pMax
#
# 	function _setnewsol(_x, _p0, _pMin, _pMax, _sign = -1)
# 		@show iter.contParams.maxSteps
# 		_iter2 = @set iter.par = setParam(iter, _p0)
# 		_iter2 = @set _iter2.contParams = setproperties(_iter2.contParams; ds = abs(iter.contParams.ds) * _sign, pMin = _pMin, pMax = _pMax)
# 		copyto!(_iter2.x0, _x)
# 		return _iter2
# 	end
#
# 	# test if the branch is done
# 	doneBranch(_br::ContResult) = _br.branch[1].param in [pMin, pMax] && _br.branch[end].param in [pMin, pMax]
# 	doneBranch(_brs::Vector{ <: ContResult}) = mapreduce(doneBranch, +, _brs)
#
# 	# test if (x,p) belongs to a branch
# 	function isin(_br::ContResult, x, p)
# 		tol = _br.contparams.newtonOptions.tol
# 		for (_id, _sol) in pairs(_br.sol)
# 			if max(iter.normC(_sol.x - x), abs(_sol.p - p)) < tol
# 				return true, _id
# 			end
# 		end
# 		 return false, 0
# 	end
#
# 	# function to find the branch to which (x, p) belongs
# 	# return found?, which?, end?
# 	function intersectBranch(_brs::Vector{ <: ContResult}, x, p, idavoid = 0)
# 		for (_id, _br) in pairs(_brs)
# 			_res = isin(_br, x, p)
# 			if (_res[1] && _id != idavoid)
# 				return (_res[1], _id, _res[2])  # return if branch id found
# 			end
# 		end
# 		return false, 0, 0
# 	end
#
# 	function getParamValues(_brs::Vector{ <: ContResult})
# 		ps = [(p = sol.p, id = ii) for (ii, br) in pairs(brs) if length(br) >1 for sol in br.sol[[1,end]]]
# 		sort!(ps, by = x -> x.p, rev = true )
# 		return ps
# 	end
# 	# @assert doneBranch(brs[1]) "error"
#
# 	printstyled(color=:blue,"#"^50*"\n")
# 	iterbrs = 0
#
# 	while doneBranch(brs) < length(brs) && iterbrs < iterbrsmax
# 		printstyled(color=:green, "#"^50*" - $iterbrs, #br = $(length(brs)) \n")
# 		for (id, br) in pairs(brs)
# 			print("--> branch $id, # = ", length(br), ", isdone = ")
# 			printstyled("$(doneBranch(br)) \n", color = (doneBranch(br) ? :green : :red))
# 			if ~doneBranch(br) && length(br) > 1 # if there is a single point, discard
# 				printstyled(color=:magenta,"--> branch $id not finished\n")
# 				# we found an un-finished branch
# 				iterbr = 0
# 				# we use as starting point the one
# 				startingpoint = ((br.branch[1].param > br.branch[2].param) & (br.branch[1].param < pMax)) ? (br.sol[1]..., first = true) : (br.sol[end]..., first = false)
# 				@show startingpoint.p
# 				if length(br) > 1
# 					# δp is where we want to go:
# 					δp = startingpoint.first ? br.branch[1].param - br.branch[2].param : br.branch[end].param - br.branch[end-1].param
# 				else
# 					@error "branch not large enough"
# 					δp = -abs(ds)
# 				end
# 				δp = δp == 0 ? -abs(ds) : δp
# 				_ps = getParamValues(brs)
# 				display(_ps)
# 				if δp > 0
# 					@assert 1==1 "searchsorted?"
# 					idp = findfirst(x -> x.p > startingpoint.p, _ps)
# 					@show  idp
# 					pspan = (startingpoint.p, ~isnothing(idp) ? _ps[idp].p : pMax)
# 					printstyled(color=:blue, "--> to the right\n")
# 				else
# 					@assert 1==1 "searchsorted?"
# 					idp = findfirst(x -> x.p < startingpoint.p, _ps)
# 					pspan = (~isnothing(idp) ? _ps[idp].p : pMin, startingpoint.p)
# 					printstyled(color=:green, "--> to the left\n")
# 				end
#
# 				# we define the parameter span
# 				iter2 = _setnewsol(startingpoint.x, startingpoint.p, pspan..., sign(δp))
# 				while ~doneBranch(brs[id]) && iterbr < 1
# 					brright, = continuation(iter2)
# 					brs[id] = _merge(br, brright)
# 					println("--> branch from ",brs[id].sol[1].p," to ", brright.sol[end].p)
# 					# return brright
#
# 					res = intersectBranch(brs, brright.sol[end].x, brright.sol[end].p,id)
# 					if res[1]  #we found a curve to merge with
# 							printstyled(color=:red, "--> Merge branch with ", res[2],"\n")
# 						brs[id] = _merge(brs[res[2]], brs[id])
# 							printstyled(color=:green,"\n--> the branch is complete = ", doneBranch(brs[id]),"\n")
# 						# we remove the other branch that we merged
# 						deleteat!(brs, res[2])
# 							println("--> #br = $(length(brs))\n")
# 					end
# 					iterbr += 1
# 				end
#
# 				break
# 			end
# 		end
# 		iterbrs += 1
# 	end
# 	iterbrs < iterbrsmax && printstyled("--> Merge computed!!!", color=:green)
# end
