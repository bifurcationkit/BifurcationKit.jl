# iterable which contains the options associated with Deflated Continuation
@with_kw struct DefContIterable{Tit, TperturbSolution, TacceptSolution}
	it::Tit								# replicate PALC iterator
	maxBranches::Int64					# maximum number of (active) branches to be computed
	seekEveryStep::Int64 = 1			# whether to seek new (deflated) solution at every step
	perturbSolution::TperturbSolution 	# perturb function
	acceptSolution::TacceptSolution	  	# accept (solution) function
end

# state specific to Deflated Continuation, it is updated during the continuation process
mutable struct DCState{T, Tstate}
	tmp::T
	contState::Tstate
	isactive::Bool
	DCState(sol::T) where T = new{T, Nothing}(copy(sol), nothing, true)
	DCState(sol::T, state::ContState) where {T} = new{T, typeof(state)}(copy(sol), state, true)
end
# whether the branch is active
isactive(dc::DCState) = dc.isactive
# getters
getx(dc::DCState) = getx(dc.contState)
getp(dc::DCState) = getp(dc.contState)

#
function updatebranch!(iter::DefContIterable, dcstate::DCState, contResult::ContResult, defOp::DeflationOperator; current_param, step)
	isactive(dcstate) == false &&  return false, 0
	state = dcstate.contState 	# continuation state
	it = iter.it 				# continuation iterator
	@unpack step, ds, theta = state
	@unpack verbosity = it
	state.z_pred.p = current_param

	getPredictor!(state, it)
	sol1, fval, converged, itnewton = newton(it.F, it.J, getx(state), setParam(it,current_param), it.contParams.newtonOptions, defOp; normN = it.normC, callback = it.callbackN, iterationC = step, z0 = state.z_old)
	if converged
		# record previous parameter (cheap) and update current solution
		copyto!(state.z_old.u, sol1); state.z_old.p = current_param
		state.z_pred.p = current_param
		# Get tangent, it only mutates tau
		getTangent!(state.tau, state.z_pred, state.z_old, it, ds, theta, it.tangentAlgo, verbosity)
		push!(defOp.roots, sol1)
		computeEigenElements(it.contParams) && computeEigenvalues!(it, state)
		if it.contParams.detectBifurcation > 1 && detectBifucation(state)
			# we double-ckeck that the previous line, which mutated `state`, did not remove the bifurcation point
			if detectBifucation(state)
				_T  = eltype(it)
				_, bifpt = getBifurcationType(it.contParams, state, it.normC, it.printSolution, it.verbosity, :guess, getinterval(current_param, current_param-ds))
				if bifpt.type != :none; push!(contResult.bifpoint, bifpt); end
			end
		end
		state.step += 1
		save!(contResult, iter.it, state)
	else
		dcstate.isactive = false
		# save the last solution
		push!(contResult.sol, (x = copy(getx(state)), p = getp(state), step = state.step))
	end
	return converged, itnewton
end

"""
$(SIGNATURES)

The function computes the set of curves of solutions `γ(s) = (x(s), p(s))` to the equation `F(x,p)=0` based on the algorithm of **deflated continuation** as described in Farrell, Patrick E., Casper H. L. Beentjes, and Ásgeir Birkisson. “The Computation of Disconnected Bifurcation Diagrams.” ArXiv:1603.00809 [Math], March 2, 2016. http://arxiv.org/abs/1603.00809.

Depending on the options in `contParams`, it can locate the bifurcation points on each branch. Note that you can specify different predictors using `tangentAlgo`.

# Arguments:
- `F` is a function with input arguments `(x, p)`, where `p` is the set of parameters passed to `F`, and returning a vector `r` that represents the functional. For type stability, the types of `x` and `r` should match. In particular, it is not **inplace**,
- `J` is the jacobian of `F` at `(x, p)`. It can assume three forms.
    1. Either `J` is a function and `J(x, p)` returns a `::AbstractMatrix`. In this case, the default arguments of `contParams::ContinuationPar` will make `continuation` work.
    2. Or `J` is a function and `J(x, p)` returns a function taking one argument `dx` and returning `dr` of the same type as `dx`. In our notation, `dr = J * dx`. In this case, the default parameters of `contParams::ContinuationPar` will not work and you have to use a Matrix Free linear solver, for example `GMRESIterativeSolvers`,
    3. Or `J` is a function and `J(x, p)` returns a variable `j` which can assume any type. Then, you must implement a linear solver `ls` as a composite type, subtype of `AbstractLinearSolver` which is called like `ls(j, rhs)` and which returns the solution of the jacobian linear system. See for example `examples/SH2d-fronts-cuda.jl`. This linear solver is passed to `NewtonPar(linsolver = ls)` which itself passed to `ContinuationPar`. Similarly, you have to implement an eigensolver `eig` as a composite type, subtype of `AbstractEigenSolver`.
- `par` initial set of parameters,
- `lens::Lens` specifies which parameter axis among `par` is used for continuation. For example, if `par = (α = 1.0, β = 1)`, we can perform continuation w.r.t. `α` by using `lens = (@lens _.α)`. If you have an array `par = [ 1.0, 2.0]` and want to perform continuation w.r.t. the first variable, you can use `lens = (@lens _[1])`. For more information, we refer to `SetField.jl`,
- `contParams` parameters for continuation. See [`ContinuationPar`](@ref) for more information about the options,
- `defOp::DeflationOperator` a Deflation Operator (see [`DeflationOperator`](@ref)) which contains the set of solution guesses for the parameter `par`.

# Optional Arguments:
- `seekEveryStep::Int = 1` we look for additional solution, using deflated newton, every `seekEveryStep` step,
- `maxBranches::Int = 100` maximum number of branches considered,
- `showplot = false` whether to plot the solution while computing,
- `printSolution = (x, p) -> norm(x)` function used to plot in the continuation curve. It is also used in the way results are saved. It could be `norm` or `(x, p) -> x[1]`. This is also useful when saving several huge vectors is not possible for memory reasons (for example on GPU...),
- `plotSolution = (x, p; kwargs...) -> nothing` function implementing the plot of the solution,
- `callbackN` callback for newton iterations. see docs for `newton`. Can be used to change preconditioners or affect the newton iterations. In the deflation part of the algorithm, when seeking for new branches, the callback is passed the keyword argument `fromDeflatedNewton = true` to tell the user can it is not in the continuation part (regular newton) of the algorithm,
- `tangentAlgo = NaturalPred()` controls the algorithm used to predict the tangents along the curve of solutions or the corrector. Can be `NaturalPred`, `SecantPred` or `BorderedPred`,
- `verbosity::Int` controls the amount of information printed during the continuation process. Must belong to `{0,⋯,5}`,
- `normN = norm` norm used in the different Newton solves,
- `dotPALC = (x, y) -> dot(x, y) / length(x)`, dot product used to define the weighted dot product (resp. norm) ``\\|(x, p)\\|^2_\\theta`` in the constraint ``N(x, p)`` (see below). This arguement can be used to remove the factor `1/length(x)` for example in problems where the dimension of the state space changes (mesh adaptation, ...),
- `perturbSolution = (x, p, id) -> x` perturbation applied to the solution when trying to find new solutions using Deflated Newton. You can use for example `(x, p, id) -> x .+ (1 .+ 0.001 * rand(size(x)...))`

# Outputs:
- `contres::Vector{ContResult}` composite type which contains the computed branches. See [`ContResult`](@ref) for more information,
- the solutions at the last parameter value,
- current parameter value.
"""
function continuation(F, J, par, lens::Lens, contParams::ContinuationPar, defOp::DeflationOperator;
			verbosity::Int = 2,
			maxBranches::Int = 100,
			seekEveryStep::Int = 1,
			showplot::Bool = true,
			tangentAlgo = SecantPred(),
			linearAlgo = BorderingBLS(contParams.newtonOptions.linsolver),
			dotPALC = (x,y) -> dot(x,y) / length(x),
			printSolution = (x, p) -> norm(x),
			plotSolution = (x, p ;kwargs...) -> plot!(x; kwargs...),
			perturbSolution = (x, p, id) -> x,
			callbackN = (x, f, J, res, iteration, itlinear, options; kwargs...) -> true,
			acceptSolution = (x, p) -> true,
			normN = norm) where vectype

	if length(defOp) == 0; return nothing, nothing, 0; end

	# function to get new solutions based on Deflated Newton
	function getNewSolution(_st::DCState, _p::Real, _idb)
		newton(F, J, perturbSolution(getx(_st), _p, _idb), set(par, lens, _p), setproperties(optnewton; maxIter = 5optnewton.maxIter), deflationOp; normN = normN, callback = callbackN, fromDeflatedNewton = true)
	end

	# we make a copy of the deflation operator
	deflationOp = DeflationOperator(defOp.power, defOp.dot, defOp.shift, deepcopy(defOp.roots))

	verbosity > 0 && printstyled(color=:magenta, "#"^51*"\n")
	verbosity > 0 && printstyled(color=:magenta, "--> There are $(length(deflationOp)) branches\n")

	# we "hack" the saveSolEveryStep option because we always want to record the first point on each branch
	iter = DefContIterable(ContIterable(F=F, J=J, x0 = defOp[1], par=par,lens=lens, contParams = ContinuationPar(contParams, saveSolEveryStep = contParams.saveSolEveryStep == 0 ? Int(1e14) : contParams.saveSolEveryStep), tangentAlgo = tangentAlgo, linearAlgo = linearAlgo, plot=showplot, plotSolution = plotSolution, printSolution=printSolution, normC=normN,dottheta=DotTheta(dotPALC),finaliseSolution=finaliseDefault,callbackN=callbackN,verbosity=verbosity-2, filename=nothing), maxBranches, seekEveryStep, perturbSolution, acceptSolution)

	# underlying continuation iterator
	contIt = iter.it

	# allocate states, one per branch
	states = [DCState(sol, iterate(@set contIt.x0 = sol)[1]) for sol in deflationOp.roots]

	# allocate branches to hold the result
	branches = [ContResult(contIt, state.contState) for state in states]

	S1 = empty(copy(deflationOp.roots))
	current_param = get(par, lens)

	# we extract the newton options
	optnewton = contParams.newtonOptions

	ii = 0
	while ((contParams.pMin < current_param < contParams.pMax) || ii == 0) &&
		 		ii < contParams.maxSteps
		# we update the parameter value
		current_param += contParams.ds
		current_param = clampPred(current_param, contIt)

		verbosity > 0 && println("#"^51)
		verbosity > 0 && println("--> step = $ii has $(length(branches)) branche(s), p = $current_param")

		# we empty the set of known solutions
		empty!(deflationOp.roots)

		# update the known branches
		for (idb, state) in enumerate(states)
			# this computes the solution for the new parameter value current_param
			# it also updates deflationOp
			flag, itnewton = updatebranch!(iter, state, branches[idb], deflationOp;
					current_param = current_param, step = ii)
			(verbosity >= 2 && isactive(state)) && println("----> Continuation for branch $idb in $itnewton iterations")
			verbosity >= 1 && ~flag && itnewton>0 && printstyled(color=:red, "--> Fold for branch $idb ?\n")
		end

		verbosity>1 && printstyled(color = :magenta,"--> looking for new branches\n")
		# number of branches
		nc = length(states)
		# number of active branches
		nactive = mapreduce(x -> x.isactive, +, states)
		if showplot && mod(ii, contParams.plotEveryStep) == 0
			plot(branches..., label = "", title  = "$nc branches, actives = $(nactive), step = $ii")
			for br in branches
				length(br) > 1 && plot!([br.branch[end-1:end].param], [getproperty(br.branch,1)[end-1:end]], label = "", arrow = true, color = :red)
			end
			scatter!([br.branch[1].param for br in branches], [br.branch[1][1] for br in branches], marker = :cross, color=:green, label = "") |> display


		end

		# only look for new branches if the number of active branches is too small
		if mod(ii, seekEveryStep) == 0
			if nactive < maxBranches
				n_active = 0
				# we restrict to 1:nc because we don't want to update the newly found branches
				for (idb, state) in enumerate(states[1:nc])
					# TODO this could be done in parallel
					if isactive(state) && (n_active < maxBranches)
						n_active += 1
						_success = true
						verbosity >= 2 && println("----> Deflation for branch $idb")
						while _success
							sol1, hist, _success, itnewton = getNewSolution(state, current_param, idb)
							if _success && normN(sol1 - getx(state)) < optnewton.tol
								@error "Same solution found for identical parameter value!!"
								_success = false
							end
							if _success
								verbosity>=1 && printstyled(color=:green, "--> new solution for branch $idb \n")
								push!(deflationOp.roots, sol1)
								push!(states, DCState(sol1, iterate(setproperties(contIt; x0 = sol1, par = set(par,lens,current_param)))[1]))
								push!(branches, ContResult(contIt, states[end].contState))
							end
						end
					end
				end
			end
		end

		ii += 1
	end
	if showplot
		display(plot(branches..., label = ""))
	end

	return branches, contIt, [getx(c.contState) for c in states if isactive(c)], current_param
end
