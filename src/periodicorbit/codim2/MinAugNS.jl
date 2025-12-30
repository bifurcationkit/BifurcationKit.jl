"""
$(SIGNATURES)

For an initial guess from the index of a NS bifurcation point located in `ContResult.specialpoint`, returns a point which can be refined using `newtonFold`.
"""
function ns_point(br::AbstractBranchResult, index::Int)
    bptype = br.specialpoint[index].type
    if bptype != :ns 
        error("This should be a NS point.\nYou passed a $bptype point.")
    end
    specialpoint = br.specialpoint[index]
    ω = imag(br.eig[specialpoint.idx].eigenvals[specialpoint.ind_ev])
    return BorderedArray(_copy(specialpoint.x), [specialpoint.param, ω])
end

function apply_jacobian_neimark_sacker(pb, x, par, ω, dx, _transpose = false)
    if _transpose == false
        throw("Work in progress")
        return jacobian_adjoint_neimark_sacker_matrix_free(pb, x, par, ω, dx)
    else
        # if matrix-free:
        if has_adjoint(pb)
            return jacobian_adjoint_neimark_sacker_matrix_free(pb, x, par, ω, dx)
        else
            return apply(adjoint(jacobian_neimark_sacker(pb, x, par, ω)), dx)
        end
    end
end
####################################################################################################
is_symmetric(::NSMAProblem) = false

# test function for NS bifurcation
nstest(JacNS, v, w, J22, _zero, n, lsbd = MatrixBLS()) = lsbd(JacNS, v, w, J22, _zero, n)

# this function encodes the functional
function (𝐍𝐒::NeimarkSackerMinimallyAugmentedFormulation)(x, p::𝒯, ω::𝒯, params) where 𝒯
    # These are the equations of the minimally augmented (MA) formulation of the Neimark-Sacker bifurcation point
    # input:
    # - x guess for the point at which the jacobian is singular
    # - p guess for the parameter value `<: Real` at which the jacobian is singular
    # The jacobian of the MA problem is solved with a BLS method
    a = 𝐍𝐒.a
    b = 𝐍𝐒.b
    # update parameter
    par = set(params, getlens(𝐍𝐒), p)
    J = jacobian_neimark_sacker(𝐍𝐒.prob_vf, x, par, ω)
    σ1 = nstest(J, a, b, zero(𝒯), 𝐍𝐒.zero, one(𝒯), 𝐍𝐒.linbdsolver)[2]
    return residual(𝐍𝐒.prob_vf, x, par), real(σ1), imag(σ1)
end
###################################################################################################
function _compute_bordered_vectors(𝐍𝐒::NeimarkSackerMinimallyAugmentedFormulation, JNS, JNS★, ω)
    a = 𝐍𝐒.a
    b = 𝐍𝐒.b
    𝒯 = eltype(𝐍𝐒)

    # we solve N[v, σ1] = [0, 1]
    v, σ1, cv, itv = nstest(JNS, a, b, zero(𝒯), 𝐍𝐒.zero, one(𝒯), 𝐍𝐒.linbdsolver)
    ~cv && @debug "[codim2 NS] Linear solver for N did not converge."

    # we solve Nᵗ[w, σ2] = [0, 1]
    w, σ2, cv, itw = nstest(JNS★, b, a, zero(𝒯), 𝐍𝐒.zero, one(𝒯), 𝐍𝐒.linbdsolverAdjoint)
    ~cv && @debug "[codim2 NS] Linear solver for Nᵗ did not converge."

    return (; v, w, itv, itw)
end

function _get_bordered_terms(𝐍𝐒::NeimarkSackerMinimallyAugmentedFormulation, x, p::𝒯, ω::𝒯, par) where 𝒯
    # get the PO functional, ie a WrapPOSh, WrapPOTrap, WrapPOColl
    POWrap = 𝐍𝐒.prob_vf

    # parameter axis
    lens = getlens(𝐍𝐒)

    # update parameter
    par0 = set(par, lens, p)

    # Avoid doing 3 times the (possibly) costly building of J(x, p)
    JNS = jacobian_neimark_sacker(POWrap, x, par0, ω) # jacobian with period NS boundary condition
    # Avoid computing the jacobian twice in case 𝐍𝐒.Jadjoint is not provided
    JNS★ = has_adjoint(𝐍𝐒) ? jacobian_adjoint_neimark_sacker(POWrap, x, par0, ω) : adjoint(JNS)

    (; v, w, itv, itw) = _compute_bordered_vectors(𝐍𝐒, JNS, JNS★, ω)

    δ = getdelta(POWrap)
    ϵ1 = ϵ2 = ϵ3 = 𝒯(δ)
    ################### computation of σx σp ####################
    # TODO!! This is only finite differences
    dₚF = minus(residual(POWrap, x, set(par, lens, p + ϵ1)),
                residual(POWrap, x, set(par, lens, p - ϵ1))); LA.rmul!(dₚF, 𝒯(1 / (2ϵ1)))
    dJvdp = minus(apply(jacobian_neimark_sacker(POWrap, x, set(par, lens, p + ϵ3), ω), v),
                  apply(jacobian_neimark_sacker(POWrap, x, set(par, lens, p - ϵ3), ω), v));
    LA.rmul!(dJvdp, 𝒯(1/(2ϵ3)))
    σₚ = -LA.dot(w, dJvdp)

    # case of ∂σ_ω
    σω = -(LA.dot(w, apply(jacobian_neimark_sacker(POWrap, x, par, ω+ϵ2), v)) - 
           LA.dot(w, apply(jacobian_neimark_sacker(POWrap, x, par, ω), v)) )/ϵ2

    return (;JNS, JNS★, dₚF, σₚ, δ, ϵ2, ϵ3, v, w, par0, dJvdp, itv, itw, σω)
end
###################################################################################################
function jacobian(pdpb::NSMAProblem{Tprob, MinAugMatrixBased}, X, par) where {Tprob}
    𝐍𝐒 = pdpb.prob

    # get the PO functional, ie a WrapPOSh, WrapPOTrap, WrapPOColl
    POWrap = 𝐍𝐒.prob_vf

    x = @view X[begin:end-2]
    p = X[end-1]
    ω = X[end]

    𝒯 = eltype(p)

    (;JNS★, dₚF, σₚ, ϵ2, ϵ3, v, w, par0, σω) = _get_bordered_terms(𝐍𝐒, x, p, ω, par)

    cw = conj(w)
    vr = real(v); vi = imag(v)
    u1r = apply_jacobian_neimark_sacker(POWrap, x .+ ϵ2 .* vcat(vr,0), par0, ω, cw, true)
    u1i = apply_jacobian_neimark_sacker(POWrap, x .+ ϵ2 .* vcat(vi,0), par0, ω, cw, true)
    u2 = apply(JNS★, cw)
    σxv2r = @. -(u1r - u2) / ϵ2 # careful, this is a complex vector
    σxv2i = @. -(u1i - u2) / ϵ2
    σx = @. σxv2r + Complex{𝒯}(0, 1) * σxv2i

    dJvdt = minus(apply(jacobian_neimark_sacker(POWrap, x .+ ϵ2 .* vcat(0 * vr, 1), par0, ω), v),
                  apply(jacobian_neimark_sacker(POWrap, x .- ϵ2 .* vcat(0 * vr, 1), par0, ω), v));
    LA.rmul!(dJvdt, 𝒯(1/(2ϵ3)))
    σt = -LA.dot(w, dJvdt) 

    _Jpo = jacobian(POWrap, x, par0)
    Jns = hcat(_Jpo, dₚF, zero(dₚF))
    Jns = vcat(Jns, vcat(real(σx), real(σt), real(σₚ), real(σω))')
    Jns = vcat(Jns, vcat(imag(σx), imag(σt), imag(σₚ), imag(σω))')
end
###################################################################################################
# Struct to invert the jacobian of the ns MA problem.
struct NSLinearSolverMinAug <: AbstractLinearSolver; end

function NSMALinearSolver(x, p::𝒯, ω::𝒯, 𝐍𝐒::NeimarkSackerMinimallyAugmentedFormulation, par,
                            duu, dup, duω) where 𝒯
    ################################################################################################
    # Recall that the functional we want to solve is [F(x,p), σ(x,p)]
    # where σ(x,p) is computed in the above functions and F is the periodic orbit
    # functional. We recall that N⋅[v, σ] ≡ [0, 1]
    # The Jacobian Jpd of the functional is expressed at (x, p)
    # We solve here Jpd⋅res = rhs := [rhsu, rhsp, rhsω]
    # The jacobian expression of the NS problem is
    #           ┌             ┐
    #    Jns =  │  J  dpF   0 │
    #           │ σx   σp  σω │
    #           └             ┘
    # where σx := ∂ₓσ and σp := ∂ₚσ
    ########## Resolution of the bordered linear system ########
    # J * dX      + dpF * dp           = du => dX = x1 - dp * x2
    # The second equation
    #    <σx, dX> +  σp * dp + σω * dω = du[end-1:end]
    # thus becomes
    #   (σp - <σx, x2>) * dp + σω * dω = du[end-1:end] - <σx, x1>
    # This 2 x 2 system is then solved to get (dp, dω)
    ########################## Extraction of function names ########################################
    # get the PO functional, ie a WrapPOSh, WrapPOTrap, WrapPOColl
    POWrap = 𝐍𝐒.prob_vf
    (; JNS★, dₚF, σₚ, ϵ2, ϵ3, v, w, par0, σω, itv, itw) = _get_bordered_terms(𝐍𝐒, x, p, ω, par)

    # inversion of Jns 
    if has_hessian(𝐍𝐒) == false || 𝐍𝐒.usehessian == false
        cw = conj(w)
        vr = real(v); vi = imag(v)
        # u1r = jacobian_neimark_sacker(POWrap, x .+ ϵ2 .* vcat(vr,0), par0, ω).jacpb' * cw
        # u1i = jacobian_neimark_sacker(POWrap, x .+ ϵ2 .* vcat(vi,0), par0, ω).jacpb' * cw
        u1r = apply_jacobian_neimark_sacker(POWrap, x .+ ϵ2 .* vcat(vr, 0), par0, ω, cw, true)
        u1i = apply_jacobian_neimark_sacker(POWrap, x .+ ϵ2 .* vcat(vi, 0), par0, ω, cw, true)
        u2 = apply(JNS★, cw)
        σxv2r = @. -(u1r - u2) / ϵ2 # careful, this is a complex vector
        σxv2i = @. -(u1i - u2) / ϵ2
        σx = @. σxv2r + Complex{𝒯}(0, 1) * σxv2i

        dJvdt = minus(apply(jacobian_neimark_sacker(POWrap, x .+ ϵ2 .* vcat(0 * vr, 1), par0, ω), v),
                      apply(jacobian_neimark_sacker(POWrap, x .- ϵ2 .* vcat(0 * vr, 1), par0, ω), v));
        LA.rmul!(dJvdt, 𝒯(1/(2ϵ3)))
        σt = -LA.dot(w, dJvdt) 

        _Jpo = jacobian(POWrap, x, par0)

        x1, x2, cv, (it1, it2) = 𝐍𝐒.linsolver(_Jpo, duu, dₚF)
        ~cv && @debug "[codim2 NS] Linear solver for N did not converge."

        σxx1 = LA.dot(vcat(σx, σt), x1)
        σxx2 = LA.dot(vcat(σx, σt), x2)

    else
        error("WIP. Please select another jacobian method like :autodiff or :finiteDifferences. You can also pass the option usehessian = false.")
    end

    # We need to be careful here because the dot produces conjugates. 
    # Hence the + dot(σx, x2) and + imag(dot(σx, x1) and not the opposite
    LS = Matrix{𝒯}(undef, 2, 2);
    rhs = Vector{𝒯}(undef, 2);
    LS[1, 1] = real(σₚ - σxx2); LS[1, 2] = real(σω)
    LS[2, 1] = imag(σₚ + σxx2); LS[2, 2] = imag(σω)
    rhs[1] = dup - real(σxx1); rhs[2] =  duω + imag(σxx1)
    dp, dω = LS \ rhs

    return x1 .- dp .* x2, dp, dω, true, it1 + it2 + sum(itv) + sum(itw)
end

function (pdls::NSLinearSolverMinAug)(Jns, rhs::BorderedArray{vectype, 𝒯}; kwargs...) where {vectype, 𝒯}
    # kwargs is used by AbstractLinearSolver
    out = NSMALinearSolver((Jns.x).u,
                (Jns.x).p[1],
                (Jns.x).p[2],
                Jns.nspb,
                Jns.params,
                rhs.u, rhs.p[1], rhs.p[2])
    # this type annotation enforces type stability
    return BorderedArray{vectype, 𝒯}(out[1], [out[2], out[3]]), out[4], out[5]
end
###################################################################################################
get_wrap_po(pb::NSMAProblem) = get_wrap_po(pb.prob)
# we add :hopfpb in order to use HopfEig
jacobian(nspb::NSMAProblem{Tprob, Nothing}, x, p) where {Tprob} = (x = x, params = p, nspb = nspb.prob, hopfpb = nspb.prob)
###################################################################################################
function update!(probma::NSMAProblem, iter, state)
    # it is called to update the Minimally Augmented problem
    # by updating the vectors a, b
    # we first check that the continuation step was successful
    # if not, we do not update the problem with bad information!
    𝐍𝐒 = get_formulation(probma)
    𝒯 = eltype(𝐍𝐒)
    success = state.converged
    if (~mod_counter(step, 𝐍𝐒.update_minaug_every_step) || success == false)
        # we call the user update
        return update!(𝐍𝐒, iter, state)
    end
    @debug "[codim2 NS] Update a / b in NS"

    z = getsolution(state)
    x = getvec(z.u, 𝐍𝐒)   # NS point
    p1, ω = getp(z.u, 𝐍𝐒) # first parameter
    p2 = z.p              # second parameter

    lens1, lens2 = get_lenses(probma)
    newpar = set(getparams(probma), lens1, p1)
    newpar = set(newpar, lens2, p2)

    # get the PO functional
    POWrap = 𝐍𝐒.prob_vf

    JNS = jacobian_neimark_sacker(POWrap, x, newpar, ω)
    JNS★ = has_adjoint(𝐍𝐒) ? jacobian_adjoint_neimark_sacker(POWrap, x, newpar, ω) : adjoint(JNS)

    (; v, w, itv, itw) = _compute_bordered_vectors(𝐍𝐒, JNS, JNS★, ω)
    _copyto!(𝐍𝐒.a, w); LA.rmul!(𝐍𝐒.a, 1/𝐍𝐒.norm(w))
    # do not normalize with dot(newb, 𝐍𝐒.a), it prevents detection of resonances
    _copyto!(𝐍𝐒.b, v); LA.rmul!(𝐍𝐒.b, 1/𝐍𝐒.norm(v))

    # we stop continuation at R1, PD points
    # test if we jumped to PD branch
    pdjump = abs(abs(ω) - pi) < 100𝐍𝐒.newton_options.tol

    isbif = true#isnothing(contResult) ? true : isnothing(findfirst(x -> x.type in (:R1, :pd), contResult.specialpoint))
    # if the frequency is null, this is not a NS point, we halt the process
    # tolerance for detecting R1 bifurcation and stopping continuation
    ϵR1 = 100𝐍𝐒.newton_options.tol
    stop_R1 = 1-cos(ω) <= ϵR1
    if stop_R1
        @warn "[Codim 2 NS - update!]\nThe NS curve seems to be close to a R1 point: ω ≈ $ω.\n Stopping computations at ($lens1, $lens2) = ($p1, $p2).\nIf the R1 point is not detected, try lowering Newton tolerance or dsmax."
    end

    if pdjump
        @warn "[Codim 2 NS - update!] The NS curve seems to jump to a PD curve.\nStopping computations at ($p1, $p2).\nPerhaps it is close to a R2 bifurcation for example."
    end

    # call the user-passed update
    final_result = update!(𝐍𝐒, iter, state)

    return ~stop_R1 && isbif && final_result && ~pdjump
end
###################################################################################################
function finalise_solution(iter::ContIterable{NSPeriodicOrbitCont}, 
                           state::AbstractContinuationState, 
                           contres)
    isbt = isnothing(contres) ? true : isnothing(findfirst(x -> x.type in (:bt, :ghbt, :btgh), contres.specialpoint))
    fin_user = iter.finalise_solution(getsolution(state),
                                  state.τ,
                                  state.step,
                                  contres; 
                                  state,
                                  iter)
    return isbt && fin_user
end

function record_from_solution(iter::ContIterable{NSPeriodicOrbitCont, <: NSMAProblem},
                              state::AbstractContinuationState)
    probma = getprob(iter)
    𝐍𝐒 = get_formulation(probma)
    probwrap = 𝐍𝐒.prob_vf
    lens1, lens2 = get_lenses(probma)
    lenses = get_lens_symbol(lens1, lens2)
    u = getx(state)
    p = getp(state)

    return (; zip(lenses, (getp(u, 𝐍𝐒)[1], p))...,
                ωₙₛ = getp(u, 𝐍𝐒)[2],
                CH = real(𝐍𝐒.l1),
                R₁ = 𝐍𝐒.R1,
                R₂ = 𝐍𝐒.R2,
                R₃ = 𝐍𝐒.R3,
                R₄ = 𝐍𝐒.R4,
                _namedrecordfromsol(__user_record_solution_periodic_orbit(probwrap, user_passed_pofunction(probwrap.recordFromSolution), iter, state))...
                )
end

function continuation_ns(prob, alg::AbstractContinuationAlgorithm,
                        nspointguess::BorderedArray{vectype, 𝒯b}, par,
                        lens1::AllOpticTypes, lens2::AllOpticTypes,
                        eigenvec, eigenvec_ad,
                        options_cont::ContinuationPar ;
                        normC = norm,

                        update_minaug_every_step = 1,
                        bdlinsolver::AbstractBorderedLinearSolver = MatrixBLS(),
                        bdlinsolver_adjoint::AbstractBorderedLinearSolver = bdlinsolver,

                        jacobian_ma::AbstractJacobianType = AutoDiff(),
                        compute_eigen_elements = false,
                        kind = NSCont(),
                        usehessian = false,
                        plot_solution = BifurcationKit.plot_solution(prob),
                        prm::Bool = prob isa WrapPOSh,
                        kwargs...) where {𝒯b, vectype}
    @assert lens1 != lens2 "Please choose 2 different parameters. You only passed $lens1"
    @assert lens1 == getlens(prob)

    # options for the Newton Solver inheritated from the ones the user provided
    newton_options = options_cont.newton_options

    𝐍𝐒 = NeimarkSackerMinimallyAugmentedFormulation(
            prob,
            _copy(eigenvec),
            _copy(eigenvec_ad),
            newton_options.linsolver,
            # do not change linear solver if user provides it
            @set bdlinsolver.solver = (isnothing(bdlinsolver.solver) ? newton_options.linsolver : bdlinsolver.solver);
            linbdsolve_adjoint = bdlinsolver_adjoint,
            usehessian,
            _norm = normC,
            newton_options,
            prm,
            update_minaug_every_step)

    @assert jacobian_ma in (AutoDiff(), FiniteDifferences(), MinAug(), FiniteDifferencesMF(), MinAugMatrixBased())

    # Jacobian for the NS problem
    record_ns = RecordForNS(record_from_solution, BifurcationKit.record_from_solution(prob))
    if jacobian_ma in (AutoDiff(), FiniteDifferencesMF(), FiniteDifferences(), MinAugMatrixBased())
        nspointguess = vcat(nspointguess.u, nspointguess.p...)
        prob_ns = NSMAProblem(𝐍𝐒, jacobian_ma, nspointguess, par, lens2, plot_solution, record_ns)
        opt_ns_cont = deepcopy(options_cont)
    else
        prob_ns = NSMAProblem(𝐍𝐒, nothing, nspointguess, par, lens2, plot_solution, record_ns)
        opt_ns_cont = @set options_cont.newton_options.linsolver = NSLinearSolverMinAug()
    end

    # current lyapunov coefficient
    𝒯 = eltype(𝒯b)
    𝐍𝐒.l1 = Complex{𝒯}(1, 0)
    𝐍𝐒.R1 = zero(𝒯)
    𝐍𝐒.R2 = zero(𝒯)
    𝐍𝐒.R3 = zero(𝒯)
    𝐍𝐒.R4 = zero(𝒯)

    # eigen solver
    eigsolver = HopfEig(getsolver(opt_ns_cont.newton_options.eigsolver), prob_ns)

    # change the plot and record functions
    _kwargs = (;plot_solution = plot_solution)
    _plotsol = modify_po_plot(prob_ns, getparams(prob_ns), getlens(prob_ns) ; _kwargs...)
    prob_ns = re_make(prob_ns, plot_solution = _plotsol)

    # define event for detecting codim 2 bifurcations. Couple it with user passed events
    event_user = get(kwargs, :event, nothing)
    event_bif = ContinuousEvent(5, test_for_ns_ch, compute_eigen_elements, ("R1", "R2", "R3", "R4", "ch",), 0)
    event = isnothing(event_user) ? event_bif : PairOfEvents(event_bif, event_user)

    # solve the NS equations
    br_ns_po = continuation(
                    prob_ns, alg,
                    (@set opt_ns_cont.newton_options.eigsolver = eigsolver);
                    linear_algo = BorderingBLS(solver = opt_ns_cont.newton_options.linsolver, check_precision = false),
                    kwargs...,
                    kind,
                    event,
                    normC,
                    )
    _correct_event_labels(br_ns_po)
end

function test_for_ns_ch(iter, state)
    probma = getprob(iter)
    𝐍𝐒 = get_formulation(probma)
    lens1, lens2 = get_lenses(probma)

    z = getx(state)
    x = getvec(z, 𝐍𝐒)   # NS point
    p1, ω = getp(z, 𝐍𝐒) # first parameter
    p2 = getp(state)    # second parameter
    par = getparams(probma)
    newpar = set(par, lens1, p1)
    newpar = set(newpar, lens2, p2)

    prob_ns = iter.prob.prob
    pbwrap = prob_ns.prob_vf

    ns0 = NeimarkSacker(copy(x), nothing, p1, ω, newpar, lens1, nothing, nothing, nothing, :none)
    newton_options = 𝐍𝐒.newton_options
    # test if we jumped to PD branch
    pdjump = abs(abs(ω) - pi) < 100newton_options.tol
    if ~pdjump && pbwrap.prob isa ShootingProblem
        ns = neimark_sacker_normal_form(pbwrap, ns0, (1, 1), NewtonPar(newton_options, verbose = false,))
        prob_ns.l1 = ns.nf.nf.b
        prob_ns.l1 = abs(real(ns.nf.nf.b)) < 1e5 ? real(ns.nf.nf.b) : state.eventValue[2][2]
        #############
    end
    if ~pdjump && pbwrap.prob isa PeriodicOrbitOCollProblem
        if 𝐍𝐒.prm
            ns = neimark_sacker_normal_form_prm(pbwrap, ns0, NewtonPar(newton_options, verbose = true))
        else
            ns = neimark_sacker_normal_form_iooss(pbwrap, ns0)
        end
        if ns.prm
            prob_ns.l1 = ns.nf.nf.b
            prob_ns.l1 = abs(real(ns.nf.nf.b)) < 1e5 ? real(ns.nf.nf.b) : state.eventValue[2][2]
        else
            prob_ns.l1 = ns.nf.nf.d
            prob_ns.l1 = abs(real(prob_ns.l1)) < 1e5 ? real(prob_ns.l1) : state.eventValue[2][2]
        end
    end
    # Witte, Virginie De “Computational Analysis of Bifurcations of Periodic Orbits,” PhD thesis
    c = cos(ω)
    𝐍𝐒.R1 = ω    # μ = {1, 1} this is basically a BT using Iooss normal form
    𝐍𝐒.R2 = c+1  # μ = {1, -1}
    𝐍𝐒.R3 = 2c+1 # μ = {1, exp(±2iπ/3)}
    𝐍𝐒.R4 = c    # μ = {1, exp(±iπ/2)}
    return 𝐍𝐒.R1, 𝐍𝐒.R2, 𝐍𝐒.R3, 𝐍𝐒.R4, real(prob_ns.l1)
end

function compute_eigenvalues(eig::HopfEig, iter::ContIterable{NSPeriodicOrbitCont}, state, u0, par, nev = iter.contparams.nev; k...)
    probma = getprob(iter)
    lens1, lens2 = get_lenses(probma)
    x = getvec(u0, get_formulation(probma))        # ns point
    p1, ω = getp(u0, get_formulation(probma))      # first parameter
    p2 = getp(state.z)                             # second parameter
    par = getparams(probma)
    newpar = _set(par, (lens1, lens2), (p1, p2))
    compute_eigenvalues(eig.eigsolver, iter, state, x, newpar, nev; k...)
end