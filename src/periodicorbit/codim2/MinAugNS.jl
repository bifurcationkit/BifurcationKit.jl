"""
$(SIGNATURES)

For an initial guess from the index of a NS bifurcation point located in ContResult.specialpoint, returns a point which will be refined using `newtonFold`.
"""
function ns_point(br::AbstractBranchResult, index::Int)
    bptype = br.specialpoint[index].type
    if bptype != :ns 
        error("This should be a NS point")
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
@inline getvec(x, ::NeimarkSackerProblemMinimallyAugmented) = get_vec_bls(x, 2)
@inline getp(x, ::NeimarkSackerProblemMinimallyAugmented) = get_par_bls(x, 2)

is_symmetric(::NSMAProblem) = false

# test function for NS bifurcation
nstest(JacNS, v, w, J22, _zero, n; lsbd = MatrixBLS()) = lsbd(JacNS, v, w, J22, _zero, n)

# this function encodes the functional
function (𝐍𝐒::NeimarkSackerProblemMinimallyAugmented)(x, p::𝒯, ω::𝒯, params) where 𝒯
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
    σ1 = nstest(J, a, b, zero(𝒯), 𝐍𝐒.zero, one(𝒯); lsbd = 𝐍𝐒.linbdsolver)[2]
    return residual(𝐍𝐒.prob_vf, x, par), real(σ1), imag(σ1)
end

# this function encodes the functional
function (𝐍𝐒::NeimarkSackerProblemMinimallyAugmented)(x::BorderedArray, params)
    res = 𝐍𝐒(x.u, x.p[1], x.p[2], params)
    return BorderedArray(res[1], [res[2], res[3]])
end

@views function (𝐍𝐒::NeimarkSackerProblemMinimallyAugmented)(x::AbstractVector, params)
    res = 𝐍𝐒(x[begin:end-2], x[end-1], x[end], params)
    return vcat(res[1], res[2], res[3])
end
###################################################################################################
function _get_bordered_terms(𝐍𝐒::NeimarkSackerProblemMinimallyAugmented, x, p::𝒯, ω::𝒯, par) where 𝒯
    a = 𝐍𝐒.a
    b = 𝐍𝐒.b

    # get the PO functional, ie a WrapPOSh, WrapPOTrap, WrapPOColl
    POWrap = 𝐍𝐒.prob_vf

    # parameter axis
    lens = getlens(𝐍𝐒)

    # update parameter
    par0 = set(par, lens, p)

    # we define the following jacobian. It is used at least 3 times below. This avoids doing 3 times the (possibly) costly building of J(x, p)
    JNS = jacobian_neimark_sacker(POWrap, x, par0, ω) # jacobian with period NS boundary condition

    # we do the following in order to avoid computing the jacobian twice in case 𝐍𝐒.Jadjoint is not provided
    JNS★ = has_adjoint(𝐍𝐒) ? jacobian_adjoint_neimark_sacker(POWrap, x, par0, ω) : adjoint(JNS)

    # we solve N[v, σ1] = [0, 1]
    v, σ1, cv, itv = nstest(JNS, a, b, zero(𝒯), 𝐍𝐒.zero, one(𝒯); lsbd = 𝐍𝐒.linbdsolver)
    ~cv && @debug "[codim2 NS] Linear solver for N did not converge."

    # we solve Nᵗ[w, σ2] = [0, 1]
    w, σ2, cv, itw = nstest(JNS★, b, a, zero(𝒯), 𝐍𝐒.zero, one(𝒯); lsbd = 𝐍𝐒.linbdsolverAdjoint)
    ~cv && @debug "[codim2 NS] Linear solver for Nᵗ did not converge."

    δ = getdelta(POWrap)
    ϵ1 = ϵ2 = ϵ3 = 𝒯(δ)
    ################### computation of σx σp ####################
    dₚF = minus(residual(POWrap, x, set(par, lens, p + ϵ1)),
                residual(POWrap, x, set(par, lens, p - ϵ1))); rmul!(dₚF, 𝒯(1 / (2ϵ1)))
    dJvdp = minus(apply(jacobian_neimark_sacker(POWrap, x, set(par, lens, p + ϵ3), ω), v),
                  apply(jacobian_neimark_sacker(POWrap, x, set(par, lens, p - ϵ3), ω), v));
    rmul!(dJvdp, 𝒯(1/(2ϵ3)))
    σₚ = -dot(w, dJvdp)

    # case of ∂σ_ω
    σω = -(dot(w, apply(jacobian_neimark_sacker(POWrap, x, par, ω+ϵ2), v)) - 
           dot(w, apply(jacobian_neimark_sacker(POWrap, x, par, ω), v)) )/ϵ2

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
    rmul!(dJvdt, 𝒯(1/(2ϵ3)))
    σt = -dot(w, dJvdt) 

    _Jpo = jacobian(POWrap, x, par0)
    Jns = hcat(_Jpo.jacpb, dₚF, zero(dₚF))
    Jns = vcat(Jns, vcat(real(σx), real(σt), real(σₚ), real(σω))')
    Jns = vcat(Jns, vcat(imag(σx), imag(σt), imag(σₚ), imag(σω))')
end
###################################################################################################
# Struct to invert the jacobian of the ns MA problem.
struct NSLinearSolverMinAug <: AbstractLinearSolver; end

function NSMALinearSolver(x, p::𝒯, ω::𝒯, 𝐍𝐒::NeimarkSackerProblemMinimallyAugmented, par,
                            duu, dup, duω;
                            debugArray = nothing) where 𝒯
    ################################################################################################
    # debugArray is used as a temp to be filled with values used for debugging. 
	# If debugArray = nothing, then no debugging mode is entered. 
	# If it is AbstractArray, then it is populated
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
        rmul!(dJvdt, 𝒯(1/(2ϵ3)))
        σt = -dot(w, dJvdt) 

        _Jpo = jacobian(POWrap, x, par0)

        x1, x2, cv, (it1, it2) = 𝐍𝐒.linsolver(_Jpo, duu, dₚF)
        ~cv && @debug "[codim2 NS] Linear solver for N did not converge."

        σxx1 = dot(vcat(σx,σt), x1)
        σxx2 = dot(vcat(σx,σt), x2)

        dp, dω = [real(σₚ - σxx2) real(σω);
                  imag(σₚ + σxx2) imag(σω) ] \
                  [dup - real(σxx1), duω + imag(σxx1)]

        # Jns = hcat(_Jpo, dₚF, zero(dₚF))
        # Jns = vcat(Jns, vcat(real(σx), real(σt), real(σₚ), real(σω))')
        # Jns = vcat(Jns, vcat(imag(σx), imag(σt), imag(σₚ), imag(σω))')

        # sol = Jns \ vcat(duu,dup,duω)
        # return sol[1:end-2], sol[end-1],sol[end],true,2

        # Jfd = ForwardDiff.jacobian(z->𝐍𝐒(z,par0),vcat(x,p,ω))

        # display(Jfd[end, 1:19]')
        # display(vcat(imag(σx), imag(σt), imag(σₚ), imag(σω))')

        # @debug "" norm(Jns-Jfd, Inf) dp dω

        # Jns .= Jfd

        if debugArray isa AbstractArray
            Jns = hcat(_Jpo.jacpb, dₚF, zero(dₚF))
            Jns = vcat(Jns, vcat(real(σx), real(σt), real(σₚ), real(σω))')
            Jns = vcat(Jns, vcat(imag(σx), imag(σt), imag(σₚ), imag(σω))')
            debugArray .= Jns
        end
        
        return x1 .- dp .* x2, dp, dω, true, it1 + it2 + sum(itv) + sum(itw)
    else
        @assert false "WIP. Please select another jacobian method like :autodiff or :finiteDifferences. You can also pass the option usehessian = false."
    end

    return dX, dsig, true, sum(it) + sum(itv) + sum(itw)
end

function (pdls::NSLinearSolverMinAug)(Jns, rhs::BorderedArray{vectype, 𝒯}; debugArray = nothing, kwargs...) where {vectype, 𝒯}
    # kwargs is used by AbstractLinearSolver
    out = NSMALinearSolver((Jns.x).u,
                (Jns.x).p[1],
                (Jns.x).p[2],
                Jns.nspb,
                Jns.params,
                rhs.u, rhs.p[1], rhs.p[2];
                debugArray = debugArray)
    # this type annotation enforces type stability
    return BorderedArray{vectype, 𝒯}(out[1], [out[2], out[3]]), out[4], out[5]
end
###################################################################################################
residual(nspb::NSMAProblem, x, p) = nspb.prob(x, p)
residual!(nspb::NSMAProblem, out, x, p) = (copyto!(out, nspb.prob(x, p)); out)
@inline getdelta(nspb::NSMAProblem) = getdelta(nspb.prob)
save_solution(::NSMAProblem, x ,p) = x

# we add :hopfpb in order to use HopfEig
jacobian(nspb::NSMAProblem{Tprob, Nothing, Tu0, Tp, Tl, Tplot, Trecord}, x, p) where {Tprob, Tu0, Tp, Tl <: Union{AllOpticTypes, Nothing}, Tplot, Trecord} = (x = x, params = p, nspb = nspb.prob, hopfpb = nspb.prob)

jacobian(nspb::NSMAProblem{Tprob, AutoDiff, Tu0, Tp, Tl, Tplot, Trecord}, x, p) where {Tprob, Tu0, Tp, Tl <: Union{AllOpticTypes, Nothing}, Tplot, Trecord} = ForwardDiff.jacobian(z -> nspb.prob(z, p), x)

jacobian(nspb::NSMAProblem{Tprob, FiniteDifferences, Tu0, Tp, Tl, Tplot, Trecord}, x, p) where {Tprob, Tu0, Tp, Tl <: Union{AllOpticTypes, Nothing}, Tplot, Trecord} = finite_differences(z -> nspb.prob(z, p), x; δ = 1e-8)

jacobian(nspb::NSMAProblem{Tprob, FiniteDifferencesMF, Tu0, Tp, Tl, Tplot, Trecord}, x, p) where {Tprob, Tu0, Tp, Tl <: Union{AllOpticTypes, Nothing}, Tplot, Trecord} = dx -> (nspb.prob(x .+ 1e-8 .* dx, p) .- nspb.prob(x .- 1e-8 .* dx, p)) / (2e-8)
###################################################################################################
function continuation_ns(prob, alg::AbstractContinuationAlgorithm,
                        nspointguess::BorderedArray{vectype, 𝒯b}, par,
                        lens1::AllOpticTypes, lens2::AllOpticTypes,
                        eigenvec, eigenvec_ad,
                        options_cont::ContinuationPar ;
                        normC = norm,

                        update_minaug_every_step = 1,
                        bdlinsolver::AbstractBorderedLinearSolver = MatrixBLS(),
                        bdlinsolver_adjoint::AbstractBorderedLinearSolver = bdlinsolver,

                        jacobian_ma::Symbol = :autodiff,
                        compute_eigen_elements = false,
                        kind = NSCont(),
                        usehessian = false,
                        plot_solution = BifurcationKit.plot_solution(prob),
                        prm = false,
                        kwargs...) where {𝒯b, vectype}
    @assert lens1 != lens2 "Please choose 2 different parameters. You only passed $lens1"
    @assert lens1 == getlens(prob)

    # options for the Newton Solver inheritated from the ones the user provided
    options_newton = options_cont.newton_options
    # tolerance for detecting R1 bifurcation and stopping continuation
    ϵR1 = 100options_newton.tol

    𝐍𝐒 = NeimarkSackerProblemMinimallyAugmented(
            prob,
            _copy(eigenvec),
            _copy(eigenvec_ad),
            options_newton.linsolver,
            # do not change linear solver if user provides it
            @set bdlinsolver.solver = (isnothing(bdlinsolver.solver) ? options_newton.linsolver : bdlinsolver.solver);
            linbdsolve_adjoint = bdlinsolver_adjoint,
            usehessian = usehessian)

    @assert jacobian_ma in (:autodiff, :finiteDifferences, :minaug, :finiteDifferencesMF, :MinAugMatrixBased)

    # Jacobian for the NS problem
    if jacobian_ma == :autodiff
        nspointguess = vcat(nspointguess.u, nspointguess.p...)
        prob_ns = NSMAProblem(𝐍𝐒, AutoDiff(), nspointguess, par, lens2, plot_solution, prob.recordFromSolution)
        opt_ns_cont = @set options_cont.newton_options.linsolver = DefaultLS()
    elseif jacobian_ma == :finiteDifferences
        nspointguess = vcat(nspointguess.u, nspointguess.p...)
        prob_ns = NSMAProblem(𝐍𝐒, FiniteDifferences(), nspointguess, par, lens2, plot_solution, prob.recordFromSolution)
        opt_ns_cont = @set options_cont.newton_options.linsolver = options_cont.newton_options.linsolver
    elseif jacobian_ma == :finiteDifferencesMF
        nspointguess = vcat(nspointguess.u, nspointguess.p...)
        prob_ns = NSMAProblem(𝐍𝐒, FiniteDifferencesMF(), nspointguess, par, lens2, plot_solution, prob.recordFromSolution)
        opt_ns_cont = @set options_cont.newton_options.linsolver = options_cont.newton_options.linsolver
    elseif jacobian_ma == :MinAugMatrixBased
        nspointguess = vcat(nspointguess.u, nspointguess.p...)
        prob_ns = NSMAProblem(𝐍𝐒, MinAugMatrixBased(), nspointguess, par, lens2, plot_solution, prob.recordFromSolution)
        opt_ns_cont = @set options_cont.newton_options.linsolver = options_cont.newton_options.linsolver
    
    else
        prob_ns = NSMAProblem(𝐍𝐒, nothing, nspointguess, par, lens2, plot_solution, prob.recordFromSolution)
        opt_ns_cont = @set options_cont.newton_options.linsolver = NSLinearSolverMinAug()
    end

    # this functions allows to tackle the case where the two parameters have the same name
    lenses = get_lens_symbol(lens1, lens2)

    # current lyapunov coefficient
    𝒯 = eltype(𝒯b)
    𝐍𝐒.l1 = Complex{𝒯}(1, 0)
    𝐍𝐒.R1 = zero(𝒯)
    𝐍𝐒.R2 = zero(𝒯)
    𝐍𝐒.R3 = zero(𝒯)
    𝐍𝐒.R4 = zero(𝒯)

    # this function is used as a Finalizer
    # it is called to update the Minimally Augmented problem
    # by updating the vectors a, b
    function update_min_aug_ns(z, tau, step, contResult; kUP...)
        # user-passed finalizer
        finaliseUser = get(kwargs, :finalise_solution, nothing)
        # we first check that the continuation step was successful
        # if not, we do not update the problem with bad information!
        success = get(kUP, :state, nothing).converged
        if (~mod_counter(step, update_minaug_every_step) || success == false)
            # we call the user finalizer
            return _finsol(z, tau, step, contResult; prob = 𝐍𝐒, kUP...)
        end
        @debug "[codim2 NS] Update a / b dans NS"

        x = getvec(z.u, 𝐍𝐒)   # NS point
        p1, ω = getp(z.u, 𝐍𝐒) # first parameter
        p2 = z.p              # second parameter
        newpar = set(par, lens1, p1)
        newpar = set(newpar, lens2, p2)

        a = 𝐍𝐒.a
        b = 𝐍𝐒.b

        # get the PO functional
        POWrap = 𝐍𝐒.prob_vf

        # compute new b
        JNS = jacobian_neimark_sacker(POWrap, x, newpar, ω)
        newb,_,cv,it = nstest(JNS, a, b, zero(𝒯), 𝐍𝐒.zero, one(𝒯); lsbd = 𝐍𝐒.linbdsolver)
        ~cv && @debug "[codim2 NS] Linear solver for N did not converge. it = $it"

        # compute new a
        JNS★ = has_adjoint(𝐍𝐒) ? jacobian_adjoint_neimark_sacker(POWrap, x, newpar, ω) : adjoint(JNS)
        newa,_,cv,it = nstest(JNS★, b, a, zero(𝒯), 𝐍𝐒.zero, one(𝒯); lsbd = 𝐍𝐒.linbdsolverAdjoint)
        ~cv && @debug "[codim2 NS] Linear solver for N★ did not converge. it = $it"

        copyto!(𝐍𝐒.a, newa); rmul!(𝐍𝐒.a, 1/normC(newa))
        # do not normalize with dot(newb, 𝐍𝐒.a), it prevents detection of resonances
        copyto!(𝐍𝐒.b, newb); rmul!(𝐍𝐒.b, 1/normC(newb))

        # we stop continuation at R1, PD points
        # test if we jumped to PD branch
        pdjump = abs(abs(ω) - pi) < 100options_newton.tol

        isbif = isnothing(contResult) ? true : isnothing(findfirst(x -> x.type in (:R1, :pd), contResult.specialpoint))

         # if the frequency is null, this is not a NS point, we halt the process
         stop_R1 = 1-cos(ω) <= ϵR1
         if stop_R1
            @warn "[Codim 2 NS - Finalizer]\n The NS curve seems to be close to a R1 point: ω ≈ $ω.\n Stopping computations at $(get_lenses(contResult)) = ($p1, $p2).\n If the R1 point is not detected, try lowering Newton tolerance or dsmax."
        end

        if pdjump
            @warn "[Codim 2 NS - Finalizer] The NS curve seems to jump to a PD curve. Stopping computations at ($p1, $p2). Perhaps it is close to a R2 bifurcation for example."
        end

        # call the user-passed finalizer
        final_result = _finsol(z, tau, step, contResult; prob = 𝐍𝐒, kUP...)

        return ~stop_R1 && isbif && final_result && ~pdjump
    end

    function test_ch(iter, state)
        z = getx(state)
        x = getvec(z, 𝐍𝐒)   # NS point
        p1, ω = getp(z, 𝐍𝐒) # first parameter
        p2 = getp(state)    # second parameter
        newpar = set(par, lens1, p1)
        newpar = set(newpar, lens2, p2)

        prob_ns = iter.prob.prob
        pbwrap = prob_ns.prob_vf

        ns0 = NeimarkSacker(copy(x), nothing, p1, ω, newpar, lens1, nothing, nothing, nothing, :none)
        # test if we jumped to PD branch
        pdjump = abs(abs(ω) - pi) < 100options_newton.tol
        if ~pdjump && pbwrap.prob isa ShootingProblem
            ns = neimark_sacker_normal_form(pbwrap, ns0, (1, 1), NewtonPar(options_newton, verbose = false,))
            prob_ns.l1 = ns.nf.nf.b
            prob_ns.l1 = abs(real(ns.nf.nf.b)) < 1e5 ? real(ns.nf.nf.b) : state.eventValue[2][2]
            #############
        end
        if ~pdjump && pbwrap.prob isa PeriodicOrbitOCollProblem
            if prm
                ns = neimark_sacker_normal_form_prm(pbwrap, ns0, NewtonPar(options_newton, verbose = true))
            else
                ns = neimark_sacker_normal_form(pbwrap, ns0)
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

    # change the user provided functions by passing probPO in its parameters
    _finsol = modify_po_finalise(prob_ns, kwargs, prob.prob.update_section_every_step)

    # the following allows to append information specific to the codim 2 continuation to the user data
    _recordsol = get(kwargs, :record_from_solution, nothing)
    _recordsol2 = isnothing(_recordsol) ?
        (u, p; kw...) -> (; _namedrecordfromsol(record_from_solution(prob)(getvec(u, 𝐍𝐒), p; kw...))...,
                zip(lenses, (getp(u, 𝐍𝐒)[1], p))...,
                ωₙₛ = getp(u, 𝐍𝐒)[2],
                CH = real(𝐍𝐒.l1),
                R₁ = 𝐍𝐒.R1,
                R₂ = 𝐍𝐒.R2,
                R₃ = 𝐍𝐒.R3,
                R₄ = 𝐍𝐒.R4, 
                ) :
        (u, p; kw...) -> (; 
            _namedrecordfromsol(_recordsol(getvec(u, 𝐍𝐒), p; kw...))..., 
            zip(lenses, (getp(u, 𝐍𝐒)[1], p))..., 
            ωₙₛ = getp(u, 𝐍𝐒)[2], 
            CH = real(𝐍𝐒.l1), )

    # eigen solver
    eigsolver = HopfEig(getsolver(opt_ns_cont.newton_options.eigsolver), prob_ns)
	
    prob_ns = re_make(prob_ns, record_from_solution = _recordsol2)

    # change the plotter
    _kwargs = (record_from_solution = record_from_solution(prob), plot_solution = plot_solution)
    _plotsol = modify_po_plot(prob_ns, _kwargs)
    prob_ns = re_make(prob_ns, record_from_solution = _recordsol2, plot_solution = _plotsol)

    # define event for detecting bifurcations. Coupled it with user passed events
    # event for detecting codim 2 points
    event_user = get(kwargs, :event, nothing)
    if isnothing(event_user)
        event = ContinuousEvent(5, test_ch, compute_eigen_elements, ("R1", "R2", "R3", "R4", "ch",), 0)
    else
        event = PairOfEvents(
                ContinuousEvent(5, test_ch, compute_eigen_elements, ("R1", "R2", "R3", "R4", "ch",), 0),
                event_user)
    end

    # solve the NS equations
    br_ns_po = continuation(
        prob_ns, alg,
        (@set opt_ns_cont.newton_options.eigsolver = eigsolver);
        linear_algo = BorderingBLS(solver = opt_ns_cont.newton_options.linsolver, check_precision = false),
        kwargs...,
        kind = kind,
        event = event,
        normC = normC,
        finalise_solution = update_min_aug_ns,
        )
    correct_bifurcation(br_ns_po)
end
