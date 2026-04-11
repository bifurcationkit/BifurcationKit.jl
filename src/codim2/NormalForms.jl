"""
$(TYPEDSIGNATURES)

Compute the Cusp normal form.

# Arguments
- `_prob` bifurcation problem
- `pt::Cusp` Cusp bifurcation point
- `ls` linear solver

# Optional arguments
- `δ = 1e-8` used for finite differences
- `verbose` bool to print information
"""
function cusp_normal_form(_prob,
                            br::AbstractBranchResult, ind_bif::Int,
                            Teigvec::Type{𝒯eigvec} = _getvectortype(br);
                            δ = 1e-8,
                            nev = length(eigenvalsfrombif(br, ind_bif)),
                            verbose = false,
                            ζs = nothing,
                            lens = getlens(br),
                            scaleζ = norm) where {𝒯eigvec}
    if br.specialpoint[ind_bif].type != :cusp 
        error("The provided index does not refer to a Cusp Point")
    end

    verbose && println("━"^53*"\n──▶ Cusp Normal form computation")

    # MA problem formulation
    prob_ma = _prob.prob

    # get the vector field
    prob_vf = prob_ma.prob_vf

    # scalar type
    𝒯 = VI.scalartype(𝒯eigvec)

    # linear solvers
    ls = prob_ma.linsolver
    bls = prob_ma.linbdsolver

    # kernel dimension
    N = 1

    # in case nev = 0 (number of unstable eigenvalues), we increase nev to avoid bug
    nev = min(2N, nev)

    # newton parameters
    options = br.contparams.newton_options

    # bifurcation point
    bifpt = br.specialpoint[ind_bif]
    eigRes = br.eig

    # eigenvalue
    if bifpt.ind_ev > 0
        λ = eigRes[bifpt.idx].eigenvals[bifpt.ind_ev]
    else
        λ = rightmost(eigRes[bifpt.idx].eigenvals)[1]
    end

    # parameters for vector field
    x0, parbif = get_bif_point_codim2(br, ind_bif)

    # jacobian at bifurcation point
    L = jacobian(prob_vf, x0, parbif)

    # eigenvectors
    # we recompute the eigen-elements if they were not saved
    verbose && @info "Eigen-elements not saved in the branch. Recomputing them..."
    eigsolver = getsolver(options.eigsolver)
    _λ0, _ev0, _ = eigsolver(L, nev)
    Ivp = sortperm(_λ0, by = abs)
    _λ = _λ0[Ivp]
    if norm(_λ[1:N] .- 0, Inf) > br.contparams.tol_stability
        @warn "We did not find the correct eigenvalues. We found the eigenvalues:\n $(display(( _λ[1:N]))).\n Difference between the eigenvalues:"
        display(_λ[1:N] .- 0)
    end
    ζ = real.(geteigenvector(eigsolver, _ev0, Ivp[1]))
    ζ ./= scaleζ(ζ)

    # extract eigen-elements for adjoint(L), needed for spectral projector
    if is_symmetric(prob_vf)
        λ★ = br.eig[bifpt.idx].eigenvals[bifpt.ind_ev]
        ζ★ = copy(ζ)
    else
        _Jt = has_adjoint(prob_vf) ? jacobian_adjoint(prob_vf, x0, parbif) : adjoint(L)
        ζ★, λ★ = get_adjoint_basis(_Jt, conj(λ), eigsolver; nev, verbose)
    end

    ζ★ = real.(ζ★); λ★ = real.(λ★)

    @assert abs(LA.dot(ζ, ζ★)) > 1e-10 "We got ζ⋅ζ★ = $((LA.dot(ζ, ζ★))). This dot product should not be zero. Perhaps, you can increase `nev` which is currently $nev."
    ζ★ ./= LA.dot(ζ, ζ★)

    # Kuznetsov, Yu. A. “Numerical Normalization Techniques for All Codim 2 Bifurcations of Equilibria in ODE’s.” SIAM Journal on Numerical Analysis 36, no. 4 (January 1, 1999): 1104–24. https://doi.org/10.1137/S0036142998335005.
    # notations from this paper
    B(dx1, dx2) = d2F(prob_vf, x0, parbif, dx1, dx2)
    C(dx1, dx2, dx3) = d3F(prob_vf, x0, parbif, dx1, dx2, dx3)
    q = ζ
    p = ζ★

    h2 = B(q, q)
    h2 .= LA.dot(p, h2) .* q .- h2
    H2, _, cv, it = bls(L, q, p, zero(𝒯), h2, zero(𝒯))
    ~cv && @debug "[CUSP (H2)] Bordered linear solver for J did not converge. iterations = $it"

    c = LA.dot(p, C(q, q, q)) + 3LA.dot(p, B(q, H2))
    c /= 6

    pt = Cusp(
        x0, parbif,
        (getlens(prob_ma), lens),
        ζ, ζ★,
        (c = c, ),
        :none
    )
end

"""
$(TYPEDSIGNATURES)

Compute the Bogdanov-Takens normal form.

# Arguments
- `prob_ma` a `FoldProblemMinimallyAugmented` or `HopfProblemMinimallyAugmented`
- `pt::BogdanovTakens` BogdanovTakens bifurcation point
- `ls` linear solver

# Optional arguments
- `δ = 1e-8` used for finite differences
- `verbose` bool to print information
- `autodiff = true` only for Bogdanov-Takens point. Whether to use ForwardDiff for the many differentiations that are required to compute the normal form.
- `detailed = true` only for Bogdanov-Takens point. Whether to compute only a simplified normal form.
"""
function bogdanov_takens_normal_form(prob_ma, L,
                                    pt::BogdanovTakens;
                                    δ = 1e-8,
                                    verbose = false,
                                    detailed::Val{detailed_type} = Val(true),
                                    autodiff = true,
                                    # bordered linear solver
                                    bls = prob_ma.linbdsolver,
                                    bls_block = bls) where {detailed_type}
    x0 = pt.x0
    parbif = pt.params
    Ty = VI.scalartype(x0)

    # vector field
    VF = prob_ma.prob_vf
    F(x, p) = residual(VF, x, p)

    # for finite differences
    ϵ = convert(Ty, δ)
    ϵ2 = sqrt(ϵ) # for second order differential

    # linear solvers
    ls = prob_ma.linsolver

    lens1, lens2 = pt.lens

    getp(l::AllOpticTypes) = _get(parbif, l)
    setp(l::AllOpticTypes, p::Number) = set(parbif, l, p)
    setp(p1::Number, p2::Number) = set(set(parbif, lens1, p1), lens2, p2)

    ζ0, ζ1 = pt.ζ
    ζs0, ζs1 = pt.ζ★

    G = [LA.dot(xs, x) for xs in pt.ζ★, x in pt.ζ]
    norm(G - LA.I(2), Inf) > 1e-5 && @warn "G == I(2) is not valid. We built a basis such that G = $G"

    G = [LA.dot(xs, apply(L, x)) for xs in pt.ζ★, x in pt.ζ]
    norminf(G - [0 1; 0 0]) > 1e-5 && @warn "G is not close to the Jordan block of size 2. We built a basis such that G = $G. The norm of the difference is $(norminf(G - [0 1; 0 0]))"

    # second differential
    R2(dx1, dx2) = d2F(VF, x0, parbif, dx1, dx2) ./2

    # quadratic coefficients
    R20 = R2(ζ0, ζ0)
    a = LA.dot(ζs1, R20)
    b = 2LA.dot(ζs0, R20) + 2LA.dot(ζs1, R2(ζ0, ζ1))

    # return the normal form coefficients
    pt.nf = (; a, b)
    if detailed_type == false # TODO! THIS MAKES IT TYPE UNSTABLE
        return pt
    end

    ###########################
    # computation of the unfolding. We follow the procedure described in Al-Hdaibat et al. 2016

    # Al-Hdaibat, B., W. Govaerts, Yu. A. Kuznetsov, and H. G. E. Meijer. “Initialization of Homoclinic Solutions near Bogdanov--Takens Points: Lindstedt--Poincaré Compared with Regular Perturbation Method.” SIAM Journal on Applied Dynamical Systems 15, no. 2 (January 2016): 952–80. https://doi.org/10.1137/15M1017491.
    ###########################
    # to have the same notations as in the paper above
    q0 = ζ0 ; q1 = ζ1;
    p0 = ζs0; p1 = ζs1;

    # second differential notations, to be in agreement with Kuznetsov et al.
    B(dx1, dx2) = d2F(VF, x0, parbif, dx1, dx2)
    Ainv(dx) = bls(L, p1, q0, zero(Ty), dx, zero(Ty))

    H2000, _, cv, it = Ainv(2 .* a .* q1 .- B(q0, q0))
    ~cv && @debug "[BT H2000] Linear solver for J did not converge. it = $it"
    γ = (-2LA.dot(p0, H2000) + 2LA.dot(p0, B(q0, q1)) + LA.dot(p1, B(q1, q1))) / 2
    H2000 .+= γ .* q0

    H1100, _, cv, it = Ainv(b .* q1 .+ H2000 .- B(q0, q1))
    ~cv && @debug "[BT H1100] Linear solver for J did not converge. it = $it"

    H0200, _, cv, it = Ainv(2 .* H1100 .- B(q1, q1))
    ~cv && @debug "[BT H0200] Linear solver for J did not converge. it = $it"

    # first order derivatives
    pBq(p, q) = 2 .* (apply_jacobian(VF, x0 .+ ϵ .* q, parbif, p, true) .-
                      apply_jacobian(VF, x0,           parbif, p, true)) ./ ϵ
    A1(q, lens) = (apply_jacobian(VF, x0, setp(lens, _get(parbif, lens) + ϵ), q) .-
                   apply_jacobian(VF, x0, parbif, q)) ./ϵ
    pAq(p, q, lens) =  LA.dot(p, A1(q, lens))

    # second order derivative
    p10 = _get(parbif, lens1); p20 = _get(parbif, lens2);

    if autodiff
        Jp(p, l)  = ForwardDiff.derivative( P -> F(x0, setp(l, P)), p)
        Jpp(p, l) = ForwardDiff.derivative( P -> Jp(P, l), p)
        Fp(p1, p2)  = F(x0, setp(p1, p2))
        Jp1p2(p1, p2) = ForwardDiff.derivative(P1 -> ForwardDiff.derivative(P2 -> Fp(P1, P2) , p2), p1)

        J2_11 = Jpp(p10, lens1)
        J2_22 = Jpp(p20, lens2)
        J2_12 = Jp1p2(p10, p20)
    else #finite  differences. We need to be careful here because (1e-8)^2 is really small!!
        J2_11 = (F(x0, setp(lens1, p10 + ϵ2)) .- 2 .* F(x0, setp(lens1, p10)) .+
                 F(x0, setp(lens1, p10 - ϵ2)) ) ./ ϵ2^2

        J2_22 = (F(x0, setp(lens2, p20 + ϵ2)) .- 2 .* F(x0, setp(lens2, p20)) .+
                 F(x0, setp(lens2, p20 - ϵ2)) )./ ϵ2^2

        J2_12 = (F(x0, setp(p10 + ϵ2, p20 + ϵ2)) .- F(x0, setp(lens1, p10 + ϵ2)) .-
                                                    F(x0, setp(lens2, p20 + ϵ2)) .+ F(x0, parbif))./ ϵ2^2
    end

    # build the big matrix of size (n+2) x (n+2) A = [L J1s; A12 A22]
    J1 = lens -> F(x0, setp(lens, _get(parbif, lens) + ϵ)) ./ ϵ
    J1s = (J1(lens1), J1(lens2))

    A12_1 = pBq(p1, q0) ./2
    A12_2 = (pBq(p0, q0) .+ pBq(p1, q1)) ./2
    A22 = [[pAq(p1, q0, lens1), pAq(p0, q0, lens1)+pAq(p1, q1, lens1)] [pAq(p1, q0, lens2), pAq(p0, q0, lens2)+pAq(p1, q1, lens2)] ]

    # solving the linear system of size n+2
    c = 3LA.dot(p0, H1100) - LA.dot(p0, B(q1, q1))
    H0010, K10, cv, it = solve_bls_block(bls_block, L, J1s, (A12_1, A12_2), A22, q1, [LA.dot(p1, B(q1, q1))/2, c])
    ~cv && @debug "[BT K10] Linear solver for J did not converge. it = $it"
    @assert size(H0010) == size(x0)

    H0001, K11, cv, it = solve_bls_block(bls_block, L, J1s, (A12_1, A12_2), A22, zero(q1), [zero(Ty), one(Ty)])
    ~cv && @debug "[BT K11] Linear solver for J did not converge. it = $it"
    @assert size(H0001) == size(x0)

    # computation of K2
    κ1 = LA.dot(p1, B(H0001, H0001))
    κ2 = pAq(p1, H0001, lens1) * K11[1] +
         pAq(p1, H0001, lens2) * K11[2]
    J2K = @. J2_11 * K11[1]^2 + 2J2_12 * K11[1] * K11[2] + J2_11 * K11[2]^2
    κ3 = LA.dot(p1, J2K)
    K2 = -( κ1 + 2κ2 + κ3 ) .* K10

    # computation of H0002
    h0002 = B(H0001, H0001)
    h0002 .+= A1(H0001, lens1) .* (2K11[1]) .+ A1(H0001, lens2) .* (2K11[2])
    h0002 .+= J2K
    h0002 .+= J1s[1] .* K2[1] .+ J1s[2] .* K2[2]
    H0002, _, ct, it = Ainv(h0002)
    ~cv && @debug "[BT H0002] Linear solver for J did not converge. it = $it"
    H0002 .*= -1

    # computation of H1001
    h1001 = B(q0, H0001)
    h1001 .+= A1(q0, lens1) .* K11[1] .+ A1(q0, lens2) .* K11[2]
    H1001, _, cv, it = Ainv(h1001)
    ~cv && @debug "[BT H1001] Linear solver for J did not converge. it = $it"
    H1001 .*= -1

    # computation of H0101
    h0101 = B(q1, H0001)
    h0101 .+= A1(q1, lens1) .* K11[1] .+ A1(q1, lens2) .* K11[2]
    h0101 .-= H1001 .+ q1
    H0101, _, cv, it = Ainv(h0101)
    ~cv && @debug "[BT H0101] Linear solver for J did not converge. it = $it"
    H0101 .*= -1

    # computation of H3000 and d
    h3000 = d3F(VF, x0, parbif, q0, q0, q0) .+ 3 .* B(q0, H2000) .- (6a) .* H1100
    d = LA.dot(p1, h3000)/6
    h3000 .-= (6d) .* q1
    H3000, _, cv, it = Ainv(h3000)
    ~cv && @debug "[BT H3000] Linear solver for J did not converge. it = $it"
    H3000 .*= -1

    # computation of e
    e = LA.dot(p1, d3F(VF, x0, parbif, q0, q0, q0)) + 2LA.dot(p1, B(q0, H1100)) + LA.dot(p1, B(q1, H2000))
    e += -2b * LA.dot(p1, H1100) - 2a * LA.dot(p1, H0200) - LA.dot(p1, H3000)
    e /= 2

    # computation of H2001 and a1
    B1(q, p, l) = (d2F(VF, x0, setp(l, getp(l) + ϵ), q, p) .- d2F(VF, x0, parbif, q, p)) ./ ϵ
    h2001 = d3F(VF, x0, parbif, q0, q0, H0001) .+ 2 .* B(q0, H1001) .+ B(H0001, H2000)
    h2001 .+= B1(q0, q0, lens1) .* K11[1] .+ B1(q0, q0, lens2) .* K11[2]
    h2001 .+= A1(H2000, lens1)  .* K11[1] .+ A1(H2000, lens2)  .* K11[2]
    h2001 .-= (2a) .* H0101
    a1 = LA.dot(p1, h2001) / 2
    h2001 .-= (2a1) .* q1
    H2001, _, cv, it = Ainv(h2001)
    ~cv && @debug "[BT H2001] Linear solver for J did not converge. it = $it"
    H2001 .*= -1

    # computation of b1
    b1 = LA.dot(p1, d3F(VF, x0, parbif, q0, q1, H0001)) +
         LA.dot(p1, B1(q0, q1, lens1)) * K11[1] +
         LA.dot(p1, B1(q0, q1, lens2)) * K11[2] +
         LA.dot(p1, B(q1, H1001)) +
         LA.dot(p1, B(H0001, H1100)) +
         LA.dot(p1, B(q0, H0101)) +
         LA.dot(p1, A1(H1100, lens1)) * K11[1] + LA.dot(p1, A1(H1100, lens2)) * K11[2] -
         b * LA.dot(p1, H0101) - LA.dot(p1, H1100) - LA.dot(p1, H2001)

    verbose && println(pt.nf)
    return @set pt.nfsupp = (; γ, c, K10, K11, K2, d, e, a1, b1, H0001, H0010, H0002, H1001, H2000)
end

"""
$(TYPEDSIGNATURES)

Compute the predictor for the Hopf curve near the Bogdanov-Takens point.
"""
function predictor(bt::BogdanovTakens, ::Val{:HopfCurve}, ds::T; 
                    verbose = false, 
                    ampfactor = one(T)) where T
    # If we write the normal form [y2, β1 + β2 y2 + a y1^2 + b y1 y2]
    # equilibria y2 = 0, 0 = β1 + a y1^2
    # Characteristic polynomial: t^2 + (-x*b - β2)*t - 2*x*a
    # the fold curve is β1 / a < 0 with x± := ±√(-β1/a)v
    # the Hopf curve is 0 = -x*b - β2, -x⋅a > 0
    # ie β2 = -bx with ±b√(-β1/a)
    (;a, b) = bt.nf
    (;K10, K11, K2) = bt.nfsupp
    lens1, lens2 = bt.lens
    p1 = _get(bt.params, lens1)
    p2 = _get(bt.params, lens2)
    par0 = [p1, p2]
    getx(s) = a > 0 ? -sqrt(abs(s) / a) : sqrt(abs(s) / abs(a))

    function HopfCurve(s)
        if a > 0
            x = -sqrt(abs(s) / a)
            β1 = -abs(s)
        else
            x = sqrt(abs(s) / abs(a))
            β1 = abs(s)
        end
        β2 = -b * x
        ω = sqrt(-2x*a)
        return (pars = par0 .+ K10 .* β1 .+ K11 .* β2 .+ K2 .* (β2^2/2), ω = ω)
    end

    # compute eigenvector corresponding to the Hopf branch
    function EigenVec(s)
        x = getx(s)
        # the jacobian is [0 1; 2x*a b*X+β2] with b*X+β2 = 0
        F = LA.eigen([0 1; 2x*a 0])
        ind = findall(imag.(F.values) .> 0)
        hopfvec = F.vectors[:, ind]
        return bt.ζ[1] .* hopfvec[1] .+ bt.ζ[2] .* hopfvec[2]
    end

    function EigenVecAd(s)
        x = getx(s)
        # the jacobian is [0 1; 2x*a b*X+β2] with b*X+β2 = 0
        F = LA.eigen([0 1; 2x*a 0]')
        ind = findall(imag.(F.values) .< 0)
        hopfvec = F.vectors[:, ind]
        return bt.ζ★[1] .* hopfvec[1] .+ bt.ζ★[2] .* hopfvec[2]
    end

    # compute point on the Hopf curve
    x0 = getx(ds)

    return (
            hopf = t -> HopfCurve(t).pars,
            ω = t -> HopfCurve(t).ω,
            EigenVec = EigenVec,
            EigenVecAd = EigenVecAd,
            x0 = t -> getx(t) .* bt.ζ[1]
            )
end


"""
$(TYPEDSIGNATURES)

Compute the predictor for the Fold curve near the Bogdanov-Takens point.
"""
function predictor(bt::BogdanovTakens, ::Val{:FoldCurve}, ds::T; 
                    verbose = false, 
                    ampfactor = one(T)) where T
    # If we write the normal form [y2, β1 + β2 y2 + a y1^2 + b y1 y2]
    # equilibria y2 = 0, 0 = β1 + a y1^2
    # the fold curve is β1 / a < 0 with x± := ±√(-β1/a)
    # the Hopf curve is 0 = -x*b - β2, x⋅a > 0
    # ie β2 = -bx with ±b√(-β1/a)
    (;a, b) = bt.nf
    (; K10, K11, K2) = bt.nfsupp
    lens1, lens2 = bt.lens
    p1 = _get(bt.params, lens1)
    p2 = _get(bt.params, lens2)
    par0 = [p1, p2]
    getx(s) = a > 0 ? -sqrt(abs(s) / a) : sqrt(abs(s) / abs(a))
    function FoldCurve(s)
        β1 = 0
        β2 = s
        return par0 .+ K10 .* β1 .+ K11 .* β2 .+ K2 .* (β2^2/2)
    end
    return (
            fold = FoldCurve,
            EigenVec = t -> (bt.ζ[1]),
            EigenVecAd = t -> (bt.ζ★[2]),
            x0 = t -> getx(t) .* bt.ζ[1]
            )
end

"""
$(TYPEDSIGNATURES)

Compute the predictor for the curve of homoclinic orbits near the Bogdanov-Takens point.

## Reference

Al-Hdaibat, B., W. Govaerts, Yu. A. Kuznetsov, and H. G. E. Meijer. “Initialization of Homoclinic Solutions near Bogdanov--Takens Points: Lindstedt--Poincaré Compared with Regular Perturbation Method.” SIAM Journal on Applied Dynamical Systems 15, no. 2 (January 2016): 952–80. https://doi.org/10.1137/15M1017491.
"""
function predictor(bt::BogdanovTakens, ::Val{:HomoclinicCurve}, ds::T; 
                    verbose = false, 
                    ampfactor = one(T)) where T
    (;a, b) = bt.nf
    (;K10, K11, K2, b1, e, d, a1) = bt.nfsupp
    (;H0001, H0010, H0002, H1001, H2000) = bt.nfsupp

    lens1, lens2 = bt.lens
    p1 = _get(bt.params, lens1)
    p2 = _get(bt.params, lens2)
    par0 = [p1, p2]

    # formula 63
    τ2 = 4/a * (25/49*b1 - e/b) + 2/(49a^2) * (144/49b^2 - 25b*a1 + 73d)

    # formula 69
    α(ϵ) = @. par0 + (10b*ϵ^2 / (7a)) * K11 + ϵ^4/a * ( -4*K10 + 50b^2/(49a) * K2 + b * τ2 * K11)

    # formula 71
    q0, q1 = bt.ζ

    u0(ξ) = -6sech(ξ)^2 + 2
    v0(ξ) = 12sech(ξ)^2 * tanh(ξ)
    u1(ξ) = 0
    v1(ξ) = -6b/(7a) * tanh(ξ) * v0(ξ)
    u2(ξ) = -3/(49a^2) * (6b^2 - 70b*a1 + 49d) * sech(ξ)^2 - 2(5a1*b + 7d)/(7a^2)

    function xLP(t, ϵ)
        ξ = ϵ * t
        return @. bt.x0 + (ϵ^2/a) * ( (10b/7) * H0001 + u0(ξ) * q0) +
                        (ϵ^3/a) * ( v0(ξ) * q1 + u1(ξ) * q0) +
                        (ϵ^4/a) * ( -4 * H0010 + 50 * b^2/(49a) * H0002 + b*τ2 * H0001 +
                        u2(ξ) * q0 + v1(ξ) * q1 +
                        1/(2a) * u0(ξ)^2 * H2000 + 10b/(7a) * u0(ξ) * H1001)
    end

    return (α = α, orbit = xLP)
end

"""
$(TYPEDSIGNATURES)

Compute the Bogdanov-Takens normal form.

# Arguments
- `prob` bifurcation problem, typically `br.prob`
- `br` branch result from a call to [`continuation`](@ref)
- `ind_bif` index of the bifurcation point in `br`
- `options` options for the Newton solver

# Optional arguments
- `δ = 1e-8` used for finite differences with respect to parameters
- `nev = 5` number of eigenvalues to compute to estimate the spectral projector
- `verbose` bool to print information
- `autodiff = true` Whether to use ForwardDiff for the many differentiations that are required to compute the normal form.
- `detailed = true` Whether to compute only a simplified normal form where not all coefficients are computed.
- `ζs` list of vectors spanning the kernel of `dF` at the bifurcation point. Useful to enforce the basis for the normal form.
- `ζs_ad` list of vectors spanning the kernel of `transpose(dF)` at the bifurcation point. Useful to enforce the basis for the normal form. The vectors must be listed so that the corresponding eigenvalues are equals to the ones associated to each vector in ζs. 
- `scaleζ` function to normalise the kernel basis. Indeed, when used with large vectors and `norm`, it results in ζs and the normal form coefficient being super small.
- `bls` specify Bordered linear solver for dF.
- `bls_adjoint` specify Bordered linear solver for transpose(dF).
"""
function bogdanov_takens_normal_form(_prob,
                                    br::AbstractBranchResult, ind_bif::Int,
                                    Teigvec::Type{𝒯eigvec} = _getvectortype(br);
                                    δ = 1e-8,
                                    nev::Int = length(eigenvalsfrombif(br, ind_bif)),
                                    verbose = false,
                                    ζs = nothing,
                                    ζs_ad = nothing,
                                    lens = getlens(br),
                                    scaleζ = norm,
                                    # bordered linear solver
                                    bls = _prob.prob.linbdsolver,
                                    bls_adjoint = bls,
                                    bls_block = bls,
                                    detailed::Val{detailed_type} = Val(true),
                                    autodiff = true) where {𝒯eigvec, detailed_type}
    @assert br.specialpoint[ind_bif].type == :bt "The provided index does not refer to a Bogdanov-Takens Point"

    # functional
    # get the MA problem
    prob_ma = _prob.prob

    # get the initial vector field
    prob_vf = prob_ma.prob_vf

    @assert prob_ma isa AbstractProblemMinimallyAugmented

    # kernel dimension
    N = 2

    # in case nev = 0 (number of requested eigenvalues), we increase nev to avoid bug
    nev = max(2N, nev)
    verbose && println("━"^53*"\n──▶ Bogdanov-Takens Normal form computation")

    # newton parameters
    optionsN = br.contparams.newton_options

    # bifurcation point
    bifpt = br.specialpoint[ind_bif]

    # parameters for vector field
    x0, parbif = get_bif_point_codim2(br, ind_bif)
    if 𝒯eigvec <: BorderedArray
        x0 = convert(𝒯eigvec.parameters[1], getvec(bifpt.x, prob_ma))
    else
        x0 = convert(𝒯eigvec, getvec(bifpt.x , prob_ma))
    end

    𝒯 = VI.scalartype(𝒯eigvec)
    # jacobian at bifurcation point
    L = jacobian(prob_vf, x0, parbif)

    # and corresponding eigenvectors
    eigsolver = getsolver(optionsN.eigsolver)
    if isnothing(ζs) # do we have a basis for the kernel?
        if haseigenvector(br) == false # are the eigenvector saved in the branch?
            verbose && @info "No eigenvector recorded, computing them on the fly"
            # we recompute the eigen-elements if there were not saved during the computation of the branch
            _λ0, _ev, _ = eigsolver(L, nev)
            Ivp = sortperm(_λ0, by = abs)
            _λ = _λ0[Ivp]
            verbose && (println("──▶ (λs, λs (recomputed)) = "); display(( _λ[1:N])))
            if norm(_λ[1:N] .- 0, Inf) > br.contparams.tol_stability
                @warn "We did not find the correct eigenvalues (see 1st col). We found the eigenvalues displayed in the second column:\n $(display(( _λ[1:N]))).\n Difference between the eigenvalues:"
                display(_λ[1:N] .- 0)
            end
            ζs = [_copy(geteigenvector(eigsolver, _ev, ii)) for ii in Ivp[1:N]]
        else
            # "zero" eigenvalues at bifurcation point
            rightEv = br.eig[bifpt.idx].eigenvals
            # indev = br.specialpoint[ind_bif].ind_ev
            # find the 2 eigenvalues closest to zero
            Ind = sortperm(abs.(rightEv))
            ind0 = Ind[1]
            ind1 = Ind[2]
            verbose && (println("────▶ eigenvalues = ", rightEv[Ind[1:2]]))
            ζs = [_copy(geteigenvector(eigsolver, br.eig[bifpt.idx].eigenvecs, ii)) for ii in (ind0, ind1)]
        end
    end
    ###########################
    # Construction of the basis (ζ0, ζ1), (ζ★0, ζ★1). We follow the procedure described in Al-Hdaibat et al. 2016 on page 972.

    # Al-Hdaibat, B., W. Govaerts, Yu. A. Kuznetsov, and H. G. E. Meijer. “Initialization of Homoclinic Solutions near Bogdanov--Takens Points: Lindstedt--Poincaré Compared with Regular Perturbation Method.” SIAM Journal on Applied Dynamical Systems 15, no. 2 (January 2016): 952–80. https://doi.org/10.1137/15M1017491.
    ###########################
    vr = real.(ζs[1])
    Lᵗ = has_adjoint(prob_vf) ? jacobian_adjoint(prob_vf, x0, parbif) : transpose(L)
    if isnothing(ζs_ad) # do we have a basis for the kernel of the adjoint?
        _λ★, _ev★, _ = eigsolver(Lᵗ, nev)
        Ivp = sortperm(_λ★, by = abs)
        # in case the prob is HopfMA, we enforce real values
        vl = real.(geteigenvector(eigsolver, _ev★, Ivp[1]))
    else
        vl = real(ζs_ad[1])
    end

    zerov = real.(prob_ma.zero)
    q0, _, cv, it = bls(L, vl, vr, zero(𝒯), zerov, one(𝒯))
    ~cv && @debug "[BT basis] Linear solver for J  did not converge. it = $it"
    p1, _, cv, it = bls_adjoint(Lᵗ, vr, vl, zero(𝒯), zerov, one(𝒯))
    ~cv && @debug "[BT basis] Linear solver for J' did not converge. it = $it"
    q1, _, cv, it = bls(L, p1, q0, zero(𝒯), q0,    zero(𝒯))
    ~cv && @debug "[BT basis] Linear solver for J  did not converge. it = $it"
    p0, _, cv, it = bls_adjoint(Lᵗ, q0, p1, zero(𝒯), p1,    zero(𝒯))
    ~cv && @debug "[BT basis] Linear solver for J' did not converge. it = $it"

    # we want
    # A⋅q0 = 0, A⋅q1 = q0
    # At⋅p1 = 0, At⋅p0 = p1
    μ = √(abs(LA.dot(q0, q0)))
    q0 ./= μ
    q1 ./= μ
    q1 .= q1 .- LA.dot(q0, q1) .* q0
    ν = LA.dot(q0, p0)
    p1 ./= ν
    p0 .= p0 .- LA.dot(p0, q1) .* p1
    p0 ./= ν

    pt = BogdanovTakens(
        x0, parbif, (getlens(prob_ma), lens),
        (;q0, q1), (;p0, p1),
        (a = zero(𝒯), b = zero(𝒯) ),
        (K2 = zero(𝒯),),
        :none
    )

    return bogdanov_takens_normal_form(prob_ma, L, pt; 
                δ,
                verbose,
                detailed,
                autodiff,
                bls,
                bls_block)
end
####################################################################################################
function bautin_normal_form(_prob::HopfMAProblem,
                            br::AbstractBranchResult, ind_bif::Int,
                            Teigvec::Type{𝒯eigvec} = _getvectortype(br);
                            δ = 1e-8,
                            nev = length(eigenvalsfrombif(br, ind_bif)),
                            verbose = false,
                            ζs = nothing,
                            lens = getlens(br),
                            scaleζ = norm,
                            detailed = false) where {𝒯eigvec}
    @assert br.specialpoint[ind_bif].type == :gh "The provided index does not refer to a Bautin Point"

    verbose && println("━"^53*"\n──▶ Bautin Normal form computation")

    # get the MA problem
    prob_ma = _prob.prob

    # get the initial vector field
    prob_vf = prob_ma.prob_vf

    # scalar type
    𝒯 = VI.scalartype(𝒯eigvec)
    ϵ = 𝒯(δ)

    # functional
    if ~(prob_ma isa HopfProblemMinimallyAugmented)
        error("[Bautin normal form] You need to provide a curve of Hopf points.\nThe underlying problem is not a `HopfProblemMinimallyAugmented`.\nWe found the type: $(typeof(prob_ma))")
    end
    ls = prob_ma.linsolver
    bls = prob_ma.linbdsolver

    # ``kernel'' dimension
    N = 2

    # in case nev = 0 (number of unstable eigenvalues), we increase nev to avoid bug
    nev = max(N, nev)

    # newton parameters
    optionsN = br.contparams.newton_options

    # bifurcation point
    bifpt = br.specialpoint[ind_bif]
    eigRes = br.eig

    # eigenvalue
    ω = abs(getp(bifpt.x, prob_ma)[2])
    λ = Complex(0, ω)

    # parameters for vector field
    x0, parbif = get_bif_point_codim2(br, ind_bif)

    # jacobian at bifurcation point
    if 𝒯eigvec <: BorderedArray
        x0 = convert(𝒯eigvec.parameters[1], getvec(bifpt.x, prob_ma))
    else
        x0 = convert(𝒯eigvec, getvec(bifpt.x , prob_ma))
    end

    # jacobian at bifurcation point
    L = jacobian(prob_vf, x0, parbif)

    # right eigenvector
    if haseigenvector(br) == false
        # we recompute the eigen-elements if there were not saved during the computation of the branch
        verbose && @info "Recomputing eigenvector on the fly"
        _λ, _ev, _ = optionsN.eigsolver.eigsolver(L, nev)
        _ind = argmin(abs.(_λ .- λ))
        verbose && @info "The eigenvalue is $(_λ[_ind])"
        abs(_λ[_ind] - λ) > 10br.contparams.newton_options.tol && @warn "We did not find the correct eigenvalue $λ. We found $(_λ[_ind])"
        ζ = geteigenvector(optionsN.eigsolver, _ev, _ind)
    else
        _λ = br.eig[bifpt.idx].eigenvals
        _ind = argmin(abs.(_λ .- λ))
        ζ = _copy(geteigenvector(optionsN.eigsolver, br.eig[bifpt.idx].eigenvecs, _ind))
    end
    ζ ./= scaleζ(ζ)

    # left eigen-elements
    _Jt = has_adjoint(prob_vf) ? jacobian_adjoint(prob_vf, x0, parbif) : adjoint(L)
    ζ★, λ★ = get_adjoint_basis(_Jt, conj(_λ[_ind]), optionsN.eigsolver.eigsolver; nev, verbose)

    # check that λ★ ≈ conj(λ)
    abs(λ + λ★) > 1e-2 && @warn "We did not find the left eigenvalue for the Hopf point to be very close to the imaginary part, $λ ≈ $(λ★) and $(abs(λ + λ★)) ≈ 0?\n You can perhaps increase the number of computed eigenvalues, the number is nev = $nev."

    # normalise left eigenvector
    ζ★ ./= LA.dot(ζ, ζ★)
    @assert LA.dot(ζ, ζ★) ≈ 1

    # second order differential, to be in agreement with Kuznetsov et al.
    B = BilinearMap( (dx1, dx2) -> d2F(prob_vf, x0, parbif, dx1, dx2) )
    C = TrilinearMap((dx1, dx2, dx3) -> d3F(prob_vf, x0, parbif, dx1, dx2, dx3) )

    q0 = ζ
    p0 = ζ★
    cq0 = conj(q0)

    # normal form computation based on 
    # REF1 Kuznetsov, Yu. A. “Numerical Normalization Techniques for All Codim 2 Bifurcations of Equilibria in ODE’s.” https://doi.org/10.1137/S0036142998335005.

    # formula (7.2) in REF1
    H20, cv, it = ls(L, B(q0, q0); a₀ = Complex(0, 2ω), a₁ = -1)
    ~cv && @debug "[Bautin H20] Linear solver for J did not converge. it = $it"

    # formula (7.3) in REF1
    H11, cv, it = ls(L, -B(q0, cq0))
    ~cv && @debug "[Bautin H11] Linear solver for J did not converge. it = $it"

    # formula (7.4) in REF1
    H30, cv, it = ls(L, C(q0, q0, q0) .+ 3 .* B(q0, H20); a₀ = Complex(0, 3ω), a₁ = -1)
    ~cv && @debug "[Bautin H30] Linear solver for J did not converge. it = $it"

    # formula (7.5) in REF1
    h21 = C(q0, q0, cq0) .+ B(cq0, H20) .+ 2 .* B(q0, H11)
    G21 = LA.dot(p0, h21)      # (7.6)
    h21 .= G21 .* q0 .- h21 # (7.7)

    # formula (7.7) in REF1
    H21, _, cv, it = bls(L, q0, p0, zero(𝒯), h21, zero(𝒯); shift = Complex{𝒯}(0, -ω))
    ~cv && @debug "[Bautin H21] Bordered linear solver for J did not converge. it = $it"

    # 4-th order coefficient
    d4F(x0, dx1, dx2, dx3, dx4) = (d3F(prob_vf, x0 .+ ϵ .* dx4, parbif, dx1, dx2, dx3) .-
                                   d3F(prob_vf, x0 .- ϵ .* dx4, parbif, dx1, dx2, dx3)) ./(2ϵ)

    # implement 4th order differential with finite differences
    function D(x0, dx1, dx2, dx3, dx4)
        dx4r = real.(dx4); dx4i = imag.(dx4);
        # C(dx, dx4r) + i * C(dx, dx4i)
        trilin_r = TrilinearMap((_dx1, _dx2, _dx3) -> d4F(x0, _dx1, _dx2, _dx3, dx4r) )
        out1 = trilin_r(dx1, dx2, dx3)
        trilin_i = TrilinearMap((_dx1, _dx2, _dx3) -> d4F(x0, _dx1, _dx2, _dx3, dx4i) )
        out2 = trilin_i(dx1, dx2, dx3)
        return out1 .+ im .* out2
    end

    # h40 is not needed, so we compute the next formula on page 1114 in REF1
    h31 = D(x0, q0, q0, q0, cq0) .+ 3 .* C(q0, q0, H11) .+ 3 .* C(q0, cq0, H20) .+ 3 .* B(H20, H11)
    h31 .+= B(cq0, H30) .+ 3 .* B(q0, H21) .- (3 * G21) .* H20
    H31, cv, it = ls(L, h31; a₀ = Complex(0, 2ω), a₁ = -1)
    ~cv && @debug "[Bautin H31] Linear solver for J did not converge. it = $it"

    h22 = D(x0, q0, q0, cq0, cq0) .+
        4 .* C(q0, cq0, H11) .+ C(cq0, cq0, H20) .+ C(q0, q0, conj.(H20)) .+
        2 .* B(H11, H11) .+ 2 .* B(q0, conj.(H21)) .+ 2 .* B(cq0, H21) .+ B(conj.(H20), H20) .-
        (2G21 + 2conj(G21)) .* H11
    H22, cv, it = ls(L, h22)
    ~cv && @debug "[Bautin H22] Linear solver for J did not converge. it = $it"
    H22 .*= -1

    # 5-th order coefficient
    # implement 5th order differential with finite differences
    function E(dx1, dx2, dx3, dx4, dx5)
        dx5r = real.(dx5); dx5i = imag.(dx5);
        out1 = (D(x0 .+ ϵ .* dx5r, dx1, dx2, dx3, dx4) .-
                D(x0 .- ϵ .* dx5r, dx1, dx2, dx3, dx4)) ./(2ϵ)
        out2 = (D(x0 .+ ϵ .* dx5i, dx1, dx2, dx3, dx4) .-
                D(x0 .- ϵ .* dx5i, dx1, dx2, dx3, dx4)) ./(2ϵ)
        return out1 .+ im .* out2
    end

    G32 = LA.dot(p0, E(q0, q0, q0, cq0, cq0))
    G32 += LA.dot(p0, D(x0, q0, q0, q0, conj.(H20))) +
          3LA.dot(p0, D(x0, q0, cq0, cq0, H20)) +
          6LA.dot(p0, D(x0, q0, q0, cq0, H11))

    G32 += LA.dot(p0, C(cq0, cq0, H30)) +
          3LA.dot(p0, C(q0, q0, conj.(H21))) +
          6LA.dot(p0, C(q0, cq0, H21)) +
          3LA.dot(p0, C(q0, conj.(H20), H20)) +
          6LA.dot(p0, C(q0, H11, H11)) +
          6LA.dot(p0, C(cq0, H20, H11))

    G32 += 2LA.dot(p0, B(cq0, H31)) +
           3LA.dot(p0, B(q0, H22)) +
            LA.dot(p0, B(conj(H20), H30)) +
           3LA.dot(p0, B(conj(H21), H20)) +
           6LA.dot(p0, B(H11, H21))

    # second Lyapunov coefficient
    l2 = real(G32) / 12

    pt = Bautin(
        x0, parbif,
        (getlens(prob_ma), lens),
        ζ, ζ★,
        (;ω, G21, G32, l2),
        :none
    )

    # case of simplified normal form
    if detailed == false
        return pt
    end

    ###########################
    # computation of the unfolding
    # the unfolding are in 
    # REF2 “Switching to Nonhyperbolic Cycles from Codim 2 Bifurcations of Equilibria in ODEs,” 2005. https://doi.org/10.1016/j.physd.2008.06.006.

    # this part is for branching to Fold of periodic orbits
    VF = prob_ma.prob_vf
    F(x, p) = residual(prob_vf, x, p)

    lens1, lens2 = pt.lens
    _getp(l::AllOpticTypes) = _get(parbif, l)
    _setp(l::AllOpticTypes, p::Number) = set(parbif, l, p)
    _setp(p1::Number, p2::Number) = set(set(parbif, lens1, p1), lens2, p2)
    _A1(q, lens) = (apply_jacobian(VF, x0, _setp(lens, _get(parbif, lens) + ϵ), q) .-
                       apply_jacobian(VF, x0, parbif, q)) ./ϵ
    A1(q, lens) = _A1(real(q), lens) .+ im .* _A1(imag(q), lens)
    A1(q::T, lens) where {T <: AbstractArray{<: Real}} = _A1(q, lens)
    Bp(pars) = BilinearMap( (dx1, dx2) -> d2F(prob_vf, x0, pars, dx1, dx2) )
    B1(q, p, l) = (Bp(_setp(l, _getp(l) + ϵ))(q, p) .- B(q, p)) ./ ϵ
    J1(lens) = F(x0, _setp(lens, _get(parbif, lens) + ϵ)) ./ ϵ

    # formula 17 in REF2
    h₀₀₁₀, = ls(L, J1(lens1)); h₀₀₁₀ .*= -1
    h₀₀₀₁, = ls(L, J1(lens2)); h₀₀₀₁ .*= -1
    γ₁₁₀ = LA.dot(p0, A1(q0, lens1) + B(q0, h₀₀₁₀))
    γ₁₀₁ = LA.dot(p0, A1(q0, lens2) + B(q0, h₀₀₀₁))

    # compute the lyapunov coefficient l1, conform to notations from above paper
    # formulas (15a - 15c) in REF2
    h₂₀₀₀ = H20
    h₁₁₀₀ = H11
    l1 = G21/2
    h₂₁₀₀ = H21

    # formula (19) in REF2
    Ainv(dx) = bls(L, q0, p0, zero(𝒯), dx, zero(𝒯); shift = -λ)
    h₁₀₁₀, = Ainv(γ₁₁₀ .* q0 .- A1(q0, lens1) .- B(q0, h₀₀₁₀) )
    h₁₀₀₁, = Ainv(γ₁₀₁ .* q0 .- A1(q0, lens2) .- B(q0, h₀₀₀₁) )

    # formula (20a) in REF2
    tmp2010 = (2γ₁₁₀) .* h₂₀₀₀ .- (C(q0, q0, h₀₀₁₀) .+ 2 .* B(q0, h₁₀₁₀) .+ B(h₂₀₀₀, h₀₀₁₀) .+ B1(q0, q0, lens1) .+ A1(h₂₀₀₀, lens1))
    h₂₀₁₀, = ls(L, tmp2010; a₀ = Complex(0, -2ω) )

    # formula (20a) in REF2
    tmp2001 = (2γ₁₀₁) .* h₂₀₀₀ .- (C(q0, q0, h₀₀₀₁) .+ 2 .* B(q0, h₁₀₀₁) .+ B(h₂₀₀₀, h₀₀₀₁) .+ B1(q0, q0, lens2) .+ A1(h₂₀₀₀, lens2))
    h₂₀₀₁, = ls(L, tmp2001; a₀ = Complex(0, -2ω) )

    # formula (20b) in REF2
    tmp1110 = 2real(γ₁₁₀) .* h₁₁₀₀ .- (C(q0, cq0, h₀₀₁₀) .+ B(h₁₁₀₀, h₀₀₁₀) .+ 2 .* real(B(cq0, h₁₀₁₀)) .+ B1(q0, cq0, lens1) .+ A1(h₁₁₀₀, lens1))
    h₁₁₁₀, = ls(L, tmp1110)

    # formula (20b) in REF2
    tmp1101 = 2real(γ₁₀₁) .* h₁₁₀₀ .- (C(q0, cq0, h₀₀₀₁) .+ B(h₁₁₀₀, h₀₀₀₁) .+ 2 .* real(B(cq0, h₁₀₀₁)) .+ B1(q0, cq0, lens2) .+ A1(h₁₁₀₀, lens2))
    h₁₁₀₁, = ls(L, tmp1101)

    _C1(pars) = TrilinearMap((dx1, dx2, dx3) -> d3F(prob_vf, x0, pars, dx1, dx2, dx3) )
    C1(dx1, dx2, dx3, l) = (_C1(_setp(l, _getp(l) + ϵ))(dx1, dx2, dx3) .- C(dx1, dx2, dx3)) ./ ϵ 

    # formula (21) in REF2
    tmp2110 = D(x0, q0, q0, cq0, h₀₀₁₀) .+
            2 .* C(q0, h₁₁₀₀, h₀₀₁₀) .+
            2 .* C(q0, cq0, h₁₀₁₀) .+
            C(q0, q0, conj(h₁₀₁₀)) .+
            C(h₂₀₀₀, cq0, h₀₀₁₀) .+
            2 .* B(q0, h₁₁₁₀) .+
            2 .* B(h₁₁₀₀, h₁₀₁₀) .+
            B(h₂₀₀₀, conj(h₁₀₁₀)) .+
            B(h₂₁₀₀, h₀₀₁₀) .+
            B(h₂₀₁₀, cq0) .+
            C1(q0, q0, cq0, lens1) .+
            2 .* B1(h₁₁₀₀, q0, lens1) .+ B1(h₂₀₀₀, cq0, lens1) .+ A1(h₂₁₀₀, lens1)

    # formula (21) in REF2
    tmp2101 = D(x0, q0, q0, cq0, h₀₀₀₁) .+
            2 .* C(q0, h₁₁₀₀, h₀₀₀₁) .+
            2 .* C(q0, cq0, h₁₀₀₁) .+
            C(q0, q0, conj(h₁₀₀₁)) .+
            C(h₂₀₀₀, cq0, h₀₀₀₁) .+
            2 .* B(q0, h₁₁₀₁) .+
            2 .* B(h₁₁₀₀, h₁₀₀₁) .+
            B(h₂₀₀₀, conj(h₁₀₀₁)) .+
            B(h₂₁₀₀, h₀₀₀₁) .+
            B(h₂₀₀₁, cq0) .+
            C1(q0, q0, cq0, lens2) .+
            2 .* B1(h₁₁₀₀, q0, lens2) .+ B1(h₂₀₀₀, cq0, lens2) .+ A1(h₂₁₀₀, lens2)
    
    γ₂₁₀ = LA.dot(p0, tmp2110) / 2
    γ₂₀₁ = LA.dot(p0, tmp2101) / 2

    # formula (22)
    α = real.([γ₁₁₀ γ₁₀₁; γ₂₁₀ γ₂₀₁]) \ [0, 1]

    @set pt.nf = (;ω, G21, G32, l2, l1, h₂₀₀₀, h₁₁₀₀, h₀₀₁₀, h₀₀₀₁, γ₁₁₀, γ₁₀₁, γ₂₁₀, γ₂₀₁, α )
end

"""
$(TYPEDSIGNATURES)

Compute the predictor for the curve of Folds of periodic orbits near the Bautin bifurcation point.

## Reference

Kuznetsov, Yu A., H. G. E. Meijer, W. Govaerts, and B. Sautois. “Switching to Nonhyperbolic Cycles from Codim 2 Bifurcations of Equilibria in ODEs.” Physica D: Nonlinear Phenomena 237, no. 23 (December 2008): 3061–68. https://doi.org/10.1016/j.physd.2008.06.006.

"""
function predictor(gh::Bautin, ::Val{:FoldPeriodicOrbitCont}, ϵ::T; 
                    verbose = false, 
                    ampfactor = one(T)) where T
    (;h₂₀₀₀, h₁₁₀₀, h₀₀₁₀, h₀₀₀₁, α, l1, l2, ω, γ₁₁₀, γ₁₀₁) = gh.nf
    lens1, lens2 = gh.lens
    p1 = _get(gh.params, lens1)
    p2 = _get(gh.params, lens2)
    par0 = [p1, p2]
    
    # periodic orbit on the fold
    # formula in section "2.3.1. Generalized Hopf"
    x0 = @. gh.x0 + ϵ^2 * real(h₁₁₀₀ - 2l2 * (h₀₀₁₀ * α[1] + h₀₀₀₁ * α[2]))
    q0 = gh.ζ

    function FoldPO(θ)
        @. x0 + 2ϵ * real(q0 * cis(θ)) + 2ϵ^2 * real(h₂₀₀₀ * cis(2θ))
    end

    return (orbit = t -> FoldPO(t),
            ω = ω + (-2l2 * imag(α[1] * γ₁₁₀ + α[2] * γ₁₀₁) + imag(l1)) * ϵ^2,
            params = (@. par0 - 2l2 * α * ϵ^2),
            x0 = t -> x0)
end
####################################################################################################
function zero_hopf_normal_form(_prob,
                                br::AbstractBranchResult, ind_bif::Int,
                                Teigvec::Type{𝒯eigvec} = _getvectortype(br);
                                δ = 1e-8,
                                nev = length(eigenvalsfrombif(br, ind_bif)),
                                verbose = false,
                                ζs = nothing,
                                lens = getlens(br),
                                scaleζ = norm,
                                bls = _prob.prob.linbdsolver,
                                autodiff = true,
                                detailed::Val{detailed_type} = Val(false)) where {𝒯eigvec, detailed_type}
    @assert br.specialpoint[ind_bif].type == :zh "The provided index does not refer to a Zero-Hopf Point"

    verbose && println("━"^53*"\n──▶ Zero-Hopf Normal form computation")

    # scalar type
    𝒯 = VI.scalartype(Teigvec)
    ϵ = 𝒯(δ)

    # get the MA problem
    prob_ma = _prob.prob

    # get the initial vector field
    prob_vf = prob_ma.prob_vf
    if ~(prob_ma isa AbstractProblemMinimallyAugmented)
        error("[zero-hopf normal form] The underlying problem is not a `AbstractProblemMinimallyAugmented`.\nWe found the type: $(typeof(prob_ma))")
    end

    # linear solver
    ls = prob_ma.linsolver

    # bordered linear solver
    bls = prob_ma.linbdsolver

    # kernel dimension
    N = 3
    # in case nev = 0 (number of unstable eigenvalues), we increase nev to avoid bug
    nev = max(N, nev)

    # newton parameters
    optionsN = br.contparams.newton_options

    # bifurcation point
    bifpt = br.specialpoint[ind_bif]
    eigRes = br.eig

    # parameter for vector field
    x0, parbif = get_bif_point_codim2(br, ind_bif)

    if Teigvec <: BorderedArray
        x0 = convert(Teigvec.parameters[1], x0)
    else
        x0 = convert(Teigvec, x0)
    end

    # jacobian at bifurcation point
    L = jacobian(prob_vf, x0, parbif)

    # right eigenvector
    # TODO IMPROVE THIS
    if true #haseigenvector(br) == false
        # we recompute the eigen-elements if there were not saved during the computation of the branch
        verbose && @info "Recomputing eigenvector on the fly"
        _λ, _ev, _ = optionsN.eigsolver.eigsolver(L, nev)
        # null eigenvalue
        _ind0 = argmin(abs.(_λ))
        verbose && @info "The eigenvalue is $(_λ[_ind0])"
        abs(_λ[_ind0]) > br.contparams.newton_options.tol && @warn "We did not find the correct eigenvalue 0. We found $(_λ[_ind0])"
        q0 = geteigenvector(optionsN.eigsolver, _ev, _ind0)
        # imaginary eigenvalue
        tol_ev = max(1e-10, 10abs(imag(_λ[_ind0])))
        # imaginary eigenvalue iω1
        _ind2 = [ii for ii in eachindex(_λ) if ((abs(imag(_λ[ii])) > tol_ev) & (ii != _ind0))]
        verbose && (@info "Eigenvalue :" _λ _ind2)
        _indIm = argmin(abs(real(_λ[ii])) for ii in _ind2)
        λI = _λ[_ind2[_indIm]]
        q1 = geteigenvector(optionsN.eigsolver, _ev, _ind2[_indIm])
        verbose && @info "Second eigenvalue = $(λI)"
    else
        error("This case has not been done. Please open an issue on the website.")
        ζ = _copy(geteigenvector(optionsN.eigsolver ,br.eig[bifpt.idx].eigenvec, bifpt.ind_ev))
    end

    # normalise for easier debugging
    if imag(λI) < 0
        λI = conj(λI)
        q1 = conj(q1)
    end

    q0 = real(q0)
    q0 ./= scaleζ(q0)
    cq1 = conj(q1)

    # left eigen-elements
    _Jt = has_adjoint(prob_vf) ? jacobian_adjoint(prob_vf, x0, parbif) : adjoint(L)
    p0, λ★ = get_adjoint_basis(_Jt, conj(_λ[_ind0]), optionsN.eigsolver.eigsolver; nev = nev, verbose = verbose)
    p1, λ★1 = get_adjoint_basis(_Jt, conj(λI), optionsN.eigsolver.eigsolver; nev = nev, verbose = verbose)

    # normalise left eigenvectors
    p0 ./= LA.dot(p0, q0)
    p1 ./= LA.dot(q1, p1)
    @assert LA.dot(p0, q0) ≈ 1
    @assert LA.dot(p1, q1) ≈ 1

    # parameters
    lenses = (getlens(prob_ma), lens)
    lens1, lens2 = lenses
    p10 = _get(parbif, lens1); p20 = _get(parbif, lens2);

    getp(l::AllOpticTypes) = _get(parbif, l)
    setp(l::AllOpticTypes, p::Number) = set(parbif, l, p)
    setp(p1::Number, p2::Number) = set(set(parbif, lens1, p1), lens2, p2)
    if autodiff
        Jp = (p, l) -> ForwardDiff.derivative( P -> residual(prob_vf, x0, setp(l, P)), p)
    else
        # finite differences
        Jp = (p, l) -> (residual(prob_vf, x0, setp(l, p + ϵ)) .- 
                        residual(prob_vf, x0, setp(l, p - ϵ)) ) ./ (2ϵ)
    end

    dFp = [LA.dot(p0, Jp(p10, lens1)) LA.dot(p0, Jp(p20, lens2)); LA.dot(p1, Jp(p10, lens1)) LA.dot(p1, Jp(p20, lens2))]

    pt = ZeroHopf(
        x0, parbif,
        lenses,
        (;q0, q1), (;p0, p1),
        (;ω = λI, λ0 = _λ[_ind0], dFp),
        :none
    )

    if ~detailed_type
        return pt
    end

    # second order differential, to be in agreement with Kuznetsov et al.
    B = BilinearMap( (dx1, dx2) -> d2F(prob_vf, x0, parbif, dx1, dx2) )
    C = TrilinearMap((dx1, dx2, dx3) -> d3F(prob_vf, x0, parbif, dx1, dx2, dx3) )
    Ainv0(dx; kw...) = bls(L, q0, p0, zero(𝒯), dx, zero(𝒯); kw...)
    Ainv1(dx; kw...) = bls(L, q1, p1, zero(𝒯), dx, zero(𝒯); kw...)

    # REF1: Kuznetsov, Yu. A. “Numerical Normalization Techniques for All Codim 2 Bifurcations of Equilibria in ODE’s.” SIAM Journal on Numerical Analysis 36, no. 4 (January 1, 1999): 1104–24. https://doi.org/10.1137/S0036142998335005.

    # REF2: “Switching to Nonhyperbolic Cycles from Codim 2 Bifurcations of Equilibria in ODEs,” 2005. https://doi.org/10.1016/j.physd.2008.06.006.
    
    ω = imag(λI)

    # formula (8.2) in REF1
    G200 = LA.dot(p0, B(q0, q0)) |> real # it is real anyway
    G110 = LA.dot(p1, B(q0, q1))
    G011 = LA.dot(p0, B(q1, cq1)) |> real # it is real anyway

    # second order terms
    # formula (8.3) in REF1
    tmp200 = -B(q0, q0) .+ LA.dot(p0, B(q0, q0)) .* q0
    h200, = Ainv0(tmp200)

    # formula (8.4) in REF1
    h020, = ls(L, B(q1, q1); a₀ = Complex(0, -2ω)); h020 .*= -1

    # formula (8.5) in REF1
    tmp110 = B(q0, q1) .- LA.dot(p1, B(q0, q1)) .* q1
    h110, = Ainv1(tmp110; shift = Complex(0, -ω)); h110 .*= -1

    # formula (8.6) in REF1
    tmp011 = B(q1, cq1) .- LA.dot(p0, B(q1, cq1)) .* q0
    h011, = Ainv0(tmp011); h011 .*= -1

    # third order terms
    # G300 and G210 are not needed so not computed
    tmp111 = C(q0, q1, q1) .+ B(q0, h011) .+ B(q1, conj(h110)) .+ B(cq1, h110)
    G111 = LA.dot(p0, tmp111)

    # G021 needed for formula 10 in REF2
    tmp021 = C(q1, q1, cq1) .+ 2 .* B(q1, h011) .+ B(cq1, h020)
    G021 = LA.dot(p1, tmp021)

    # adapt to notations of REF2
    f011 = G011
    g021 = G021/2
    f111 = G111
    g110 = G110

    # Boolean for whether the curve of NS exists
    hasNS = real(g110) * f011 < 0 

    # additional definitions for the parameter unfolding
    VF = prob_ma.prob_vf
    F(x, p) = residual(prob_vf, x, p)

    _A1(q, lens) = (apply_jacobian(VF, x0, setp(lens, _get(parbif, lens) + ϵ), q) .-
                    apply_jacobian(VF, x0, parbif, q)) ./ϵ
    A1(q, lens) = _A1(real(q), lens) .+ im .* _A1(imag(q), lens)
    A1(q::T, lens) where {T <: AbstractArray{<: Real}} = _A1(q, lens)
    Bp(pars) = BilinearMap( (dx1, dx2) -> d2F(prob_vf, x0, pars, dx1, dx2) )
    B1(q, p, l) = (Bp(setp(l, getp(l) + ϵ))(q, p) .- B(q, p)) ./ ϵ
    J1(lens) = F(x0, setp(lens, _get(parbif, lens) + ϵ)) ./ ϵ

    # compute change in parameters
    # formulas (24) in REF2
    s1 = [LA.dot(p0, J1(lens1)), LA.dot(p0, J1(lens2))]
    s2 = [-s1[2], s1[1]]
    s1 ./= LA.dot(s1, s1)

    # computation of the matrix LL in REF2
    # there is a typo in this formula, A1(q1, r1) -> A1(q1, s1)
    # Hil Meijer personal communication
    r1, = Ainv0(q0 .- J1(lens1) .* s1[1] .- J1(lens2) .* s1[2]); #r1 .*= -1
    r2, = Ainv0(J1(lens1) .* s2[1] .+ J1(lens2) .* s2[2])
    LL = zeros(Complex{𝒯}, 2, 2)
    
    LL[1, 1] = LA.dot(p0, B(q0, r2) .+ A1(q0, lens1) .* s2[1] .+ A1(q0, lens2) .* s2[2])
    LL[2, 1] = LA.dot(p1, B(q1, r2) .+ A1(q1, lens1) .* s2[1] .+ A1(q1, lens2) .* s2[2])
    f200 = G200 / 2
    LL[1, 2] = 2*f200
    LL[2, 2] = G110

    # formula (25) in REF2
    # this is corrected by Hil Meijer, personal communication
    RR = [ -LA.dot(p0, B(q0, r1) .+ A1(q0, lens1) .* s1[1] .+ A1(q0, lens2) .* s1[2]), 
           -LA.dot(p1, B(q1, r1) .+ A1(q1, lens1) .* s1[1] .+ A1(q1, lens2) .* s1[2])]

    δ₁, δ₃ = real(LL) \ real(RR)
    δ₂, δ₄ = real(LL) \ [0, 1]

    τ1 = (LL * [δ₁, δ₃] - RR)[2] |> imag
    τ2 = (LL * [δ₂, δ₄])[2] |> imag

    # formula (24) in REF2
    v10 = @. s1 + δ₁ * s2
    v01 = @. δ₂ * s2

    h00010 = @. r1 + δ₁ * r2 + δ₃ * q1
    h00001 = @. δ₂ * r2 + δ₄ * q1

    # formula (10) in REF2
    x = -(f111 + 2*g021) / (2*f200)
    β1 = -f011
    β2 = (2real(g021)*(real(g110)-f200) + real(g110)*f111) / (2*f200) |> real

    @set pt.nf = (;ω = imag(λI), λ0 = _λ[_ind0], dFp, h200, h110, h020, h011, G111, G021, v10, v01, x, β1, β2, h00010, h00001, hasNS, G200, G110, G011, g110, f011, τ1, τ2 )
end

"""
$(TYPEDSIGNATURES)

Compute the predictor for the curve of Hopf bifurcations near the Zero-Hopf bifurcation point.
"""
function predictor(zh::ZeroHopf, ::Val{:HopfCurve}, ds::T; 
                    verbose = false, 
                    ampfactor = one(T)) where T
    (;ω, λ0) = zh.nf
    lens1, lens2 = zh.lens
    p1 = _get(zh.params, lens1)
    p2 = _get(zh.params, lens2)
    par0 = [p1, p2]

    function HopfCurve(s)
        return (pars = par0 , ω = abs(ω))
    end

    # compute eigenvector corresponding to the Hopf branch
    function EigenVec(s)
        return zh.ζ.q1
    end

    function EigenVecAd(s)
        return zh.ζ★.p1
    end

    return (hopf = t -> HopfCurve(t).pars,
            ω    = t -> HopfCurve(t).ω,
            EigenVec = EigenVec,
            EigenVecAd = EigenVecAd,
            x0 = t -> 0)
end

"""
$(TYPEDSIGNATURES)

Compute the predictor for the curve of Fold bifurcations near the Zero-Hopf bifurcation point.
"""
function predictor(zh::ZeroHopf, ::Val{:FoldCurve}, ds::T; 
                    verbose = false, 
                    ampfactor = one(T)) where T
    (;ω, λ0) = zh.nf
    lens1, lens2 = zh.lens
    p1 = _get(zh.params, lens1)
    p2 = _get(zh.params, lens2)
    par0 = [p1, p2]

    function FoldCurve(s)
        return (pars = par0 , λ0 = λ0)
    end

    # compute eigenvector corresponding to the Hopf branch
    function EigenVec(s)
        return zh.ζ.q0
    end

    function EigenVecAd(s)
        return zh.ζ★.p0
    end

    return (fold = t -> FoldCurve(t).pars,
            λ0   = t -> FoldCurve(t).λ0,
            EigenVec = EigenVec,
            EigenVecAd = EigenVecAd,
            x0 = t -> 0)
end

"""
$(TYPEDSIGNATURES)

Compute the predictor for the curve of Neimark-Sacker bifurcations near the Zero-Hopf bifurcation point.

## Reference

Kuznetsov, Yu A., H. G. E. Meijer, W. Govaerts, and B. Sautois. “Switching to Nonhyperbolic Cycles from Codim 2 Bifurcations of Equilibria in ODEs.” Physica D: Nonlinear Phenomena 237, no. 23 (December 2008): 3061–68. https://doi.org/10.1016/j.physd.2008.06.006.
"""
function predictor(zh::ZeroHopf, ::Val{:NS}, ϵ::T; 
                    verbose = false, 
                    ampfactor = one(T)) where T
    (;x, β1, β2, v10, v01, h00010, h00001, h011, ω, h020, g110, f011, hasNS) = zh.nf
    lens1, lens2 = zh.lens
    p1 = _get(zh.params, lens1)
    p2 = _get(zh.params, lens2)
    par0 = [p1, p2]

    q0 = zh.ζ.q0
    q1 = zh.ζ.q1

    # formula (27) in REF2. There is a typo for the coefficient of β2
    x = @. real(zh.x0 + ϵ^2 * (h00010 * β1 + h00001 * β2 + x * q0 + h011))

    # approximation of the multiplier
    o1 = sqrt(2abs(real(g110) * f011)) * ϵ
    o2 = ω
    k = 1 - (2pi*o1/o2)^2/2 |> acos

    function NS(θ)
        @. x + 2ϵ * real(q1 * cis(θ)) + 2ϵ^2 * real(h020 * cis(2θ))
    end
    
    return (orbit = t -> NS(t),
            hasNS = hasNS,
            params = (@. real(par0 + (β1 * v10 + β2 * v01) * ϵ^2)),
            T = 2pi / (ω),
            k = k
    )
end
####################################################################################################
function hopf_hopf_normal_form(_prob,
                                br::AbstractBranchResult, ind_bif::Int,
                                Teigvec::Type{𝒯eigvec} = _getvectortype(br);
                                δ = 1e-8,
                                nev = length(eigenvalsfrombif(br, ind_bif)),
                                verbose = false,
                                ζs = nothing,
                                lens = getlens(br),
                                scaleζ = norm,
                                autodiff = true,
                                detailed::Val{detailed_type} = Val(false)) where {𝒯eigvec, detailed_type}
    @assert br.specialpoint[ind_bif].type == :hh "The provided index does not refer to a Hopf-Hopf Point"

    verbose && println("━"^53*"\n──▶ Hopf-Hopf Normal form computation")

    # scalar type
    𝒯 = VI.scalartype(𝒯eigvec)
    ϵ = 𝒯(δ)

    # get the MA problem
    prob_ma = _prob.prob

    # get the initial vector field
    prob_vf = prob_ma.prob_vf

    if ~(prob_ma isa AbstractProblemMinimallyAugmented)
        error("[Hopf-Hopf normal form] The underlying problem is not a `AbstractProblemMinimallyAugmented`.\n\nWe found the type: $(typeof(prob_ma))")
    end

    # linear solver
    ls = prob_ma.linsolver

    # kernel dimension
    N = 4

    # in case nev = 0 (number of unstable eigenvalues), we increase nev to avoid bug
    nev = max(N, nev)

    # newton parameters
    optionsN = br.contparams.newton_options

    # bifurcation point
    bifpt = br.specialpoint[ind_bif]
    eigRes = br.eig

    # parameter for vector field
    x0, parbif = get_bif_point_codim2(br, ind_bif)
    if 𝒯eigvec <: BorderedArray
        x0 = convert(𝒯eigvec.parameters[1], getvec(bifpt.x, prob_ma))
    else
        x0 = convert(𝒯eigvec, getvec(bifpt.x , prob_ma))
    end

    # jacobian at bifurcation point
    L = jacobian(prob_vf, x0, parbif)

    p0, ω0 = getp(bifpt.x, prob_ma)

    # right eigenvector
    # TODO IMPROVE THIS
    if true#haseigenvector(br) == false
        # we recompute the eigen-elements if there were not saved during the computation of the branch
        verbose && @info "Recomputing eigenvector on the fly"
        _λ, _ev, _ = optionsN.eigsolver.eigsolver(L, nev)
        # imaginary eigenvalue iω0
        _ind0 = argmin(abs.(_λ .- im * ω0))
        λ1 = _λ[_ind0]
        verbose && @info "The first eigenvalue  is $(λ1), ω0 = $ω0"
        q1 = geteigenvector(optionsN.eigsolver, _ev, _ind0)
        tol_ev = max(1e-10, 10abs(ω0 - imag(_λ[_ind0])))
        # imaginary eigenvalue iω1
        _ind2 = [ii for ii in eachindex(_λ) if abs(abs(imag(_λ[ii])) - abs(ω0)) > tol_ev]
        _indIm = argmin(real(_λ[ii]) for ii in _ind2)
        λ2 = _λ[_ind2[_indIm]]
        verbose && @info "The second eigenvalue is $(λ2)"
        q2 = geteigenvector(optionsN.eigsolver, _ev, _ind2[_indIm])
    else
        @assert false "Case not handled yet. Please open an issue on the website of BifurcationKit.jl"
    end

    # for easier debugging, we normalise the case to ω1 > ω2 > 0
    if imag(λ1) < 0
        λ1 = conj(λ1)
        q1 = conj(q1)
    end

    if imag(λ2) < 0
        λ2 = conj(λ2)
        q2 = conj(q2)
    end

    if imag(λ1) < imag(λ2)
        q1, q2 = q2, q1
        λ1, λ2 = λ2, λ1
    end

    q1 ./= scaleζ(q1)
    q2 ./= scaleζ(q2)

    cq1 = conj(q1); cq2 = conj(q2)
    ω1 = imag(λ1); ω2 = imag(λ2);

    # left eigen-elements
    _Jt = has_adjoint(prob_vf) ? jacobian_adjoint(prob_vf, x0, parbif) : adjoint(L)
    p1, λ★1 = get_adjoint_basis(_Jt, conj(λ1), optionsN.eigsolver.eigsolver; nev = nev, verbose = verbose)
    p2, λ★2 = get_adjoint_basis(_Jt, conj(λ2), optionsN.eigsolver.eigsolver; nev = nev, verbose = verbose)

    # normalise left eigenvectors
    p1 ./= LA.dot(q1, p1)
    p2 ./= LA.dot(q2, p2)

    @assert LA.dot(p1, q1) ≈ 1 "we found $(LA.dot(p1, q1)) instead of 1."
    @assert LA.dot(p2, q2) ≈ 1 "we found $(LA.dot(p2, q2)) instead of 1."

    # parameters
    lenses = (getlens(prob_ma), lens)
    lens1, lens2 = lenses
    p10 = _get(parbif, lens1); p20 = _get(parbif, lens2);

    # _getp(l::AllOpticTypes) = _get(parbif, l)
    # _setp(l::AllOpticTypes, p::Number) = set(parbif, l, p)
    # _setp(p1::Number, p2::Number) = set(set(parbif, lens1, p1), lens2, p2)
    if autodiff
        Jp = (p, l) -> ForwardDiff.derivative( P -> residual(prob_vf, x0, setp(l, P)) , p)
    else
        # finite differences
        Jp = (p, l) -> (residual(prob_vf, x0, setp(l, p + ϵ2)) .- 
                        residual(prob_vf, x0, setp(l, p - ϵ2)) ) ./ (2ϵ2)
    end

    pt = HopfHopf(
        x0, parbif,
        lenses,
        (;q1, q2), (;p1, p2),
        (;λ1, λ2),
        :none
    )

    # case of simplified normal form
    if detailed_type == false
        return pt
    end

    # second order differential, to be in agreement with Kuznetsov et al.
    B = BilinearMap( (dx1, dx2) -> d2F(prob_vf, x0, parbif, dx1, dx2) )
    C = TrilinearMap((dx1, dx2, dx3) -> d3F(prob_vf, x0, parbif, dx1, dx2, dx3) )

    # REF1: Kuznetsov, Yu. A. “Numerical Normalization Techniques for All Codim 2 Bifurcations of Equilibria in ODE’s.” SIAM Journal on Numerical Analysis 36, no. 4 (January 1, 1999): 1104–24. https://doi.org/10.1137/S0036142998335005.

    # REF2 “Switching to Nonhyperbolic Cycles from Codim 2 Bifurcations of Equilibria in ODEs,” 2005. https://doi.org/10.1016/j.physd.2008.06.006.

    # second order, formulas 9.2 - 9.6 in REF1
    h₂₀₀₀, = ls(-L, B(q1, q1), a₀ = 2λ1)
    h₀₀₂₀, = ls(-L, B(q2, q2), a₀ = 2λ2)

    h₁₀₁₀, = ls(-L, B(q1, q2),  a₀ = Complex(0, ω1 + ω2))
    h₁₀₀₁, = ls(-L, B(q1, cq2), a₀ = Complex(0, ω1 - ω2))

    h₁₁₀₀, = ls(L, B(q1, cq1)); h₁₁₀₀ .*= -1
    h₀₀₁₁, = ls(L, B(q2, cq2)); h₀₀₁₁ .*= -1

    # for implementing forumla 28 in REF2, we need G2100, G1110 from REF1, on page 1117
    tmp2100 = C(q1, q1, cq1) .+ B(h₂₀₀₀, cq1) .+ 2 .* B(h₁₁₀₀, q1)
    G2100 = LA.dot(p1, tmp2100)
    tmp0021 = C(q2, q2, cq2) .+ B(h₀₀₂₀, cq2) .+ 2 .* B(h₀₀₁₁, q2)
    G0021 = LA.dot(p2, tmp0021)
    tmp1110 = C(q1, cq1, q2) .+ B(h₁₁₀₀, q2) .+ B(h₁₀₁₀, cq1) .+ B(conj(h₁₀₀₁), q1)
    G1110 = LA.dot(p2, tmp1110)
    tmp1011 = C(q1, q2, cq2) .+ B(h₁₀₁₀, cq2) .+ B(h₁₀₀₁, q2) .+ B(h₀₀₁₁, q1)
    G1011 = LA.dot(p1, tmp1011)

    # some more definitions
    VF = prob_ma.prob_vf
    F(x, p) = residual(prob_vf, x, p)

    lens1, lens2 = pt.lens
    _getp(l::AllOpticTypes) = _get(parbif, l)
    _setp(l::AllOpticTypes, p::Number) = set(parbif, l, p)
    _setp(p1::Number, p2::Number) = set(set(parbif, lens1, p1), lens2, p2)
    _A1(q, lens) = (apply_jacobian(VF, x0, _setp(lens, _get(parbif, lens) + ϵ), q) .-
                      apply_jacobian(VF, x0, parbif, q)) ./ϵ
    A1(q, lens) = _A1(real(q), lens) .+ im .* _A1(imag(q), lens)
    A1(q::T, lens) where {T <: AbstractArray{<: Real}} = _A1(q, lens)
    Bp(pars) = BilinearMap( (dx1, dx2) -> d2F(prob_vf, x0, pars, dx1, dx2) )
    B1(q, p, l) = (Bp(_setp(l, _getp(l) + ϵ))(q, p) .- B(q, p)) ./ ϵ
    J1(lens) = F(x0, _setp(lens, _get(parbif, lens) + ϵ)) ./ ϵ

    # implement formula 26 from REF2
    h₀₀₀₀₁₀, = ls(L, J1(lens1)); h₀₀₀₀₁₀ .*= -1
    h₀₀₀₀₀₁, = ls(L, J1(lens2)); h₀₀₀₀₀₁ .*= -1
    
    # implement formula 26 from REF2, Fredholm alternative
    γ₁₁₀ = LA.dot(p1, B(q1, h₀₀₀₀₁₀) .+ A1(q1, lens1))
    γ₂₁₀ = LA.dot(p2, B(q2, h₀₀₀₀₁₀) .+ A1(q2, lens1))
    γ₁₀₁ = LA.dot(p1, B(q1, h₀₀₀₀₀₁) .+ A1(q1, lens2))
    γ₂₀₁ = LA.dot(p2, B(q2, h₀₀₀₀₀₁) .+ A1(q2, lens2))

    # this matrix is written V in 2.3.3 Double Hopf
    Γ = [γ₁₁₀ γ₁₀₁; γ₂₁₀ γ₂₀₁]
    
    # formula (22) for Neimark-Sacker1, from formula (12)
    f2100 = real(G2100)/2 # conform to notations of REF2
    α = real.(Γ) \ [f2100, real(G1110)] # formula (22)
    dω1, dω2 =  [imag(G2100)/2, imag(G1110)] .- (imag.(Γ) * α) # formula (28) in REF2
    ns1 = (; dω1, dω2, α)

    # formula (22) for Neimark-Sacker2, from formula (13)
    f0021 = real(G0021)/2 # conform to notations of REF2
    α = real.(Γ) \ [real(G1011), f0021] # formula (22)
    dω1, dω2 = [imag(G1011), imag(G0021)/2] .- (imag.(Γ) * α) # formula (28) in REF2
    ns2 = (; dω1, dω2, α)

    return @set pt.nf = (;λ1, λ2, G2100, G0021, G1110, G1011, γ₁₁₀, γ₁₀₁, γ₂₁₀, γ₂₀₁, Γ, h₁₁₀₀, h₀₀₁₁, h₀₀₀₀₁₀, h₀₀₀₀₀₁, h₂₀₀₀, h₀₀₂₀, ns1, ns2)
end

"""
$(TYPEDSIGNATURES)

Compute the predictor for the Hopf curve near the Hopf-Hopf bifurcation point.
"""
function predictor(hh::HopfHopf, ::Val{:HopfCurve}, ds::T; 
                    verbose = false, 
                    ampfactor = one(T)) where T
    (;λ1, λ2) = hh.nf
    lens1, lens2 = hh.lens
    p1 = _get(hh.params, lens1)
    p2 = _get(hh.params, lens2)
    par0 = [p1, p2]

    function HopfCurve(s)
        return (pars = par0 , ω = imag(λ2))
    end

    # compute eigenvector corresponding to the Hopf branch
    function EigenVec(s)
        return hh.ζ.q2
    end

    function EigenVecAd(s)
        return hh.ζ★.p2
    end

    return (hopf = t -> HopfCurve(t).pars,
            ω    = t -> HopfCurve(t).ω,
            EigenVec = EigenVec,
            EigenVecAd = EigenVecAd,
            x0 = t -> 0)
end

"""
$(TYPEDSIGNATURES)

Compute the predictor for the curve of Neimark-Sacker points near the Hopf-Hopf bifurcation point.

## Reference

Kuznetsov, Yu A., H. G. E. Meijer, W. Govaerts, and B. Sautois. “Switching to Nonhyperbolic Cycles from Codim 2 Bifurcations of Equilibria in ODEs.” Physica D: Nonlinear Phenomena 237, no. 23 (December 2008): 3061–68. https://doi.org/10.1016/j.physd.2008.06.006.
"""
function predictor(hh::HopfHopf, ::Val{:NS}, ϵ::T; 
                    verbose = false, 
                    ampfactor = one(T)) where T
    (;λ1, λ2, h₁₁₀₀, h₀₀₁₁, h₀₀₀₀₁₀, h₀₀₀₀₀₁, h₂₀₀₀, h₀₀₂₀, ns1, ns2) = hh.nf
    lens1, lens2 = hh.lens
    p1 = _get(hh.params, lens1)
    p2 = _get(hh.params, lens2)
    par0 = [p1, p2]

    # formula in section "2.1.3. Double-Hopf"
    x1 = @. hh.x0 + ϵ^2 * real(h₁₁₀₀ - (h₀₀₀₀₁₀ * ns1.α[1] + h₀₀₀₀₀₁ * ns1.α[2]))
    x2 = @. hh.x0 + ϵ^2 * real(h₀₀₁₁ - (h₀₀₀₀₁₀ * ns2.α[1] + h₀₀₀₀₀₁ * ns2.α[2]))

    q1 = hh.ζ.q1
    q2 = hh.ζ.q2

    ω1 = imag(λ1)
    ω2 = imag(λ2)

    ω11 = ω1 + ns1.dω1 * ϵ^2
    ω12 = ω2 + ns1.dω2 * ϵ^2
    ω21 = ω1 + ns2.dω1 * ϵ^2
    ω22 = ω2 + ns2.dω2 * ϵ^2

    # Floquet multipliers for NS associated to the periodic orbit 
    k1 = mod(ω22 / ω11 * 2pi, 2pi)
    k2 = mod(ω11 / ω22 * 2pi, 2pi)

    function NS1(θ)
        @. x1 + 2ϵ * real(q1 * cis(θ)) + 2ϵ^2 * real(h₂₀₀₀ * cis(2θ))
    end

    function NS2(θ)
        @. x2 + 2ϵ * real(q2 * cis(θ)) + 2ϵ^2 * real(h₀₀₂₀ * cis(2θ))
    end
    
    return (;ns1 = t -> NS1(t),
            ns2 = t -> NS2(t),
            params1 = (@. par0 - ns1.α * ϵ^2),
            params2 = (@. par0 - ns2.α * ϵ^2),
            ω11,
            ω12,
            ω21,
            ω22,
            T1 = 2pi / ω11,
            T2 = 2pi / ω22,
            k1,
            k2,
    )
end

