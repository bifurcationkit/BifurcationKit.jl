# This function is very important for the computation of Floquet multipliers: it checks that the eigensolvers compute the eigenvalues with largest modulus instead of their default behaviour which is with largest real part. If this option is not properly set, bifurcations of periodic orbits will be wrong.
function check_floquet_options(eigls::AbstractEigenSolver)
    if eigls isa DefaultEig
        return @set eigls.which = abs
    elseif eigls isa EigArpack
        return setproperties(eigls; which = :LM, by = abs)
    elseif eigls isa EigArnoldiMethod
        return setproperties(eigls; which = ArnoldiMethod.LM(), by = abs)
    end
    if eigls isa EigKrylovKit
        return @set eigls.which = :LM
    end
    eigls
end

# see https://discourse.julialang.org/t/uniform-scaling-inplace-addition-with-matrix/59928/5

####################################################################################################
"""
    floquet = FloquetQaD(eigsolver::AbstractEigenSolver)

This composite type implements the computation of the eigenvalues of the monodromy matrix in the case of periodic orbits problems (based on the Shooting method or Finite Differences (Trapeze method)), also called the Floquet multipliers. The method, dubbed Quick and Dirty (QaD), is not numerically very precise for large / small Floquet exponents when the number of time sections is large because of many matrix products. It allows, nevertheless, to detect bifurcations. The arguments are as follows:
- `eigsolver::AbstractEigenSolver` solver used to compute the eigenvalues.

If `eigsolver == DefaultEig()`, then the monodromy matrix is formed and its eigenvalues are computed. Otherwise, a Matrix-Free version of the monodromy is used.

!!! danger "Floquet multipliers computation"
    The computation of Floquet multipliers is necessary for the detection of bifurcations of periodic orbits (which is done by analyzing the Floquet exponents obtained from the Floquet multipliers). Hence, the eigensolver `eigsolver` needs to compute the eigenvalues with largest modulus (and not with largest real part which is their default behavior). This can be done by changing the option `which = :LM` of `eigsolver`. Nevertheless, note that for most implemented eigensolvers in the current Package, the proper option is set.
"""
struct FloquetQaD{E <: AbstractEigenSolver } <: AbstractFloquetSolver
    eigsolver::E
    function FloquetQaD(eigls::AbstractEigenSolver)
        eigls2 = check_floquet_options(eigls)
        return new{typeof(eigls2)}(eigls2)
    end
    FloquetQaD(eigls::AbstractFloquetSolver) = eigls
end
geteigenvector(eig::FloquetQaD, vecs, n::Union{Int, AbstractVector{Int64}}) = geteigenvector(eig.eigsolver, vecs, n)

function (fl::FloquetQaD)(J, nev; kwargs...)
    if fl.eigsolver isa AbstractDirectEigenSolver
        monodromy = MonodromyQaD(J)
    else
        # Matrix Free version
        monodromy = dx -> MonodromyQaD(J, dx)
    end
    vals, vecs, cv, info = fl.eigsolver(monodromy, nev)
    Inf in vals && @warn "Detecting infinite eigenvalue during the computation of Floquet coefficients"

    # the `vals` should be sorted by largest modulus, but we need the log of them sorted this way
    logvals = log.(complex.(vals))
    I = sortperm(logvals, by = real, rev = true)

    # floquet exponents
    σ = logvals[I]
    vp0 = minimum(abs, σ)
    if (J isa FloquetWrapper{ShootingProblem}) && vp0 > 1e-8
        @warn "The precision on the Floquet multipliers is $vp0. Either decrease `tol_stability` in the option ContinuationPar or use a different method than `FloquetQaD`"
    end
    return σ, geteigenvector(fl.eigsolver, vecs, I), cv, info
end
####################################################################################################
# ShootingProblem
# Matrix free monodromy operators
function MonodromyQaD(JacSH::FloquetWrapper{Tpb, Tjacpb, Torbitguess, Tp}, du::AbstractVector) where {Tpb <: ShootingProblem, Tjacpb, Torbitguess, Tp}
    sh = JacSH.pb
    x = JacSH.x
    p = JacSH.par

    # period of the cycle
    T = getperiod(sh, x)

    # extract parameters
    M = get_mesh_size(sh)
    N = div(length(x) - 1, M)

    # extract the time slices
    xv = @view x[begin:end-1]
    xc = reshape(xv, N, M)

    out = copy(du)

    for ii in 1:M
        # call the jacobian of the flow
        @views out .= evolve(sh.flow, Val(:SerialdFlow), xc[:, ii], p, out, sh.ds[ii] * T).du
    end
    return out
end

# Compute the monodromy matrix at `x` explicitly, not suitable for large systems
function MonodromyQaD(JacSH::FloquetWrapper{Tpb, Tjacpb, Torbitguess, Tp}) where {Tpb <: ShootingProblem, Tjacpb, Torbitguess, Tp}
    sh = JacSH.pb
    x = JacSH.x
    p = JacSH.par

    # period of the cycle
    T = getperiod(sh, x)

    # extract parameters
    M = get_mesh_size(sh)
    N = div(length(x) - 1, M)

    Mono = zeros(N, N)

    # extract the time slices
    xv = @view x[begin:end-1]
    xc = reshape(xv, N, M)
    du = zeros(N)

    for ii in 1:N
        du[ii] = 1
        # call jacobian of the flow
        @views Mono[:, ii] .= evolve(sh.flow, Val(:SerialdFlow), xc[:, 1], p, du, T).du
        du[ii] = 0
    end

    return Mono
end

# Compute the monodromy matrix at `x` explicitly, not suitable for large systems
# it is based on a matrix expression of the Jacobian of the shooting functional. We
# just extract the blocks needed to compute the monodromy
function MonodromyQaD(JacSH::FloquetWrapper{Tpb, Tjacpb, Torbitguess, Tp}) where {Tpb <: ShootingProblem, Tjacpb <: AbstractMatrix, Torbitguess, Tp}
    J = _get_matrix(JacSH)
    sh = JacSH.pb
    M = get_mesh_size(sh)
    N = div(length(JacSH.x) - 1, M)
    mono = copy(J[1:N, 1:N])
    if M == 1
        return mono + I
    end
    tmp = similar(mono)
    r = N
    for ii = 1:M-1
        # mono .= J[r+1:r+N, r+1:r+N] * mono
        @views mul!(tmp, J[r+1:r+N, r+1:r+N], mono)
        mono .= tmp
        r += N
    end
    return mono
end

# This function is used to reconstruct the spatio-temporal eigenvector of the shooting functional sh
# at position x from the Floquet eigenvector ζ
@views function (fl::FloquetQaD)(::Val{:ExtractEigenVector}, powrap::WrapPOSh{ <: ShootingProblem}, x::AbstractVector, par, ζ::AbstractVector)
    # get the shooting problem
    sh = powrap.prob

    # period of the cycle
    T = getperiod(sh, x)

    # extract parameters
    M = get_mesh_size(sh)
    N = div(length(x) - 1, M)

    # extract the time slices
    xv = x[begin:end-1]
    xc = reshape(xv, N, M)

    out = evolve(sh.flow, Val(:SerialdFlow), xc[:, 1], par, ζ, sh.ds[1] * T).du
    out_a = [copy(out)]

    for ii in 2:M
        # call the jacobian of the flow
        out .= evolve(sh.flow, Val(:SerialdFlow), xc[:, ii], par, out, sh.ds[ii] * T).du
        push!(out_a, copy(out))
    end
    return out_a
end
####################################################################################################
# PoincareShooting

# matrix free evaluation of monodromy operator
function MonodromyQaD(JacSH::FloquetWrapper{Tpb, Tjacpb, Torbitguess, Tp}, dx_bar::AbstractVector) where {Tpb <: PoincareShootingProblem, Tjacpb, Torbitguess, Tp}
    psh = JacSH.pb
    x_bar = JacSH.x
    p = JacSH.par

    M = get_mesh_size(psh)
    Nm1 = div(length(x_bar), M)

    # reshape the period orbit guess into a Matrix
    x_barc = reshape(x_bar, Nm1, M)
    @assert length(dx_bar) == Nm1 "Please provide the right dimension to your matrix-free eigensolver, it must be $Nm1."

    xc = similar(x_bar, Nm1 + 1)
    outbar = copy(dx_bar)
    outc = similar(dx_bar, Nm1 + 1)

    for ii in 1:M
        E!(psh.section,  xc,  view(x_barc, :, ii), ii)
        dE!(psh.section, outc, outbar, ii)
        outc .= diff_poincare_map(psh, xc, p, outc, ii)
        # check to <outc, normals[ii]> = 0
        # println("--> ii=$ii, <out, normali> = ", dot(outc, sh.section.normals[ii]))
        dR!(psh.section, outbar, outc, ii)
    end
    return outbar

end

# matrix based formulation of monodromy operator, not suitable for large systems
# it is based on a matrix expression of the Jacobian of the shooting functional. We thus
# just extract the blocks needed to compute the monodromy
function MonodromyQaD(JacSH::FloquetWrapper{Tpb, Tjacpb, Torbitguess, Tp}) where {Tpb <: PoincareShootingProblem, Tjacpb <: AbstractMatrix, Torbitguess, Tp}
    J = _get_matrix(JacSH)
    sh = JacSH.pb
    T = eltype(J)

    M = get_mesh_size(sh)
    Nj = length(JacSH.x)
    N = div(Nj, M)

    if M == 1
        return I - J
    end

    mono = copy(J[N+1:2N, 1:N])
    tmp = similar(mono)
    r1 = mod(2N, Nj)
    r2 = N
    for ii = 1:M-1
        # mono .= J[r+1:r+N, r+1:r+N] * mono
        @views mul!(tmp, J[r1+1:r1+N, r2+1:r2+N], mono)
        mono .= tmp
        r1 = mod(r1 + N, Nj)
        r2 += N
    end
    # the structure of the functional imposes to take into account the sign
    sgn = iseven(M) ? one(T) : -one(T)
    mono .*= sgn
    return mono
end

# This function is used to reconstruct the spatio-temporal eigenvector of the shooting functional sh
# at position x from the Floquet eigenvector ζ
@views function (fl::FloquetQaD)(::Val{:ExtractEigenVector}, powrap::WrapPOSh{ <: PoincareShootingProblem}, x_bar::AbstractVector, p, ζ::AbstractVector)
    # get the shooting problem
    psh = powrap.prob

    #  ζ is of size (N-1)
    M = get_mesh_size(psh)
    Nm1 = length(ζ)
    dx = similar(x_bar, length(ζ) + 1)

    x_barc = reshape(x_bar, Nm1, M)
    xc = similar(x_bar, Nm1 + 1)
    dx_bar = similar(x_bar, Nm1)
    outbar = copy(dx_bar)
    outc = similar(dx_bar, Nm1 + 1)
    out_a = typeof(xc)[]

    for ii in 1:M
        E!(psh.section,  xc,  view(x_barc, :, ii), ii)
        dE!(psh.section, outc, outbar, ii)
        outc .= diff_poincare_map(psh, xc, p, outc, ii)
        # check to <outc, normals[ii]> = 0
        # println("--> ii=$ii, <out, normali> = ", dot(outc, sh.section.normals[ii]))
        dR!(psh.section, outbar, outc, ii)
        push!(out_a, copy(outbar))
    end
    return out_a
end
####################################################################################################
# PeriodicOrbitTrapProblem

# Matrix-Free version of the monodromy operator
@views function MonodromyQaD(JacFW::FloquetWrapper{Tpb, Tjacpb, Torbitguess, Tp}, du::AbstractVector) where {Tpb <: PeriodicOrbitTrapProblem, Tjacpb, Torbitguess, Tp}
    poPb = JacFW.pb
    u0 = JacFW.x
    par = JacFW.par

    # extraction of various constants
    M, N = size(poPb)

    # period of the cycle
    T = getperiod(poPb, u0)

    # time step
    h =  T * get_time_step(poPb, 1)
    Typeh = typeof(h)

    out = copy(du)

    u0c = get_time_slices(u0, N, M)

    out .= out .+ h/2 .* apply(jacobian(poPb.prob_vf, u0c[:, M-1], par), out)
    # res = (I - h/2 * jacobian(poPb.prob_vf, u0c[:, 1])) \ out
    res, _ = poPb.linsolver(jacobian(poPb.prob_vf, u0c[:, 1], par), out; a₀ = convert(Typeh, 1), a₁ = -h/2)
    out .= res

    for ii in 2:M-1
        h =  T * get_time_step(poPb, ii)
        out .= out .+ h/2 .* apply(jacobian(poPb.prob_vf, u0c[:, ii-1], par), out)
        # res = (I - h/2 * jacobian(poPb.prob_vf, u0c[:, ii])) \ out
        res, _ = poPb.linsolver(jacobian(poPb.prob_vf, u0c[:, ii], par), out; a₀ = convert(Typeh, 1), a₁ = -h/2)
        out .= res
    end

    return out
end

# This function is used to reconstruct the spatio-temporal eigenvector of the Trapezoid functional
# at position x from the Floquet eigenvector ζ
function (fl::FloquetQaD)(::Val{:ExtractEigenVector}, powrap::WrapPOTrap, u0::AbstractVector, par, ζ::AbstractVector)
    # get the Trapezoid problem
    poPb = powrap.prob

    # extraction of various constants
    M, N = size(poPb)

    # period of the cycle
    T = getperiod(poPb, u0)

    # time step
    h =  T * get_time_step(poPb, 1)
    Typeh = typeof(h)

    out = copy(ζ)

    u0c = get_time_slices(u0, N, M)

    @views out .= out .+ h/2 .* apply(jacobian(poPb.prob_vf, u0c[:, M-1], par), out)
    # res = (I - h/2 * poPb.J(u0c[:, 1])) \ out
    @views res, _ = poPb.linsolver(jacobian(poPb.prob_vf, u0c[:, 1], par), out; a₀ = convert(Typeh, 1), a₁ = -h/2)
    out .= res
    out_a = [copy(out)]
    # push!(out_a, copy(out))

    for ii in 2:M
        h =  T * get_time_step(poPb, ii)
        @views out .= out .+ h/2 .* apply(jacobian(poPb.prob_vf, u0c[:, ii-1], par), out)
        # res = (I - h/2 * poPb.J(u0c[:, ii])) \ out
        @views res, _ = poPb.linsolver(jacobian(poPb.prob_vf, u0c[:, ii], par), out; a₀ = convert(Typeh, 1), a₁ = -h/2)
        out .= res
        push!(out_a, copy(out))
    end
    # push!(out_a, copy(ζ))

    return out_a
end

# Compute the monodromy matrix at `u0` explicitly, not suitable for large systems
function MonodromyQaD(JacFW::FloquetWrapper{Tpb, Tjacpb, Torbitguess, Tp})  where {Tpb <: PeriodicOrbitTrapProblem, Tjacpb, Torbitguess, Tp}

    poPb = JacFW.pb
    u0 = JacFW.x
    par = JacFW.par

    # extraction of various constants
    M, N = size(poPb)

    # period of the cycle
    T = getperiod(poPb, u0)

    # time step
    h =  T * get_time_step(poPb, 1)

    u0c = get_time_slices(u0, N, M)

    @views mono = Array(I - h/2 * (jacobian(poPb.prob_vf, u0c[:, 1], par))) \ Array(I + h/2 * jacobian(poPb.prob_vf, u0c[:, M-1], par))
    temp = similar(mono)

    for ii in 2:M-1
        # for some reason, the next line is faster than doing (I - h/2 * (poPb.J(u0c[:, ii]))) \ ...
        # also I - h/2 .* J seems to hurt (a little) the performances
        h =  T * get_time_step(poPb, ii)
        @views temp = Array(I - h/2 * (jacobian(poPb.prob_vf, u0c[:, ii], par))) \ Array(I + h/2 * jacobian(poPb.prob_vf, u0c[:, ii-1], par))
        mono .= temp * mono
    end
    return mono
end
####################################################################################################
"""

Computation of Floquet coefficients for the orthogonal collocation method. The method is based on a formulation through a generalised eigenvalue problem (GEV). Relatively slow but quite precise.

This is a simplified version of [1].

## Arguments
- `eigls` an eigensolver
- `ntot` total number of unknowns (without counting the period), ie `length(::PeriodicOrbitOCollProblem)`
- `n` space dimension
- `array_zeros` useful for sparse matrices

## Example

You can create such solver like this (here `n = 2`):

    eigfloquet = BifurcationKit.FloquetCollGEV(DefaultEig(), (30 * 4 + 1) * 2, 2))

## References
[1] Fairgrieve, Thomas F., and Allan D. Jepson. “O. K. Floquet Multipliers.” SIAM Journal on Numerical Analysis 28, no. 5 (October 1991): 1446–62. https://doi.org/10.1137/0728075.
"""
struct FloquetCollGEV{E <: AbstractEigenSolver, Tb} <: AbstractFloquetSolver
    eigsolver::E
    B::Tb
    function FloquetCollGEV(eigls::AbstractEigenSolver, ntot::Int, n::Int; array_zeros = zeros)
        eigls2 = check_floquet_options(eigls)
        # build the mass matrix
        B = array_zeros(ntot, ntot)
        B[end-n+1:end, end-n+1:end] .= I(n)
        return new{typeof(eigls2), typeof(B)}(eigls2, B)
    end
    FloquetCollGEV(eigls::FloquetCollGEV) = eigls
end

geteigenvector(fl::FloquetCollGEV, vecs, n::Union{Int, AbstractVector{Int64}}) = geteigenvector(fl.eigsolver, vecs, n)

@views function (fl::FloquetCollGEV)(JacColl::FloquetWrapper{Tpb, Tjacpb, Torbitguess, Tp}, nev; kwargs...) where {Tpb, Tjacpb <: AbstractMatrix, Torbitguess, Tp}
    prob = JacColl.pb
    _J = _get_matrix(JacColl)
    n = get_state_dim(prob)
    J = copy(_J[1:end-1, 1:end-1]) # we cannot mess-up with the linear solver
    # case of v(0)
    J[end-n+1:end, 1:n] .= I(n)
    # case of v(1)
    J[end-n+1:end, end-n+1:end] .= -I(n)
    # solve generalized eigenvalue problem
    values, vecs = gev(fl.eigsolver, J, fl.B, nev;)
    # remove infinite eigenvalues
    indvalid = findall(x -> abs(x) < 1e9, values)
    vals = values[indvalid]
    # these are the Floquet multipliers
    μ = @. Complex(1 / (1 + vals))
    vp0 = minimum(abs ∘ log, μ)
    if vp0 > 1e-8
        @warn "The precision on the Floquet multipliers is $vp0. Either decrease `tol_stability` in the option `ContinuationPar` or use a different method than `FloquetCollGEV`"
    end
    Ind = sortperm(log.(μ); by = real, rev = true)
    nev2 = min(nev, length(Ind))
    Ind = Ind[begin:nev2]
    return log.(μ[Ind]), geteigenvector(fl.eigsolver, vecs, indvalid[Ind]), true, 1
end

"""
    eigfloquet = BifurcationKit.FloquetColl()

Computation of Floquet coefficients for the orthogonal collocation method. The method is based on the condensation of parameters described in [1] and used in Auto07p with a twist from [2] in which we form the monodromy matrix with a product of `Ntst` matrices.

This is much faster than `FloquetCollGEV` but less precise. The best version uses a Periodic Schur decomposition instead of the product of `Ntst` matrices. This is provided in the package `PeriodicSchurBifurcationKit.jl`.

## References
[1] Doedel, Eusebius, Herbert B. Keller, et Jean Pierre Kernevez. «NUMERICAL ANALYSIS AND CONTROL OF BIFURCATION PROBLEMS (II): BIFURCATION IN INFINITE DIMENSIONS». International Journal of Bifurcation and Chaos 01, nᵒ 04 (décembre 1991): 745‑72. https://doi.org/10.1142/S0218127491000555.

[2] Lust, Kurt. «Improved Numerical Floquet Multipliers». International Journal of Bifurcation and Chaos 11, nᵒ 09 (septembre 2001): 2389‑2410. https://doi.org/10.1142/S0218127401003486.
"""
struct FloquetColl{E <: AbstractEigenSolver} <: AbstractFloquetSolver
    eigsolver::E
    function FloquetColl(eigls::AbstractEigenSolver = DefaultEig())
        eigls2 = check_floquet_options(eigls)
        return new{typeof(eigls2)}(eigls2)
    end
    FloquetColl(eigls::FloquetColl) = eigls
end

@views function (eig::FloquetColl)(JacColl, nev; kwargs...)
    pbcoll = JacColl.pb
    J = _get_matrix(JacColl)
    n, m, Ntst = size(pbcoll)
    _eig_floquet_col(J, n, m, Ntst, nev)
end

@views function _eig_floquet_col(J::AbstractMatrix{𝒯}, n, m, Ntst, nev) where 𝒯
    nbcoll = n * m
    N = n
    In = LinearAlgebra.I(N)

    # condensation of parameters
    # this removes the internal unknowns of each mesh interval
    # this matrix is diagonal by blocks and each block is the L Matrix
    # which makes the corresponding J block upper triangular
    # the following matrix 𝐅𝐬 collects the LU factorizations by blocks
    # recall that if F = lu(A) then
    # F.L * F.U = F.P * A
    # (F.P⁻¹ * F.L) * F.U = A
    # hence 𝐅𝐬⁻¹ = (P⁻¹ * L)⁻¹ = L⁻¹ * P
    
    blockⱼ = zeros(𝒯, nbcoll, nbcoll)
    blockₙ = zeros(𝒯, nbcoll, N)
    blockₙ₂ = copy(blockₙ)
    rg = 1:nbcoll
    rN = 1:N

    Jcop = copy(J)
    Jcop[end-N:end-1,end-N:end-1] .= In
    Jcop[end-N:end-1,1:N] .= (-1) .* In
    Jcop[end, end] = J[end, end]
    Lₜ = LowerTriangular(copy(blockⱼ))
    for 𝐢 in 1:Ntst
        blockⱼ .= J[rg, rg .+ N]
        F = lu!(blockⱼ)
        p = F.p
        # Lₜ = LowerTriangular(F.L) # zero allocation?
        Lₜ.data .= F.factors
        for i in axes(Lₜ, 1); Lₜ[i,i] = one(𝒯); end

        # we put the blocks in Jcop
        Jcop[rg, rg .+ N] .= UpperTriangular(F.factors)
        # Jcop[rg, rN] .= F.L \ (F.P * J[rg, rN])
        blockₙ .= J[rg, rN][p,:]
        ldiv!(blockₙ₂, Lₜ, blockₙ)
        Jcop[rg, rN] .= blockₙ₂

        rg = rg .+ nbcoll
        rN = rN .+ nbcoll
    end

    Ai = Matrix{𝒯}(undef, n, n)
    Bi = Matrix{𝒯}(undef, n, n)
    r1 = 1:n
    r2 = n*(m-1)+1:(m*n)

    # monodromy matrix
    M = Matrix{𝒯}(LinearAlgebra.I(n))

    for 𝐢 in 1:Ntst
        Ai .= Jcop[r2, r1]
        Bi .= Jcop[r2, r1 .+ n * m]
        M .= (Bi \ Ai) * M
        r1  = r1 .+ m * n
        r2  = r2 .+ m * n
    end

    # in theory, it should be multiplied by (-1)ᴺᵗˢᵗ
    factor = iseven(Ntst) ? 1 : -1

    # floquet multipliers
    vals, vecs = eigen(M)

    nev = min(n, nev)
    logvals = @. log(Complex(factor * vals))
    I = sortperm(logvals, by = real, rev = true)[1:nev]

    # floquet exponents
    σ = logvals[I]

    # give indications on the precision on the Floquet coefficients
    vp0 = minimum(abs, σ)
    if vp0 > 1e-9
        @debug "The precision on the Floquet multipliers is $vp0.\n It may be not enough to allow for precise bifurcation detection.\n Either decrease `tol_stability` in the option ContinuationPar or use a different method than `FloquetColl` for computing Floquet coefficients."
    end
    return σ, Complex.(vecs[I, :]), true, 1
end

@views function _eig_floquet_col(J::AbstractSparseMatrix{𝒯}, n, m, Ntst, nev) where 𝒯
    nbcoll = n * m
    N = n
    In = LinearAlgebra.I(N)

    # condensation of parameters
    # this removes the internal unknowns of each mesh interval
    # this matrix is diagonal by blocks and each block is the L Matrix
    # which makes the corresponding J block upper triangular
    # the following matrix 𝐅𝐬 collects the LU factorizations by blocks
    # recall that if F = lu(A) then
    # F.L * F.U = F.P * A
    # (F.P⁻¹ * F.L) * F.U = A
    # hence 𝐅𝐬⁻¹ = (P⁻¹ * L)⁻¹ = L⁻¹ * P
    
    blockⱼ = zeros(𝒯, nbcoll, nbcoll)
    blockₙ = zeros(𝒯, nbcoll, N)
    blockₙ₂ = copy(blockₙ)
    rg = 1:nbcoll
    rN = 1:N

    Jcop = copy(J)
    Jcop[:, rN] .= 0
    Jcop[end-N:end-1,end-N:end-1] .= In
    Jcop[end-N:end-1,1:N] .= (-1) .* In
    Jcop[end, end] = J[end, end]
    Lₜ = LowerTriangular(copy(blockⱼ))

    first_column_block = Matrix{𝒯}[]
    upper_triangular = SparseMatrixCSC{𝒯, Int64}[]

    for 𝐢 in 1:Ntst
        # blockⱼ .= J[rg, rg .+ N]
        F = lu(Array(J[rg, rg .+ N]))
        p = F.p
        # Lₜ = LowerTriangular(F.L) # zero allocation?
        Lₜ = F.L
        for i in axes(Lₜ, 1); Lₜ[i,i] = one(𝒯); end

        # # we put the blocks in Jcop
        # Jcop[rg, rg .+ N] .= UpperTriangular(F.factors)
        push!(upper_triangular, F.U)

        # # Jcop[rg, rN] .= F.L \ (F.P * J[rg, rN])
        # blockₙ .= J[rg, rN][p,:]
        # ldiv!(blockₙ₂, Lₜ, blockₙ)
        # Jcop[rg, rN] .= blockₙ₂

        blockₙ = J[rg, rN][p,:]
        blockₙ₂ = Lₜ \ Array(blockₙ)
        # Jcop[rg, rN] .= blockₙ₂
        push!(first_column_block, blockₙ₂)

        # @error "" size(blockₙ₂) size(F.U) typeof(F.U)


        rg = rg .+ nbcoll
        rN = rN .+ nbcoll
    end

    Ai = Matrix{𝒯}(undef, n, n)
    Bi = Matrix{𝒯}(undef, n, n)
    r1 = 1:n
    r2 = n*(m-1)+1:(m*n)

    # monodromy matrix
    M = Matrix{𝒯}(LinearAlgebra.I(n))

    for 𝐢 in 1:Ntst
        Ai = first_column_block[𝐢][end-n+1:end, 1:n]#Jcop[r2, r1]
        # Bi = Jcop[r2, r1 .+ n * m] #upper_triangular[𝐢][end-n+1:end, end-n+1:end]#
        Bi = upper_triangular[𝐢][end-n+1:end, end-n+1:end]
        # @error "" 𝐢 Ai Bi 
        # @error "" first_column_block[𝐢]
        M .= (Bi \ Array(Ai)) * M
        r1  = r1 .+ m * n
        r2  = r2 .+ m * n
    end

    # in theory, it should be multiplied by (-1)ᴺᵗˢᵗ
    factor = iseven(Ntst) ? 1 : -1

    # floquet multipliers
    vals, vecs = eigen(M)

    nev = min(n, nev)
    logvals = @. log(Complex(factor * vals))
    I = sortperm(logvals, by = real, rev = true)[1:nev]

    # floquet exponents
    σ = logvals[I]

    # give indications on the precision on the Floquet coefficients
    vp0 = minimum(abs, σ)
    if vp0 > 1e-9
        @debug "The precision on the Floquet multipliers is $vp0.\n It may be not enough to allow for precise bifurcation detection.\n Either decrease `tol_stability` in the option ContinuationPar or use a different method than `FloquetColl` for computing Floquet coefficients."
    end
    return σ, Complex.(vecs[I, :]), true, 1
end

