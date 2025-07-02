using Revise
using ForwardDiff, IncompleteLU
using BifurcationKit, LinearAlgebra, Plots, SparseArrays
const BK = BifurcationKit

function Laplacian2D(Nx, Ny, lx, ly)
    hx = 2lx/Nx
    hy = 2ly/Ny
    D2x = spdiagm(0 => -2ones(Nx), 1 => ones(Nx-1), -1 => ones(Nx-1) ) / hx^2
    D2y = spdiagm(0 => -2ones(Ny), 1 => ones(Ny-1), -1 => ones(Ny-1) ) / hy^2

    D2x[1,1] = -2/hx^2
    D2x[end,end] = -2/hx^2

    D2y[1,1] = -2/hy^2
    D2y[end,end] = -2/hy^2

    D2xsp = sparse(D2x)
    D2ysp = sparse(D2y)
    A = kron(sparse(I, Ny, Ny), D2xsp) + kron(D2ysp, sparse(I, Nx, Nx))
    return A, D2x
end

@views function NL(u, p)
    (;r, μ, ν, c3, c5, γ) = p
    n = div(length(u), 2)
    u1 = u[1:n]
    u2 = u[n+1:2n]

    ua = u1.^2 .+ u2.^2

    f = similar(u)
    f1 = f[1:n]
    f2 = f[n+1:2n]

    f1 .= @. r * u1 - ν * u2 - ua * (c3 * u1 - μ * u2) - c5 * ua^2 * u1 + γ
    f2 .= @. r * u2 + ν * u1 - ua * (c3 * u2 + μ * u1) - c5 * ua^2 * u2

    return f
end

d1NL(x, p, dx) = ForwardDiff.derivative(t -> NL(x .+ t .* dx, p), 0.)

function Fcgl!(f, u, p)
    mul!(f, p.Δ, u)
    f .= f .+ NL(u, p)
end

function dFcgl(x, p, dx)
    f = similar(dx)
    mul!(f, p.Δ, dx)
    nl = d1NL(x, p, dx)
    f .= f .+ nl
end

# remark: I checked this against finite differences
@views function Jcgl(u, p)
   (;r, μ, ν, c3, c5, Δ) = p

    n = div(length(u), 2)
    u1 = u[1:n]
    u2 = u[n+1:2n]

    ua = u1.^2 .+ u2.^2

    f1u = zero(u1)
    f2u = zero(u1)
    f1v = zero(u1)
    f2v = zero(u1)

    @. f1u =  r - 2 * u1 * (c3 * u1 - μ * u2) - c3 * ua - 4 * c5 * ua * u1^2 - c5 * ua^2
    @. f1v = -ν - 2 * u2 * (c3 * u1 - μ * u2)  + μ * ua - 4 * c5 * ua * u1 * u2
    @. f2u =  ν - 2 * u1 * (c3 * u2 + μ * u1)  - μ * ua - 4 * c5 * ua * u1 * u2
    @. f2v =  r - 2 * u2 * (c3 * u2 + μ * u1) - c3 * ua - 4 * c5 * ua * u2 ^2 - c5 * ua^2

    jacdiag = vcat(f1u, f2v)

    Δ + spdiagm(0 => jacdiag, n => f1v, -n => f2u)
end

####################################################################################################
factor = 1
Nx = 41*factor
Ny = 21*factor
n = Nx*Ny
lx = pi
ly = pi/2

Δ, = Laplacian2D(Nx, Ny, lx, ly)
par_cgl = (r = 0.5, μ = 0.1, ν = 1.0, c3 = -1.0, c5 = 1.0, Δ = blockdiag(Δ, Δ), γ = 0.)
sol0 = zeros(2Nx, Ny)

# we group the differentials together
prob = BK.BifurcationProblem(Fcgl!, vec(sol0), par_cgl, (@optic _.r); J = Jcgl)

eigls = EigArpack(1.0, :LM)
# eigls = eig_MF_KrylovKit(tol = 1e-8, dim = 60, x₀ = rand(ComplexF64, Nx*Ny), verbose = 1)
opt_newton = NewtonPar(tol = 1e-9, verbose = true, eigsolver = eigls, max_iterations = 20)
out = @time BK.solve(prob, Newton(), opt_newton, normN = norminf)
####################################################################################################
opts_br = ContinuationPar(dsmin = 0.001, dsmax = 0.15, ds = 0.001, p_max = 2.5, detect_bifurcation = 3, nev = 9, plot_every_step = 50, newton_options = (@set opt_newton.verbose = false), max_steps = 1060, n_inversion = 6)
br = @time continuation(prob, PALC(), opts_br, verbosity = 0)
####################################################################################################
# normal form computation
hopfpt = get_normal_form(br, 2; autodiff = false)
####################################################################################################
# Continuation of the Hopf Point using Jacobian expression

ind_hopf = 1
optnew = NewtonPar(opts_br.newton_options, verbose = true)
hopfpoint = newton(br, ind_hopf;
                    options = optnew, 
                    normN = norminf, 
                    start_with_eigen = true)
BK.converged(hopfpoint) && printstyled(color=:red, "--> We found a Hopf Point at l = ", hopfpoint.u.p[1], ", ω = ", hopfpoint.u.p[2], ", from l = ", br.specialpoint[ind_hopf].param, "\n")

br_hopf = continuation(br, 1, (@optic _.γ),
    ContinuationPar(dsmin = 0.001, dsmax = 0.02, ds= 0.01, p_max = 6.5, p_min = -10.0, detect_bifurcation = 1, newton_options = optnew, plot_every_step = 5, tol_stability = 1e-7, nev = 15); plot = true,
    update_minaug_every_step = 1,
    start_with_eigen = false, 
    bothside = false,
    detect_codim2_bifurcation = 2,
    verbosity = 3, normC = norminf,
    jacobian_ma = BK.MinAug(),
    bdlinsolver = BorderingBLS(solver = DefaultLS(), check_precision = false))

plot(br_hopf, branchlabel = "Hopf curve", legend = :top)

# normal form of BT point
get_normal_form(br_hopf, 2; autodiff = false)

# improve estimation of BT point
btsol = BK.newton(br_hopf, 2; jacobian_ma = BK.MinAug(),)

# find the index of the BT point
indbt = findfirst(x -> x.type == :bt, br_hopf.specialpoint)
# branch from the BT point
brfold = continuation(br_hopf, indbt, 
                    setproperties(br_hopf.contparams; detect_bifurcation = 1, max_steps = 20, save_sol_every_step = 1);
                    update_minaug_every_step = 1,
                    detect_codim2_bifurcation = 2,
                    callback_newton = BK.cbMaxNorm(1e5),
                    bdlinsolver = BorderingBLS(solver = DefaultLS(), check_precision = false),
                    jacobian_ma = BK.MinAug(),
                    bothside = true, normC = norminf)

br_hopf2 = @set br_hopf.specialpoint = br_hopf.specialpoint[1:1]
plot(br_hopf2, brfold; legend = :topleft, branchlabel = ["Hopf", "Fold"])

# normal form of Zero-Hopf point
get_normal_form(brfold, 4; autodiff = false, nev = 15)

hopf_from_zh = continuation(brfold, 5, setproperties(brfold.contparams; detect_bifurcation = 1, max_steps = 40, save_sol_every_step = 1);
    update_minaug_every_step = 1,
    detect_codim2_bifurcation = 2,
    callback_newton = BK.cbMaxNorm(1e5),
    start_with_eigen = true,
    bdlinsolver = BorderingBLS(solver = DefaultLS(), check_precision = false),
    jacobian_ma = BK.MinAug(),
    bothside = false, 
    normC = norminf)

plot!(hopf_from_zh)
####################################################################################################
ind_hopf = 1
# number of time slices
M = 30
r_hopf, Th, orbitguess2, hopfpt, vec_hopf = BK.guess_from_hopf(br, ind_hopf, opt_newton.eigsolver, M, 22*sqrt(0.1); phase = 0.25)

orbitguess_f2 = reduce(hcat, orbitguess2)
orbitguess_f = vcat(vec(orbitguess_f2), Th) |> vec

poTrap = PeriodicOrbitTrapProblem(re_make(prob, params = (@set par_cgl.r = r_hopf - 0.01)), real.(vec_hopf), hopfpt.u, M, 2n; jacobian = BK.MatrixFree())

ls0 = GMRESIterativeSolvers(N = 2n, reltol = 1e-9)#, Pl = lu(I + par_cgl.Δ))
poTrapMF = setproperties(poTrap; linsolver = ls0)
@reset poTrapMF.prob_vf.VF.J = (x, p) ->  (dx -> dFcgl(x, p, dx))

BK.residual(poTrap, orbitguess_f, @set par_cgl.r = r_hopf - 0.1) |> plot
BK.residual(poTrapMF, orbitguess_f, @set par_cgl.r = r_hopf - 0.1) |> plot


plot();BK.plot_periodic_potrap(orbitguess_f, M, Nx, Ny; ratio = 2);title!("")
deflationOp = DeflationOperator(2, (x,y) -> dot(x[1:end-1],y[1:end-1]), 1.0, [zero(orbitguess_f)])
####################################################################################################
#
#                                     slow version DO NOT RUN!!!
#
####################################################################################################
# opt_po = (@set opt_po.eigsolver = eig_MF_KrylovKit(tol = 1e-4, x₀ = rand(2Nx*Ny), verbose = 2, dim = 20))
opt_po = (@set opt_po.eigsolver = DefaultEig())
opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.03, ds= 0.001, p_max = 2.5, max_steps = 250, plot_every_step = 3, newton_options = (@set opt_po.linsolver = DefaultLS()), nev = 5, tol_stability = 1e-7, detect_bifurcation = 0)
@assert 1==0 "Too much memory will be used!"
br_pok2 = continuation(PeriodicOrbitTrapProblem(poTrap; jacobian = :FullLU),
            orbitguess_f, PALC(),
            opts_po_cont;
            verbosity = 2,    plot = true,
            plot_solution = (x ;kwargs...) -> plot_periodic_potrap(x, M, Nx, Ny; kwargs...),
            record_from_solution = (u, p) -> BK.amplitude(u, Nx*Ny, M), normC = norminf)
###################################################################################################
# we use an ILU based preconditioner for the newton method at the level of the full Jacobian of the PO functional
Jpo = @time poTrap(Val(:JacFullSparse), orbitguess_f, @set par_cgl.r = r_hopf - 0.01); # 0.5sec

Precilu = @time ilu(Jpo, τ = 0.005); # ~2 sec

ls = GMRESIterativeSolvers(verbose = false, reltol = 1e-3, N = size(Jpo,1), restart = 40, maxiter = 50, Pl = Precilu, log=true)
ls(Jpo, rand(ls.N))

opt_po = @set opt_newton.verbose = true
@reset opt_po.linsolver = ls

outpo_f = @time newton(poTrapMF, orbitguess_f, opt_po; normN = norminf);
BK.converged(outpo_f) && printstyled(color=:red, "--> T = ", outpo_f.u[end])
plot();BK.plot_periodic_potrap(outpo_f.u, M, Nx, Ny; ratio = 2);title!("")

opt_po = @set opt_po.eigsolver = EigKrylovKit(tol = 1e-3, x₀ = rand(2n), verbose = 2, dim = 25)
opt_po = @set opt_po.eigsolver = EigArpack(; tol = 1e-3, v0 = rand(2n))
opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.03, ds = 0.001, p_max = 1.2, max_steps = 250, plot_every_step = 3, newton_options = (@set opt_po.linsolver = ls), nev = 5, tol_stability = 1e-5, detect_bifurcation = 3)

br_po = @time continuation(poTrapMF, outpo_f.u, PALC(), opts_po_cont;
        verbosity = 3,
        plot = true,
        plot_solution = (x, p;kwargs...) -> BK.plot_periodic_potrap(x, M, Nx, Ny; ratio = 2, kwargs...),
        record_from_solution = (u, p; k...) -> begin
                solpo = BK.get_periodic_orbit(p.prob, u, nothing)
                maximum(solpo.u)
        end,
        normC = norminf)

branches = Any[deepcopy(br_po)]
# push!(branches, br_po)
plot(branches[1]; putspecialptlegend = false, label="", xlabel="r", ylabel="Amplitude", legend = :bottomright)
###################################################################################################
# automatic branch switching from Hopf point
br_po = continuation(
    # arguments for branch switching
    br, 1,
    # arguments for continuation
    opts_po_cont, poTrapMF;
    # ampfactor = 3.,
    verbosity = 3, 
    plot = true,
    # callback_newton = (x, f, J, res, iteration, itl, options; kwargs...) -> (println("--> amplitude = ", BK.amplitude(x, n, M; ratio = 2));true),
    finalise_solution = (z, tau, step, contResult; k...) ->
    (BK.haseigenvalues(contResult) && Base.display(contResult.eig[end].eigenvals) ;true),
    plot_solution = (x, p; kwargs...) -> BK.plot_periodic_potrap(x, M, Nx, Ny; ratio = 2, kwargs...),
    record_from_solution = (u, p; k...) -> BK.amplitude(u, Nx*Ny, M; ratio = 2), 
    normC = norminf)
####################################################################################################
# Experimental, full Inplace
@views function NL!(f, u, p, t = 0.)
    (; r, μ, ν, c3, c5) = p
    n = div(length(u), 2)
    u1v = u[1:n]
    u2v = u[n+1:2n]

    f1 = f[1:n]
    f2 = f[n+1:2n]

    @inbounds for ii = 1:n
        u1 = u1v[ii]
        u2 = u2v[ii]
        ua = u1^2+u2^2
        f1[ii] = r * u1 - ν * u2 - ua * (c3 * u1 - μ * u2) - c5 * ua^2 * u1
        f2[ii] = r * u2 + ν * u1 - ua * (c3 * u2 + μ * u1) - c5 * ua^2 * u2
    end
    return f
end

@views function dNL!(f, u, p, du)
    (; r, μ, ν, c3, c5) = p
    n = div(length(u), 2)
    u1v = u[1:n]
    u2v = u[n+1:2n]

    du1v = du[1:n]
    du2v = du[n+1:2n]

    f1 = f[1:n]
    f2 = f[n+1:2n]

    @inbounds for ii = 1:n
        u1 = u1v[ii]
        u2 = u2v[ii]
        du1 = du1v[ii]
        du2 = du2v[ii]
        ua = u1^2+u2^2
        f1[ii] = (-5*c5*u1^4 + (-6*c5*u2^2 - 3*c3)*u1^2 + 2*μ*u1*u2 - c5*u2^4 - c3*u2^2 + r) * du1 +
        (-4*c5*u2*u1^3 + μ*u1^2 + (-4*c5*u2^3 - 2*c3*u2)*u1 + 3*u2^2*μ - ν) * du2

        f2[ii] = (-4*c5*u2*u1^3 - 3*μ*u1^2 + (-4*c5*u2^3 - 2*c3*u2)*u1 - u2^2*μ + ν) * du1 + (-c5*u1^4 + (-6*c5*u2^2 - c3)*u1^2 - 2*μ*u1*u2 - 5*c5*u2^4 - 3*c3*u2^2 + r) * du2
    end
    return f
end

function Fcgl!(f, u, p, t = 0.)
    NL!(f, u, p)
    mul!(f, p.Δ, u, 1., 1.)
end

function dFcgl!(f, x, p, dx)
    # 19.869 μs (0 allocations: 0 bytes)
    dNL!(f, x, p, dx)
    mul!(f, p.Δ, dx, 1., 1.)
end

sol0f = vec(sol0)
out_ = similar(sol0f)
@time Fcgl!(out_, sol0f, par_cgl)
@time dFcgl!(out_, sol0f, par_cgl, sol0f)

probInp = BifurcationProblem(Fcgl!, vec(sol0), (@set par_cgl.r = r_hopf - 0.01), (@optic _.r); J = dFcgl!, inplace = true)

ls = GMRESIterativeSolvers(verbose = false, reltol = 1e-3, N = size(Jpo,1), restart = 40, maxiter = 50, Pl = Precilu, log=true)
ls(Jpo, rand(ls.N))

ls0 = GMRESIterativeSolvers(N = 2Nx*Ny, reltol = 1e-9)#, Pl = lu(I + par_cgl.Δ))
poTrapMFi = PeriodicOrbitTrapProblem(
            probInp,
            real.(vec_hopf), hopfpt.u,
            M, 2n, ls0; jacobian = BK.MatrixFree())

# ca ne devrait pas allouer!!!
out_po = copy(orbitguess_f)
@time BK.residual!(poTrapMFi, out_po, orbitguess_f, par_cgl);
@time BK.potrap_functional_jac!(poTrapMFi, out_po, orbitguess_f, par_cgl, orbitguess_f)
opt_po_inp = @set opt_po.linsolver = ls
outpo_ = @time newton(poTrapMFi, orbitguess_f, opt_po_inp; normN = norminf);


lsi = BK.KrylovLSInplace(rtol = 1e-3; S = Vector{Float64}, n = length(orbitguess_f), m = length(orbitguess_f), is_inplace = true, memory = 40, Pl = Precilu, ldiv = true)
opt_po_inp_kl = @set opt_po.linsolver = lsi
# outpo_f = @time newton(poTrapMFi, 
#                         orbitguess_f,
#                         opt_po_inp_kl; 
#                         normN = norminf, 
#                         )

####################################################################################################
# Computation of Fold of limit cycle
function d2Fcglpb(f, x, dx1, dx2)
    return ForwardDiff.derivative(t2 -> ForwardDiff.derivative( t1 -> f(x .+ t1 .* dx1 .+ t2 .* dx2), 0.), 0.)
end

# we look at the second fold point
indfold = 1
foldpt = BK.foldpoint(br_po, indfold)

Jpo = poTrap(Val(:JacFullSparse), orbitguess_f, (@set par_cgl.r = r_hopf - 0.1));
Precilu = @time ilu(Jpo, τ = 0.005);
ls = GMRESIterativeSolvers(verbose = false, reltol = 1e-5, N = size(Jpo, 1), restart = 40, maxiter = 60, Pl = Precilu, log = true)
ls(Jpo, rand(ls.N))

probFold = BifurcationProblem((x, p) -> BK.residual(poTrap, x, p), 
                        foldpt, getparams(br), getlens(br);
                        J = (x, p) -> poTrap(Val(:JacFullSparse), x, p),
                        )
outfold = @time BK.newton_fold(
        br_po, indfold; #index of the fold point
        prob = probFold,
        options = (@set opt_po.linsolver = ls),
        bdlinsolver = BorderingBLS(solver = ls, check_precision = false))
BK.converged(outfold) && printstyled(color=:red, "--> We found a Fold Point at α = ", outfold.u.p," from ", br_po.specialpoint[indfold].param,"\n")

optcontfold = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, p_max = 40.1, p_min = -10., newton_options = (@set opt_po.linsolver = ls), max_steps = 20, detect_bifurcation = 0)

# optcontfold = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, p_max = 40.1, p_min = -10., newton_options = opt_po, max_steps = 10)

outfoldco = @time BK.continuation_fold(probFold,
    br_po, indfold, (@optic _.c5),
    optcontfold;
    jacobian_ma = BK.MinAug(),
    bdlinsolver = BorderingBLS(solver = ls, check_precision = false),
    plot = true, verbosity = 2)

plot(outfoldco, label="", xlabel="c5", ylabel="r")


####################################################################################################
# Continuation of periodic orbits on the GPU
# using CUDA, Test
# CUDA.allowscalar(false)
using Metal, Test

import LinearAlgebra: mul!, axpby!
mul!(x::CuArray, y::CuArray, α::T) where {T <: Number} = (x .= α .* y)
mul!(x::CuArray, α::T, y::CuArray) where {T <: Number} = (x .= α .* y)
axpby!(a::T, X::CuArray, b::T, Y::CuArray) where {T <: Number} = (Y .= a .* X .+ b .* Y)

par_cgl_gpu = @set par_cgl.Δ = CUDA.CUSPARSE.CuSparseMatrixCSC(par_cgl.Δ);
Jpo = poTrap(Val(:JacFullSparse), orbitguess_f, (@set par_cgl.r = r_hopf - 0.01))
Precilu = @time ilu(Jpo, τ = 0.003);

struct LUperso{Tl, Tu}
    L::Tl
    Ut::Tu    # transpose of U in LU decomposition
end

# https://github.com/JuliaDiffEq/DiffEqBase.jl/blob/master/src/init.jl#L146-L150
function LinearAlgebra.ldiv!(_lu::LUperso, rhs::Array)
    @show "bla"
    _x = (_lu.Ut) \ ((_lu.L) \ rhs)
    rhs .= vec(_x)
    # CuArrays.unsafe_free!(_x)
    rhs
end

function LinearAlgebra.ldiv!(_lu::LUperso, rhs::CuArray)
    _x = UpperTriangular(_lu.Ut) \ (LowerTriangular(_lu.L) \ rhs)
    rhs .= vec(_x)
    CUDA.unsafe_free!(_x)
    rhs
end

# test if we can run Fcgl on GPU
sol0_f = vec(sol0)
sol0gpu = CuArray(sol0_f)
_dxh = rand(length(sol0_f))
_dxd = CuArray(_dxh)

outh = Fcgl(sol0_f, par_cgl);
outd = Fcgl(sol0gpu, par_cgl_gpu);
@test norm(outh-Array(outd), Inf) < 1e-12

outh = dFcgl(sol0_f, par_cgl, _dxh);
outd = dFcgl(sol0gpu, par_cgl_gpu, _dxd);
@test norm(outh-Array(outd), Inf) < 1e-12


orbitguess_cu = CuArray(orbitguess_f)
norm(orbitguess_f - Array(orbitguess_cu), Inf)


Precilu_gpu = LUperso(LowerTriangular(CUDA.CUSPARSE.CuSparseMatrixCSR(I+Precilu.L)), UpperTriangular(CUDA.CUSPARSE.CuSparseMatrixCSR(sparse(Precilu.U'))));

Precilu_host = LUperso((I+Precilu.L), (sparse(Precilu.U')));

rhs = rand(size(Jpo,1))
    sol_0 = Precilu \ rhs
    sol_1 = UpperTriangular(sparse(Precilu.U')) \ (LowerTriangular(I+Precilu.L)  \ (rhs))
    # sol_2 = LowerTriangular(Precilu.U') \ (LowerTriangular(sparse(I+Precilu.L))  \ (rhs))
    norm(sol_1-sol_0, Inf64)
    # norm(sol_2-sol_0, Inf64)

sol_0 = (I+Precilu.L) \ rhs
    sol_1 = LowerTriangular(CUDA.CUSPARSE.CuSparseMatrixCSR(I+Precilu.L)) \ CuArray(rhs)
    @assert norm(sol_0-Array(sol_1), Inf64) < 1e-10

sol_0 = (Precilu.U)' \ rhs
    sol_1 = UpperTriangular(CUDA.CUSPARSE.CuSparseMatrixCSR(sparse(Precilu.U'))) \ CuArray(rhs)
    norm(sol_0-Array(sol_1), Inf64)
    @assert norm(sol_0-Array(sol_1), Inf64) < 1e-10


sol_0 = Precilu \ rhs
    sol_1 = ldiv!(Precilu_host, copy(rhs))
    @assert norm(sol_1-sol_0, Inf64) < 1e-10

sol_0 = ldiv!(Precilu_gpu, copy(CuArray(rhs)));
    sol_1 = ldiv!(Precilu_host, copy(rhs))
    norm(sol_1-Array(sol_0), Inf64)
    @assert norm(sol_1-Array(sol_0), Inf64) < 1e-10


# matrix-free problem on the gpu
ls0gpu = GMRESKrylovKit(rtol = 1e-9)
poTrapMFGPU = PeriodicOrbitTrapProblem(
    re_make(prob; J = (x,p) -> (dx -> dFcgl(x,p,dx))),
    CuArray(real.(vec_hopf)),
    CuArray(hopfpt.u),
    M, 2n, ls0gpu; ongpu = true)

poTrapMFGPU(orbitguess_cu, @set par_cgl_gpu.r = r_hopf - 0.1);
poTrapMFGPU(orbitguess_cu, (@set par_cgl_gpu.r = r_hopf - 0.1), orbitguess_cu);

ls = GMRESKrylovKit(verbose = 2, Pl = Precilu, rtol = 1e-3, dim  = 20)
outh, = @time ls((Jpo), orbitguess_f) #0.4s

lsgpu = GMRESKrylovKit(verbose = 2, Pl = Precilu_gpu, rtol = 1e-3, dim  = 20)
Jpo_gpu = CUDA.CUSPARSE.CuSparseMatrixCSR(Jpo);
outd, = @time lsgpu(Jpo_gpu, orbitguess_cu);

@test norm(outh-Array(outd), Inf) < 1e-12

outh = @time pb(orbitguess_f);
    outd = @time poTrapMFGPU(orbitguess_cu);
    norm(outh-Array(outd), Inf)

_dxh = rand(length(orbitguess_f));
    _dxd = CuArray(_dxh);
    outh = @time pb(orbitguess_f, _dxh);
    outd = @time pbgpu(orbitguess_cu, _dxd);
    norm(outh-Array(outd), Inf)

outpo_f = @time newton(
            poTrapMF, orbitguess_f, (@set par_cgl.r = r_hopf - 0.01),
            (@set opt_po.linsolver = ls); jacobianPO = :FullMatrixFree,
            normN = x -> maximum(abs.(x)),
            # callback = (x, f, J, res, iteration, options) -> (println("--> amplitude = ", amplitude(x));true)
            ) #14s
converged(outpo_f) && printstyled(color=:red, "--> T = ", outpo_f.u[end], ", amplitude = ", amplitude(outpo_f.u, Nx*Ny, M),"\n")

opt_po = @set opt_newton.verbose = true
    outpo_f, hist, flag = @time newton(
            poTrapMFGPU,
            orbitguess_cu, (@set par_cgl_gpu.r = r_hopf - 0.01),
            (@set opt_po.linsolver = lsgpu); jacobianPO = :FullMatrixFree,
            normN = x -> maximum(abs.(x)),
            # callback = (x, f, J, res, iteration, options) -> (println("--> amplitude = ", BK.amplitude(x, Nx*Ny, M));true)
            ) #7s
    flag && printstyled(color=:red, "--> T = ", outpo_f[end:end], ", amplitude = ", amplitude(outpo_f, Nx*Ny, M),"\n")


opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.03, ds= 0.001, p_max = 2.2, max_steps = 35, plot_every_step = 3, newton_options = (@set opt_po.linsolver = lsgpu))
    br_pok2, upo , _= @time BK.continuation(
        poTrapMFGPU,
        orbitguess_cu, (@set par_cgl_gpu.r = r_hopf - 0.01), (@optic _.r),
        opts_po_cont; jacobianPO = :FullMatrixFree,
        verbosity = 2,
        record_from_solution = (u, p) -> (u2 = norm(u), period = sum(u[end:end])),
        normC = x -> maximum(abs.(x)))
