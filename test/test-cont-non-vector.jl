# this test is designed to test the ability of the package to use a state space that is not an AbstractArray.
using Test, Revise
using PseudoArcLengthContinuation
const Cont = PseudoArcLengthContinuation

function F(x::BorderedVector, r)
    out = similar(x)
    out.u = (r - (x.u)^2)
    out.p = x.p - 0.1
    out
end

# there is no finite differences defined, so we need to provide a linearsolve
# we could also have used GMRES

struct linsolveBd <: Cont.LinearSolver end

function (l::linsolveBd)(J, dx)
    x, r = J
    out = similar(dx)
    out.u = dx.u ./ (- 2 .* (x.u))
    out.p = dx.p
    out, true, 1
end

sol = BorderedVector([0.01], 0.01)

opt_newton = Cont.NewtonPar(tol = 1e-11, verbose = true,
        linsolve = linsolveBd())
out, hist, flag = @time Cont.newton(
                        x -> F(x, 1.5),
                        x -> (x, 1.5),
                        sol,
                        opt_newton)

opts_br0 = Cont.ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= -0.01, pMax = 4.1, newtonOptions = opt_newton, detect_fold = true, detect_bifurcation = false)
	opts_br0.newtonOptions.maxIter = 70
	opts_br0.newtonOptions.tol = 1e-8
	opts_br0.maxSteps = 150

	br, u1 = @time Cont.continuation(
					(x, p) -> F(x, p),
					(x, p) -> (x, p),
					out, 1.5, opts_br0, printsolution = x-> x.u[1])

plotBranch(br);title!("")

outfold, hist, flag = @time Cont.newtonFold((x, p) -> F(x, p),
										(x, p) -> (x, p),
										br, 1, #index of the fold point
										opts_br0.newtonOptions)
		flag && printstyled(color=:red, "--> We found a Fold Point at α = ", outfold.p, ", β = 0.01, from ", br.bifpoint[indfold][3],"\n")
