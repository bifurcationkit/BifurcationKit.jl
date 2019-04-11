abstract type PeriodicOrbit end

include("SimpleShooting.jl")
include("PeriodicOrbitFD.jl")


function Flow(Fper, dFper, x0, M::Int, options::NewtonPar)
	options.verbose = false
	xnew = similar(x0)
	for ii=1:M
		xnew, hist, flag = newton(
			u -> Fper(u, x0),
			u -> dFper(u, x0),
			x0,
			options)
		@assert flag == true "Newton method for computing the Flow did not converge at i = $ii, hist = $hist, norm(x0) = $(norm(x0, Inf64))"
		x0 .= xnew
	end
	return xnew
end
