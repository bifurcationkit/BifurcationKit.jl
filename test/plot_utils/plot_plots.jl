using BifurcationKit, Plots
BifurcationKit.set_plot_backend!(BK.BK_Plots())
const BK = BifurcationKit

let
    BK.plot_periodic_potrap(rand(10*100+1)|>vec, 10, 100; ratio = 1)
    BK.plot_periodic_potrap(rand(2*10*100+1)|>vec, 10, 100; ratio = 2)
    BK.plot_periodic_potrap(rand(2*20*100*110+1)|>vec, 20, 100, 110; ratio = 2)
    BK.plot_periodic_shooting(rand(2*10*100+1)|>vec, 10; ratio = 2)
end

let 
    F_simple(x, p; k = 2) = @. p[1] * x + x^(k+1)/(k+1) + 0.01
    opts = ContinuationPar(p_min = -3., detect_bifurcation = 3)
    prob = ODEBifProblem(F_simple, zeros(10), -1.5, (@optic _))
    br = continuation(prob, PALC(tangent=Bordered()), opts, plot = true)
    plot(br)
    plot(br, br)
    BK.plot_eigenvals(br, true)
    BK.plot_eigenvals(br, false)
end