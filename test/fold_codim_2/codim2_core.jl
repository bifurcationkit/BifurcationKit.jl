using Test

let
    lens1 = @optic _.a
    lens2 = @optic _.b
    prob = BifurcationProblem((x,p)->x, rand(2),(a=1., b=2., c=3.),lens1)
    fold_ma = BK.FoldProblemMinimallyAugmented(prob)
    wrap = BK.FoldMAProblem(fold_ma, lens2)
    @test lens1 == getlens(prob)
    @test BK.get_lenses(wrap) == (lens1, lens2)
end

begin
    hh = BK.HopfHopf(0,zeros(2),((@optic _[1]), (@optic _[2])),zeros(1),zeros(1),(λ1 = 0, λ2 = 0),:nothing)
    show(stdout, hh)
end