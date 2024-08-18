using Test

let
    lens1 = @optic _.a
    lens2 = @optic _.b
    prob = BifurcationProblem((x,p)->x,rand(2),(a=1., b=2., c=3.),lens1)
    fold_ma = BK.FoldProblemMinimallyAugmented(prob)
    wrap = BK.FoldMAProblem(fold_ma, lens2)
    @test lens1 == getlens(prob)
    @test BK.get_lenses(wrap) == (lens1, lens2)
end