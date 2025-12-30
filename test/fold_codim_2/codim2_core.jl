using Test

let
    lens1 = @optic _.a
    lens2 = @optic _.b
    prob = BifurcationProblem((x,p)->x, rand(2), (a=1., b=2., c=3.),lens1)
    𝐌𝐚 = BK.FoldMinimallyAugmentedFormulation(prob)
    𝐏𝐛 = BK.FoldMAProblem(𝐌𝐚, lens2)
    @test lens1 == getlens(prob)
    @test BK.get_lenses(𝐏𝐛) == (lens1, lens2)

    # test the function getparams
    np = BK.getparams(zeros(3), 0.2, 𝐏𝐛)
    @test np.a == 0
    @test np.b == 0.2
    @test np.c == prob.params.c
    BK.getparams(BK.BorderedArray(zeros(3), 0.2), 𝐏𝐛)
    @test np.a == 0
    @test np.b == 0.2
    @test np.c == prob.params.c
    BK.getparams(BK.BorderedArray(zeros(3), 0.1), 0.2, 𝐏𝐛)
    @test np.a == 0
    @test np.b == 0.2
    @test np.c == prob.params.c

    hh = BK.HopfHopf(0, zeros(2), ((@optic _[1]), (@optic _[2])), zeros(1), zeros(1), (λ1 = 0, λ2 = 0), :nothing)
    show(stdout, hh)
end