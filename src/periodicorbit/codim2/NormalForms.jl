function get_normal_form(prob::AbstractBifurcationProblem,
                        br::AbstractResult{ <: TwoParamPeriodicOrbitCont}, id_bif::Int ;
                        nev = length(eigenvalsfrombif(br, id_bif)),
                        verbose = false,
                        ζs = nothing,
                        lens = getlens(br),
                        Teigvec = _getvectortype(br),
                        scaleζ = norm,
                        autodiff = false,
                        δ = getdelta(prob),
                        k...
                    )
    bifpt = br.specialpoint[id_bif]
    if (bifpt.type in (:endpoint,)) || ~(bifpt.type in (:cusp, :R1, :R2, :R3, :R4, :gpd))
        error("Normal form for $(bifpt.type) not implemented")
    end

    # parameters for normal form
    kwargs_nf = (;nev, verbose, lens, Teigvec, scaleζ)

    if bifpt.type == :R1
        return R1_normal_form(prob, br, id_bif; kwargs_nf..., autodiff)
    elseif bifpt.type == :R2
        return R2_normal_form(prob, br, id_bif; kwargs_nf..., autodiff)
    elseif bifpt.type == :R3
        return R3_normal_form(prob, br, id_bif; kwargs_nf..., autodiff)
    elseif bifpt.type == :R4
        return R4_normal_form(prob, br, id_bif; kwargs_nf..., autodiff)
    elseif bifpt.type == :gpd
        return GPD_normal_form(prob, br, id_bif; kwargs_nf..., autodiff)
    elseif bifpt.type == :cusp
        return CuspPO_normal_form(prob, br, id_bif; kwargs_nf..., autodiff)
    end
    error("Normal form for $(bifpt.type) not yet implemented.")
end

for op in (:CuspPO, :R1, :R2, :R3, :R4, :GPD, :FoldNS, :FoldPD)
    @eval begin
        function $(Symbol(op, :_normal_form))(probma::AbstractMABifurcationProblem{Tprob}, 
                                br,ind_bif;
                                nev = length(eigenvalsfrombif(br, ind_bif)),
                                lens = getlens(br),
                                Teigvec = vectortype(br),
                                scaleζ = norminf,
                                kwargs_nf...) where {Tprob}
            prob_ma = probma.prob
            powrap = prob_ma.prob_vf

            x0, parbif = get_bif_point_codim2(br, ind_bif)

            bifpt = br.specialpoint[ind_bif]
            po = get_periodic_orbit(powrap, getvec(bifpt.x, prob_ma), nothing)
            period = getperiod(powrap, getvec(bifpt.x, prob_ma), nothing)

            $op(po, period, parbif, get_lenses(br), nothing, nothing, nothing, powrap, false)
        end
    end
end