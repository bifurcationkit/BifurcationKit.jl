module FFTExt

    using FFTW
    import LinearAlgebra: ldiv!
    import BifurcationKit: POTrapCirculantPrec, _cyclic_length

    # apply P⁻¹ = (F ⊗ I)⁻¹ diag(Λ_k⁻¹) (F ⊗ I)   (real input/output)
    # The block-circulant inverse is only applied to the leading `_cyclic_length(P)` components.
    # Any extra components (closure slice x_M and the period) are left untouched, so that P can
    # also be applied to a full bordered vector, the border being handled by the bordered solver.
    function ldiv!(y, P::POTrapCirculantPrec, x)
        Lc = _cyclic_length(P)
        length(x) ≥ Lc || throw(DimensionMismatch("POTrapCirculantPrec acts on the $Lc leading (cyclic) components but received an input of length $(length(x))."))
        length(y) == length(x) || throw(DimensionMismatch("ldiv!(y, P, x) requires length(y) == length(x), got $(length(y)) and $(length(x))."))
        @views P.Rhat .= reshape(x[1:Lc], P.N, P.m)
        FFTW.fft!(P.Rhat, 2)
        @inbounds for k in 1:P.m
            P.Zhat[:, k] = P.facts[k] \ P.Rhat[:, k]
        end
        FFTW.ifft!(P.Zhat, 2)
        @views y[1:Lc] .= vec(real(P.Zhat))
        if y !== x
            @views y[Lc+1:end] .= x[Lc+1:end]
        end
        return y
    end

    function ldiv!(P::POTrapCirculantPrec, x)
        Lc = _cyclic_length(P)
        length(x) ≥ Lc || throw(DimensionMismatch("POTrapCirculantPrec acts on the $Lc leading (cyclic) components but received an input of length $(length(x))."))
        @views P.Rhat .= reshape(x[1:Lc], P.N, P.m)
        FFTW.fft!(P.Rhat, 2)
        @inbounds for k in 1:P.m
            P.Zhat[:, k] = P.facts[k] \ P.Rhat[:, k]
        end
        FFTW.ifft!(P.Zhat, 2)
        # x[Lc+1:end] (closure slice x_M + period) is left untouched
        @views x[1:Lc] .= vec(real(P.Zhat))
        return x
    end

end
