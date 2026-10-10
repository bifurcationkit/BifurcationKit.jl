module FFTExt

    using FFTW
    import LinearAlgebra: ldiv!
    import BifurcationKit: POTrapCirculantPrec, _cyclic_length

    # apply P⁻¹ = (F ⊗ I)⁻¹ diag(Λ_k⁻¹) (F ⊗ I)   (real input/output)
    # The block-circulant inverse is only applied to the leading `_cyclic_length(P)` components.
    # Any extra components (closure slice x_M and the period) are left untouched, so that P can
    # also be applied to a full bordered vector, the border being handled by the bordered solver.
    function _apply_cyclic_inverse!(dest::AbstractVector, P::POTrapCirculantPrec, x::AbstractVector)
        Lc = _cyclic_length(P)
        length(x) ≥ Lc || throw(DimensionMismatch("POTrapCirculantPrec acts on the $Lc leading (cyclic) components but received an input of length $(length(x))."))
        @views P.Rhat .= reshape(x[1:Lc], P.N, P.m)
        FFTW.fft!(P.Rhat, 2)
        @inbounds for k in 1:P.m
            P.Zhat[:, k] = P.facts[k] \ (@views P.Rhat[:, k])
        end
        FFTW.ifft!(P.Zhat, 2)
        @views dest[1:Lc] .= vec(real(P.Zhat))
        return dest
    end

    function ldiv!(y::AbstractVector, P::POTrapCirculantPrec, x)
        length(y) == length(x) || throw(DimensionMismatch("ldiv!(y, P, x) requires length(y) == length(x), got $(length(y)) and $(length(x))."))
        _apply_cyclic_inverse!(y, P, x)
        if y !== x
            @views y[_cyclic_length(P)+1:end] .= x[_cyclic_length(P)+1:end]
        end
        return y
    end

    function ldiv!(P::POTrapCirculantPrec, x)
        _apply_cyclic_inverse!(x, P, x)
        return x
    end

end
