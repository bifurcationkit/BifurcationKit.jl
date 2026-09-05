function _axpy(MaJ::MassAndJacobian{IdentityOperator, TJ}, a₀, a₁) where {TJ}
    return _axpy(MaJ.J, a₀, a₁)
end

function _axpy(MaJ::MassAndJacobian{TM, TJ}, a₀, a₁) where {TM, TJ}
    return _axpy(MaJ.J, a₀, a₁, MaJ.M)
end

function _axpy_op(MaJ::MassAndJacobian{IdentityOperator, TJ}, v::AbstractArray, a₀, a₁) where {TJ}
    return _axpy_op(MaJ.J, v, a₀, a₁)
end

function _axpy_op(MaJ::MassAndJacobian{TM, TJ}, v::AbstractArray, a₀, a₁) where {TM, TJ}
    return _axpy_op(MaJ.J, v, a₀, a₁, MaJ.M)
end

function _axpy_op!(MaJ::MassAndJacobian{IdentityOperator, TJ}, v::AbstractArray, a₀, a₁) where {TJ}
    return _axpy_op!(MaJ.J, v, a₀, a₁)
end

function _axpy_op!(MaJ::MassAndJacobian{TM, TJ}, v::AbstractArray, a₀, a₁) where {TM, TJ}
    return _axpy_op!(MaJ.J, v, a₀, a₁, MaJ.M)
end