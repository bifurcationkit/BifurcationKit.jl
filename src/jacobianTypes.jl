abstract type AbstractJacobianType end
abstract type AbstractJacobianFree <: AbstractJacobianType end
abstract type AbstractJacobianMatrix <: AbstractJacobianType end
abstract type AbstractJacobianSparseMatrix <: AbstractJacobianMatrix end

# :autodiffMF, :MatrixFree, :autodiffDense, :autodiffDenseAnalytical, :FiniteDifferences, :FiniteDifferencesDense

"""
Singleton type to trigger the computation of the jacobian Matrix using ForwardDiff.jl. It can be used for example in newton or in deflated newton.
"""
struct AutoDiff <: AbstractJacobianType end

"""
Singleton type to trigger the computation of the jacobian vector product (jvp) using ForwardDiff.jl. It can be used for example in newton or in deflated newton.
"""
struct AutoDiffMF <: AbstractJacobianFree end

"""
Singleton type to trigger the computation of the jacobian vector product (jvp) using finite differences. It can be used for example in newton or in deflated newton.
"""
struct FiniteDifferencesMF <: AbstractJacobianFree end

"""
Singleton type to trigger the computation of the jacobian using finite differences. It can be used for example in newton or in deflated newton.
"""
struct FiniteDifferences <: AbstractJacobianType end
