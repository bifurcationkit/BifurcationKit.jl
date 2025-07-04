abstract type AbstractNonLinearSolver end

struct Newton <: AbstractNonLinearSolver end
################################################################################################
abstract type AbstractJacobianType end
abstract type AbstractJacobianFree <: AbstractJacobianType end
abstract type AbstractJacobianMatrix <: AbstractJacobianType end
abstract type AbstractJacobianSparseMatrix <: AbstractJacobianMatrix end

# :FiniteDifferencesDense

"""
Singleton type to trigger the computation of the jacobian Matrix using ForwardDiff.jl. It can be used for example in newton or in deflated newton.
"""
struct AutoDiff <: AbstractJacobianType end

"""
Singleton type to trigger the computation of the jacobian vector product (jvp) using ForwardDiff.jl. It can be used for example in newton, in deflated newton or in continuation.
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

"""
For periodic orbits. The jacobian is a sparse matrix which is expressed with a custom analytical formula.
"""
struct FullSparse <: AbstractJacobianType end

"""
Same as FullSparse but the Jacobian is allocated only once and updated inplace. This is much faster than FullSparse but the sparsity pattern of the vector field must be constant.
"""
struct FullSparseInplace <: AbstractJacobianType end


# different jacobian types which parametrize the way jacobians of PO are computed

"""
The jacobian is computed with automatic differentiation, works for dense matrices. Can be used for debugging.
"""
struct AutoDiffDense <: AbstractJacobianMatrix end

"""
The jacobian is computed with an analytical formula works for dense matrices. This is the default algorithm.
"""
struct DenseAnalytical <: AbstractJacobianMatrix end

"""
The jacobian is computed with an analytical formula works for dense matrices.
"""
struct DenseAnalyticalInplace <: AbstractJacobianMatrix end


"""
Same as for `AutoDiffDense` but the jacobian is formed using a mix of AD and analytical formula. Mainly used for Shooting.
"""
struct AutoDiffDenseAnalytical <: AbstractJacobianMatrix end

"""
The jacobian is computed using Jacobian-Free method, namely a jacobian vector product (jvp).
"""
struct MatrixFree <: AbstractJacobianMatrix end

struct BorderedLU <: AbstractJacobianMatrix end
struct FullLU <: AbstractJacobianMatrix end
struct DenseAD <: AbstractJacobianMatrix end
struct Dense <: AbstractJacobianMatrix end
struct BorderedMatrixFree <: AbstractJacobianMatrix end
struct BorderedSparseInplace <: AbstractJacobianMatrix end

################################################################################################
"""
The jacobian for Minimally Augmented problem is based on an analytical formula and is matrix based.
"""
struct MinAugMatrixBased <: AbstractJacobianMatrix end

"""
The jacobian for Minimally Augmented problem is based on an analytical formula.
"""
struct MinAug <: AbstractJacobianType end