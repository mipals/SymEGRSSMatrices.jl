module SymEGRSSMatrices

import LinearAlgebra: cholesky,inv!, tr, mul!, Adjoint, det, logdet, dot,
                      Factorization, tril, triu, ldiv!,
                      LowerTriangular, UpperTriangular, Diagonal
import Base: \, inv, size, eltype, Matrix, getindex, getproperty

export SymEGRSSMatrix, SymEGRSSCholesky,
       SymEGRQSMatrix, SymEGRQSCholesky,
       trinv, cholesky, det, logdet

# Matrices
include("SymEGRSSMatrix.jl")
include("SymEGRSSCholesky.jl")

include("SymEGRQSMatrix.jl")
include("SymEGRQSCholesky.jl")

# Extra functions
include("spline_kernel.jl")

end # module
