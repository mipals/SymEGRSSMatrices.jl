using SymEGRSSMatrices
using LinearAlgebra
using Test
include("spline_kernel.jl")

@testset "SymEGRSSMatrices.jl" begin
    include("test_SymEGRSSMatrix.jl")
    include("test_SymEGRSSCholesky.jl")
    include("test_SymEGRQSMatrix.jl")
    include("test_SymEGRQSCholesky.jl")
end
