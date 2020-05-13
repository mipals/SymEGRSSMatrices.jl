using SymEGRSSMatrices
using LinearAlgebra
using Test

@testset "SymEGRSSMatrix" begin
    include("test_SymEGRSSMatrix.jl")
end

@testset "SymEGRSSCholesky" begin
    include("test_SymEGRSSCholesky.jl")
end

@testset "SymEGRQSMatrix" begin
    include("test_SymEGRQSMatrix.jl")
end

@testset "SymEGRQSCholesky" begin
    include("test_SymEGRQSCholesky.jl")
end
