# Removing t = 0, such that Σ is invertible
t = Vector(0.1:0.1:100); p = 2;

# Creating generators U,V that result in a positive-definite matrix Σ
U, V = SymEGRSSMatrices.spline_kernel(t, p)

K = SymEGRQSMatrix(U,V,ones(size(U,1)))
x = randn(size(K,1))
Kfull = Matrix(K)

# Testing multiplication
@test K*x ≈ Kfull*x
@test K'*x ≈ Kfull'*x

# Testing linear solve
@test K\x ≈ Kfull\x

# Testing (log)determinant
@test logdet(K) ≈ logdet(Kfull)
@test det(K) ≈ det(Kfull)

# Testing show
@test Matrix(K) ≈ tril(K.U*K.V') + triu(K.V*K.U',1) + Diagonal(K.d)
@test Kfull[3,1] ≈ K[3,1]
@test Kfull[2,2] ≈ K[2,2]
@test Kfull[1,3] ≈ K[1,3]
