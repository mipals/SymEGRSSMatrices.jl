# Removing t = 0, such that Σ is invertible
t = Vector(0.1:0.1:10)

# Creating a test matrix Σ = tril(UV') + triu(VU',1) that is PSD
p = 2;
U, V = spline_kernel(t, p)
# Creating a symmetric exended generator representable semiseperable matrix
K  = SymEGRQSMatrix(copy(U'),copy(V'),ones(size(U,1)))
# Creating a dense replica
Σ    = Matrix(K)
chol = cholesky(Σ)

# Calculating its Cholesky factorization
L = cholesky(K)
# Creating a test vector
x = randn(size(K,1))

# Testing inverses (Using Cholesky factorizations)
@test L\x ≈ chol.L\x
@test L'\x ≈ chol.U\x

# Testing logdet
@test logdet(L) ≈ logdet(chol)
@test det(L) ≈ det(chol)

# Testing traces and norm
M = SymEGRSSMatrix(copy(U'),copy(V'));
@test isapprox(tr(L,M), tr(chol\Matrix(M)), atol=1e-6)
@test trinv(L) ≈ tr(chol\Diagonal(ones(size(L,1))))
@test SymEGRSSMatrices.fro_norm_L(L) ≈ norm(chol.L[:])^2

# Testing show
@test L.L ≈ tril(U*L.Wt,-1) + Diagonal(L.d)
@test L.U ≈ triu(L.Wt'*U',1) + Diagonal(L.d)
@test Matrix(L) ≈ tril(L.Ut'*L.Wt,-1) + Diagonal(L.d)
@test L[3,1] ≈ chol.L[3,1]
@test L[2,2] ≈ chol.L[2,2]
@test L[1,3] ≈ chol.L[1,3]
