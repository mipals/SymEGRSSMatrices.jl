# Removing t = 0, such that Σ is invertible
t = Vector(0.1:0.1:1); p = 2;

# Creating generators U,V that result in a positive-definite matrix Σ
U, V = SymEGRSSMatrices.spline_kernel(t, p)
K = SymEGRSSMatrix(U,V)
Σ    = Matrix(K)
chol = cholesky(Σ)

# Creating a symmetric extended generator representable semiseperable matrix
# Calculating its Cholesky factorization
L = cholesky(K)
# Creating a test vector
xt = ones(size(K,1))

# Testing inverses (Using Cholesky factorizations)
@test isapprox(chol\xt, L\xt, atol=1e-6)

# Testing logdet and det
@test isapprox(logdet(L), logdet(chol), atol=1e-10)
@test isapprox(det(L), det(chol), atol=1e-10)

# Testing show
@test isapprox(L.L, tril(getfield(L,:U)*getfield(L,:W)'))
@test isapprox(L.U, triu(getfield(L,:W)*getfield(L,:U)'))
