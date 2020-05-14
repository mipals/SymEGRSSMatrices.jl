# Removing t = 0, such that Σ is invertible
t = Vector(0.1:0.1:10); p = 2;

# Creating generators U,V that result in a positive-definite matrix Σ
U, V = spline_kernel(t, p)
K = SymEGRSSMatrix(U,V)
Σ    = Matrix(K)
chol = cholesky(Σ)

# Creating a symmetric extended generator representable semiseperable matrix
# Calculating its Cholesky factorization
L = cholesky(K)
# Creating a test vector
xt = ones(size(K,1))

# Testing inverses (Using Cholesky factorizations)
@test chol\xt ≈ L\xt

# Testing logdet and det
@test logdet(L) ≈ logdet(chol)
@test det(L) ≈ det(chol)

# Testing show
@test L.L ≈ tril(getfield(L,:U)*getfield(L,:W)')
@test L.U ≈ triu(getfield(L,:W)*getfield(L,:U)')
@test Matrix(L) ≈ tril(getfield(L,:U)*getfield(L,:W)')
