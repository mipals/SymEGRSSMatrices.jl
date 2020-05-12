# Removing t = 0, such that Σ is invertible
t = Vector(0.1:0.1:1)
n = length(t);

# Creating a test matrix Σ = tril(UV') + triu(VU',1) that is PSD
p = 2;
U, V = SymEGRSSMatrices.spline_kernel(t, p);
# Creating a symmetric exended generator representable semiseperable matrix
K  = SymEGRQSMatrix(U,V,ones(n))
# Creating a dense replica
Σ    = Matrix(K)
chol = cholesky(Σ)

# Calculating its Cholesky factorization
L = cholesky(K)
# Creating a test vector
x = randn(n);

# Testing inverses (Using Cholesky factorizations)
@test isapprox(L\x, chol\x, atol=1e-6)

# Testing logdet
@test isapprox(logdet(L), logdet(chol), atol=1e-10)
@test isapprox(det(L), det(chol), atol=1e-10)

# Testing traces



# Testing show
@test isapprox(L.L, tril(getfield(L,:U)*getfield(L,:W)',-1) + Diagonal(getfield(L,:d)))
@test isapprox(L.U, triu(getfield(L,:W)*getfield(L,:U)',1) + Diagonal(getfield(L,:d)))
