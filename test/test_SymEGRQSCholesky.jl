# Removing t = 0, such that Σ is invertible
t = Vector(0.1:0.1:1)
n = length(t);

# Creating a test matrix Σ = tril(UV') + triu(VU',1) that is PSD
p = 2;
U, V = spline_kernel(t, p);
Σ    = spline_kernel_matrix(U, V) + I;
chol = cholesky(Σ)

# Creating a symmetric exended generator representable semiseperable matrix
K  = SymEGRQSMatrix(U,V,ones(n))
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
