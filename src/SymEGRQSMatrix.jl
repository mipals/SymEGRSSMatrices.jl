#export SymEGRQSMatrix

struct SymEGRQSMatrix{T,UT<:AbstractArray,VT<:AbstractArray,dT<:AbstractArray} <: AbstractMatrix{T}
    U::UT
    V::VT
    d::dT
	n::Int
	p::Int
	function SymEGRQSMatrix{T,UT,VT,dT}(U,V,d,n,p) where
			 {T,UT<:AbstractArray,VT<:AbstractArray,dT<:AbstractArray}
        Un, Um = size(U)
		Vn, Vm = size(V)
		(Un == Vn && Um == Vm && Vn == length(d)) || throw(DimensionMismatch())
    	new(U,V,d,n,p)
	end
end

# Constuctors
SymEGRQSMatrix(U::AbstractArray{T,N}, V::AbstractArray{T,N}, d::AbstractArray{T,M}) where {T,N,M}=
	SymEGRQSMatrix{T,typeof(U),typeof(V),typeof(d)}(U,V,d,size(V,1),size(V,2));


########################################################################
#### Helpful properties. Not nessecarily computionally efficient    ####
########################################################################
Matrix(K::SymEGRQSMatrix) = tril(K.U*K.V') + triu(K.V*K.U',1) + Diagonal(K.d)

size(K::SymEGRQSMatrix) = (K.n, K.n)

function getindex(K::SymEGRQSMatrix{T}, i::Int, j::Int) where T
	i > j && return dot(K.U[i,:],K.V[j,:])
	j == i && return dot(K.V[i,:],K.U[j,:]) + K.d[i]
	return dot(K.V[i,:],K.U[j,:])
end

Base.propertynames(F::SymEGRQSMatrix, private::Bool=false) =
    ( private ? fieldnames(typeof(F)) : ())



########################################################################
#### Linear Algebra routines	                                    ####
########################################################################

#### Matrix-matrix product ####
function dss_mul_mat!(Y::Array, K::SymEGRQSMatrix, X::Array)
	U = getfield(K,:U)
	V = getfield(K,:V)
	d = getfield(K,:d)
    n, m = size(U);
    mx = size(X,2);
    Vbar = zeros(m,mx);
    Ubar = U'*X;
    for i = 1:n
        tmpV = V[i,:];
        tmpU = U[i,:];
        tmpX = X[i:i,:];
        Ubar -= tmpU .* tmpX;
        Vbar += tmpV .* tmpX;
        Y[i,:] = tmpU'*Vbar + tmpV'*Ubar + d[i]*tmpX;
    end
	return Y
end

#### Log-determinant ####
logdet(K::SymEGRQSMatrix) = logdet(cholesky(K))

#### Determinant ####
det(K::SymEGRQSMatrix) = det(cholesky(K))




mul!(y::AbstractVecOrMat, K::SymEGRQSMatrix, x::AbstractVecOrMat) =
	dss_mul_mat!(y,K,x)
mul!(y::AbstractVecOrMat, K::Adjoint{<:Any,<:SymEGRQSMatrix}, x::AbstractVecOrMat) =
	dss_mul_mat!(y,K.parent,x)
(\)(K::SymEGRQSMatrix, x::AbstractVecOrMat) = ldiv!(cholesky(K), x)
