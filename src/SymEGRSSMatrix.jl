struct SymEGRSSMatrix{T,UT<:AbstractArray,VT<:AbstractArray} <: AbstractMatrix{T}
	U::UT
	V::VT
	n::Int
	p::Int

	function SymEGRSSMatrix{T,UT,VT}(U,V,n,p) where
				{T,UT<:AbstractArray,VT<:AbstractArray}
		Un, Um = size(U)
		Vn, Vm = size(V)
		(Un == Vn && Um == Vm) || throw(DimensionMismatch())
		new(U,V,n,p)
	end
end

SymEGRSSMatrix(U::AbstractArray{T,N},V::AbstractArray{T,N}) where {T,N}  =
	SymEGRSSMatrix{T,typeof(U),typeof(V)}(U,V,size(U)[1],size(U)[2])

########################################################################
#### Helpful properties. Not nessecarily computionally efficient    ####
########################################################################
Matrix(K::SymEGRSSMatrix) = tril(K.U*K.V') + triu(K.V*K.U',1)

size(K::SymEGRSSMatrix) = (K.n,K.n)

function getindex(K::SymEGRSSMatrix{T}, i::Int, j::Int) where T
	i > j && return dot(K.U[i,:],K.V[j,:])
	return dot(K.V[i,:],K.U[j,:])
end

Base.propertynames(F::SymEGRSSMatrix, private::Bool=false) =
    ( private ? fieldnames(typeof(F)) : ())

########################################################################
#### Linear Algebra routines 							   			####
########################################################################
function symegrss_mul!(Y::AbstractVecOrMat{T}, K::SymEGRSSMatrix{Q,UT,VT},
		  X::AbstractVecOrMat{S}) where
		  {T,Q,UT<:AbstractArray,VT<:AbstractArray,S}
	U = K.U;
	V = K.V;
	n = K.n;
	m = K.p;
    mx = size(X,2);
    Vbar = zeros(m,mx);
    Ubar = U'*X;
    @inbounds for i = 1:n
        tmpV = V[i,:];
        tmpU = U[i,:];
        Ubar -= tmpU .* X[i:i,:];
        Vbar += tmpV .* X[i:i,:];
        Y[i,:] = Vbar'*tmpU + Ubar'*tmpV;
    end
	return Y
end


logdet(K::SymEGRSSMatrix) = logdet(cholesky(K))
det(K::SymEGRSSMatrix) = det(cholesky(K))


#### Matrix-matrix product ####
mul!(y::AbstractVecOrMat, K::SymEGRSSMatrix, x::AbstractVecOrMat) =
		symegrss_mul!(y,K,x)
mul!(y::AbstractVecOrMat, K::Adjoint{<:Any,<:SymEGRSSMatrix}, x::AbstractVecOrMat) =
		symegrss_mul!(y,K.parent,x)
(\)(K::SymEGRSSMatrix, x::AbstractVecOrMat) = ldiv!(cholesky(K), x)
