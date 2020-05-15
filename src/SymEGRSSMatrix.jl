struct SymEGRSSMatrix{T,UT<:AbstractArray,VT<:AbstractArray} <: AbstractMatrix{T}
	Ut::UT
	Vt::VT
	n::Int
	p::Int

	function SymEGRSSMatrix{T,UT,VT}(Ut,Vt,n,p) where
				{T,UT<:AbstractArray,VT<:AbstractArray}
		Un, Um = size(Ut)
		Vn, Vm = size(Vt)
		(Un == Vn && Um == Vm) || throw(DimensionMismatch())
		new(Ut,Vt,n,p)
	end
end

SymEGRSSMatrix(Ut::AbstractArray{T,N},Vt::AbstractArray{T,N}) where {T,N}  =
	SymEGRSSMatrix{T,typeof(Ut),typeof(Vt)}(Ut,Vt,size(Ut)[2],size(Ut)[1])

########################################################################
#### Helpful properties. Not nessecarily computionally efficient    ####
########################################################################
Matrix(K::SymEGRSSMatrix) = tril(K.Ut'*K.Vt) + triu(K.Vt'*K.Ut,1)

size(K::SymEGRSSMatrix) = (K.n,K.n)

function getindex(K::SymEGRSSMatrix{T}, i::Int, j::Int) where T
	i > j && return dot(K.Ut[:,i],K.Vt[:,j])
	return dot(K.Vt[:,i],K.Ut[:,j])
end

Base.propertynames(F::SymEGRSSMatrix, private::Bool=false) =
    (private ? fieldnames(typeof(F)) : ())

########################################################################
#### Linear Algebra routines 							   			####
########################################################################
function symegrss_mul!(Y::AbstractVecOrMat{T}, K::SymEGRSSMatrix{Q,UT,VT},
		  X::AbstractVecOrMat{S}) where
		  {T,Q,UT<:AbstractArray,VT<:AbstractArray,S}
	Ut = K.Ut;
	Vt = K.Vt;
	n = K.n;
	m = K.p;
    mx = size(X,2);
    Vbar = zeros(m,mx);
    Ubar = Ut*X;
    @inbounds for i = 1:n
        tmpV = Vt[:,i];
        tmpU = Ut[:,i];
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
function (\)(K::SymEGRSSMatrix, x::AbstractVecOrMat)
	L = cholesky(K);
	return L'\(L\x)
end
