struct SymEGRQSMatrix{T,UT<:AbstractArray,VT<:AbstractArray,dT<:AbstractArray} <: AbstractMatrix{T}
    Ut::UT
    Vt::VT
    d::dT
	n::Int
	p::Int
	function SymEGRQSMatrix{T,UT,VT,dT}(Ut,Vt,d,n,p) where
			 {T,UT<:AbstractArray,VT<:AbstractArray,dT<:AbstractArray}
        Up, Un = size(Ut)
		Vp, Vn = size(Vt)
		(Un == Vn && Up == Vp && Vn == length(d)) || throw(DimensionMismatch())
    	new(Ut,Vt,d,n,p)
	end
end

# Constuctor
SymEGRQSMatrix(Ut::AbstractArray{T,N}, Vt::AbstractArray{T,N}, d::AbstractArray{T,M}) where {T,N,M}=
	SymEGRQSMatrix{T,typeof(Ut),typeof(Vt),typeof(d)}(Ut,Vt,d,size(Vt,2),size(Vt,1));


########################################################################
#### Helpful properties. Not nessecarily computionally efficient    ####
########################################################################
Matrix(K::SymEGRQSMatrix) = tril(K.Ut'*K.Vt) + triu(K.Vt'*K.Ut,1) + Diagonal(K.d)

size(K::SymEGRQSMatrix) = (K.n, K.n)

function getindex(K::SymEGRQSMatrix{T}, i::Int, j::Int) where T
	i > j && return dot(K.Ut[:,i],K.Vt[:,j])
	j == i && return dot(K.Vt[:,i],K.Ut[:,j]) + K.d[i]
	return dot(K.Vt[:,i],K.Ut[:,j])
end

Base.propertynames(F::SymEGRQSMatrix, private::Bool=false) =
    (private ? fieldnames(typeof(F)) : ())



########################################################################
#### Linear Algebra routines	                                    ####
########################################################################

#### Matrix-matrix product ####
function dss_mul_mat!(Y::Array, K::SymEGRQSMatrix, X::Array)
	Ut = K.Ut
	Vt = K.Vt
	d = getfield(K,:d)
    p, n = size(Ut);
    mx = size(X,2);
    Vbar = zeros(p,mx);
    Ubar = Ut*X;
    @inbounds for i = 1:n
        tmpV = Vt[:,i];
        tmpU = Ut[:,i];
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
function (\)(K::SymEGRQSMatrix, x::AbstractVecOrMat)
	L = cholesky(K);
	return L'\(L\x)
end
