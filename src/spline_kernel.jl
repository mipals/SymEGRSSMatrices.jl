function alpha(p::Int)
    a = zeros(p, 1);
    for i = 0:p-1
        for j = 0:i
            for k = i:p-1
                a[i+1] = a[i+1] + (-1.0)^(k-j)/((k+j+1)*
                        factorial(j)*
                        factorial(p-1)*
                        factorial(p-1-k)*
                        factorial(i-j)*
                        factorial(k-i));
            end
        end
    end
    return a;
end

function spline_kernel(t::AbstractArray, p::Int)

    fp = factorial.(p-1:-1:0)
    a = alpha(p).*fp;
    Ut = (repeat(t,p,1).^Vector(p-1:-1:0)')./fp';
    Vt = (repeat(t,p,1).^Vector(p:2*p-1)').*a';

    return Ut,Vt;
end
