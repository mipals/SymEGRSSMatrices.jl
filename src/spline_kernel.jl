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

function spline_kernel(t::Array{Float64}, p::Int)

    #TO-DO
    # Check inputs
    # Maybe the Vector(range) can be done more eleganly.

    if all(diff(t) .> 0)
        monotonic = 1;
    elseif all(diff(t) .< 0)
        monotonic = 0;
    else
        throw(DomainError("t is not strictly monotonic"));
    end

    if size(t,2) != 1 && size(t,1) == 1;
        t = t';
    end

    fp = factorial.(p-1:-1:0)
    a = alpha(p).*fp;
    if monotonic == 1;
        U = (repeat(t,1,p).^Vector(p-1:-1:0)')./fp';
        V = (repeat(t,1,p).^Vector(p:2*p-1)').*a';
    elseif monotonic == 0
        U = (repeat(t,1,p).^Vector(p:2*p-1)')./fp';
        V = (repeat(t,1,p).^Vector(p-1:-1:0)').*a';
    end

    return U,V;
end
