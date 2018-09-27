% Kernel Function.
% K(x, x') = (x_T * x' + 1) ^ q;

function f = kernel(X_1, X_2, q)
        f = (X_1' * X_2 + 1) ^ q;
end
