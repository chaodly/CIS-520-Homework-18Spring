function k = rbf(x1, x2, gamma)
    k = exp(- gamma * (x1 - x2)' * (x1 - x2));
end