function [mu, sigma, pai] = GMM(X, k, mu, sigma, pai)

    num = length(X);
    
    for i = 1 : k
        temp(:, i) = gaussian_pdf(X, mu(i, :), sigma{i}) * pai(i);
        gamma(:, i) = temp(:, i) ./ sum(temp, 2);
    end
    
    for x = 1 : k
        
        nk(x) = sum(gamma(:, x));
        
        mu(x, :) = 1 / nk(x) * gamma(:, x)' * X;
        
        sum_result = zeros(2, 2);
        for y = 1 : num
            sum_result = sum_result + gamma(y, x) * (X(y, :) - mu(x, :))' * (X(y, :) - mu(x, :));
        end
        
        sigma{x} = 1 / nk(x) * sum_result;
        pai(x) = nk(x) / num;
    end
end