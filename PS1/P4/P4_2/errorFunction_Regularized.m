function E = errorFunction_Regularized(w, data, labels, lambda)
    num = length(labels);
    E = sum(log(1 + exp(-labels .* data * w))) / num + lambda * sum(w.^2) ;
end