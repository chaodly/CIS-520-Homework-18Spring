function E = errorFunction(w, data, labels)
    num = length(labels);
    E = sum(log(1 + exp(-labels .* data * w))) / num;
end


% E = 0;
% num = length(labels);
% for i = 1 : num
%     E = E + log(1 + exp(- labels(i) * w' * data(i, :)')); 
% end