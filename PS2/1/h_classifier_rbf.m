% Classifier function, i.e., h(x) = sign(...)
function prediction = h_classifier_rbf(alpha, Labels, Data, x, gamma)
       prediction = 0;
       for i = 1 :  size(Data, 2)
            temp = alpha(i) * Labels(i) * rbf(Data(:, i), x, gamma);
            prediction = prediction + temp;
       end
end
