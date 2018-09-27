% Classifier function, i.e., h(x) = sign(...)
function prediction = h_classifier(alpha, Labels, Data, x, q)
       prediction = 0;
       for i = 1 :  size(Data, 2)
            temp = alpha(i) * Labels(i) * kernel(Data(:, i), x, q);
            prediction = prediction + temp;
       end
end
