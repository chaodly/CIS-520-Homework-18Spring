function [w, b] = LogisticRegression(traindata, trainlabels)
    % INPUT : 
    % traindata   - m X n matrix, where m is the number of training points
    % trainlabels - m X 1 vector of training labels for the training data
    % OUTPUT
    % returns learnt model: w - n x 1 weight vector, b - bias term
    % Fill in your code here 
    options = optimset('Algorithm','trust-region');
    w0 = ones(58, 1);
    w1 = fminunc(@(w)(errorFunction(w, traindata, trainlabels)), w0, options);
    b = w1(58);
    w = w1(1 : 57);
    % Consider using fminunc MATLAB function for solving the logistic regression optimization problem.
end
