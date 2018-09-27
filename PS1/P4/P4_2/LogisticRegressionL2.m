function [w, b] = LogisticRegressionL2(traindata, trainlabels, lambda)
    % INPUT : 
    % traindata   - m X n matrix, where m is the number of training points
    % trainlabels - m X 1 vector of training labels for the training data
    % lambda      - regularization parameter (positive real number)
        
    % OUTPUT
    % returns learnt model: w - n x 1 weight vector, b - bias term
    
    options = optimset('Algorithm','trust-region');
    w0 = ones(58, 1);
    w1 = fminunc(@(w)(errorFunction_Regularized(w, traindata, trainlabels, lambda)), w0, options);
    b = w1(58);
    w = w1(1 : 57);
    
    % Fill in your code here    
    % Consider using the fminunc MATLAB function for solving the L2- regularized logistic regression optimization problem. 
end
