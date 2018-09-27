function [alpha, b] = kernel_svm(trainData, trainLabels, C, q)
  
      % Compute H
        for i = 1 : size(trainData, 2)
            for j = 1 : size(trainData, 2)
                H(i, j) = trainLabels(i) * trainLabels(j) * kernel(trainData(:, i), trainData(:, j), q);
            end
        end
        
        f = - ones(size(trainData, 2), 1);
        A = [];
        B = [];
        Aeq = trainLabels;
        beq = 0;
        lb = zeros(size(trainData, 2), 1);
        ub = C * ones(size(trainData, 2), 1);
        alpha = quadprog(H, f, A, B, Aeq, beq, lb, ub);
        
        k = 1;
        for i = 1 : size(trainData,2)
             temp = 0;
            for j = 1: size(trainData, 2)
                temp = temp + alpha(j) * trainLabels(j) * kernel(trainData(:, i), trainData(:, j), q);
            end
            b(k) = trainLabels(i) - temp;
            k = k + 1;
        end
        b = mean(b);
end