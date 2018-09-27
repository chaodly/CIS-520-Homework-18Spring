function [w, b] = linear_svm(trainData, trainLabels, C)
        H = (trainData .* trainLabels)' * (trainData .* trainLabels);
        f = - ones(size(trainData, 2), 1);
        A = [];
        B = [];
        Aeq = trainLabels;
        beq = 0;
        lb = zeros(size(trainData, 2), 1);
        ub = C * ones(size(trainData, 2), 1);
        alpha = quadprog(H, f, A, B, Aeq, beq, lb, ub);

        w = trainLabels .* trainData * alpha;
        b = mean(trainLabels - w' * trainData);
end