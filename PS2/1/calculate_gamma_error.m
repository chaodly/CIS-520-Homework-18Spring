% This function take datapath and q as input.
% returns the cvError of each C.

function [testError, minerror, min_index] = calculate_gamma_error(mainPath, trainpath, testpath, gamma)
C = [1e-3, 1e-2, 1e-1, 1, 10, 100];
for i = 1 : 6
        testError_temp = 0;
        for j = 1 : 5
              [trainData, trainLabels] = loadData(mainPath, j, trainpath);
              [testData, testLabels] = loadData(mainPath, j, testpath);

              [alpha, b] = rbf_svm(trainData, trainLabels, C(i), gamma);

              for k = 1 : size(testData, 2)
                    testPredict(k) = sign(h_classifier_rbf(alpha, trainLabels, trainData, testData(:,  k), gamma) + b);
              end
              testError_temp = testError_temp + classification_error(testPredict, testLabels);
        end
        testError(i) = testError_temp / 5;
end
[minerror, min_index] = min(testError);

end