% This function take datapath and q as input.
% returns the cvError of each C.

function [testError, minerror, min_index] = calculate_q_error(mainPath, trainpath, testpath, q)
C = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100];
for i = 1 : 7
        testError_temp = 0;
        for j = 1 : 5
              [trainData, trainLabels] = loadData(mainPath, j, trainpath);
              [testData, testLabels] = loadData(mainPath, j, testpath);

              [alpha, b] = kernel_svm(trainData, trainLabels, C(i), q);

              for k = 1 : size(testData, 2)
                    testPredict(k) = sign(h_classifier(alpha, trainLabels, trainData, testData(:,  k), q) + b);
              end
              testError_temp = testError_temp + classification_error(testPredict, testLabels);
        end
        testError(i) = testError_temp / 5;
end
[minerror, min_index] = min(testError);

end