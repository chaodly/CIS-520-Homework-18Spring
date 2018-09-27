%% Problem 4 (a)
clear;
clc 
% 
% trainDataPath_main = 'C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS1\ps1_kit\Problem-4\Spambase\TrainSubsets\train-';
% trainDataPath = '0%.txt';
% testDataPath = 'C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS1\ps1_kit\Problem-4\Spambase\test.txt';
% 
% 
% trainingError = ones(10, 1);
% testingError = ones(10, 1);
% n = 1: 10;
% 
% % load training data and testing data
% for i = 1:10
%     [trainingError(i), testingError(i)] = calculateErr([trainDataPath_main, num2str(i), trainDataPath], testDataPath);
% end
% figure;
% plot(10 * n, trainingError, 'linewidth', 3);
% title('Training data')
% xlabel('Percentage(%)');
% ylabel('Training data error');
% 
% hold on;
% plot(10 * n, testingError, 'linewidth', 3);
% title('Test data')
% xlabel('Percentage(%)');
% ylabel('Testing data error');
% legend('Training Error', 'Testing Error');
% 

%% Problem 4 (b)


trainDataPath = 'C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS1\ps1_kit\Problem-4\Spambase\TrainSubsets\train-100%.txt';
testDataPath = 'C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS1\ps1_kit\Problem-4\Spambase\test.txt';
lambda = [10^(-7) 10^(-6) 10^(-5) 10^(-4) 10^(-3) 10^(-2) 10^(-1) 1];
n1 = -7: 0;
errTrain_r = ones(8, 1);
errTest_r = ones(8, 1);
for i = 1: 8
        [errTrain_r(i), errTest_r(i)] = calculateErr_Regularized(trainDataPath, testDataPath, lambda(i));
end
[min_Test, lambda_min] = min(errTest_r);
lambda_min = 10^(lambda_min - 8);
figure;
plot(n1, errTrain_r);
title('Training Data Regularized')
xlabel('log(lambda)');
ylabel('Training data error');

hold on;
plot(n1, errTest_r);
title('Test Data Regularized')
xlabel('log(lambda)');
ylabel('Test data error');
legend('Training Error', 'Testing Error');

% Select from 5-fold cross-validation
cvpath1 = 'C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS1\ps1_kit\Problem-4\Spambase\CrossValidation\Fold';
cvpath2 = '\cv-test.txt';
cvpath0 = '\cv-train.txt';
Average_CV_Error = ones(8, 1);
Average_CV_trainError = ones(8, 1);
CV_errTrain_Reg = ones(5, 1);
CV_errTest_Reg = ones(5, 1);
for i = 1:8
    for j = 1:5
        path_train = [cvpath1, num2str(j), cvpath0];
        path_test = [cvpath1, num2str(j), cvpath2];
        [CV_errTrain_Reg(j), CV_errTest_Reg(j)] = calculateErr_Regularized(path_train, path_test, lambda(i));
    end
    Average_CV_Error(i) = mean(CV_errTest_Reg);
    Average_CV_trainError(i) = mean(CV_errTrain_Reg);
end
[min_CV_Err, lambda_min_CV] = min(Average_CV_Error);
lambda_min_CV = 10^(lambda_min_CV - 8);

fprintf('The lambda with minimum training data error is %f\n', lambda_min);
fprintf('The lambda acquired after cross validation is %f\n', lambda_min_CV);
