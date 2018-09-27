% Linear SVM, main function

clear
clc

%%
mainPath = 'C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS2\ps2-kit\ps2-kit\Problems-1-2-3\Synthetic-Dataset\CrossValidation\Fold';
trainpath = '\cv-train.txt';
testpath = '\cv-test.txt';

%% Calculate C, Cross Validation

trainError = zeros(1, 7);
testError = zeros(1, 7);

for i = 1:5
    
        [trainData, trainLabels] = loadData(mainPath, i, trainpath);
        [testData, testLabels] = loadData(mainPath, i, testpath);
        C = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100];
      
       for j = 1:7
                
                [w, b] = linear_svm(trainData, trainLabels, C(j));
                testPredict = sign(w' * testData + b * ones(1, size(testData, 2)));
                testError(j) = testError(j) + classification_error(testPredict, testLabels);

        end
end

cvError = testError / 5;
[min, min_index] = min(cvError);

Co= C(min_index); % optimal C

%% Calculate [w, b]

completeTrainData = load('C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS2\ps2-kit\ps2-kit\Problems-1-2-3\Synthetic-Dataset\train.txt');
completeTrainLabels = completeTrainData (:, 3)';
completeTrainData = completeTrainData(:, 1:2)';

completeTestData = load('C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS2\ps2-kit\ps2-kit\Problems-1-2-3\Synthetic-Dataset\test.txt');
completeTestLabels = completeTestData (:, 3)';
completeTestData = completeTestData(:, 1:2)';

ww = zeros(2, 7);
bb = zeros(1, 7);
completeTrainError = zeros(1, 7);
completeTestError = zeros(1, 7);

for i = 1 : 7
    [w, b] = linear_svm(completeTrainData, completeTrainLabels, C(i));
    ww(:, i) = w;
    bb(i) = b;
    
    completeTrainPredict = sign(w' * completeTrainData + b * ones(1, size(completeTrainData, 2)));
    completeTrainError(i) = classification_error(completeTrainPredict, completeTrainLabels);     
    completeTestPredict = sign(w' * completeTestData + b * ones(1, size(completeTestData, 2)));
    completeTestError(i) = classification_error(completeTestPredict, completeTestLabels);  
    
end

figure
plot(log10(C), completeTrainError)
hold on
plot(log10(C), completeTestError)
hold on
plot(log10(C), cvError)
xlabel('lg(C)')
ylabel('Error Rate')
title('Error Curve')
legend('Complete Training Error', 'Complete Testing Error','Cross Validation Error')
decisionBoundary(w', b, completeTestData', completeTestLabels')