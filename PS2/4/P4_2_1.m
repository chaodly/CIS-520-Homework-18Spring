

clc, clear;

d1 = [7, 10, 15, 17, 20];
eta = 3.5 * 10^(-6);
iter = 20000;

cvData = load('C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS2\ps2-kit\ps2-kit\Problem-4\Data-set-2\CV_Data.mat');
cvDataTemp = cvData.cv_data_10;
cvDataTotal = cvData.cv_data_all;

mainPath = 'C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS2\ps2-kit\ps2-kit\Problem-4\Data-set-2\param_';
postfix = '.mat';

Data = load('C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS2\ps2-kit\ps2-kit\Problem-4\Data-set-2\Data.mat');
train = Data.train;
trainData = train(:, 1 : end - 1);
trainLabels = train(:, end);

test = Data.test;
testData = test(:, 1: end - 1);
testLabels = test(:, end);

for i = 1: 5
        parameters = load(strcat(mainPath, num2str(d1(i)), postfix));
        iw1 = parameters.W1;
        iw2 = parameters.W2;
        ib1 = parameters.b1;
        ib2 = parameters.b2;

        %% cross-validation for 10% and all data
        currentCVerrorTemp = 0;
        currentCVerrorTotal= 0;
        for j = 1: 5
            [cvTrain, cvTest] = crossValidation(cvDataTemp , j);
            cvTrainDataTemp = cvTrain(:, 1: end - 1);
            cvTrainLabelsTemp = cvTrain(:, end);
            cvTestDataTemp = cvTest(:, 1: end - 1);
            cvTestLabelsTemp = cvTest(:, end);

            [w1, w2, b1, b2] = NeuralNetworkRegressionReLU(cvTrainDataTemp, cvTrainLabelsTemp, d1, eta, iter, iw1, iw2, ib1, ib2);

            yTest = ReLU(cvTestDataTemp * w1 + b1) * w2 + b2;
            currentCVerrorTemp = currentCVerrorTemp + mean_squared_error(yTest, cvTestLabelsTemp);

            [cvTrain, cvTest] = crossValidation(cvDataTotal , j);
            cvTrainDataTotal= cvTrain(:, 1: end - 1);
            cvTrainLabelsTotal= cvTrain(:, end);
            cvTestDataTotal= cvTest(:, 1: end - 1);
            cvTestLabelsTotal= cvTest(:, end);

            [w1, w2, b1, b2] = NeuralNetworkRegressionReLU(cvTrainDataTotal, cvTrainLabelsTotal, d1, eta, iter, iw1, iw2, ib1, ib2);

            yTest = ReLU(cvTestDataTotal* w1 + b1) * w2 + b2;
            currentCVerrorTotal= currentCVerrorTotal+ mean_squared_error(yTest, cvTestLabelsTotal);

        end
        cvErrorTemp(i) = currentCVerrorTemp / 5;
        cvErrorTotal(i) = currentCVerrorTotal/ 5;

        %% train and test error;
        [w1, w2, b1, b2] = NeuralNetworkRegressionReLU(trainData, trainLabels, d1, eta, iter, iw1, iw2, ib1, ib2);
        yTrain = ReLU(trainData * w1 + b1) * w2 + b2;
        trainError(i) = mean_squared_error(yTrain, trainLabels);
        yTest = ReLU(testData * w1 + b1) * w2 + b2;
        testError(i) = mean_squared_error(yTest, testLabels);
end

figure;
plot(1:5, cvErrorTemp,'r');
hold on;
plot(1:5, cvErrorTotal,'b');
hold on;
plot(1:5, trainError, 'g');
hold on;
plot(1:5, testError, 'k');
xlabel('d_1 = 7, 10, 15, 17, 20');
ylabel('Errors');
legend('Cross-Validation error of 10% data', 'Cross-Validation error of all data', 'Training error', 'Testing error');