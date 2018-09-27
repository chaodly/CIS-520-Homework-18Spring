clear;
clc;

%% w1, w2, b1, b2 load path
postfix = '.mat';
w1Path = 'C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS2\ps2-kit\ps2-kit\Problems-1-2-3\Spam-Dataset\setting-files\w1_';
b1Path = 'C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS2\ps2-kit\ps2-kit\Problems-1-2-3\Spam-Dataset\setting-files\b1_';
w2Path = 'C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS2\ps2-kit\ps2-kit\Problems-1-2-3\Spam-Dataset\setting-files\w2_';
b2Path = 'C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS2\ps2-kit\ps2-kit\Problems-1-2-3\Spam-Dataset\setting-files\b2_';

mainPath = 'C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS2\ps2-kit\ps2-kit\Problems-1-2-3\Spam-Dataset\CrossValidation\Fold';
trainpath = '\cv-train.txt';
testpath = '\cv-test.txt';

%% Parameter
eta = 0.1;
iter = 5000;
d1 = [1, 5, 10, 15, 25, 50];

%% Cross validation Error
for i = 1 : 6
    currentCVerror = 0;
    for j = 1 : 5
        % load parameter initial value
        w1 = load(strcat(w1Path, num2str(d1(i)), postfix));
        w2 = load(strcat(w2Path, num2str(d1(i)), postfix));
        b1 = load(strcat(b1Path, num2str(d1(i)), postfix));
        b2 = load(strcat(b2Path, num2str(d1(i)), postfix));
        w1 = w1.w_1;
        w2 = w2.w_2;
        b1 = b1.b1;
        b2 = b2.b2;
        
        % load data
        trainDataOriginal = load([mainPath, num2str(j), trainpath]);
        trainData = trainDataOriginal(:, 1:57);
        trainLabels = trainDataOriginal(:, 58);
        trainLabels = (trainLabels + 1) / 2;
        
        testDataOriginal = load([mainPath, num2str(j), testpath]);
        testData = testDataOriginal(:, 1:57);
        testLabels = testDataOriginal(:, 58);
        testLabels = (testLabels + 1) / 2;
        
        %train parameter
        [w1, w2, b1, b2] = NeuralNetwork(trainData, trainLabels, d1(i), eta, iter, w1, w2, b1, b2);
        
        %predict value
        predict = sign(sigmoid(sigmoid(testData * w1 + b1) * w2 + b2) - 0.5);
        predict = (predict + 1) / 2;
        currentCVerror = currentCVerror + classification_error(predict, testLabels);
    end
    cvError(i) = currentCVerror / 5;
end

[minCVerror, minindex] = min(cvError);

%% train & test error
completeTrainData = load('C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS2\ps2-kit\ps2-kit\Problems-1-2-3\Spam-Dataset\train.txt');
completeTrainLabels = completeTrainData(:, 58);
completeTrainLabels = (completeTrainLabels + 1) / 2;
completeTrainData = completeTrainData(:, 1:57);

completeTestData = load('C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS2\ps2-kit\ps2-kit\Problems-1-2-3\Spam-Dataset\test.txt');
completeTestLabels = completeTestData(:, 58);
completeTestLabels = (completeTestLabels + 1) / 2;
completeTestData = completeTestData(:, 1:57);


for i = 1: 6
      w1 = load(strcat(w1Path, num2str(d1(i)), postfix));
      w2 = load(strcat(w2Path, num2str(d1(i)), postfix));
      b1 = load(strcat(b1Path, num2str(d1(i)), postfix));
      b2 = load(strcat(b2Path, num2str(d1(i)), postfix));
      w1 = w1.w_1;
      w2 = w2.w_2;
      b1 = b1.b1;
      b2 = b2.b2;
    
    [w1, w2, b1, b2] = NeuralNetwork(completeTrainData, completeTrainLabels, d1(i), eta, iter, w1, w2, b1, b2);
    
    
    yTrain = sign(sigmoid(sigmoid(completeTrainData * w1 + b1) * w2 + b2) - 0.5);
    yTrain = (yTrain + 1) / 2;
    trainError(i) = classification_error(yTrain, completeTrainLabels);
    
    yTest = sign(sigmoid(sigmoid(completeTestData * w1 + b1) * w2 + b2) - 0.5);
    yTest = (yTest + 1) / 2;
    testError(i) = classification_error(yTest, completeTestLabels);
    
end

% Choose d1:15
% corresponding train error: 0.0320  test error: 0.1292 

figure;
plot(1:6, cvError, 'r');
hold on;
plot(1:6, trainError, 'g');
hold on;
plot(1:6, testError, 'b');
xlabel('d_1 = 1, 5, 10, 15, 25, 50', 'FontSize', 12);
ylabel('Errors', 'FontSize', 12);
legend('Cross-Validation Error', 'Train error', 'Test error' );
