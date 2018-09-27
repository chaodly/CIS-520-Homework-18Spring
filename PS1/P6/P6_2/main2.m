%% (b) i
clc,clear;

trainingData = load('C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS1\ps1_kit\Problem-6\Data-set-2\Subsets.mat');
testingData = load('C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS1\ps1_kit\Problem-6\Data-set-2\Data.mat');
 
trainingRes = ones(2, 1);
testingRes = ones(2,1);

train = trainingData.subs(1);
train_org = cell2mat(train);

traindata = train_org(:,1: 8);
trainlabels = train_org(:,9);

test_org = testingData.test;
testdata = test_org(:, 1: 8);
testlabels = test_org(:, 9);

[w1, b1] = findParameters2(traindata, trainlabels);
y_hat_Train = traindata * w1 + b1;
y_hat_Test = testdata * w1 + b1;
trainingRes(1) = mean_squared_error(trainlabels, y_hat_Train);
testingRes(1) = mean_squared_error(testlabels, y_hat_Test);
 
train = trainingData.subs(10);
train_org = cell2mat(train);
traindata = train_org(:,1: 8);
trainlabels = train_org(:,9);
test_org = testingData.test;
testdata = test_org(:, 1: 8);
testlabels = test_org(:, 9);
[w2, b2] = findParameters2(traindata, trainlabels);
y_hat_Train = traindata * w2 + b2;
y_hat_Test = testdata * w2 + b2;
trainingRes(2) = mean_squared_error(trainlabels, y_hat_Train);
testingRes(2) = mean_squared_error(testlabels, y_hat_Test);

