clear;
clc;

%%
subsets = load('C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS2\ps2-kit\ps2-kit\Problem-4\Data-set-1\Subsets.mat');
parameters = load('C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS2\ps2-kit\ps2-kit\Problem-4\Data-set-1\param.mat');
% struct -> matrix
subsets = subsets.subs;

Data = load('C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS2\ps2-kit\ps2-kit\Problem-4\Data-set-1\Data.mat');
train = Data.train;
trainData = train(:, 1);
trainLabels = train(:, 2);

test = Data.test;
testData = test(:, 1);
testLabels = test(:, 2);

d1 = 15;
eta = 0.1;
iter = 20000;
iw1 = parameters.W1;
iw2 = parameters.W2;
ib1 = parameters.b1;
ib2 = parameters.b2;

%%
for i = 1:10
    subdata = subsets{1, i}; %%
    subtrainData = subdata(:,1);
    subtrainLabels = subdata(:,2);
    [w1(i,:), w2(:,i), b1(i,:), b2(i)] = NeuralNetworkRegression(subtrainData, subtrainLabels, d1, eta, iter, iw1, iw2, ib1, ib2);
    
    yTrain = sigmoid(trainData * w1(i,:) + b1(i,:)) * w2(:,i) + b2(i);
    trainError(i) = mean_squared_error(yTrain, trainLabels);
    
    yTest = sigmoid(testData * w1(i,:) + b1(i,:)) * w2(:,i) + b2(i);
    testError(i) = mean_squared_error(yTest, testLabels);
end

figure;
plot(10:10:100, trainError);
hold on;
plot(10:10:100, testError);
xlabel('Percentage of train data');
ylabel('Train error and test error');
legend('Train error', 'Test error');

[w1f, w2f, b1f, b2f] = NeuralNetworkRegression(trainData, trainLabels, d1, eta, iter, iw1, iw2, ib1, ib2);
figure;
x = 0: 0.001: 1;
y = sigmoid(x' * w1f + b1f) * w2f + b2f;
plot(x, y);
hold on;
scatter(testData, testLabels, 'b');