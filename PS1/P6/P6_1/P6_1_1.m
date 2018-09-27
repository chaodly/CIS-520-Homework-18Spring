%% HW6 (a) i

clear
clc

load('C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS1\ps1_kit\Problem-6\Data-set-1\Subsets.mat');
load('C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS1\ps1_kit\Problem-6\Data-set-1\Data.mat')

training_result = zeros(1, 10);
test_result = zeros(1, 10);

for i = 1 : 10
    traindata = cell2mat(subs(i));
    traindata_x = [traindata(: , 1) ones(length(traindata(:, 1)), 1)];
    traindata_y = traindata(: , 2);
    w0 = pinv(traindata_x' * traindata_x) * traindata_x' * traindata_y;
    w = w0(1);
    b = w0(2);
    training_result(i) = mean_squared_error(w * traindata_x(:, 1) + b , traindata_y);
    
    testdata = test;
    testdata_x = [testdata(: , 1) ones(length(testdata(:, 1)), 1)];
    testdata_y = testdata(: , 2);
    test_result(i) = mean_squared_error(w * testdata_x(:, 1) + b , testdata_y);
end

figure
plot(1:10, training_result, 'linewidth', 3);
hold on
plot(1:10, test_result, 'linewidth', 3);
title('Error Curve')
xlabel('x');
ylabel('Error');
legend('Training Error', 'Test Error')

figure
scatter(test(:, 1), test(:, 2))
hold on
x = 0:0.1:1;
plot(x, w * x + b, 'linewidth', 3)