%% (b) ii 100%
clc, clear;
CV_data_train = load('C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS1\ps1_kit\Problem-6\Data-set-2\CV_Data.mat');
data_train = load('C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS1\ps1_kit\Problem-6\Data-set-2\Subsets.mat');
data_Test = load('C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS1\ps1_kit\Problem-6\Data-set-2\Data.mat');
data_train_100 = cell2mat(data_train.subs(10));
traindata = data_train_100(:,1:8);
trainlabels = data_train_100(:, 9);
    
lambda = [0.1, 1, 10, 100, 500, 1000];
CV_data_100 = CV_data_train.cv_data_all;
data2 = data_Test.test;
testdata = data2(:, 1: 8);
testlabels = data2(:, 9);
CV_error = ones(5,1);
average_CV_error = ones(6:1);
training_error = ones(5:1);

%% Cross Validation
for i = 1:6
CV_remain = cell2mat(CV_data_100(1));
CV_data1 = [cell2mat(CV_data_100(2)); cell2mat(CV_data_100(3)); cell2mat(CV_data_100(4)); cell2mat(CV_data_100(5))];
CV_test = CV_remain(:, 1:8);
CV_testlabels = CV_remain(:,9);
CV_train = CV_data1(:,1:8);
CV_labels = CV_data1(:,9);
[w_CV, b_CV] = ridgeRegression(CV_train, CV_labels, lambda(i));
y_hat = CV_test * w_CV + b_CV;
CV_error(1) = mean_squared_error(CV_testlabels, y_hat);

CV_remain = cell2mat(CV_data_100(2));
CV_data1 = [cell2mat(CV_data_100(1)); cell2mat(CV_data_100(3)); cell2mat(CV_data_100(4)); cell2mat(CV_data_100(5))];
CV_test = CV_remain(:, 1:8);
CV_testlabels = CV_remain(:,9);
CV_train = CV_data1(:,1:8);
CV_labels = CV_data1(:,9);
[w_CV, b_CV] = ridgeRegression(CV_train, CV_labels, lambda(i));
y_hat = CV_test * w_CV + b_CV;
CV_error(2) = mean_squared_error(CV_testlabels, y_hat);

CV_remain = cell2mat(CV_data_100(3));
CV_data1 = [cell2mat(CV_data_100(1)); cell2mat(CV_data_100(2)); cell2mat(CV_data_100(4)); cell2mat(CV_data_100(5))];
CV_test = CV_remain(:, 1:8);
CV_testlabels = CV_remain(:,9);
CV_train = CV_data1(:,1:8);
CV_labels = CV_data1(:,9);
[w_CV, b_CV] = ridgeRegression(CV_train, CV_labels, lambda(i));
y_hat = CV_test * w_CV + b_CV;
CV_error(3) = mean_squared_error(CV_testlabels, y_hat);

CV_remain = cell2mat(CV_data_100(4));
CV_data1 = [cell2mat(CV_data_100(1)); cell2mat(CV_data_100(2)); cell2mat(CV_data_100(3)); cell2mat(CV_data_100(5))];
CV_test = CV_remain(:, 1:8);
CV_testlabels = CV_remain(:,9);
CV_train = CV_data1(:,1:8);
CV_labels = CV_data1(:,9);
[w_CV, b_CV] = ridgeRegression(CV_train, CV_labels, lambda(i));
y_hat = CV_test * w_CV + b_CV;
CV_error(4) = mean_squared_error(CV_testlabels, y_hat);

CV_remain = cell2mat(CV_data_100(5));
CV_data1 = [cell2mat(CV_data_100(1)); cell2mat(CV_data_100(2)); cell2mat(CV_data_100(3)); cell2mat(CV_data_100(4))];
CV_test = CV_remain(:, 1:8);
CV_testlabels = CV_remain(:,9);
CV_train = CV_data1(:,1:8);
CV_labels = CV_data1(:,9);
[w_CV, b_CV] = ridgeRegression(CV_train, CV_labels, lambda(i));
y_hat = CV_test * w_CV + b_CV;
CV_error(5) = mean_squared_error(CV_testlabels, y_hat);

average_CV_error(i) = mean(CV_error);
end

%% Count training error and test error
w_train = ones(8, 6);
b_train = ones(6, 1);
train_error = ones(6:1);
test_error = ones(6:1);
for i = 1:6
    [w_train(:,i), b_train(i)] = ridgeRegression(traindata, trainlabels, lambda(i));
    y_hat = traindata * w_train(:, i) + b_train(i);
    y_hat_test = testdata * w_train(:, i) + b_train(i);
    train_error(i) = mean_squared_error(trainlabels, y_hat);
    test_error(i) = mean_squared_error(testlabels, y_hat_test);
end
figure;
plot(log10(lambda), average_CV_error);
hold on ;
plot(log10(lambda), train_error);
hold on;
plot(log10(lambda), test_error);
xlabel('log_1_0lambda');
ylabel('Errors');
legend('Cross Validation Error', 'Train Error', 'Test Error');