%% k-NN
clc, clear;
k = [1, 5, 9, 49, 99];

%% cross-validation
path0 = 'C:\Users\wuyan\Desktop\CIS520\PS2\ps2-kit\ps2-kit\Problems-1-2-3\Spam-Dataset\CrossValidation\Fold';
path1 = '\cv-train.txt';
path2 = '\cv-test.txt';

for i = 1: 5
    currentCVerror = 0;
    for j = 1: 5
        cv_train = load([path0, num2str(j), path1]);
        train_CV = cv_train(:,1:57);
        label_CV_train = cv_train(:,58);
        CV_test_data = load([path0, num2str(j), path2]);
        CV_test_labels = CV_test_data(:,58);
        CV_test_data = CV_test_data(:, 1:57);        
        y_label = NearestNeighbor(CV_test_data, train_CV, label_CV_train, k(i));
        currentCVerror = currentCVerror + classification_error(y_label, CV_test_labels);
    end
    CV_error(i) = currentCVerror / 5;
end

[minCVerror, minCVindex] = min(CV_error);
disp('Selected K values is:');
disp(k(minCVindex));

%% training error & test error
train_data = load('C:\Users\wuyan\Desktop\CIS520\PS2\ps2-kit\ps2-kit\Problems-1-2-3\Spam-Dataset\train.txt');
test_data = load('C:\Users\wuyan\Desktop\CIS520\PS2\ps2-kit\ps2-kit\Problems-1-2-3\Spam-Dataset\test.txt');
train_labels = train_data(:,58);
train_data = train_data(:,1:57);
test_labels = test_data(:,58);
test_data = test_data(:, 1:57);

for i = 1: 5
    y_train = NearestNeighbor(train_data, train_data, train_labels, k(i));
    error_train(i) = classification_error(y_train, train_labels);
    y_test = NearestNeighbor(test_data, train_data, train_labels, k(i));
    error_test(i) = classification_error(y_test, test_labels);
end

%% plot learning curve
figure;
plot(1:5,CV_error, 'r');
hold on;
plot(1:5,error_train, 'g');
hold on;
plot(1:5,error_test, 'b');
xlabel('k = 1, 5, 9, 49, 99', 'FontSize', 12);
ylabel('Different Kinds of Errors', 'FontSize', 12);
legend('Cross Validation Error', 'Trainind Error', 'Testing error');

