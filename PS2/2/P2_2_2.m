%% k-NN
clc, clear;
k = [1, 5, 9, 49, 99];

%% cross-validation
mainPath = 'C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS2\ps2-kit\ps2-kit\Problems-1-2-3\Spam-Dataset\CrossValidation\Fold';
trainpath = '\cv-train.txt';
testpath = '\cv-test.txt';

for i = 1 : 5
    tempError = 0;
    for j = 1: 5
        
        trainData = load(strcat(mainPath, num2str(j), trainpath));
        trainLabels = trainData(:, 58);
        trainData = trainData(:, 1:57);
        
        testData = load(strcat(mainPath, num2str(j), testpath));
        testLabels = testData(:, 58);
        testData = testData(:, 1:57);
        
        
        yLabel = knn(testData, trainData, trainLabels, k(i));
        tempError = tempError + classification_error(yLabel, testLabels);
    end
    cvError(i) = tempError / 5;
end

[minCVerror, minCVindex] = min(cvError);
disp('Selected K values is:');
disp(k(minCVindex));

%% training error & test error
completeTrainData = load('C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS2\ps2-kit\ps2-kit\Problems-1-2-3\Spam-Dataset\train.txt');
completeTrainLabels = completeTrainData (:, 58);
completeTrainData = completeTrainData(:, 1:57);

completeTestData = load('C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS2\ps2-kit\ps2-kit\Problems-1-2-3\Spam-Dataset\test.txt');
completeTestLabels = completeTestData (:, 58);
completeTestData = completeTestData(:, 1:57);

for i = 1: 5
    trainPredict = knn(completeTrainData, completeTrainData, completeTrainLabels, k(i));
    trainError(i) = classification_error(trainPredict, completeTrainLabels);
    testPredict = knn(completeTestData, completeTrainData, completeTrainLabels, k(i));
    testError(i) = classification_error(testPredict, completeTestLabels);
end

%% plot learning curve
figure;
plot(1:5, cvError, 'r');
hold on;
plot(1:5, trainError, 'g');
hold on;
plot(1:5, testError, 'b');
xlabel('k = 1, 5, 9, 49, 99', 'FontSize', 12);
ylabel('Different Kinds of Errors', 'FontSize', 12);
legend('Cross Validation Error', 'Trainind Error', 'Testing error');

%% plot Decision Boundary
% xrange = [0 25];
% yrange = [0 12];
% inc = 0.01;
% [x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
% image_size = size(x);
% xy = [x(:) y(:)]; 
% 
% xy = [reshape(x, image_size(1)*image_size(2),1) reshape(y, image_size(1)*image_size(2),1)];
% scores = knn(xy, completeTrainData, completeTrainLabels, k(minCVindex));
% idx = scores>0;
% %-----------------------------------
% decisionmap = reshape(idx, image_size);
% figure;
% imagesc(xrange,yrange,decisionmap);
% hold on;
% set(gca,'ydir','normal');
% cmap =  [0.0 0.8 1; 1 0.8 0.8];
% colormap(cmap);
% temp1 = completeTestData((completeTestLabels==-1),:);
% temp2 = completeTestData((completeTestLabels==1),:);
% plot(temp1(:,1),temp1(:,2),'bx','linewidth',0.3);
% plot(temp2(:,1),temp2(:,2), 'ro','linewidth',0.3);
% legend('Class -1','Class +1')
% xlabel('x1');
% ylabel('x2');