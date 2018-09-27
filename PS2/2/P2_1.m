%% 1-NN
clc,clear;

mainPath =  'C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS2\ps2-kit\ps2-kit\Problems-1-2-3\Synthetic-Dataset';
trainPath = '\train.txt';
testPath = '\test.txt';

trainData = load(strcat(mainPath, trainPath));
testData = load(strcat(mainPath, testPath));

trainLabels = trainData(:, 3);
trainData = trainData(:, 1:2);
testLabels = testData(:, 3);
testData = testData(:, 1:2);


yLabel = knn(testData, trainData, trainLabels, 1);
err = classification_error(yLabel, testLabels);

%% 1- NN decision-boundary
xrange = [0 25];
yrange = [0 12];
inc = 0.01;
[x, y] = meshgrid(xrange(1) : inc:xrange(2), yrange(1) : inc:yrange(2));
image_size = size(x);
xy = [x(:) y(:)]; 

xy = [reshape(x, image_size(1) * image_size(2), 1) reshape(y, image_size(1) * image_size(2), 1)];
scores = knn(xy, trainData, trainLabels, 1);
idx = scores > 0;
%-----------------------------------
decisionmap = reshape(idx, image_size);
figure;
imagesc(xrange,yrange,decisionmap);
hold on;
set(gca,'ydir','normal');
cmap =  [0.0 0.8 1; 1 0.8 0.8];
colormap(cmap);
temp1 = testData((testLabels==-1) , :);
temp2 = testData((testLabels==1) , :);
plot(temp1(: ,1), temp1(: , 2), 'bx', 'linewidth', 0.3);
plot(temp2(: ,1), temp2(: , 2), 'ro', 'linewidth', 0.3);
legend('Class -1','Class +1')
xlabel('x1');
ylabel('x2');