% Kernel SVM, main function

clear
clc

%%
mainPath = 'C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS2\ps2-kit\ps2-kit\Problems-1-2-3\Synthetic-Dataset\CrossValidation\Fold';
trainpath = '\cv-train.txt';
testpath = '\cv-test.txt';

%% Calculate q and corresponding C, Cross Validation

q = [1, 2, 3, 4, 5];
C = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100];

for index = 1 : 5
      [cvError, minerror, min_index] = calculate_q_error(mainPath, trainpath, testpath, q(index));
      q(index) = C(min_index);
      q_error(index) = minerror;
end

%% q C

completeTrainData = load('C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS2\ps2-kit\ps2-kit\Problems-1-2-3\Synthetic-Dataset\train.txt');
completeTrainLabels = completeTrainData (:, 3)';
completeTrainData = completeTrainData(:, 1:2)';

completeTestData = load('C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS2\ps2-kit\ps2-kit\Problems-1-2-3\Synthetic-Dataset\test.txt');
completeTestLabels = completeTestData (:, 3)';
completeTestData = completeTestData(:, 1:2)';

for i = 1 : 5
      [alpha_c(: , i), b_c(i)] = kernel_svm(completeTrainData, completeTrainLabels, q(i), i); % C: q(i), i: q;
      for k = 1 : size(completeTrainData, 2)
            completeTrainPredict(k) = sign(h_classifier(alpha_c(: , i), completeTrainLabels, completeTrainData, completeTrainData(:,  k), i) + b_c(: , i));
      end
      trainError(i) = classification_error(completeTrainPredict, completeTrainLabels);
      
      for k = 1 : size(completeTestData, 2)
            completeTestPredict(k) = sign(h_classifier(alpha_c(: , i), completeTrainLabels, completeTrainData, completeTestData(:,  k), i) + b_c(: , i));
      end
      testError(i) = classification_error(completeTestPredict, completeTestLabels);
end

[mintestError, minindex] = min(testError);
alpha_m = alpha_c(:, minindex);
b_m = b_c(minindex);


figure
plot(1:5, q_error);
hold on
plot(1:5, trainError);
hold on
plot(1:5, testError);
xlabel('q')
ylabel('Error')
legend('Cross Validation Error', 'Training Error', 'Testing Error')


xrange = [0 25];
yrange = [0 12];
inc = 0.01;
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
image_size = size(x);
xy = [x(:) y(:)]; 

xy = [reshape(x, image_size(1)*image_size(2),1) reshape(y, image_size(1)*image_size(2),1)];
for j = 1: size(xy, 1)
        scores(j) = sign(h_classifier(alpha_m, completeTrainLabels, completeTrainData, xy(j,:)',  minindex) + b_m);
end
idx = scores > 0;
%-----------------------------------
decisionmap = reshape(idx, image_size);
figure;
imagesc(xrange,yrange,decisionmap);
hold on;
set(gca,'ydir','normal');
cmap =  [0.0 0.8 1; 1 0.8 0.8];
colormap(cmap);
temp1 = completeTestData(: , (completeTestLabels == -1));
temp2 = completeTestData(: , (completeTestLabels == 1));
plot(temp1(1 , :),temp1(2 , :),'bx','linewidth',0.3);
plot(temp2(1 , :),temp2(2 , :), 'ro','linewidth',0.3);
legend('Class -1','Class +1')
xlabel('x1');
ylabel('x2');

