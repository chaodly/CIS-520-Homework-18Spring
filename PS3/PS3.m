% Problem Set 3, P7

%% (1)
clear
clc
path = 'C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS3\ps3-kit\ps3-kit\MNIST_train.mat';
load(path)
[m, d] = size(X_train); % fn, numbers of feature
% last_image = reshape(X_train(end, :), 28, 28)';
% imagesc(last_image);
% colormap gray

%% (2)

coeff = pca(X_train);
% v1 = coeff(:, 1);
% figure
% imagesc(reshape(v1, 28, 28));
% colormap gray
% 
% v2 = coeff(:, 2);
% figure
% imagesc(reshape(v2, 28, 28));
% colormap gray
% 
% v3 = coeff(:, 3);
% figure
% imagesc(reshape(v3, 28, 28));
% colormap gray

%% (3)

X_train_s = X_train - mean(X_train); % Standardized
% pca_scatter(coeff, X_train_s , m, 1, 2)
% pca_scatter(coeff, X_train_s , m, 100, 101)

%% (4)

S = (1/(m - 1)) * X_train_s' * X_train_s ;
eigs = eig(S);
eig_sum = sum(eigs);

for i = 1 : 9
    flag(i) = find_k(eigs, eig_sum, (10 - i)/10);
end
figure
plot (flag, 0.9 : -0.1 : 0.1, 'LineWidth', 2);
xlabel('Number of Principal Components')
ylabel('Reconstruction Accuracy')
title('Accuracy Curve')

%% (5)

% % 500, 6000, 10000
% draw_final_plot(X_train, X_train_s, flag, 500)
% draw_final_plot(X_train, X_train_s, flag, 6000)
% draw_final_plot(X_train, X_train_s, flag, 10000)