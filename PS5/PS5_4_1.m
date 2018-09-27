clear
clc

%% Load Data

load('C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS5\ps5-kit\ps5-kit\Problem 4\data.mat');

[l, w] = size(labeled_train_data);
Pk = zeros(2, 1);
Pjk = zeros(w, 2);

Pk(1) = length(find(train_labels == 1));
Pk(2) = length(find(train_labels == -1));

Pk = Pk ./ l;

%% 
flag = find(train_labels == const(num)); 
cur = labeled_train_data(flag, :);
xks = zeros(w, 1);

for j = 1 : w  
    xks(j) = sum(cur(:,  j));
end

t = length(flag);
Pjk(:, num) = xks ./ t;

Tl = length(test_data);
labels = zeros(Tl, 1);

Pxk = zeros(2,1);
for j = 1 : 2
    c1 = Pjk(:, j) .* test_data(num, :)';
    c2 = (1 - Pjk(:, j)) .* (1 - test_data(num, :))';
    c1(find(c1==0)) = 1;
    c2(find(c2==0)) = 1;
    Pxk(j) = sum(log(c1), 1) + sum(log(c2), 1);
end

prob = log(Pk) + Pxk;

[~, labels(num)] = max(prob);

labels(find(labels == 2)) = -1;

accuracy =  length(find(labels == test_labels)) / 1000;