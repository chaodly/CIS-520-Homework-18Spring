clear;
clc;

%% Load Data
load('C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS5\ps5-kit\ps5-kit\Problem 4\data.mat');

[l, w] = size(labeled_train_data);
lu = length(unlabeled_data);

Pk = zeros(2, 1);
Pjk = zeros(w, 2);

q = zeros(lu, 2);
lk = zeros(2, 1);
ljk = zeros(100, 2);

%% 

Pk(1) = length(find(train_labels== 1));
Pk(2) = length(find(train_labels== -1));

Pk = Pk ./ l;

flag = find(train_labels == 1); 
cur = labeled_train_data(flag, :);
for j = 1 : w  
    ljk(j, 1) = sum(cur(:, j));
end
t = length(flag);
lk(1) = t;
Pjk(:, 1) = ljk(:, 1)' ./ t;


%%

threshold = -1e-8;
flag = 0;
csdn = 0;
while(~flag)
    csdn = csdn + 1;
    q = zeros(lu, 2);
    
    for x = 1 : lu 
        p = zeros(2, 1);
        for y = 1:2
            c1 = Pjk(:, y) .* unlabeled_data(x, :)';
            c2 = (1 - Pjk(:, y)) .* (1 - unlabeled_data(x, :))';
            c1(find(c1==0)) = 1;
            c2(find(c2==0)) =1;
            p(y) = prod(c1) * prod(c2);
        end
        t = Pk' * p;
        q(x, :) = (Pk .* p ./ t)';
    end
    
    Pk = zeros(2, 1);
    Pjk = zeros(100, 2);
    for i = 1 : 2
        Pk(i) = lk(i) + sum(q(:, i), 1);
    end
    
    for k = 1 : 2
        for j = 1 : 100
            Pjk(j, k) = ljk(j, k)+ sum(q((unlabeled_data(:, j)==1), k), 1);
        end
        Pjk(:, k) = Pjk(:, k) ./ Pk(k);
    end
    Pk = Pk ./ (l + lu);
    
    temp = llh(labeled_train_data, train_labels, unlabeled_data, Pk, Pjk, q);
    
    if ((threshold - temp) / threshold < 0.001)
        flag = 1;
    end
    
    if(csdn>4)
        flag = 1;
    end
    
    threshold = temp;
end

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
accuracy =  length(find(label==test_labels)) / lu;