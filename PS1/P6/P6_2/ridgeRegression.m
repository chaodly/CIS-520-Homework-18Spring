function[w, b] = ridgeRegression(traindata, trainlabels, lambda)
    aug = [traindata, ones(size((traindata),1), 1)];
    multi = aug' * aug;
    I = eye(8);
    R = [I zeros(8,1); zeros(1,9)];
    m = size(traindata, 1);
    w0 = pinv(multi + lambda * m * R) * aug' * trainlabels;
    b = w0(9);
    w = w0(1:8, :);
end
