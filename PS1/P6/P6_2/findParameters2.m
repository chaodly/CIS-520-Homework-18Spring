function [w, b] = findParameters2(traindata, trainlabels)
    aug = [traindata, ones(size((traindata),1), 1)];
    w0 = pinv((aug)' * aug) * aug' * trainlabels;
    b = w0(9);
    w = w0(1:8, :);
end