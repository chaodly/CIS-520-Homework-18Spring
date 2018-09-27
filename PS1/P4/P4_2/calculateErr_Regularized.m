function [errTrain, err] = calculateErr_Regularized(trainDataPath, testDataPath, lambda)

    [traindata, trainlabels] = readIn(trainDataPath);
%     trainlabels = changeLabel(trainlabels);
    [w, b] = LogisticRegressionL2(traindata, trainlabels, lambda);
    predictTrain = sigmoid(traindata * [w; b]);
    yTrain = classification(predictTrain);
    errTrain = classification_error(yTrain, trainlabels);

    [testdata, testlabels] = readIn(testDataPath);
%     testlabels = changeLabel(testlabels);
    predict = sigmoid(testdata * [w; b]);
    y = classification(predict);
    err = classification_error(y, testlabels);
end