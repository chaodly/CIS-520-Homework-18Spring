% This function returns Training Error, Testing Error
function [trainingError, testingError] = calculateErr(trainDataPath, testDataPath)

    [trainData, trainLabels] = readIn(trainDataPath);
    
    % {-1, +1} --> {0, 1}
%     trainLabels = changeLabel(trainLabels);
    
    [w, b] = LogisticRegression(trainData, trainLabels);
    trainingResult = sigmoid(trainData * [w; b]);
    
    trainingClass = classification(trainingResult);
    trainingError = classification_error(trainingClass, trainLabels);

    [testData, testLabels] = readIn(testDataPath);
%     testLabels = changeLabel(testLabels);
    
    testingResult = sigmoid(testData * [w; b]);
    testingClass = classification(testingResult);
    testingError = classification_error(testingClass, testLabels);
end