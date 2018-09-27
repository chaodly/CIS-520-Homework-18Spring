function y = knn(testData, trainData, trainLabels, k)
    
y = ones(size(testData, 1), 1);
    for i = 1 : size(testData, 1)
        
        % replicate each row of testData (instance) and compute with the
        % trainData matrix.
        tempData = repmat(testData(i,:), size(trainData, 1), 1);
        temp = (tempData - trainData) .^ 2;
        d = 0;
        for m = 1: size(temp , 2)
            d= d + temp(: , m);
        end
		
        d = sqrt(d);
		
        [Distance, index] = sort(d);
		
        total = 0;
        for j = 1: k
            total = total + trainLabels(index(j));
        end
        if total >= 0
            y(i) = 1;
        else
            y(i) = -1;
        end
    end
end