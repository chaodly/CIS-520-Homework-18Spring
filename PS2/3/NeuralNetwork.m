function [w1, w2, b1, b2] = NeuralNetwork(Data, Labels, d1, eta, iter, w1, w2, b1, b2)
    
    d = size(Data,2);
    m = size(Data,1);
    
    current = [Data, ones(size(Data,1),1)];
    
for count = 1: iter
    z1 = current * [w1; b1];
    sigma = sigmoid(sigmoid(z1) * w2 + b2) - Labels;
    
    gradient_w1 = w1;    
    temp = sigmoid(z1) - sigmoid(z1).^2;
    
    for j = 1:d
        for k = 1:d1
            cur = 0;
            for i = 1:m
                add = sigma(i,:)*w2(k)*temp(i,k)*current(i,j);
                cur = cur + add;
            end
            gradient_w1(j,k) = cur / m;
        end
    end
	
    gradient_b1 = b1;
	
    for j = 1: d1
        cur = 0;
        for i = 1: m
            add = sigma(i,:) * w2(j) * temp(i,j);
            cur = cur + add;
        end
        gradient_b1(j) = cur / m;
    end
    gradient_w2 = w2;
    for j = 1:d1
        cur = 0;
        for i = 1:m
            add = sigma(i,:) * sigmoid(z1(i,j));
            cur = cur + add;
        end
        gradient_w2(j) = cur / m;
    end
    cur = 0;
    for i = 1: m
        cur = cur + sigma(i,:);
    end
    gradient_b2 = cur / m;
    
    w1 = w1 - eta * gradient_w1;
    w2 = w2 - eta * gradient_w2;
    b1 = b1 - eta * gradient_b1;
    b2 = b2 - eta * gradient_b2;
    
end
end