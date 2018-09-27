function [w1, w2, b1, b2] = NeuralNetworkRegressionReLU(Data, Labels, d1, eta, iter, w1, w2, b1, b2)

    current = [Data, ones(size(Data,1),1)];
    d = size(Data,2);
    m = size(Data,1);
    
    
    for find = 1:iter
        temp = current * [w1; b1];
        a1 = ReLU(temp);
        f = a1 * w2 + b2;
        gradient_w1 = w1;
        g_d = (sign(temp) + 1) / 2;
        for j = 1:d
            for k = 1:d1
                cur = 0;
                for i = 1:m
                    add = (f(i) - Labels(i) )* w2(k) * g_d(i,k) * current(i,j);
                    cur = cur + add;
                end
                gradient_w1(j,k) = 2 * cur / m;
            end
        end
        
        gradient_b1 = b1;
        for j = 1: d1
        cur = 0;
            for i = 1: m
                add = (f(i) - Labels(i)) * w2(j) * g_d(i,j);
                cur = cur + add;
            end
            gradient_b1(j) = 2 * cur / m;
        end
        
        gradient_w2 = w2;
        for j = 1:d1
        cur = 0;
            for i = 1:m
                add = (f(i) - Labels(i)) * sigmoid(temp(i,j));
                cur = cur + add;
            end
            gradient_w2(j) = 2 * cur / m;
        end
        
        gradient_b2 = 2 *sum(f - Labels) / m;
        
        w1 = w1 - eta * gradient_w1;
        w2 = w2 - eta * gradient_w2;
        b1 = b1 - eta * gradient_b1;
        b2 = b2 - eta * gradient_b2;
    end
end