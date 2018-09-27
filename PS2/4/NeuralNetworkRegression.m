function [w1, w2, b1, b2] = NeuralNetworkRegression(Data, Labels, d1, eta, iter, w1, w2, b1, b2)

    parameters = [Data, ones(size(Data,1),1)];
    c = size(Data,2);
    m = size(Data,1);
    
    
    for count = 1 : iter
        
        temp = parameters * [w1; b1];
        a1 = sigmoid(temp);
        f = a1 * w2 + b2;
        
        Grad_w1 = w1;
        g_d = sigmoid(temp) - sigmoid(temp).^2;
        
        for j = 1:c
            for k = 1:d1
                cur = 0;
                for i = 1:m
                    add = (f(i) - Labels(i) )* w2(k) * g_d(i,k) * parameters(i,j);
                    cur = cur + add;
                end
                Grad_w1(j,k) = 2 * cur / m;
            end
        end
        
        Grad_b1 = b1;
        for j = 1: d1
        cur = 0;
            for i = 1: m
                add = (f(i) - Labels(i)) * w2(j) * g_d(i,j);
                cur = cur + add;
            end
            Grad_b1(j) = 2 * cur / m;
        end
        
        Grad_w2 = w2;
        for j = 1:d1
        cur = 0;
            for i = 1:m
                add = (f(i) - Labels(i)) * sigmoid(temp(i,j));
                cur = cur + add;
            end
            Grad_w2(j) = 2 * cur / m;
        end
        
        Grad_b2 = 2 *sum(f - Labels) / m;
        
        w1 = w1 - eta * Grad_w1;
        w2 = w2 - eta * Grad_w2;
        b1 = b1 - eta * Grad_b1;
        b2 = b2 - eta * Grad_b2;
    end
end