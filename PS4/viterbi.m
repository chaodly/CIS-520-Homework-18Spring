function [z, tag, prob] = viterbi(K, init, trans, emit, voc, pos, test1)
    T = size(test1, 1);
    test_index = zeros(1, T);
    
    for i = 1 : length(test1)
        test_index(i) = find(voc == string(test1{i}));
    end
    
    delta = zeros(K, T);
    psai = zeros(K, T);
    delta(:, 1) = init .* emit(:, test_index(1));

    for i = 2 : T
        [temp, psai(:, i)] = max(delta(:, i - 1) .* trans);
        delta(:, i) = temp' .* emit(:, test_index(i));
    end

    z = zeros(T, 1);
    [prob, z(T)] = max(delta(:, T));

    for t = T - 1 : -1 : 1
        z(t) = psai(z(t + 1), t +1);
    end
    tag = pos(z);
end