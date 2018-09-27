% This function use to find k, when calculating reconstruction accuracy
% num, the percentage
% l, length of the eigs
function flag = find_k(eigs, eig_sum, num)
    sum = 0;
    flag = 0;
    for i = 1 : length(eigs)
        sum = sum + eigs(i);
        if sum >= num * eig_sum
            flag = i;
            break;
        end
    end
end