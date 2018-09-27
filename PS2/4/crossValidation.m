function [Train, Test] = crossValidation(Data, j)

Test = Data{1,j};
Train = [];
for i = 1: 5
    if i == j
        continue;
    else
        Train = [Train; Data{1, i}];
    end
end

end