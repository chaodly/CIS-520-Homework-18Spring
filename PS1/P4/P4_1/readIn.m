% This function reads in the training data
% And at the same time, combines the training data and the corresponding
% labels.

function [Data, Labels] = readIn(path)

    fid = fopen(path);
    data = fscanf(fid, '%f');
    fclose(fid);
    data = reshape(data, 58, []);
    data = data';
    
    % traindata, add ones (column-vector) in the end
    Data = data(:, 1: 57);
    Data = [Data ones(size(Data, 1), 1)];
    
    % trainlabel
    Labels = data(:, 58);
end