% This function is used for loading data from CrossValidation Folders.
% Specifically, i is the index of the Folders.
% return Data, Labels.

function [Data, Labels] = loadData(mainPath, i, path)

    DataPath = strcat(mainPath, num2str(i), path);
    
    % train, 200 * 3 double, test, 1800 * 3 double
    originalData = load(DataPath)';
    Data = originalData(1 : size(originalData, 1) - 1 , :);
    Labels = originalData(end, :);
end