% This function is used to change the {+1, -1} label to standard binary
% label, that is {0, +1}.
function f = changeLabel(y)
f = ones(length(y), 1);
for i = 1: length(y)
    if y(i) == -1
        f(i) = 0;
    end
end