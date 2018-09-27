% This function is used to classify an [0, 1] variable into {0, 1}.
function f = classification(x)
f = x;
    for i = 1: size(x, 1)
        if x(i) >= 0.5
            f(i) = 1;
        else
            f(i) = -1;
        end
    end
end
        