% Returns the first index of the mins.
function f = search_min(X)
        target = min(X);
        for i = 1: length(X)
                if (X(i) == target)
                      f = i;
                      break;
                end
        end
end