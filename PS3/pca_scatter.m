% Used for (2)

function pca_scatter(coeff, X_train, nums, s, e)

    trans = coeff( :, s : e); % 784 * 2
    zero = X_train(nums/10 - 1199 : nums/10, :); % 1200 * 784   
    seven = X_train(nums/10 * 8 - 1199 : nums/10 * 8, :);
    zero_t = zero * trans;
    seven_t = seven * trans;

    figure
    scatter(zero_t(:, 1), zero_t(:, 2))
    hold on
    scatter(seven_t(:, 1), seven_t(:, 2))
end
