% Used for (5), draw the subplots
function draw_final_plot(X_train, X_train_s, flag, num)
    figure
    
    for i = 1 : 9
        coeff = pca(X_train_s);
        T = coeff(:, 1 : flag(i));
        Z = X_train_s * T;
        X_recover = Z * T' + mean(X_train);
        subplot(3, 4, 1)
        imagesc(reshape(X_train(num, :), 28, 28)')
        title('Original Image')
        subplot(3, 4, i + 1)
        imagesc(reshape(X_recover(num, :), 28, 28)');
        colormap gray
        title(strcat(num2str(10 * (10 - i)), '% Reconstruction'))
    end
%     title(strcat('Image Number', num2str(num)));
end