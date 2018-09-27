clear
clc

mainPath = 'C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS4\ps4-kit\ps4-kit\Problem-2\';

for i = 1 : 5
      cvTrainStruct(i) = load([mainPath, 'CrossValidation\Fold' , num2str(i), '\cv-train.mat']);
      cvTestStruct(i) = load([mainPath, 'CrossValidation\Fold' , num2str(i), '\cv-test.mat']);
      muStruct(i) = load([mainPath, 'MeanInitialization\Part_b\mu_k_' , num2str(i), '.mat']);
end

load([mainPath, 'X.mat']);
load([mainPath, 'X_test.mat']);

for k = 1 : 5
    sigma = cell(1, k);
    for i = 1 : 5
        cvTrainData = cvTrainStruct(i).X_train;
        cvTestData = cvTestStruct(i).X_test;
        mu= muStruct(k).mu;
        for j = 1 : k
            sigma{j} = eye(2);
        end
        pai = ones(1, k) / k;

        init_llh= compute_nllh(cvTrainData, k, mu, sigma, pai);

        [mu, sigma, pai] = GMM(cvTrainData, k, mu, sigma, pai);
        temp_llh = compute_nllh(cvTrainData, k, mu, sigma, pai);

        num = 1;
        while (num <= 1000 && (temp_llh - init_llh) > 1e-6)
            init_llh = temp_llh;
            [mu, sigma, pai] = GMM(cvTrainData, k, mu, sigma, pai);
            temp_llh = compute_nllh(cvTrainData, k, mu, sigma, pai);    
            num = num + 1;
        end
        
        cv(i) = compute_nllh(cvTestData, k, mu, sigma, pai);
    end
    
    train_llh(k) = compute_nllh(X_full, k, mu, sigma, pai);
    test_llh(k) = compute_nllh(X_test, k, mu, sigma, pai);
    cv_llh(k) = mean(cv);
end
%% From Sample Plot