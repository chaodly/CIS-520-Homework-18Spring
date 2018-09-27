clear
clc

%% Load Data
mainPath = 'C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS4\ps4-kit\ps4-kit\Problem-2\';
load([mainPath, 'X_test.mat']);

K = 3;
samples = [1:10] * 10; 
for i = 1 : 10
      trainStruct(i) = load([mainPath, 'TrainSubsets\X_' , num2str(0.1 * i), '.mat']);
      muStruct(i) = load([mainPath, 'MeanInitialization\Part_a\mu_' , num2str(0.1 * i), '.mat']);
end

%% GMM

for i = 1 : 10
    
    trainData = trainStruct(i).X;
    mu = muStruct(i).mu;
    sigma = {eye(2), eye(2), eye(2)};
    pai = ones(1, K) / K;
    
    init_llh= compute_nllh(trainData, K, mu, sigma, pai);
    
    [mu, sigma, pai] =GMM(trainData, K, mu, sigma, pai);
    temp_llh = compute_nllh(trainData, K, mu, sigma, pai);
    
    num = 1;
    while (num <= 1000 && (temp_llh - init_llh) / size(trainData, 1) > 1e-6)
        init_llh = temp_llh;
        [mu, sigma, pai] =GMM(trainData, K, mu, sigma, pai);
        temp_llh= compute_nllh(trainData, K, mu, sigma,pai);    
        num = num + 1;
    end
    
    train_llh(i) = compute_nllh(trainData, K, mu, sigma, pai);
    test_llh(i) = compute_nllh(X_test, K, mu, sigma, pai);

    figure;

%% From Sample Plot

