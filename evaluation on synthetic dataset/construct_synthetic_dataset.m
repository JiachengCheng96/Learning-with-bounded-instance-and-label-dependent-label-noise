function [ output_args ] = construct_synthetic_dataset(rho_1max, rho_0max, W1, W2)
    % generate 3 structs: train_clean, train_noisy, test.
    % each stuct is a dataset consisting of a observation matrix and a label matrix
    mu = 2 * [-1 1];
    SIGMA = diag([1,1]);

    P_total = 1000; %1000
    P_train = 500; %250
    P_test = P_total - P_train;

    Positve = mvnrnd(mu, SIGMA, P_total); % positive examples

    N_total = 1000;
    N_train = 500;
    N_test = N_total - N_train;

    Negative = mvnrnd((-1)*mu, SIGMA,N_total); % negative examples

    train_clean.X = [(Positve(1:P_train,:))', (Negative(1:N_train,:))'];
    train_clean.y = [ones(1,P_train), zeros(1,N_train)];

    test.X=[(Positve(P_train+1:P_train+P_test,:))',(Negative(N_train+1:N_train+N_test,:))'];
    test.y=[ones(1,P_test),zeros(1,N_test)];

    train_clean.X = [ones(1,size(train_clean.X,2)); train_clean.X]; 
    test.X = [ones(1,size(test.X,2)); test.X];

    save dataset_clean.mat train_clean test

    train_noisy.X = train_clean.X;
    train_noisy.y_tilde = zeros(size(train_clean.y));

    rho1 = rho_1max *(sigmoid(W1' * train_clean.X(:, find(train_clean.y==1))));
    rho0 = rho_0max*sigmoid(W2' * train_clean.X(:, find(train_clean.y==0)));
    for i=1:P_train
        if_noisy = binornd(1,rho1(i));
        train_noisy.y_tilde(i)=if_noisy * 0 +(1 - if_noisy) * 1;
    end
    for i=1:N_train
        if_noisy = binornd(1, rho0(i));
        train_noisy.y_tilde(P_train+i) = if_noisy * 1 + (1 - if_noisy) * 0;
    end

    save dataset_noisy.mat train_noisy
end

