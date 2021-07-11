function [ beta ] = Empirical_KMM(X_refined, X_original)
    % Reference: 
    % Huang et.al.: Correcting sample selection bias by unlabeled data. In NeurIPS, 2007.

    beta = zeros(size(X_refined,2), 1);
    K = zeros(size(X_refined,2), size(X_refined,2));
    kappa = zeros(size(X_refined, 2), 1);


    epsilon = (sqrt(size(X_refined, 2))-1) / sqrt(size(X_refined, 2));
    B = 1000;
    A = [1 / size(X_refined, 2) * ones(1, size(X_refined, 2)); -1 / size(X_refined, 2) * ones(1, size(X_refined, 2))];
    b = [1 + epsilon; -(1-epsilon)];
    lb = zeros(size(beta));
    ub = B * ones(size(beta));


    sigma = -1;
    K = exp(sigma * squareform(pdist(X_refined')));
    kappa = sum(exp(sigma * pdist2(X_refined', X_original')), 2) *  (size(X_refined, 2) / size(X_original, 2));
    
    options = optimset('Display', 'off');
    [beta, ~, exitflag, output] = quadprog(K,(-1*kappa), A, b, [], [], lb, ub, [], options);
end

