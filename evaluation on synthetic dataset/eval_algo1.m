% Reference: 
% Cheng et.al.: Learning with bounded instance and label-dependent label noise. In ICML, 2020.

clear; clc

% please download and compile the minFunc package 
% https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html
addpath ../minFunc_2012
addpath ../minFunc_2012/minFunc
addpath ../minFunc_2012/minFunc/compiled/

% settings
rho_1max = 0.49
rho_0max = 0.49
n_act = 3 % the number of actively labeled examples

W = [mvnrnd([0,0,0], diag([1,1,1]), 2)];   % W: parameters of noise rate functions
W1 = W(1,:)';
W2 = W(2,:)';

% dataset construction
% construct_synthetic_dataset(rho_1max, rho_0max, W1, W2);
load('dataset_clean.mat')
load('dataset_noisy.mat')


% initialize the parameters of classifiers
theta_clean = rand(size(train_noisy.X, 1), 1) * 0.001;
theta_noisy = rand(size(train_noisy.X, 1), 1) * 0.001;
theta_auto = rand(size(train_noisy.X, 1), 1) * 0.001;
theta_auto_act = rand(size(train_noisy.X, 1), 1) * 0.001;
theta_auto_rd = rand(size(train_noisy.X, 1), 1) * 0.001;
theta_noisy_act = rand(size(train_noisy.X, 1), 1) * 0.001;
theta_noisy_rd = rand(size(train_noisy.X, 1), 1) * 0.001;

% learning of $\tilde_{\eta}$
options = struct('MaxIter', 20, 'Display', 'off');
theta_noisy = minFunc(@logistic_regression, theta_noisy, options, train_noisy.X, train_noisy.y_tilde);

% collect distilled examples
eta_tilde_hat_X = sigmoid(theta_noisy' * train_noisy.X);
X_auto = [train_noisy.X(:, find(eta_tilde_hat_X>(1+rho_0max)/2)), train_noisy.X(:, find(eta_tilde_hat_X<(1-rho_1max)/2))];
y_auto = [ones(1, length(find(eta_tilde_hat_X>(1+rho_0max)/2))), zeros(1, length(find(eta_tilde_hat_X<(1-rho_1max)/2)))];

idx_uncollected = find(eta_tilde_hat_X<=(1+rho_0max)/2 & eta_tilde_hat_X>=(1-rho_1max)/2);
X_uncollected = train_noisy.X(:, idx_uncollected);


% actively label uncollected examples
if (n_act <= length(idx_uncollected))

    idx_random = randperm(numel(idx_uncollected));
    X_act = train_clean.X(:, idx_uncollected(idx_random(1:n_act)));
    y_act = train_clean.y(:, idx_uncollected(idx_random(1:n_act)));

 
    train_noisy_act.X = train_noisy.X;
    train_noisy_act.y = train_noisy.y_tilde;
    train_noisy_act.y(idx_random(1:n_act)) = train_clean.y(:, idx_random(1:n_act));
else
    % if the number of uncollected examples is smaller than n_act, then actively label all uncollected examples
    X_act = train_clean.X(:, idx_uncollected(1:end));
    y_act = train_clean.y(:, idx_uncollected(1:end));

    train_noisy_act.X = train_noisy.X;
    train_noisy_act.y = train_noisy.y_tilde;
    train_noisy_act.y(idx_uncollected(1:end)) = train_clean.y(:, idx_uncollected(1:end));
end

train_auto_act.X = [X_auto, X_act];
train_auto_act.y = [y_auto, y_act];
importances = Empirical_KMM(train_auto_act.X, train_noisy.X);

% training
theta_clean = minFunc(@logistic_regression, theta_clean, options, train_clean.X, train_clean.y);
theta_auto = minFunc(@logistic_regression, theta_auto, options, X_auto, y_auto);
theta_auto_act = minFunc(@weighted_logistic_regression, theta_auto_act, options, train_auto_act.X, train_auto_act.y, importances');
theta_noisy_act = minFunc(@logistic_regression, theta_noisy_act, options, train_noisy_act.X, train_noisy_act.y);

% test
acc_clean = binary_classifier_accuracy(theta_clean,test.X,test.y);
fprintf('clean test accuracy: %2.1f%%\n', 100 * acc_clean);

acc_noisy = binary_classifier_accuracy(theta_noisy,test.X,test.y);
fprintf('noisy test accuracy: %2.1f%%\n', 100 * acc_noisy);

acc_auto = binary_classifier_accuracy(theta_auto,test.X,test.y);
fprintf('auto test accuracy: %2.1f%%\n', 100 * acc_auto);

acc_noisy_act = binary_classifier_accuracy(theta_noisy_act,test.X,test.y);
fprintf('noisy_act test accuracy: %2.1f%%\n', 100 * acc_noisy_act);

acc_auto_act = binary_classifier_accuracy(theta_auto_act,test.X,test.y);
fprintf('auto_act test accuracy: %2.1f%%\n', 100 * acc_auto_act);



