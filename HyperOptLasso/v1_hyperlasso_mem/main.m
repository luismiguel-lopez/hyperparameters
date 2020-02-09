% Sample demonstration for Hyperparameter optimization for LASSO
%% Setup the data
rng(3);
n_train = 200;
n_test = 2000;
p = 100;

A_train = randn(n_train, p); % sensing matrix
A_test = randn(n_test, p);
x = double(randn(p, 1) > 1); % sparse coefficients
epsilon = randn(n_train, 1) * 0.3; % noise
epsilon_test = randn(n_test, 1) * 0.3;
y_train = A_train * x + epsilon;
y_test  = A_test * x + epsilon_test;


%% Run the LASSO regression
hl_mem = HyperLasso2; %memory
hl_nomem = HyperLasso2; 
hl_nomem.b_mem = 0; %memoryless
hl_mem_shuffle = HyperLasso2; 
hl_mem_shuffle.b_shuffle = 1; %memory, shuffle
hl_nomem_shuffle = HyperLasso2; 
hl_nomem_shuffle.b_shuffle = 1;
hl_nomem_shuffle.b_mem = 0; %memoryless, shuffle
hl_mem_impl24 = HyperLasso2; 
hl_mem_impl24.b_implement_24 = 1;

[beta_nomem, lambda_nomem, lambda_24_nomem] = hl_nomem.solve(A_train, y_train);

[beta_mem, lambda_mem, lambda_24_mem] = hl_mem.solve(A_train, y_train);

[beta_nomem_shuffle, lambda_nomem_shuffle, lambda_24_nomem_shuffle] = ...
    hl_nomem_shuffle.solve(A_train, y_train);

[beta_mem_shuffle, lambda_mem_shuffle, lambda_24_mem_shuffle] = ...
    hl_mem_shuffle.solve(A_train, y_train);

[beta_mem_impl24, lambda_mem_impl24, lambda_24_mem_impl24] = ...
    hl_mem_impl24.solve(A_train, y_train);


%%
figure(1); clf
plot([lambda_nomem lambda_24_nomem])
hold on
plot([lambda_mem lambda_24_mem])
plot([lambda_nomem_shuffle lambda_24_nomem_shuffle])
plot([lambda_mem_shuffle lambda_24_mem_shuffle])
plot([lambda_mem_impl24 lambda_24_mem_impl24], '--');

my_cap = max(lambda_mem(ceil(9*length(lambda_mem)/10)));
ylim(my_cap*[1/2,2])
legend('Stoch. memoryless', 'Eq. 24, memoryless', ...
    'Stoch. memory',  'Eq. 24, memory', ...
    'Stoch. ml+shuffle', 'Eq. 24, ml+shuffle', ...
    'Stoch. memory+shuffle',  'Eq. 24, memory+shuffle', ...
    'Trust region', 'Eq.24, trust region')

%TODO:try a non-stochastic proximal algorithm that minimizes the linear
%approximation of the cost function minimized by (24) plus a proximal term,
%that may be quadratic (gradient descent) or not (mirror descent).

%%
figure(2); clf
[beta, fitinfo] = lasso(A_train, y_train, 'lambda', ...
    linspace(1, 10, 100)/n_train);
test_mse = zeros(size(beta, 2), 1);
for k = 1:size(beta, 2)
    yhat_test = A_test * beta(:, k);
    test_mse(k) = mean((y_test-yhat_test).^2);
end
plot(fitinfo.Lambda*n_train, test_mse);

% TODO: compare with AloCV
% TODO: compare with exact Leave-One-Out