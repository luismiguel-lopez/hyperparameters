function [m_W, m_lambda, v_it_count, v_timing, ...
     v_trainSequence] = ...
    crossvalidated_lasso(m_X, v_y, lambda_0)
% Crossvalidated lasso using Hypergradient.
factor_initial_lambda = 0;
max_iter_outer = 200;
stepsize_lambda = 5e-3;
tol_w = 1e-3;
b_fast_method = 1;
n_pVal = 1000;
b_adaptive_lasso = 0;

[P,    N]    = size(m_X); assert(length(v_y) == N);
m_F = m_X*m_X'/N;    % Phi matrix
v_r = m_X*v_y(:)/N;  % r   vector
stepsize_w = 1/(20*eigs(m_F,1));

[v_it_count, v_timing, v_trainSequence] = ....
    deal(nan(1, max_iter_outer)); % iteration counts
if b_adaptive_lasso
    m_lambda = nan(P, max_iter_outer); % sequence of lambdas
else
    m_lambda = nan(1, max_iter_outer);
end

try
    m_lambda(:,1) = lambda_0;
catch ME %#ok<NASGU>
    lambda_max = max(abs(v_r));
    m_lambda(:,1) = lambda_max*factor_initial_lambda;
end

m_W = spalloc(P, max_iter_outer, P*max_iter_outer);
v_support = zeros(P,1);
t_reference = tic;
for f = 1:n_folds
    %TODO: prepare m_F and v_r cells.
end
for k = 1:max_iter_outer
    f = mod(k, n_folds);
    m_F = c_m_F{f};
    v_r = c_v_r{f};
    m_Xval = 
    [v_w, v_w_f, it_inner] = ista(m_W(:,k), m_F, v_r, ...
        m_lambda(:,k), stepsize_w, tol_w);
    v_support_now = (abs(v_w_f)  >=  stepsize_w*m_lambda(:,k) - 0*tol_w);
    %v_support_now   = v_w~=0; %!
    v_b = -sign(v_w_f);
    v_b(not(v_support_now)) = 0; %TODO?
    v_hypergradient = zeros(P,1);
    %if any(v_support - v_support_now)
    v_support = v_support_now;
    v_v = v_b(v_support_now);
    m_Psi = m_F(v_support, v_support);
    %end
    m_T = m_X(v_support,mb_val_inFold(:,f));
    m_Psi_inv_T = m_Psi\m_T;
    v_errors = v_w'*m_X(:,mb_val_inFold(:,f))-v_y;
    v_denominators = 1-sum(m_T.*m_Psi_inv_T);
    if b_adaptive_lasso
        m_numerators = diag(v_v)*m_Psi_inv_T;
        v_hypergradient(v_support) = ...
            (m_numerators./v_denominators)*v_errors';
    else
        v_numerators = v_v'*m_Psi_inv_T;
        v_hypergradient = ... % a scalar in this case
            (v_numerators./v_denominators)*v_errors';
    end    
    m_lambda(:,k+1) = max(0, m_lambda(:,k) - stepsize_lambda*v_hypergradient);
    m_W(:,k+1) = v_w;
    v_timing(k) = toc(t_reference);
    v_it_count(k) = it_inner;
%     v_trainSequence(k) = mean((v_w'*m_X - v_y).^2); %REMOVE?
end

function [v_w_out, v_w_f, it_count] = ista(v_w_in, m_F, v_r, v_lambda, alpha, tol_w)
max_iter_inner = 100;
min_iter_inner = 1;

v_w = v_w_in;
it_count = max_iter_inner;
for k_inner = 1:max_iter_inner
    v_grad = m_F*v_w - v_r; % gradient
    
    % forward
    v_w_f = v_w - alpha*(v_grad); 
    
    %check stopping criterion
    v_grad_violations = (v_grad.*sign(v_w) + v_lambda).*(v_w~=0);
    if       max(abs(v_grad_violations)) < tol_w   ...
            && all(abs(v_grad).*(v_w==0) <= v_lambda)   ... !% '<' --> '<='
            && k_inner > min_iter_inner
        it_count = k_inner-1;
        break;
    end
    
    % backward (prox)
    v_factors = max(0, 1-alpha*v_lambda./abs(v_w_f));
    v_w = v_w_f.*v_factors; 
end
v_w_out = v_w;
