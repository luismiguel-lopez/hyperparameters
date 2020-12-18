function [m_W, m_lambda, v_it_count, v_timing, ....
    v_valCurve, v_lambdasCurve, v_valSequence, v_trainSequence] = ...
    holdout_validated_lasso(m_X, v_y, m_X_val, v_y_val, lambda_0)
% Holdout validated lasso using Hypergradient.
factor_initial_lambda = 0;
max_iter_outer = 200;
stepsize_lambda = 5e-3;
tol_w = 1e-3;
b_fast_method = 0;
n_pVal = 1000;
b_adaptive_lasso = 1;

[P,    N]    = size(m_X); assert(length(v_y) == N);
[Pval, Nval] = size(m_X_val); assert(length(v_y_val) == Nval); %#ok<ASGLU>
m_F = m_X*m_X'/N;    % Phi matrix
v_r = m_X*v_y(:)/N;  % r   vector
m_F_val = m_X_val*m_X_val'/Nval;    % Phi matrix
v_r_val = m_X_val*v_y_val(:)/Nval;  % r   vector
stepsize_w = 1/(20*eigs(m_F,1));

[v_it_count, v_timing, v_valSequence, v_trainSequence] = ....
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
for k = 1:max_iter_outer
    [v_w, v_w_f, it_inner] = ista(m_W(:,k), m_F, v_r, m_lambda(:,k), stepsize_w, tol_w);
    v_support_now = (abs(v_w_f)  >=  stepsize_w*m_lambda(:,k) - 0*tol_w);   
    %v_support_now   = v_w~=0; %!
    v_b = -sign(v_w_f);
    v_b(not(v_support_now)) = 0; %TODO?
    if b_fast_method        
        %if any(v_support - v_support_now)
            v_support = v_support_now;
            v_v = v_b(v_support_now);
            m_Psi = m_F(v_support, v_support);
        %end
        grad_val = m_F_val*v_w - v_r_val;
        m_Z = m_Psi\grad_val(v_support);
    else
        m_A = diag(v_support);
        m_toInvert = eye(P) - m_A*(eye(P)-stepsize_w*m_F);
        grad_val = m_F_val*v_w - v_r_val;
        v_v = v_b;
        m_Z = m_toInvert\grad_val;
    end
    if b_adaptive_lasso
        v_hypergradient = diag(v_v)*m_Z;
    else
        v_hypergradient = v_v'*m_Z;
    end
        
    m_lambda(:,k+1) = max(0, m_lambda(:,k) - stepsize_lambda*v_hypergradient);
    m_W(:,k+1) = v_w;
    v_timing(k) = toc(t_reference);
    v_it_count(k) = it_inner;
    v_valSequence(k) = mean((v_w'*m_X_val - v_y_val).^2);
    v_trainSequence(k) = mean((v_w'*m_X - v_y).^2);
end

v_lambdasCurve = linspace(0, lambda_max, n_pVal);
v_valCurve  = nan(1, n_pVal);

for p = 1:n_pVal
    lambda = v_lambdasCurve(p);
    v_w = ista(v_w, m_F, v_r, lambda, stepsize_w, tol_w);
    v_valCurve(p) = mean((v_w'*m_X_val - v_y_val).^2);
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
