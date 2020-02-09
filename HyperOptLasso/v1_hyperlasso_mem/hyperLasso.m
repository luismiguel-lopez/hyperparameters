function [v_w, lambda] = hyperLasso(m_X, v_y, alpha, beta_in)
% Solves y = X*w using Lasso with hyperparameter optimization
max_iter = 20000;
max_iter_inner = 10;
assert(iscolumn(v_y))
N = length(v_y);
assert(size(m_X, 1)==N);
P = size(m_X, 2);
m_Phi = m_X'*m_X;
v_r   = m_X'*v_y;
v_w = zeros(P, 1);
alpha = 1/trace(m_Phi);
lambda_max = max(abs(v_r));
lambda = zeros(max_iter, 1);
lambda(1) = lambda_max/100;
beta = beta_in./sqrt(1:max_iter)';
for k = 2:max_iter
    j = mod(k-1, N)+1;
    v_x_j = m_X(j,:)';
    m_Phi_j = m_Phi - v_x_j * v_x_j';
    v_r_j   = v_r   - v_y(j)* v_x_j;
    % obtain w(k) via ISTA
    for k_inner = 1:max_iter_inner
        w_f = v_w - alpha*(m_Phi_j*v_w - v_r_j);
        v_w = soft_thresholding(w_f, alpha*lambda(k-1));
        % TODO: check convergence and exit loop
    end
    v_s_j = max(-1, min(1, 1/(alpha*lambda(k-1))*w_f));
    d = alpha*v_x_j'*v_s_j*(v_y(j)-v_x_j'*(v_w - alpha*(...
        m_Phi_j*v_w - v_r_j + lambda(k-1)*v_s_j)));
    if d*beta(k) < -0.9
        warning 'negative d times beta too large: rate of increase capped at 10'
        % beta(k) = beta./abs(2*beta(k)*d); % this is the old line
        lambda(k) = lambda(k-1)*10;
    else
        lambda(k) = lambda(k-1)/(1+beta(k)*d);
    end
end

end

function w_out = soft_thresholding(w_in, rho)
v_factors = max(0, 1-rho./abs(w_in));
w_out = w_in.*v_factors;
end