classdef CrossValidatedLasso
properties
    factor_initial_lambda = 0.1
    factor_stepsize_w     = 1;
    max_iter_outer = 10000;
    stepsize_lambda = .003;
    tol_w = 1e-4;
    minifold_size = 10;
    b_adaptive_lasso = 0;
    b_efficient = 0;
    
    crazy_param = 1;
    decay_factor = 1;
    
    max_iter_inner = 100;
    min_iter_inner = 1;

end

methods
    function [m_W, m_lambda, v_it_count, v_timing, mb_valFold, ...
            v_valLoss, v_maxDev, v_w_final] = ...
            interleave(obj, m_X, v_y)
        [P,    N]    = size(m_X); assert(length(v_y) == N);
        m_F = m_X*m_X'/N;    % Phi matrix
        v_r = m_X*v_y(:)/N;  % r   vector
        stepsize_w = 1/(2*obj.factor_stepsize_w*eigs(m_F,1));

        [v_it_count, v_timing, v_valLoss, v_maxDev] = deal(nan(1, obj.max_iter_outer));
        v_allValErrors = nan(N, 1);
        m_lambda = nan(P, obj.max_iter_outer); % sequence of lambdas
        lambda_max = max(abs(v_r));
        m_lambda(:,1) = lambda_max*obj.factor_initial_lambda;
        
        m_W = spalloc(P, obj.max_iter_outer, P*obj.max_iter_outer);
        v_support = false(P,1); %keep to add lines later for efficiency
        t_reference = tic;
        n_folds = ceil(N/obj.minifold_size);
        mb_valFold = logical(kron(eye(n_folds), ones(1, obj.minifold_size)));
        mb_valFold(:,N+1:end) = [];

        for k = 1:obj.max_iter_outer
            v_hypergradient = zeros(P,1);
            f = mod(k, n_folds)+1;
            vb_val = mb_valFold(f,:);
            s = nnz(vb_val); % mini-fold size
            if k < n_folds
                v_w_initial = m_W(:,k);
            else
                v_w_initial = m_W(:,k-n_folds+1);
            end
            [v_w_star, v_w_f, v_it_count(k)] = obj.ista(v_w_initial, m_F, v_r, ...
                m_lambda(:,k), stepsize_w);
            v_support = (v_w_star~=0);
            v_b = -sign(v_w_f);
            v_b(not(v_support)) = 0;
            if obj.b_adaptive_lasso
                m_V = diag(v_b(v_support));
                v_Vlambda = m_V*m_lambda(v_support,k);
            else
                m_V = v_b(v_support);
                assert(var(m_lambda(:,k))<1e-8);
                v_Vlambda = m_V.*m_lambda(1,k);
            end
            m_Rs = N*m_F(v_support, v_support)/obj.crazy_param;
            m_T = m_X(v_support,mb_valFold(f,:));
            m_Rs_inv_T = m_Rs\m_T;
            v_alpha_star = v_w_star(v_support);
            
            v_hv_efficient = 1/s*m_V'*m_Rs_inv_T * ...
                ( ((eye(s)-m_T'*m_Rs_inv_T)^2)\(m_T'*v_alpha_star ...
                - vec(v_y(vb_val)) - s*(m_Rs_inv_T'*v_Vlambda)) );
            v_hypergradient_values = v_hv_efficient;
            if obj.b_adaptive_lasso
                v_hypergradient(v_support) = v_hypergradient_values;
                v_hypergradient(not(v_support)) = obj.decay_factor * ...
                    m_lambda(not(v_support),k);
            else
                v_hypergradient = v_hypergradient_values;
                assert(isscalar(v_hypergradient));
            end
            m_lambda(:,k+1) = max(0, m_lambda(:,k) - obj.stepsize_lambda*v_hypergradient);
            m_W(:,k+1) = v_w_star;
            v_timing(k) = toc(t_reference);
%             v_valErrors = (m_T'*v_w(v_support)-vec(v_y(vb_val)));
%             v_allValErrors(vb_val) = v_valErrors;
%           TODO:: compute alpha_j_star
%            v_valLoss(k) = mean(v_allValErrors.^2, 'omitnan');
        end
        v_w_final = obj.ista(m_W(:,end), m_F, v_r, m_lambda(:,k+1), stepsize_w);
    end

    function [m_W, m_lambda, v_it_count, v_timing, mb_valFold, ...
            v_valLoss, v_maxDev, v_w_final] = ...
            solve(obj, m_X, v_y, v_lambda_in)
        [P,    N]    = size(m_X); assert(length(v_y) == N);
        m_F = m_X*m_X'/N;    % Phi matrix
        v_r = m_X*v_y(:)/N;  % r   vector
        stepsize_w = 1/(2*obj.factor_stepsize_w*eigs(m_F,1));

        [v_it_count, v_timing, v_valLoss, v_maxDev] = deal(nan(1, obj.max_iter_outer));
        v_allValErrors = nan(N, 1);
        m_lambda = nan(P, obj.max_iter_outer); % sequence of lambdas
        if exist('v_lambda_in', 'var')
            m_lambda(:,1) = v_lambda_in;
        else
            lambda_max = max(abs(v_r));
            m_lambda(:,1) = lambda_max*obj.factor_initial_lambda;
        end
        
        m_W = spalloc(P, obj.max_iter_outer, P*obj.max_iter_outer);
        v_support = zeros(P,1); %keep to add lines later for efficiency
        t_reference = tic;
        n_folds = ceil(N/obj.minifold_size);
        mb_valFold = logical(kron(eye(n_folds), ones(1, obj.minifold_size)));
        mb_valFold(:,N+1:end) = [];
        t_corr_mat = zeros(P, P, n_folds);
        m_xcorr_vec = zeros(P, n_folds);
        for f = 1:n_folds
            N_f = nnz(not(mb_valFold(f,:)));
            t_corr_mat(:,:,f) = (N*m_F - m_X(:,mb_valFold(f,:))*....
                m_X(:, mb_valFold(f,:))')/N_f;
            m_xcorr_vec(:,f)  = (N*v_r - m_X(:,mb_valFold(f,:))*...
                vec(v_y(mb_valFold(f,:))))/N_f;
        end
        for k = 1:obj.max_iter_outer
            f = mod(k, n_folds)+1;
            vb_val = mb_valFold(f,:);
            m_F_f = t_corr_mat(:,:,f);
            v_r_f = m_xcorr_vec(:,f) ;
            if k < n_folds
                v_w_initial = m_W(:,k);
            else
                v_w_initial = m_W(:,k-n_folds+1);
            end
            [v_w, v_w_f, v_it_count(k)] = obj.ista(v_w_initial, m_F_f, v_r_f, ...
                m_lambda(:,k), stepsize_w);
                        
            v_support_now = (abs(v_w_f)  >=  stepsize_w*m_lambda(:,k));
            if not(all(v_support_now==(v_w~=0)))
                disp(k)
                warning('support detection conflict.')
            end
            v_b = -sign(v_w_f);
            v_b(not(v_support_now)) = 0;
            v_hypergradient = zeros(P,1);
            %if any(v_support - v_support_now)
            v_support = v_support_now;
            if obj.b_adaptive_lasso
                m_V = diag(v_b(v_support_now));
                v_Vlambda = m_V*m_lambda(v_support_now,k);
            else
                m_V = v_b(v_support_now);
                assert(var(m_lambda(:,k))<1e-8);
                v_Vlambda = m_V.*m_lambda(1,k);
            end
            m_Rs = m_F(v_support, v_support); %!!! wrong, but works better!?
            m_Rs = N*m_F(v_support, v_support)/obj.crazy_param;
            m_T = m_X(v_support,mb_valFold(f,:));
            m_Rs_inv_T = m_Rs\m_T;
            s = nnz(vb_val); % mini-fold size
            v_valErrors = (m_T'*v_w(v_support)-vec(v_y(vb_val)));
            v_hypergradient_values = 1/s*m_V'*m_Rs_inv_T * ...
                ( (eye(s)-m_T'*m_Rs_inv_T)\v_valErrors );
            if obj.b_efficient
                %experimental, efficient hypergradient computation
                if k==1
                    v_w_star = v_w;
                end
                v_w_star = obj.ista(v_w_star, m_F, v_r, m_lambda(:,k), stepsize_w);
                v_alpha_star = v_w_star(v_support);
                %TODO: compute alpha_star using the trick
                v_hv_efficient = 1/s*m_V'*m_Rs_inv_T * ...
                    ( ((eye(s)-m_T'*m_Rs_inv_T)^2)\(m_T'*v_alpha_star ...
                    - vec(v_y(vb_val)) - s*(m_Rs_inv_T'*v_Vlambda)) );
                v_maxDev(k) = max(abs(v_hv_efficient - v_hypergradient_values));
                
                v_hypergradient_values = v_hv_efficient;
            end
            %trials2
            %alt_hypergrad;
            if obj.b_adaptive_lasso
                v_hypergradient(v_support) = v_hypergradient_values;
                v_hypergradient(not(v_support)) = obj.decay_factor * ...
                    m_lambda(not(v_support),k);
            else
                v_hypergradient = v_hypergradient_values;
                assert(isscalar(v_hypergradient));
            end
            m_lambda(:,k+1) = max(0, m_lambda(:,k) - obj.stepsize_lambda*v_hypergradient);
            m_W(:,k+1) = v_w;
            v_timing(k) = toc(t_reference);
            v_allValErrors(vb_val) = v_valErrors;
            v_valLoss(k) = mean(v_allValErrors.^2, 'omitnan');
        end
        v_w_final = obj.ista(m_W(:,end), m_F, v_r, m_lambda(:,k+1), stepsize_w);

    end
    
    function [m_W, m_lambda] = solve_reweighted(obj, m_X, v_y, lambda)
        [P,    N]    = size(m_X); assert(length(v_y) == N);
        m_F = m_X*m_X'/N;    % Phi matrix
        v_r = m_X*v_y(:)/N;  % r   vector
        stepsize_w = 1/(2*obj.factor_stepsize_w*eigs(m_F,1));

        [v_it_count, v_timing, v_valLoss, v_maxDev] = ...
            deal(nan(1, obj.max_iter_outer));
        %v_allValErrors = nan(N, 1);
        m_lambda = nan(P, obj.max_iter_outer); % sequence of lambdas
        %lambda_max = max(abs(v_r));
        m_lambda(:,1) = lambda; %_max*obj.factor_initial_lambda;
        my_epsilon = 1e-2; %obj.tol_w;
        m_W = spalloc(P, obj.max_iter_outer, P*obj.max_iter_outer);
        for k = 1:100
            v_w = obj.ista(m_W(:,k), m_F, v_r, m_lambda(:,k), stepsize_w);
            m_lambda(:,k+1) = m_lambda(:,1)./(my_epsilon+abs(v_w));
            m_W(:,k+1) = v_w;
        end
    end
    
    function [v_w_out, v_w_f, it_count] = ista(obj, ...
            v_w_in, m_F, v_r, v_lambda, alpha)
        
        v_w = v_w_in;
        it_count = obj.max_iter_inner;
        for k_inner = 1:obj.max_iter_inner
            v_grad = m_F*v_w - v_r; % gradient
            
            %check stopping criterion
            v_grad_violations = (v_grad.*sign(v_w) + v_lambda).*(v_w~=0);
            if       max(abs(v_grad_violations)) < obj.tol_w   ...
                    && all(abs(v_grad).*(v_w==0) <= v_lambda)   ... !% '<' --> '<='
                    && k_inner > obj.min_iter_inner
                it_count = k_inner-1;
                break;
            end
            
            % forward (grad step)
            v_w_f = v_w - alpha*(v_grad);
            % backward (prox op)
            v_factors = max(0, 1-alpha*v_lambda./abs(v_w_f));
            v_w = v_w_f.*v_factors;
        end
        v_w_out = v_w;
    end
end

end