classdef HyperGradientLasso
    % Online Hyperparameter optimization for the Lasso
    
properties
    stepsize_w
    stepsize_lambda = 1e-3;
    
    tol_w
    tol_g_lambda
    
    max_iter_outer = 200;
    max_iter_inner = 100;
    min_iter_inner = 1;
    
    normalized_lambda_0 = 0.01;
    
    b_online = 0;
    b_memory = 1;
    
end

methods
    
    function [m_W, v_lambda, v_it_count] = solve_gradient(obj, m_X, v_y, lambda_0)
        
        N = length(v_y); assert(size(m_X, 2) == N);
        P = size(m_X, 1);
        m_F = m_X*m_X'/N;
        v_r = m_X*v_y(:)/N;
        
        if isempty(obj.stepsize_w)
            obj.stepsize_w = 1/trace(m_F);
        end
        
        v_it_count  = nan(1, obj.max_iter_outer);
        v_lambda    = nan(1, obj.max_iter_outer);
        try
            v_lambda(1) = lambda_0;
        catch ME %#ok<NASGU>
            lambda_max = max(abs(v_r));
            v_lambda(1) = lambda_max*obj.normalized_lambda_0;
        end
        
        [v_w_0, v_c_0, v_it_count(1)] = obj.ista_fg(...
            zeros(P, 1), m_F, v_r, v_lambda(1), zeros(P, 1));
        
        if obj.b_memory %initialize m_W
            m_W = repmat(sparse(v_w_0), [1 N]);
            m_C = repmat(sparse(v_c_0), [1 N]);
        else
            v_w_j = sparse(v_w_0);
            v_c_j = sparse(v_c_0);
        end
        
        v_j = mod(0:obj.max_iter_outer-1, N)+1; % so the online is cyclic
        running_average_g = 0;
        ltc = LoopTimeControl(obj.max_iter_outer); ltc.b_erase = 0;        
        for k_outer = 2:obj.max_iter_outer
            sum_g = 0;
            ltc_j = LoopTimeControl(N);
            if obj.b_online,  v_indices_k = v_j(k_outer); 
            else,             v_indices_k = 1:N; 
            end
            v_it_count(k_outer) = 0;
            for j = v_indices_k
                v_x_j = m_X(:, j);
                if obj.b_memory
                    v_w_j = sparse(m_W(:, j));
                    v_c_j = sparse(m_C(:, j));
                end
                m_F_j = m_F - v_x_j*v_x_j'/N;
                v_r_j = v_r - v_x_j*v_y(j)/N;
                
                [v_w_j, v_c_j, v_it_count_j] = obj.ista_fg(...
                    v_w_j, m_F_j, v_r_j, v_lambda(k_outer-1), v_c_j);
                
                sum_g = sum_g - (v_y(j) - v_w_j'*v_x_j)*(v_x_j'*v_c_j);
                if obj.b_memory
                    m_W(:, j) = v_w_j;
                    m_C(:, j) = v_c_j;
                else
                    v_w_j = sparse(v_w_j);
                    v_c_j = sparse(v_c_j);
                end
                v_it_count(k_outer) = v_it_count(k_outer) + v_it_count_j;
                ltc_j.go(j);
            end
            mean_g = sum_g/length(v_indices_k);
            v_lambda(k_outer) = max(0,...
                v_lambda(k_outer-1) - obj.stepsize_lambda*mean_g);
            
            if ~obj.b_online
                b_stopping_criterion = all(abs(mean_g)<obj.tol_g_lambda);
            else
                running_average_g = 0.99*running_average_g + 0.01*mean_g;
                if abs(running_average_g) < obj.tol_g_lambda  
                              combo = combo+1; 
                else,         combo = 0;
                end
                b_stopping_criterion = (combo > N);
            end
            if b_stopping_criterion
                disp 'Reached Stopping criterion.'
                break
            end
            ltc.go(k_outer);
        end
        if k_outer == obj.max_iter_outer
            disp 'Maximum number of iterations exceeded.'
        end
        if not(obj.b_memory)
            m_W = v_w_j;
        end
    end
    
    function [v_w, v_c, it_count] = ista_fg(obj, ...
            v_w, m_F, v_r, lambda, v_c)
        % Iterative soft thresholding algorithm (ISTA)
        % and Franceschi's forward gradient 
        
        alpha = obj.stepsize_w;
        it_count = obj.max_iter_inner;
        for k_inner = 1:obj.max_iter_inner
            v_grad = m_F*v_w - v_r; % gradient
            
            %check stopping criterion
            v_grad_violations = (v_grad.*sign(v_w) + lambda).*(v_w~=0);
            if       norm(v_grad_violations) < obj.tol_w   ...
                  && all(abs(v_grad).*(v_w==0) < lambda)   ...
                  && k_inner > obj.min_iter_inner
                it_count = k_inner;
                break;
            end
            
            v_w_f = v_w - alpha*(v_grad);
            v_w = obj.soft_thresholding(v_w_f, alpha*lambda);
            v_z_argument = v_w_f/(alpha*lambda);
            v_z = (v_z_argument > 1) - (v_z_argument < -1);
            v_c = v_z.^2.*(v_c - alpha*m_F*v_c) - alpha*v_z;
            % we do not check convergence of v_c, not sure if  important
        end
    end
end

methods (Static)
    function w_out = soft_thresholding(w_in, rho)
        v_factors = max(0, 1-rho./abs(w_in));
        w_out = w_in.*v_factors;
    end
end
end

