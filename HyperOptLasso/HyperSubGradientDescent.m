classdef HyperSubGradientDescent
    % Online Hyperparameter optimization for the Lasso
    
properties
    
    estimator = 'lasso';
    
    stepsize_w
    stepsize_lambda = 1e-3;
    momentum_lambda = 0; %! 1e-5
    
    tol_w
    tol_g_lambda
    
    max_iter_outer = 200;
    max_iter_inner = 1000;
    min_iter_inner = 1;
    
    normalized_lambda_0 = 0.01;
    
    b_online = 0;
    b_memory = 1;
    method_Z = 'linsolve'; %options: invert, or iterate
    
    param_c = 0; % We multiply the c variable (Jacobian of the validation 
                 % cost with respect to w) by this parameter(factor)
    
end

methods
    
    function [m_W, v_lambda, v_it_count] = solve_gradient(obj, m_X, v_y, lambda_0)
        
        N = length(v_y); assert(size(m_X, 2) == N);
        P = size(m_X, 1);
        m_F = m_X*m_X'/N;
        v_r = m_X*v_y(:)/N;
        
        if isempty(obj.stepsize_w)
            obj.stepsize_w = 1/(2*eigs(m_F,1));
        end
        
        v_it_count  = nan(1, obj.max_iter_outer);
        v_lambda    = nan(1, obj.max_iter_outer);
        v_velocity  = zeros(1, obj.max_iter_outer);
        try
            v_lambda(1) = lambda_0;
        catch ME %#ok<NASGU>
            lambda_max = max(abs(v_r));
            v_lambda(1) = lambda_max*obj.normalized_lambda_0;
        end
        
        [v_w_0, m_z_0, v_it_count(1)] = obj.ista_fg(...
            zeros(P, 1), m_F, v_r, v_lambda(1), zeros(P, 1));
        
        if obj.b_memory %initialize m_W
            m_W = repmat(sparse(v_w_0), [1 N]);
            t_C = repmat(m_z_0, [1 1 N]); %TODO: consider using a cell if we want to exploit sparsity
        else
            error 'this branch not ready yet'
            v_w_j = sparse(v_w_0);
            m_Z_j = sparse(m_z_0);
        end
        
        v_j = mod(0:obj.max_iter_outer-1, N)+1; % so the online is cyclic
        running_average_g = 0;
        ltc = LoopTimeControl(obj.max_iter_outer); %ltc.b_erase = 0;    
        for k_outer = 2:obj.max_iter_outer
            sum_g = 0;
            %ltc_j = LoopTimeControl(N);
            if obj.b_online  %  warning 'branch under test'
                v_indices_k = v_j(k_outer); 
            else          
                v_indices_k = 1:N; 
            end
            v_it_count(k_outer) = 0;
            for j = v_indices_k
                v_x_j = m_X(:, j);
                m_F_j = m_F - v_x_j*v_x_j'/N;
                v_r_j = v_r - v_x_j*v_y(j)/N;
                v_c_previous = t_C(:,:,j); %! for debugging
                if obj.b_memory 
                    [v_w_j, m_Z_j, v_it_count_j, v_w_f_j] = obj.ista_fg(...
                        m_W(:, j), m_F_j, v_r_j, v_lambda(k_outer-1), t_C(:,:,j));
                    
                    m_W(:, j)  = v_w_j;
                    t_C(:,:,j) = m_Z_j;
                else                    
                    [v_w_j, m_Z_j, v_it_count_j, v_w_f_j] = obj.ista_fg(...
                        v_w_j, m_F_j, v_r_j, v_lambda(k_outer-1), ...
                        obj.param_c*m_Z_j); 
                    % the output value of m_z_j is used only if obj.method_Z
                    % is 'iterate', otherwise it is left there in case it 
                    % is needed for debugging
                    
                    % TODO: if obj.method_Z is 'invert', ista_fg should not
                    % waste time in iterating over v_c/v_Z
                    
                    v_w_j = sparse(v_w_j); %TODO: check if this is really needed
                end
                switch obj.method_Z 
                    case 'linsolve'
                        %TODO: these two lines could be encoded into their
                        %own method and/or integrated in ista_fg
                        [m_A_tilde, m_B_tilde] = obj.compute_subderivatives(...
                            v_w_f_j, v_lambda(k_outer-1));
                        m_toInvert = (eye(P)-m_A_tilde*(eye(P)-obj.stepsize_w*m_F_j));
                        if N>P^1.5
                            m_Z_now = m_toInvert\m_B_tilde;
                        else
                            m_Z_now = lsqminnorm(m_toInvert, m_B_tilde);
                        end
                        
                        if k_outer >Inf %! debug
                            norm(m_Z_j-m_Z_now)/norm(m_Z_j)
                            figure(999); clf;
                            stem(m_Z_j); hold on
                            stem(v_c_paper);
                            legend ('iterative', 'inversion')
                            pause
                        end
                    case 'iterate'
                        m_Z_now = m_Z_j;
                    otherwise, error 'Only valid options for method_Z are: invert, iterate'
                end
                
                sum_g = sum_g + (m_Z_now'*v_x_j)*(v_x_j'*v_w_j - v_y(j));
                v_it_count(k_outer) = v_it_count(k_outer) + v_it_count_j;
                %ltc_j.go(j);
            end
            mean_g = sum_g/length(v_indices_k);
            
            %%
            % TODO: optimizer class and subclasses should be used here for
            % universality
            v_velocity(k_outer) = ...
                obj.momentum_lambda*v_velocity(k_outer-1) + (1-obj.momentum_lambda)*mean_g;
            v_lambda(k_outer) = max(0,...
                v_lambda(k_outer-1) - obj.stepsize_lambda*v_velocity(k_outer));
            
            %%
            % TODO: stopping criterion should be internalized in the optimizer
            % object
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
    
    function [v_w, v_z, it_count, v_w_f] = ista_fg(obj, ...
            v_w, m_F, v_r, lambda, v_z)
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
                it_count = k_inner-1;
                break;
            end
            
            v_w_f = v_w - alpha*(v_grad);
            v_w = obj.soft_thresholding(v_w_f, alpha*lambda);
            
            %% This section corresponds to the 
            v_b_argument = v_w_f./(alpha*lambda);
            v_b = (v_b_argument > 1) - (v_b_argument < -1);
            v_z = v_b.*(v_b.*(v_z-alpha*m_F*v_z)-alpha);
            % v_z = (v_b.^2).*(v_z - alpha*m_F*v_z) - alpha*v_b;
            % we do not check convergence of v_z when we use the 'iterate'
            % method
            
            % TODO: split the 'iterate' method for z into two:
            % a) 'interleave' 
            % b) 'iterate', with two options: until convergence, for fixed
            % num of iterates
        end
    end
    
    function [m_A_tilde, m_B_tilde] = compute_subderivatives(...
            obj, v_w_f, lambda)
        alpha = obj.stepsize_w;
        switch obj.estimator
            case 'lasso'
                m_A_tilde = diag(abs(v_w_f)>=(alpha*lambda));
                m_B_tilde = alpha*(...
                    (v_w_f <= (-alpha*lambda)) -...
                    (v_w_f >= ( alpha*lambda)) );
            otherwise, error 'unrecognized option'
        end      
    end

end

methods (Static)
    function w_out_2 = soft_thresholding(v_w_in, rho)
        
%         v_signs = sign(v_w_in);
%         w_out = v_w_in - v_signs*rho;
%         w_out(v_signs~=sign(w_out)) = 0;
        
        v_factors = max(0, 1-rho./abs(v_w_in));
        w_out_2 = v_w_in.*v_factors;
%         norm(w_out-w_out_2)/norm(w_out_2)
%         keyboard
    end
end
end

