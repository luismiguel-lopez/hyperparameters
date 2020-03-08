classdef HyperSubGradientDescent
    % Online Hyperparameter optimization for several estimators.
    % In the current version, it works for Lasso and it is being
    % developed to work with Group Lasso as well.
    
properties
    
    s_estimator = 'lasso';
    
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
    s_method_Z = 'linsolve'; %options: 
    % linsolve, lsqminnorm, interleave, interleave-fast, iterate
    
    v_group_structure = [];
    
    param_c = 0; % We multiply the c variable (Jacobian of the validation 
                 % cost with respect to w) by this parameter(factor)
                 % (this is for the memoryless)
                
    
end

methods
    
    function [m_W, v_lambda, v_it_count] = solve_gradient(obj, ...
            m_X, v_y, lambda_0)
        % Hyperparameter optimization [lopez2020online]
        % solves the outer problem via gradient descent (with momentum)
              
        [P, N] = size(m_X); assert(length(v_y) == N);
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
        
        if contains(obj.s_estimator, 'group')
            validateGroupStructure(obj.v_group_structure);
        end
        
        [v_w_0, m_Z_0, v_it_count(1)] = obj.ista_fg(...
            zeros(P, 1), m_F, v_r, v_lambda(1), zeros(P, 1));
        
        if obj.b_memory %initialize m_W
            m_W = repmat(sparse(v_w_0), [1 N]);
            t_C = repmat(m_Z_0, [1 1 N]); %TODO: consider using a cell 
            % if m_z_0 has sparsity and we want to use it
        else
            error 'memoryless branch not ready yet'
            v_w_j = sparse(v_w_0);
            m_Z_j = sparse(m_Z_0);
        end
        
        v_j_cyclic = mod(0:obj.max_iter_outer-1, N)+1;
        running_average_g = 0;
        combo = 0; %for online alg's stopping criterion
        ltc = LoopTimeControl(obj.max_iter_outer); %ltc.b_erase = 0;    
        for k_outer = 2:obj.max_iter_outer
            sum_g = 0;
            %ltc_j = LoopTimeControl(N);
            if obj.b_online
                v_indices_k = v_j_cyclic(k_outer); 
            else          
                v_indices_k = 1:N; 
            end
            v_it_count(k_outer) = 0;
            for j = v_indices_k
                v_x_j = m_X(:, j);
                m_F_j = m_F - v_x_j*v_x_j'/N;
                v_r_j = v_r - v_x_j*v_y(j)/N;
                % v_c_previous = t_C(:,:,j); %! for debugging
                if obj.b_memory 
                    [v_w_j, m_Z_j, v_it_count_j] = obj.ista_fg(...
                        m_W(:, j), m_F_j, v_r_j, v_lambda(k_outer-1), t_C(:,:,j));
                    
                    m_W(:, j)  = v_w_j;
                    t_C(:,:,j) = m_Z_j;
                else                    
                    [v_w_j, m_Z_j, v_it_count_j] = obj.ista_fg(...
                        v_w_j, m_F_j, v_r_j, v_lambda(k_outer-1), ...
                        obj.param_c*m_Z_j); 
                                        
                    v_w_j = sparse(v_w_j); %TODO: check if this is really needed
                end
                
                sum_g = sum_g + (m_Z_j'*v_x_j)*(v_x_j'*v_w_j - v_y(j));
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
    
    function [v_w, m_Z_tilde_out, it_count, v_w_f] = ista_fg(obj, ...
            v_w, m_F, v_r, v_lambda, m_Z_tilde_in)
        % Iterative soft thresholding algorithm (ISTA)
        % and Franceschi's forward gradient 
        
        [P, P2] = size(m_F); assert(P==P2 && length(v_r)==P && length(v_w)==P);
        [Pz, D] = size(m_Z_tilde_in); assert(P==Pz && length(v_lambda)==D);
        
        alpha = obj.stepsize_w;
        it_count = obj.max_iter_inner;
        assert(any(strcmp(obj.s_method_Z, { 'iterate', 'linsolve', ...
            'lsqminnorm', 'interleave', 'interleave-fast', 'none' } )), ...
            'unrecognized option for s_method_Z');
        m_Z_tilde = m_Z_tilde_in;
        for k_inner = 1:obj.max_iter_inner
            v_grad = m_F*v_w - v_r; % gradient
            
            %check stopping criterion
            v_grad_violations = (v_grad.*sign(v_w) + v_lambda).*(v_w~=0);
            if       norm(v_grad_violations) < obj.tol_w   ...
                  && all(abs(v_grad).*(v_w==0) < v_lambda)   ...
                  && k_inner > obj.min_iter_inner
                it_count = k_inner-1;
                break;
            end
            
            v_w_f = v_w - alpha*(v_grad); % forward
            v_w = obj.prox_Omega(v_w_f, alpha*v_lambda); %backward (prox)
            
            % Methods that interleave computation of m_Z_tilde with the
            % ISTA iterations:
            switch obj.s_method_Z
                case 'interleave'
                    [m_A_tilde, m_B_tilde] = compute_subderivatives(...
                        obj, v_w_f, v_lambda);
                    m_JF = eye(P)-alpha*m_F;
                    m_Z_tilde = m_A_tilde*m_JF*m_Z_tilde + m_B_tilde;
                case 'interleave-fast'
                    m_Z_tilde = obj.fast_Z_iteration(v_w_f, v_lambda, ...
                        m_Z_tilde, m_F);              
            end
        end
        
        % Methods that compute m_Z_tilde after the loop:
        if any(strcmp(obj.s_method_Z, {'linsolve', 'lsqminnorm'}))
            [m_A_tilde, m_B_tilde] = obj.compute_subderivatives(...
                v_w_f, v_lambda);
            m_toInvert = (eye(P)-m_A_tilde*(eye(P)-alpha*m_F));
            m_Z_tilde_out = feval(obj.s_method_Z, m_toInvert, m_B_tilde);
        elseif strcmp(obj.s_method_Z, {'iterate'})
            error 'not implemented yet'
            %
            % TODO: iteratively solve for Z based on (10)
            %
        else, m_Z_tilde_out = m_Z_tilde;
        end
    end
    
end

methods %that will be transferred to sub-classes in next version
    
    function [m_Z_tilde_out] = fast_Z_iteration(obj, ...
            v_w_f, v_lambda, m_Z_tilde_previous, m_F)
        alpha = obj.stepsize_w;
        switch obj.s_estimator
            case 'lasso'
                v_b_argument = v_w_f./(alpha*v_lambda);
                v_b = (v_b_argument > 1) - (v_b_argument < -1);
                m_Z_tilde_out = v_b.*( v_b.*(m_Z_tilde_previous...
                    -alpha*m_F*m_Z_tilde_previous) - alpha );
            case 'group-lasso'
                error 'not implemented yet'
            case 'weighted-lasso'
                error 'not implemented yet'
            case 'weighted-group-lasso'
                error 'not implemented yet'
            otherwise, error 'unrecognized option'
        end
    end
    
    function [m_A_tilde, m_B_tilde] = compute_subderivatives(...
            obj, v_w_f, lambda)
        alpha = obj.stepsize_w;
        switch obj.s_estimator
            case 'lasso'
                m_A_tilde = diag(abs(v_w_f)>=(alpha*lambda));
                m_B_tilde = alpha*(...
                    (v_w_f <= (-alpha*lambda)) -...
                    (v_w_f >= ( alpha*lambda)) );
            case 'group-lasso'
                v_gs = obj.v_group_structure;
                v_a_tilde = zeros(size(v_w_f)); m_B_tilde = v_a_tilde;
                for gr = 1:max(v_gs)
                    vb_idx = v_gs==gr;
                    my_norm = norm(v_w_f(vb_idx));
                        if my_norm >= alpha*lambda
                            v_a_tilde(vb_idx) = 1;
                            m_B_tilde(vb_idx) = -alpha*v_w_f(vb_idx)./my_norm;
                        % else, 0
                        end
                end
                m_A_tilde = diag(v_a_tilde);
            case 'weighted-lasso'
                error 'not implemented yet'
            case 'weighted-group-lasso'
                error 'not implemented yet'
            otherwise, error 'unrecognized option'
        end      
    end

    function v_w_out = prox_Omega(obj, v_w_in, rho)
        switch obj.s_estimator
            case 'lasso' % soft thresholding
                v_factors = max(0, 1-rho./abs(v_w_in));
                v_w_out = v_w_in.*v_factors;
                % v_signs = sign(v_w_in); 
                % % this version could be faster when compiled
                % w_out_alt = v_w_in - v_signs*rho;
                % w_out_alt(v_signs~=sign(w_out)) = 0;
            case 'group-lasso' %multidimensional soft thresholding
                v_gs = obj.v_group_structure; % assumed to have been checked
                v_factors = zeros(size(v_w_in));
                for gr = 1:max(v_gs)
                    vb_idx = v_gs==gr;
                    v_factors(vb_idx) = max(0, ...
                        1-rho./norm(v_w_in(vb_idx))); %TODO: this line is slow
                end
                v_w_out = v_w_in.*v_factors;   
            case 'weighted-lasso'
                error 'not implemented yet'
            case 'weighted-group-lasso'
                error 'not implemented yet'
            otherwise, error 'unrecognized option'
        end
    end
      
end
end

