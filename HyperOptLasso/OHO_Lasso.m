classdef OHO_Lasso
    % Algorithm for Online Hyperparameter Optimization for the Lasso
    % regularization parameter
properties
    max_iter_outer = 200;  % maximum of outer iterations (over the hyperparam)
    max_iter_inner = 1000; % maximum of inner iterations (over model weights)
    min_iter_inner = 1;    % minimum of inner iterations (over model weights)
    tol_w = 1e-3;   % tolerance for checking convergence of (inner alg) ISTA
    tol_g = 1e-5; % tolerance for checking convergence of outer algorithm
    mirror_type = 'grad' %type of mirror descent (grad or log)
    b_online = 0; % flag for running online, 0 by default
    stepsize_policy StepsizePolicy = DiminishingStepsize;
    b_memory = 1; % flag for running with memory (saving the weights for each cross validation fold)
    normalized_lambda_0 = 1/100; % parameter for initializing the lambda par
    c_alpha = 1; % proportionality constant of stepsize (w.r.t. trace(Phi))
    
    approx_type = 'soft'; % type of connvex approximation (hard or soft)
    
    b_DA = 0; %dual averaging
        
    b_debug = 1; %flag for debugging mode
    
end

methods   
    function [v_lambda, v_it_count, w_out, v_eta] ... %TODO: output w_jackknife
            = solve_approx_mirror(obj, m_X, v_y, initial_lambda)
        
        N = length(v_y); assert(size(m_X, 1)==N);
        P = size(m_X, 2); % m_X is NxP
        m_Phi = m_X'*m_X/N;
        v_r   = m_X'*v_y/N;
        alpha = obj.c_alpha/trace(m_Phi);
        lambda_max = max(abs(v_r));
        v_lambda = zeros(1, obj.max_iter_outer);
        if exist('initial_lambda', 'var')
            v_lambda(1) = initial_lambda;
        else
            v_lambda(1) = lambda_max.*obj.normalized_lambda_0;
        end
        
        v_w_0 = obj.ista(zeros(P,1), m_Phi, v_r, alpha, v_lambda(1));
        %v_w_0 = zeros(P,1); %used this line to look in the oos-error
        if obj.b_memory
            m_W = repmat(sparse(v_w_0), [1 N]); % fast initialization
        else
            v_w_j = sparse(v_w_0);
        end
        v_eta = zeros(1, obj.max_iter_outer);
        v_g_bar = zeros(1, obj.max_iter_outer);
        
        my_sp = obj.stepsize_policy;
        v_eta(1)= my_sp.eta_0;
        v_it_count = zeros(obj.max_iter_outer, 1);
        v_kappas= zeros(1, obj.max_iter_outer); %only for debug plots
        v_crossval_error = zeros(obj.max_iter_outer,1);
        
        v_j = mod(0:obj.max_iter_outer-1, N)+1;
        my_sp.k = 1;
        ltc = LoopTimeControl(obj.max_iter_outer);
        combo = 0;
        for k = 2:obj.max_iter_outer
            g = 0;
            v_crossval_error(k) = 0;
            v_it_count(k) = 0;
            if obj.b_online
                v_indices_k = v_j(k);
            else
                v_indices_k = 1:N;
            end
            for j = v_indices_k
                v_x_j = m_X(j,:)';
                m_Phi_j = (m_Phi - v_x_j * v_x_j'/N);
                v_r_j   = (v_r   - v_y(j)* v_x_j/N);
                if obj.b_memory
                    [v_w_j, v_w_f, niter_out] = obj.ista(m_W(:,j), ...
                        m_Phi_j, v_r_j, alpha, v_lambda(k-1));
                    m_W(:,j) = v_w_j;
                else
                    [v_w_j, v_w_f, niter_out] = obj.ista(v_w_j, ...
                         m_Phi_j, v_r_j, alpha, v_lambda(k-1));
                    v_w_j = sparse(v_w_j);
                end
                v_z_j = max(-1, min(1, 1/(alpha*v_lambda(k-1))*v_w_f));
                v_zHard_j = sign(round(1/(2*alpha*v_lambda(k-1))*v_w_f));
                my_alpha = alpha; % version that actually works
                %my_alpha = -alpha; % this does not work
                % with the derivations as of feb 1, 2020, the use of -alpha
                % could be justified but I don't have the theoretical 
                % explanation for the algorithm working with alpha and not
                % with -alpha
                switch obj.approx_type
                    case 'soft'
                        g = g + my_alpha* v_x_j'*v_z_j*(v_y(j)- v_x_j'*(v_w_j- ...
                            my_alpha*(m_Phi_j*v_w_j - v_r_j + v_lambda(k-1)*v_z_j)));
                    case 'hard'
                        g = g + my_alpha* v_x_j'*v_zHard_j*(v_y(j)- v_x_j'*(v_w_j- ...
                            my_alpha*(m_Phi_j*v_w_j - v_r_j + v_lambda(k-1)*v_z_j)));
                    otherwise, error 'misspecified approximation type'
                end
                
                v_it_count(k) = v_it_count(k) + niter_out;
%                 if obj.b_debug
%                     aiej = 1:N; aiej(j) = []; %all indices except j
%                     v_crossval_error(k) = v_crossval_error(k) + sum(...
%                         (m_X(aiej,:)*v_w_j-v_y(aiej)).^2);
%                 end
            end

            v_eta(k) = my_sp.update_stepsize(g, v_lambda(k-1));
            if obj.b_DA
                [v_lambda(k), v_g_bar(k)] = obj.dual_averaging_step(g, v_eta(k), v_g_bar(k-1), k);
            else                               
                v_lambda(k) = obj.mirror_step(v_lambda(k-1), g, v_eta(k));
            end
            
%             b_stopping_criterion = norm(my_sp.v_u)<obj.tol_g &&  ...
%                     k > N && all( abs(my_sp.v_q-0.5) < my_sp.dqg );
            b_stopping_criterion = 0;
            if ~obj.b_online
                b_stopping_criterion = all(abs(g)<obj.tol_g);
            else
                try
                    b_improvement_is_small = all( ...
                        abs(my_sp.v_u)*N*v_eta(k)<obj.tol_g*lambda_max );
                catch
                    b_improvement_is_small = 0;
                end
                if b_improvement_is_small
                    combo = combo + b_improvement_is_small;
                else
                    combo = 0;
                end
                if combo > N ...
                    && all( abs(my_sp.v_q-0.5) < my_sp.dqg )
                    b_stopping_criterion = 1;
                end
            end
            % set breakpoint here and set b_stopping criterion manually
            % to 1 to end execution
            if b_stopping_criterion
                disp 'Reached Stopping criterion.'
                break
            end

            ltc.go(k);
            if obj.b_debug
                try
                    v_sigma2 = my_sp.v_v...
                        - my_sp.v_u.^2; assert( all(v_sigma2>0) )                  
                    v_kappas(k) = qfunc(my_sp.v_u ...
                        ./sqrt(v_sigma2));
                end                
                if obj.b_debug && mod(k, 100) == 0
                    figure(101); clf
                    subplot(411);
                    plot(v_lambda); title '\lambda'
                    subplot(412);
                    if obj.b_DA
                        plot(v_g_bar); title 'g bar'
                    else
                        plot(v_eta);    title '\eta'
                    end
                    subplot(413);
                    plot(v_kappas); title '\kappa'
%                     subplot(414)
%                     plot(v_crossval_error); title 'cross-validation error'
                    drawnow
                end
            end
        end
        %output
        if obj.b_memory
            w_jackknife = mean(m_W, 2); %jackknife estimator?
            w_out = obj.ista(w_jackknife, m_Phi, v_r, alpha, v_lambda(k));
        else
            obj.tol_w = 1e-4;
            %TODO: w_jackknife = running average of w_j's
            w_out = obj.ista(v_w_j, m_Phi, v_r, alpha, v_lambda(k));
        end
    end
    
    function [v_w_j, v_w_f, it_count] = ista(obj,...
            v_w_initial, m_Phi_j, v_r_j, alpha, lambda)
        v_w_j = v_w_initial;
        v_w_f = v_w_j;
        it_count = obj.max_iter_inner;
        for k_inner = 1:obj.max_iter_inner
            v_v_j = m_Phi_j*v_w_j - v_r_j; %gradient
            
            %check stopping criterion
            v_grad_violations = (v_v_j.*sign(v_w_j) + lambda).*(v_w_j~=0);
            if norm(v_grad_violations) < obj.tol_w && ...
                    all(abs(v_v_j).*(v_w_j==0) < lambda) && ...
                    k_inner>obj.min_iter_inner
                it_count = k_inner;
                break; 
            end
            
            v_w_f = v_w_j - alpha*v_v_j;
            v_w_j = obj.soft_thresholding(v_w_f, alpha*lambda);
            
            %debug
%             if mod(k_inner,5) == 0
%                 figure(2); clf
%                 stem(v_v_j); hold on
%                 stem(v_grad_violations)
%                 plot([1 length(v_v_j)], lambda*[1 1; -1 -1]')
%                 legend('gradient');%, 'violations')
%                 drawnow
%             end           
        end
    end
    
    function [x_out, v_g_bar_out] = dual_averaging_step(obj, g, beta, v_g_bar_t, t)
            % update the average subgradient
            v_g_bar_out = ((t-1)*v_g_bar_t + g)/t;
            % compute the next iterate
            %w_next = arg min (dot(g_bar, w) + beta/t*h(w)
            x_out = max(0, -v_g_bar_out*t/beta);
        end
    
    function x_out = mirror_step(obj, x_in, g, beta)
        assert(iscolumn(x_in), 'x must be a column vector')
        assert(iscolumn(g),    'g must be a column vector')
        assert(iscolumn(beta), 'step size must be a column vector')
        switch obj.mirror_type
            case 'grad'
                x_out = max(0, x_in - beta.*g);
            case 'log'
                if g*beta < -0.9
                    warning 'negative d times beta too large'
                    warning 'rate of increase capped at 10'
                    x_out = 10*x_in;
                else
                    x_out = x_in/(1+beta.*g);
                end
                % TODO: maybe also interesting to check whether x exceeds
                % a maximum value (in this case, lambda_max)
            otherwise, error 'misspecified mode'
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