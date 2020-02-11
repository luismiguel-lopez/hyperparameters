classdef OHO_ElasticNet
properties
    max_iter_outer = 200;
    max_iter_inner = 1000;
    min_iter_inner = 1;
    tol = 1e-3; % tolerance for checking convergence of ISTA
    tol_g = 1e-5;
    mirror_type = 'grad'
    b_online = 0;
    stepsize_policy StepsizePolicy = DiminishingStepsize;
    b_memory = 1;
    normalized_lambda_0 = 1/100;
        
    debug = 1;
    
end

methods   
    function [m_lambda, v_it_count, w_out] ... %TODO: output w_jackknife
            = solve_approx_mirror(obj, m_X, v_y)
        
        N = length(v_y); assert(size(m_X, 1)==N);
        P = size(m_X, 2); % m_X is NxP
        m_Phi = m_X'*m_X;
        v_r   = m_X'*v_y;
        alpha = 10/trace(m_Phi);
        lambda_max = max(abs(v_r));
        m_lambda = zeros(2, obj.max_iter_outer); % EN: now 2 hyperparams
        m_lambda(1, 1) = lambda_max.*obj.normalized_lambda_0;
        m_lambda(2, 1) = m_lambda(1,1)/10;
        v_w_0 = obj.ista(zeros(P,1), m_Phi+eye(P)*m_lambda(2,1), ...
            v_r, alpha, m_lambda(1, 1));
        %v_w_0 = zeros(P,1); %used this line to look in the oos-error
        if obj.b_memory
            m_W = repmat(sparse(v_w_0), [1 N]); % fast initialization
        else
            v_w_j = sparse(v_w_0);
        end
        m_eta = zeros(2, obj.max_iter_outer);
        my_sp = obj.stepsize_policy;
        m_eta(:, 1)= my_sp.eta_0;
        v_it_count = zeros(obj.max_iter_outer, 1);
        v_kappas= zeros(obj.max_iter_outer, 1); %only for debug plots
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
                m_Phi_j_plus_rhoI = m_Phi - v_x_j * v_x_j' + m_lambda(2, k-1)*eye(P);
                v_r_j   = v_r   - v_y(j)* v_x_j;
                if obj.b_memory
                    [v_w_j, v_w_f, niter_out] = obj.ista(m_W(:,j), ...
                        m_Phi_j_plus_rhoI, v_r_j, alpha, m_lambda(1, k-1));
                    m_W(:,j) = v_w_j;
                else
                    [v_w_j, v_w_f, niter_out] = obj.ista(v_w_j, ...
                         m_Phi_j_plus_rhoI, v_r_j, alpha, m_lambda(1, k-1));
                    v_w_j = sparse(v_w_j);
                end
                v_s_j = max(-1, min(1, ...
                    1/(alpha*m_lambda(1, k-1))*v_w_f));
                b_elasticNet = 1;
                if b_elasticNet
                    g = g + alpha*[v_s_j';v_w_j']*v_x_j*(v_y(j)- v_x_j'*(v_w_j- ...
                        alpha*(m_Phi_j_plus_rhoI*v_w_j - v_r_j + m_lambda(1, k-1)*v_s_j)));
                else %old code
                    g = g + alpha* v_x_j'*v_s_j*(v_y(j)- v_x_j'*(v_w_j- ...
                        alpha*(m_Phi_j_plus_rhoI*v_w_j - v_r_j + m_lambda(1, k-1)*v_s_j)));
                end
                v_it_count(k) = v_it_count(k) + niter_out;
                aiej = 1:N; aiej(j) = []; %all indices except j
                v_crossval_error(k) = v_crossval_error(k) + sum(...
                    (m_X(aiej,:)*v_w_j-v_y(aiej)).^2);
            end
%             if obj.b_memory % to be removed
%                 v_crossval_error(k) = sum((sum(m_W.*m_X')-v_y').^2);
%             else
%                 v_crossval_error(k) = sum((m_X*v_w_j-v_y).^2);
%             end
            m_eta(:, k) = my_sp.update_stepsize(g, ...
                m_lambda(:, k-1));
            m_lambda(:, k) = obj.mirror_step(m_lambda(:, k-1), g, m_eta(:, k));
            
%             b_stopping_criterion = norm(my_sp.v_u)<obj.tol_g &&  ...
%                     k > N && all( abs(my_sp.v_q-0.5) < my_sp.dqg );
            try
                b_improvement_is_small = all( ...
                    abs(my_sp.v_u)*N*m_eta(:, k)<obj.tol_g*lambda_max );
            catch
                b_improvement_is_small = 0;
            end
                    
            if b_improvement_is_small
                combo = combo + b_improvement_is_small;
            else
                combo = 0;
            end
            if combo > N && all( abs(my_sp.v_q-0.5) < my_sp.dqg )
                disp 'Reached Stopping criterion.'
                break
            end

            ltc.go(k);
            if obj.debug
                try
                    v_sigma2 = my_sp.v_v...
                        - my_sp.v_u.^2; assert (v_sigma2>0)                  
                    v_kappas(k) = qfunc(my_sp.v_u ...
                        ./sqrt(v_sigma2));
                end                
                if obj.debug && mod(k, 100) == 0
                    figure(101); clf
                    subplot(411);
                    plot(m_lambda'); title 'Hyperparams'
                    legend('\lambda', '\rho')
                    subplot(412);
                    plot(m_eta');    title '\eta'
                    subplot(413);
                    plot(v_kappas); title '\kappa'
                    subplot(414)
                    plot(v_crossval_error); title 'cross-validation error'
                    drawnow
%                   if isa(my_sp, 'LinearRegressionStepsize')
%                       my_sp.plot_state();
%                       ylim([0 lambda_max/50]);
%                   end
                end
            end
        end
        %output
        if obj.b_memory
            w_jackknife = mean(m_W, 2); %jackknife estimator?
            w_out = obj.ista(w_jackknife, m_Phi+m_lambda(2, k)*eye(P), v_r, alpha, m_lambda(1, k));
        else
            obj.tol = 1e-4;
            %TODO: w_jackknife = running average of w_j's
            w_out = obj.ista(v_w_j, m_Phi+m_lambda(2, k)*eye(P), v_r, alpha, m_lambda(1, k));
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
            if norm(v_grad_violations) < obj.tol && ...
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
            otherwise error 'misspecified mode'
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