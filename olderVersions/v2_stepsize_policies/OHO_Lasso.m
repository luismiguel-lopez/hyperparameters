classdef OHO_Lasso
properties
    max_iter_outer = 200;
    max_iter_inner = 1000;
    tol = 1e-3; % tolerance for checking convergence of ISTA
    mirror_type = 'grad'
    b_online = 0;
    stepsize_policy_object = DiminishingStepsize;
    
    stepsize_policy = 'decreasing';
    beta = 500; % initial stepsize
    rms_beta = 0.99;
    adagrad_eta = 0.05
    rmsprop_eta = 0.003;
    lrag_nu  = 0.001;
    lrag_k   = 3/200; %! /200
    lrag_rmin = 0.99
    lrag_rmax = 1.01
    
    debug = 1;
    
end

methods   
    function v_lambda = solve_approx_mirror(obj, m_X, v_y)
        N = length(v_y); assert(size(m_X, 1)==N);
        P = size(m_X, 2);
        m_Phi = m_X'*m_X;
        v_r   = m_X'*v_y;
        alpha = 10/trace(m_Phi);
        lambda_max = max(abs(v_r));
        v_lambda = zeros(obj.max_iter_outer, 1);
        v_lambda(1) = lambda_max/100;
        v_w = obj.ista(zeros(P,1), m_Phi, v_r, alpha, v_lambda(1));
        m_W = repmat(v_w, [1 N]); % fast initialization
        v_beta = obj.beta./sqrt(1:obj.max_iter_outer);
        v_j = mod(0:obj.max_iter_outer-1, N)+1;
        v_eta = zeros(obj.max_iter_outer, 1); v_eta(1) = obj.beta;
        try
            v_eta(1)= obj.stepsize_policy_object.eta_0;
        end
        adagrad_u = 0;%adagrad
        rmsprop_u = 1/obj.beta;
        r = rls(obj.rms_beta);
        r.set_nWeights(2);
        p_hat = 0;
        lambda0_hat = 0;
        [lrag_a, lrag_b] = give_me_params(obj.lrag_k, obj.lrag_rmin, obj.lrag_rmax);
        obj.stepsize_policy_object.k = 1;
        if isa(obj.stepsize_policy_object, 'RmsPropStepsize')
            obj.stepsize_policy_object.v_u = 1/obj.beta;
        end
        for k = 2:obj.max_iter_outer
            g = 0;
            for j = 1:N
                if obj.b_online && j ~= v_j(k)
                    continue
                end
                v_x_j = m_X(j,:)';
                m_Phi_j = m_Phi - v_x_j * v_x_j';
                v_r_j   = v_r   - v_y(j)* v_x_j;
                [v_w_j, v_w_f] = obj.ista(m_W(:,j), m_Phi_j, v_r_j, ...
                    alpha, v_lambda(k-1));
                m_W(:,j) = v_w_j;
                v_s_j = max(-1, min(1, 1/(alpha*v_lambda(k-1))*v_w_f));
                g = g + alpha* v_x_j'*v_s_j*(v_y(j)- v_x_j'*(v_w_j- ...
                    alpha*(m_Phi_j*v_w_j - v_r_j + v_lambda(k-1)*v_s_j)));
                v_g_j(j) = g; %!, debug
            end
            adagrad_u = adagrad_u + g^2; %used in 2 cases
            switch obj.stepsize_policy
                case 'constant'
                    v_eta(k) = obj.beta;
                case 'decreasing' 
                    v_eta(k) = v_beta(k);
                case 'adagrad'
                    v_eta(k) = obj.adagrad_eta/(sqrt(adagrad_u)+1/obj.beta);
                case 'adagrad_modified'
                    v_eta(k) = k.^(-1/3)/(sqrt(adagrad_u)+1/obj.beta);
                    
                case 'rmsprop'
                    rmsprop_u = obj.rms_beta*rmsprop_u + (1-obj.rms_beta)*g^2;
                    v_eta(k) = obj.rmsprop_eta/(sqrt(rmsprop_u)+1/obj.beta);
                case 'adam'
                    assert(isa(obj.stepsize_policy_object,'AdamStepsize'));
                    v_eta(k) = obj.stepsize_policy_object.update_stepsize(g, v_lambda(k-1));
                case 'uftml'
                    assert(isa(obj.stepsize_policy_object,'UftmlStepsize'));
                    v_eta(k) = obj.stepsize_policy_object.update_stepsize(g, v_lambda(k-1));
                case 'linear_regression'
                    assert(isa(obj.stepsize_policy_object,'LinearRegressionStepsize'));
                    v_eta(k) = obj.stepsize_policy_object.update_stepsize(g, v_lambda(k-1));
                case 'linear_regression_old'
                    ww_in = [p_hat; lambda0_hat];
                    ww_out = r.update_weights(ww_in, [k; 1], ...
                        v_lambda(k-1)-[k 1]*ww_in);
                    p_hat = ww_out(1);
                    lambda0_hat = ww_out(2);
                    weights= obj.rms_beta.^(k-1-(1:k-1));
                    normalized_weights= weights/sum(weights);
                    sigma2_hat = normalized_weights*...
                        (v_lambda(1:k-1)-[1:k-1; ones(1, k-1)]'*ww_out).^2;
                    sigma_hat = sqrt(sigma2_hat);
                    if obj.lrag_rmin == 0
                        v_eta(k) = v_eta(k-1)*abs(p_hat/(obj.lrag_k*sigma_hat)).^obj.lrag_nu;
                    elseif obj.lrag_rmax == 0
                        v_eta(k) = v_eta(k-1)*obj.rmin*(1+ abs(p_hat/(obj.lrag_k*sigma_hat)));
                    else
                        q = abs(p_hat/sigma_hat);
                        factor = lrag_a*q^2 + lrag_b*q + obj.lrag_rmax;
                        v_eta(k) = v_eta(k-1)*factor;
                    end
            end
            %v_eta2 = obj.stepsize_policy_object.update_stepsize(g, v_lambda(k-1));
            %assert(v_eta2==v_eta(k));
            v_lambda(k) = obj.mirror_step(v_lambda(k-1), g, v_eta(k));
            v_kappa(k) = obj.stepsize_policy_object.kappa;
            v_g(k) = g;

            if obj.debug && mod(k, 10) == 0
                figure(101); clf
                subplot(311);
                plot(v_lambda);
                if isequal(obj.stepsize_policy, 'linear_regression_old')
                    hold on
                    plot(1:k, [1:k; ones(1, k)]'*ww_out); 
                    plot(1:k, [1:k; ones(1, k)]'*ww_out + sigma_hat, '--r')
                    plot(1:k, [1:k; ones(1, k)]'*ww_out - sigma_hat, '--r')
                    ylim([0 lambda_max/20])
                    drawnow
                end
                if isa(obj.stepsize_policy_object, 'LinearRegressionStepsize')
                    obj.stepsize_policy_object.plot_state();
                    ylim([0 lambda_max/20]);
                    subplot(312);
                    plot(v_eta);
                    subplot(313);
                    plot(v_kappa);
                end
%                 figure(102); clf
%                 plot(v_g);
                drawnow
            end
        end
    end
    
    function [v_w_j, v_w_f] = ista(obj,...
            v_w_initial, m_Phi_j, v_r_j, alpha, lambda)
        v_w_j = v_w_initial;
        v_w_f = v_w_j;
        for k_inner = 1:obj.max_iter_inner
            v_v_j = m_Phi_j*v_w_j - v_r_j; %gradient
            %check stopping criterion
            v_grad_violations = (v_v_j.*sign(v_w_j) + lambda).*(v_w_j~=0);
            if norm(v_grad_violations) < obj.tol && ...
                all(abs(v_v_j).*(v_w_j==0) < lambda)
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
        switch obj.mirror_type
            case 'grad'
                x_out = max(0, x_in - beta*g);
            case 'log'
                if g*beta < -0.9
                    warning 'negative d times beta too large'
                    warning 'rate of increase capped at 10'
                    x_out = 10*x_in;
                else
                    x_out = x_in/(1+beta*g);
                end
                % TODO: maybe also interesting to check whether x exceeds
                % a maximum value (in this case, lambda_max)
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