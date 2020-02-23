classdef DynamicLassoHyper_alt
    % Algorithm for Online Hyperparmeter Optimization for the Lasso
    % regularizaction parameter in the DYNAMIC setting
properties
    stepsize_w
    stepsize_lambda
    
    approx_type = 'soft' %approximation type (soft or hard)
    
    forgettingFactor = 0.99 % for estimation of L only
    
    bias_correction = 0;
    %debug = 1
    
end

methods
    

    % Update derived on Feb 15, 2020
    function [v_w_t, lambda_next, loss_t, v_wf_next] = update(obj, ...
        lambda_t, v_x_t, y_t, v_wf_t)
    

        alpha = obj.stepsize_w;
        beta  = obj.stepsize_lambda;
        
        v_w_t = obj.soft_thresholding(v_wf_t, alpha*lambda_t);
        prediction_error = y_t - v_x_t'*v_w_t;

        loss_t = prediction_error.^2;
        v_z_t = max(-1, min(1, v_wf_t./(alpha*lambda_t))); %soft
        grad_t = prediction_error*alpha * v_x_t'*v_z_t;
        lambda_next = max(0, lambda_t - beta*(grad_t+obj.bias_correction));
        m_phi_t = v_x_t*v_x_t';
        v_r_t   = v_x_t*y_t;
        v_wf_next = v_w_t - alpha*(m_phi_t*v_w_t - v_r_t);
    end
    
    function [v_w_next, lambda_next, loss_t, v_wf_next] = update_alt(obj, ...
        lambda_t, v_x_t, y_t, v_wf_t, v_x_next, y_next)

        alpha = obj.stepsize_w;
        beta  = obj.stepsize_lambda;
        
        v_w_next = obj.soft_thresholding(v_wf_t, alpha*lambda_t);
        % prediction_error = y_t - v_x_t'*v_w_next;
        prediction_error = y_next - v_x_next'*v_w_next;

        loss_t = prediction_error.^2;
        v_z_t = max(-1, min(1, v_wf_t./(alpha*lambda_t))); %soft
        grad_t = prediction_error*alpha * v_x_next'*v_z_t;
        lambda_next = max(0, lambda_t - beta*grad_t);
        m_phi_t = v_x_t*v_x_t';
        v_r_t   = v_x_t*y_t;
        v_wf_next = v_w_next - alpha*(m_phi_t*v_w_next - v_r_t);
    end
    
    % modified from the one from report
    function [v_w_next, lambda_next, loss, v_wf_next] = update_modified(obj, ...
            v_w_t, lambda_t, v_x_t, y_t, v_x_next, y_next, v_wf_t_DUMMY)
               
        beta  =    obj.stepsize_lambda;       
        alpha =    obj.stepsize_w;
        
        %Online ISTA produces w^f[t];
        %v_wf_t = v_w_t - alpha*(v_x_t*(v_x_t'*v_w_t) - y_t*v_x_t);
        v_wf_t = v_wf_t_DUMMY;
        % Learner provides a hypothesis of the optimal lambda: lambda[t]
        
        % Nature reveals next time series sample: v_x_next, y_next
        
        % Online ISTA uses the lambda provided to generate w_next
        v_w_next = obj.soft_thresholding(v_wf_t, alpha*lambda_t);
        
        % Learner suffers loss
        prediction_error = y_next - v_x_next'*v_w_next;
        loss = prediction_error^2;
        
        % we produce lambda[t+1] to be used in the next iteration:
        z_argument = v_wf_t/(alpha*lambda_t);
        switch obj.approx_type
            case 'soft'
                v_z_t = max(-1, min(1, z_argument));
                g_t = prediction_error*alpha*v_x_next'*v_z_t;
            case 'hard'
                v_zHard_t = (z_argument > 1)  -  (z_argument < -1);
                g_t = prediction_error*alpha*v_x_next'*v_zHard_t;
        end
        lambda_next = max(0, lambda_t - beta*g_t);
        
        v_wf_next = v_w_next - alpha*(v_x_next*(v_x_next'*v_w_next) - y_next*v_x_next);
    end

        % update taken from report
    function [v_w_next, lambda_next, loss, L_hat_next] = update_original(obj, v_w_t, lambda_t, ...
            v_x_t, y_t, v_x_next, y_next, L_hat_t, t)
               
        beta = obj.stepsize_lambda;       
        L_now = v_x_t'*v_x_t;
        try
            alpha = double(obj.stepsize_w);
            L_hat_next = L_now;
        catch ME
            assert(strcmp(ME.identifier, 'MATLAB:invalidConversion') &... 
                isa(obj.stepsize_w, 'function_handle'), ... 
            'stepsize_w must either be a number or a Function Handle')           
            gamma = obj.forgettingFactor;
            if isnan(L_hat_t)
                L_hat_next = sqrt(length(v_x_t))*L_now;
            else
                L_hat_next = gamma*L_hat_t + (1-gamma)*L_now;
            end
            alpha = feval(obj.stepsize_w, L_hat_next, t);
        end
        
        %Online ISTA produces w^f[t];
        v_wf_t = v_w_t - alpha*(v_x_t*(v_x_t'*v_w_t) - y_t*v_x_t);
        
        % Learner provides a hypothesis of the optimal lambda: lambda[t]
        
        % Nature reveals next time series sample: v_x_next, y_next
        
        % Online ISTA uses the lambda provided to generate w_next
        v_w_next = obj.soft_thresholding(v_wf_t, alpha*lambda_t);
        
        % Learner suffers loss
        prediction_error = y_next - v_x_next'*v_w_next;
        loss = prediction_error^2;
        
        % we produce lambda[t+1] to be used in the next iteration:
        z_argument = v_wf_t/(alpha*lambda_t);
        switch obj.approx_type
            case 'soft'
                v_z_t = max(-1, min(1, z_argument));
                v_g_t = prediction_error*alpha*v_x_next'*v_z_t;
            case 'hard'
                v_zHard_t = (z_argument > 1)  -  (z_argument < -1);
                v_g_t = prediction_error*alpha*v_x_next'*v_zHard_t;
        end
        lambda_next = max(0, lambda_t - beta*v_g_t);
        
        % v_wf_next = v_w_next -
        % alpha(v_x_next*(v_x_next*(v_x_next'*v_w_next) - y_next*v_x_next)
    end
    
    % update taken from report, with two steps  
    function [v_w_next, lambda_next, loss, L_hat_next] = update_twoStepish(obj, v_w_t, lambda_t, ...
            v_x_t, y_t, v_x_next, y_next, L_hat_t, t)
               
        beta = obj.stepsize_lambda;       
        L_now = v_x_t'*v_x_t;
        try
            alpha = double(obj.stepsize_w);
            L_hat_next = L_now;
        catch ME
            assert(strcmp(ME.identifier, 'MATLAB:invalidConversion') &... 
                isa(obj.stepsize_w, 'function_handle'), ... 
            'stepsize_w must either be a number or a Function Handle')           
            gamma = obj.forgettingFactor;
            if isnan(L_hat_t)
                L_hat_next = sqrt(length(v_x_t))*L_now;
            else
                L_hat_next = gamma*L_hat_t + (1-gamma)*L_now;
            end
            alpha = feval(obj.stepsize_w, L_hat_next, t);
        end
        
        %Online ISTA produces w^f[t];
        v_wf_t = v_w_t - alpha*(v_x_t*(v_x_t'*v_w_t) - y_t*v_x_t);
        
        % Learner provides a hypothesis of the optimal lambda: lambda[t]
        
        % Nature reveals next time series sample: v_x_next, y_next
        
        % Online ISTA uses the lambda provided to generate w_next
        v_w_next = obj.soft_thresholding(v_wf_t, alpha*lambda_t);
        
        % Learner suffers loss
        first_prediction_error = y_next - v_x_next'*v_w_next;
        loss = first_prediction_error^2;
                
        z_argument = v_wf_t/(alpha*lambda_t);
        v_zSoft_t = max(-1, min(1, z_argument));
        v_zHard_t = (z_argument > 1)  -  (z_argument < -1);

        second_prediction_error = y_next - v_x_next'*(v_w_next - alpha*...
            (v_x_t*(v_x_t'*v_w_next) - y_t*v_x_t + lambda_t*v_zSoft_t));
        
        % we produce lambda[t+1] to be used in the next iteration:
        switch obj.approx_type
            case 'soft'               
%                 v_g_t = first_prediction_error*alpha*v_x_next'*v_zSoft_t;
                v_g_t = second_prediction_error*alpha*v_x_next'*v_zSoft_t;
            case 'hard'
%                 v_g_t = first_prediction_error*alpha*v_x_next'*v_zHard_t;
                v_g_t = second_prediction_error*alpha*v_x_next'*v_zHard_t;
        end
        lambda_next = max(0, lambda_t - beta*v_g_t);
        
        % v_wf_next = v_w_next -
        % alpha(v_x_next*(v_x_next*(v_x_next'*v_w_next) - y_next*v_x_next)
    end

end

methods (Static)
    function w_out = soft_thresholding(w_in, rho)
        v_factors = max(0, 1-rho./abs(w_in));
        w_out = w_in.*v_factors;
    end
end

end