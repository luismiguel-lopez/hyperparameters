classdef DynamicRecursiveLassoHyper_alt
    % Algorithm for Online Hyperparmeter Optimization for the Lasso
    % regularizaction parameter in the DYNAMIC setting
    % Recursive objective (TIRSO style)
properties
    stepsize_w
    stepsize_lambda
    
    approx_type = 'soft' %approximation type (soft or hard)
    
    forgettingFactor = 0.9;
    
    bias_correction = 0;
    b_QL = 0;
    %debug = 1
    
end

methods
    
    % Update derived on Feb 16, 2020
    function [v_w_t, lambda_next, loss_t, v_wf_next, m_Phi_next, v_r_next] = update(obj, ...
        lambda_t, v_x_t, y_t, v_wf_t, m_Phi_t, v_r_t)
    
        gamma = obj.forgettingFactor;
        alpha = obj.stepsize_w;
        beta  = obj.stepsize_lambda;
        
        m_Phi_next = gamma*m_Phi_t + (1-gamma)*(v_x_t*v_x_t');
        v_r_next   = gamma*v_r_t   + (1-gamma)*v_x_t*y_t;

        
        v_w_t = obj.soft_thresholding(v_wf_t, alpha*lambda_t);
        prediction_error = y_t - v_x_t'*v_w_t;

        loss_t = prediction_error.^2;
        v_z_t = max(-1, min(1, v_wf_t./(alpha*lambda_t))); %soft
        grad_t = prediction_error*alpha * v_x_t'*v_z_t;
        grad_DP = -2*alpha*v_z_t'*(m_Phi_next*(v_w_t-alpha*(m_Phi_next*v_w_t-v_r_next+lambda_t*v_z_t))-v_r_next);
        grad_QL = -alpha*v_z_t'*(m_Phi_next*v_w_t - 2*v_r_next);
        if obj.b_QL
            lambda_next = max(0, lambda_t + beta*(grad_QL+obj.bias_correction)); %!!+
        else
            lambda_next = max(0, lambda_t - beta*(grad_t+obj.bias_correction));
        end
        v_wf_next = v_w_t - alpha*(m_Phi_next*v_w_t - v_r_next);
    end

    function [v_w_next, lambda_next, loss, m_Phi_next, v_r_next] = update_twoStepish(obj, v_w_t, lambda_t, ...
            v_x_t, y_t, v_x_next, y_next, m_Phi_t, v_r_t, t)
        
        % update recursive figures Phi and r
        gamma = obj.forgettingFactor;
        m_Phi_next = gamma*m_Phi_t + (1-gamma)*(v_x_t*v_x_t');
        v_r_next   = gamma*v_r_t   + (1-gamma)*(v_x_t*y_t);
        try
            alpha = double(obj.stepsize_w);
        catch ME
            assert(strcmp(ME.identifier, 'MATLAB:invalidConversion') &... 
                isa(obj.stepsize_w, 'function_handle'), ... 
            'stepsize_w must either be a number or a Function Handle')
            alpha = feval(obj.stepsize_w, trace(m_Phi_next), t);         
        end
        
        %Online ISTA produces w^f[t];
        v_wf_t = v_w_t - alpha*(m_Phi_next*v_w_t - v_r_next);
        
        % Learner provides a hypothesis of the optimal lambda: lambda[t]     
        % Nature reveals next time series sample: v_x_next, y_next
        
        % Online ISTA uses the lambda provided to generate w_next
        v_w_next = obj.soft_thresholding(v_wf_t, alpha*lambda_t);
        
        % Learner suffers loss
        prediction_error = y_next - v_x_next'*v_w_next;
        loss = prediction_error^2;
        
        z_argument = v_wf_t/(alpha*lambda_t);
        v_zSoft_t = max(-1, min(1, z_argument));
        v_zHard_t = (z_argument > 1)  -  (z_argument < -1);
        
        second_prediction_error = y_next - v_x_next'*(v_w_next - alpha*...
            (m_Phi_next*v_w_next - v_r_next + lambda_t*v_zSoft_t));
        
        % we produce lambda[t+1] to be used in the next iteration:
        switch obj.approx_type
            case 'soft'
                v_g_t = second_prediction_error*alpha*v_x_next'*v_zSoft_t;
            case 'hard'
                v_g_t = second_prediction_error*alpha*v_x_next'*v_zHard_t;
        end
        beta = obj.stepsize_lambda;
        lambda_next = max(0, lambda_t - beta*v_g_t);
    end
    
    function [v_w_next, lambda_next, loss, m_Phi_next, v_r_next] = update_original(obj, v_w_t, lambda_t, ...
            v_x_t, y_t, v_x_next, y_next, m_Phi_t, v_r_t, t)
        
        % update recursive figures Phi and r
        gamma = obj.forgettingFactor;
        m_Phi_next = gamma*m_Phi_t + (1-gamma)*(v_x_t*v_x_t');
        v_r_next   = gamma*v_r_t   + (1-gamma)*(v_x_t*y_t);
        try
            alpha = double(obj.stepsize_w);
        catch ME
            assert(strcmp(ME.identifier, 'MATLAB:invalidConversion') &... 
                isa(obj.stepsize_w, 'function_handle'), ... 
            'stepsize_w must either be a number or a Function Handle')
            alpha = feval(obj.stepsize_w, trace(m_Phi_next), t);         
        end
        
        %Online ISTA produces w^f[t];
        v_wf_t = v_w_t - alpha*(m_Phi_next*v_w_t - v_r_next);
        
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
        beta = obj.stepsize_lambda;
        lambda_next = max(0, lambda_t - beta*v_g_t);
    end
end

methods (Static)
    function w_out = soft_thresholding(w_in, rho)
        v_factors = max(0, 1-rho./abs(w_in));
        w_out = w_in.*v_factors;
    end
end

end