classdef DynamicRecursiveLassoHyper
    % Algorithm for Online Hyperparmeter Optimization for the Lasso
    % regularizaction parameter in the DYNAMIC setting
properties
    stepsize_w
    stepsize_lambda
    
    approx_type = 'soft' %approximation type (soft or hard)
    forgettingFactor = 0.9;
    
    %debug = 1
    
end

methods
    function [v_w_next, lambda_next, loss, m_Phi_next, v_r_next] = update(obj, v_w_t, lambda_t, ...
            v_x_t, y_t, v_x_next, y_next, m_Phi_t, v_r_t)
        
        alpha = obj.stepsize_w;
        gamma = obj.forgettingFactor;
        
        % update recursive figures Phi and r
        m_Phi_next = gamma*m_Phi_t + (1-gamma)*(v_x_t*v_x_t');
        v_r_next   = gamma*v_r_t   + (1-gamma)*(v_x_t*y_t);
        
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