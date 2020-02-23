classdef FranceschiRecursiveLasso
    % Algorithm for Online Hyperparmeter Optimization for the Lasso
    % regularizaction parameter in the DYNAMIC setting
    % Recursive objective (TIRSO style)
properties
    stepsize_w
    stepsize_lambda
        
    forgettingFactor = 0.99;
        
end

methods
    
    % Update derived on Feb 20, 2020
    function [v_w_t, lambda_next, loss_t, v_wf_next, m_Phi_next, v_r_next, v_c_next] = update(obj, ...
        lambda_t, v_x_t, y_t, v_wf_t, m_Phi_t, v_r_t, v_c_t)
    
        gamma = obj.forgettingFactor;
        alpha = obj.stepsize_w;
        beta  = obj.stepsize_lambda;
        
        m_Phi_next = gamma*m_Phi_t + (1-gamma)*(v_x_t*v_x_t');
        v_r_next   = gamma*v_r_t   + (1-gamma)*v_x_t*y_t;

        
        v_w_t = obj.soft_thresholding(v_wf_t, alpha*lambda_t);
        prediction_error = y_t - v_x_t'*v_w_t;

        loss_t = prediction_error.^2;
        v_z_t = max(-1, min(1, v_wf_t./(alpha*lambda_t))); %soft
        v_c_next = v_c_t - alpha*m_Phi_t*v_c_t - alpha*v_z_t;
        grad_t = -prediction_error* v_x_t'*v_c_t;
        lambda_next = max(0, lambda_t - beta*grad_t);
        v_wf_next = v_w_t - alpha*(m_Phi_next*v_w_t - v_r_next);
    end
end

methods (Static)
    function w_out = soft_thresholding(w_in, rho)
        v_factors = max(0, 1-rho./abs(w_in));
        w_out = w_in.*v_factors;
    end
end

end