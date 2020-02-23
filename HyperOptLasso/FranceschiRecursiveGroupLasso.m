classdef FranceschiRecursiveGroupLasso
    % Algorithm for Online Hyperparmeter Optimization for the Lasso
    % regularizaction parameter in the DYNAMIC setting
    % Recursive objective (TIRSO style)
properties
    stepsize_w
    stepsize_lambda
        
    forgettingFactor = 0.99;
    
    v_group_structure
        
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

        
        v_w_t = obj.group_soft_thresholding(v_wf_t, alpha*lambda_t, ...
            obj.v_group_structure);
        prediction_error = y_t - v_x_t'*v_w_t;

        loss_t = prediction_error.^2;
        v_z_t = obj.zGroup(v_wf_t, alpha*lambda_t, obj.v_group_structure);
        v_c_next = v_c_t - alpha*m_Phi_t*v_c_t - alpha*v_z_t;
        grad_t = -prediction_error* v_x_t'*v_c_t;
        lambda_next = max(0, lambda_t - beta*grad_t);
        v_wf_next = v_w_t - alpha*(m_Phi_next*v_w_t - v_r_next);
    end
end

methods (Static)
    function w_out = group_soft_thresholding(w_in, rho, v_group_structure)
        v_factors = zeros(size(w_in));
        for gr = 1:max(v_group_structure)
            indices = v_group_structure==gr;
            v_factors(indices) = max(0, 1-rho./norm(w_in(indices)));
        end
        w_out = w_in.*v_factors;
    end
    
     function [v_z, v_zHard] = zGroup(wf_in, rho, v_group_structure)
         v_z = zeros(size(wf_in));
         v_zHard = v_z;
         for gr = 1:max(v_group_structure)
             indices = v_group_structure==gr;
             group_norm = norm(wf_in(indices));
             if group_norm == 0
                 v_z(indices) = 0;
                 v_zHard(indices) = 0;
             elseif group_norm < rho && group_norm > 0
                 v_z(indices) = wf_in(indices)./rho;
                 v_zHard(indices) = 0;
             elseif group_norm >= rho && group_norm > 0
                 v_z(indices) = wf_in(indices)./group_norm;
                 v_zHard(indices) = v_z(indices);
             else
                 error fatal
             end
         end
     end
end

end