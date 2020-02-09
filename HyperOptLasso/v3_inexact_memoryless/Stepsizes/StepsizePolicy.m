classdef StepsizePolicy <handle
    properties
        eta_0  = 1
        beta_2 = 0.99 % parameter for recursive estimation 
        % of second-order moments
%         beta_1 = 0.9% parameter for recursive estimation 
%         % of first-order moments
        
        v_eta
        k = 0
    end
    
    methods
        v_eta_out = update_stepsize(obj, v_gradient, v_x_prev)
        
        function reset(obj)
            obj.k = 0;
            obj.v_eta = obj.eta_0;
        end
    end
end