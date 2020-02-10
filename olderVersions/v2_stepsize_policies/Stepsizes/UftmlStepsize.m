classdef UftmlStepsize < StepsizePolicy
    % Stepsize policy from unconstrained FTML
    % zheng2017follow, equation (14)
    properties
        epsilon = 1e-6
        beta_1 = 0.9;

        v_v = 0.2;
        v_u = 0
        eta_policy StepsizePolicy
    end
    methods
        function v_eta_out = update_stepsize(obj, v_g, ~)
            obj.k = obj.k+1;
            obj.v_v = obj.beta_2*obj.v_v +(1-obj.beta_2)*v_g.^2;
            % same as in Adam;
            internal_eta = obj.eta_policy.update_stepsize(v_g);
            obj.v_eta = (1-obj.beta_1)./(1-obj.beta_1.^obj.k) ...
                .* internal_eta./...
                (sqrt(obj.v_v/(1-obj.beta_2^obj.k))+obj.epsilon);
            v_eta_out = obj.v_eta;
        end
    end
end