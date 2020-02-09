classdef QStepsize_old < StepsizePolicy
    % Proposed by Luismi on Aug 10, 2019
    properties
        epsilon = 1e-6
        beta_1 = 0.9;

        v_v = 0;
        v_v_begin %used to make sure the step size does not grow too large
        v_u = 0 
        %eta_policy StepsizePolicy
        nu
        dqg = 0.01; %delta q goal
        N
    end
    methods
        function v_eta_out = update_stepsize(obj, v_g, ~)
            obj.k = obj.k+1;
            obj.v_v = obj.beta_2*obj.v_v +(1-obj.beta_2)*v_g.^2;
            % same as in Adam;
            
            obj.v_u = obj.beta_1*obj.v_u + (1-obj.beta_1)*v_g; 
            v_sigma2 = obj.v_v-obj.v_u.^2;
            v_sigma  = sqrt(v_sigma2);
            v_q      = qfunc(obj.v_u./v_sigma);
            eta_ftml = (1-obj.beta_1)./(1-obj.beta_1.^obj.k) ...
                .* obj.eta_0./...
                (sqrt(obj.v_v/(1-obj.beta_2^obj.k))+obj.epsilon);
            if isempty(obj.v_eta) || obj.k<obj.N
                obj.v_eta = obj.eta_0;
                obj.v_v_begin = obj.v_v/(1-obj.beta_2^obj.k);
            else
                decrease_factor = 1-obj.nu*...
                    (v_q-0.5+obj.dqg)*(v_q-0.5-obj.dqg);
                increase_factor = 1/decrease_factor;
                v_v_cap = obj.v_v_begin;%! *max(1, obj.k/obj.N);
                if obj.v_v > v_v_cap
                    beta_cap = (1+obj.beta_2)/2;
                    obj.v_v_begin = beta_cap*obj.v_v_begin + ...
                        (1-beta_cap)*v_g.^2;
                    increase_factor = 0.99; % property
                end
                obj.v_eta = obj.v_eta*increase_factor;
            end
            v_eta_out = obj.v_eta;
        end
    end
end