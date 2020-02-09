classdef QStepsize < StepsizePolicy
    % Proposed by Luismi on Aug 10, 2019
    % Variance control with the Q function
    properties
        epsilon = 1e-6
        beta_1;

        v_v = 0;
        v_u = 0 
        nu          % Stepsize of the stepsize
        dqg = 0.01; % delta q goal: we want v_q \in 0.5 +- dqg
        v_q
        v_sigma
    end
    
    methods
        function v_eta_out = update_stepsize(obj, v_g, ~)
            obj.k = obj.k+1;
            if isempty(obj.v_eta)
                obj.v_eta = obj.eta_0;
            end

            obj.v_v = obj.beta_2*obj.v_v +(1-obj.beta_2)*v_g.^2;            
            obj.v_u = obj.beta_1*obj.v_u + (1-obj.beta_1)*v_g; 
            v_sigma2 = obj.v_v-obj.v_u.^2;
            if v_sigma2<0 
                warning 'Obtained a negative variance estimate'
                obj.v_q = 0.5+obj.dqg; % so the increase factor is 1
            else
                obj.v_sigma = sqrt(v_sigma2);
                obj.v_q = qfunc(obj.v_u./obj.v_sigma);
            end
            
            eta_ftml = (1-obj.beta_1)./(1-obj.beta_1.^obj.k) ...
                .* obj.eta_0./...
                (sqrt(obj.v_v/(1-obj.beta_2^obj.k))+obj.epsilon);
            increase_factor = 1./(1-obj.nu*...
                (obj.v_q-0.5+obj.dqg)*(obj.v_q-0.5-obj.dqg));
            obj.v_eta = min(eta_ftml, obj.v_eta*increase_factor);
            v_eta_out = obj.v_eta;
        end
    end
end