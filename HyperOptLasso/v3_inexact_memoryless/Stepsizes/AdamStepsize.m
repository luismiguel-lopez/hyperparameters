classdef AdamStepsize < StepsizePolicy
    properties
        beta_1 = 0.9;
        epsilon = 1e-12
        v_m = 0
        v_v = 0.2
    end
    methods
        function v_eta_out = update_stepsize(obj, v_g, ~)
            obj.k = obj.k+1;
            obj.v_m = obj.beta_1*obj.v_m +(1-obj.beta_1)*v_g;
            obj.v_v = obj.beta_2*obj.v_v +(1-obj.beta_2)*v_g.^2;
            m_hat = obj.v_m/(1-obj.beta_1.^obj.k);
            v_hat = obj.v_v/(1-obj.beta_2.^obj.k);
            obj.v_eta = obj.eta_0.*obj.v_m./(v_g.*sqrt(obj.v_v)+obj.epsilon);
            v_eta_out = obj.v_eta;
        end
    end
end