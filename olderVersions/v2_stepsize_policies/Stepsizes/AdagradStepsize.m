classdef AdagradStepsize < StepsizePolicy
    properties
        epsilon = 1e-3
        v_u = 0
    end
    methods
        function v_eta_out = update_stepsize(obj, v_g, ~)
            obj.k = obj.k+1;
            obj.v_u = obj.v_u +v_g.^2;
            obj.v_eta = obj.eta_0/(sqrt(obj.v_u)+obj.epsilon);
            v_eta_out = obj.v_eta;
        end
    end
end