classdef ConstantStepsize < StepsizePolicy
    methods
        function v_eta_out = update_stepsize(obj, ~, ~)
            v_eta_out = obj.eta_0;
        end
    end
end