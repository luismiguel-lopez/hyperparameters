classdef IncreasingStepsize < StepsizePolicy
    properties
        law = 'sqrt'
    end
    methods
        function v_eta_out = update_stepsize(obj, ~, ~)
            obj.k = obj.k+1;
            obj.v_eta = obj.eta_0*feval(obj.law, obj.k);
            v_eta_out = obj.v_eta;
        end
    end
end