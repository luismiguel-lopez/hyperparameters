classdef SmoothStepsize < StepsizePolicy
    properties
        F % factor of decrease after N steps
        N % number of steps that takes to reduce in factor F
    end
    methods
        function v_eta_out = update_stepsize(obj, ~, ~)
            obj.k = obj.k+1;           
            if obj.k<obj.N
                obj.v_eta = obj.eta_0./exp(log(obj.F)*...
                    ((1-cos(pi*obj.k/obj.N))/2));
            else
                obj.v_eta = obj.eta_0./obj.F;
            end
            v_eta_out = obj.v_eta;
        end
    end
end