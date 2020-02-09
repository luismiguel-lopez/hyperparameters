classdef CrazyStepsize < StepsizePolicy
    properties
       m_g_history
       m_stepsizes 
    end
    
    methods
        function v_eta_out = update_stepsize(obj, v_gradient)
            my_degree = size(obj.m_g_history,2);
            d = size(obj.m_g_history, 1);
            assert(iscolumn(v_gradient) && length(v_gradient)==d);
            m_p = pascal(my_degree); %pascal matrix contains the exponents
            for k = 2:my_degree
                v_p = diag(m_p, k); %select secondary diagonal with correct
                % exponents
                m_stepsizes(k,:) = m_stepsizes(k,:) + m_stepsizes(k-1,:).*...
                    prod(m_g_history.^m_p(:,my_degree+1-k));
                % TODO: implement first a simple version with small degree
                % and hardcoded updates, and then make the most general one
            end
        end
    end
end
    