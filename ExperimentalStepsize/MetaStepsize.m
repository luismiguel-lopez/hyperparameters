classdef MetaStepsize < StepsizePolicy
    properties
       m_g_history
       m_stepsizes 
       degree = 3;
    end
    
    methods
        function v_eta_out = update_stepsize(obj, v_gradient)
            my_degree = obj.degree;
            assert(iscolumn(v_gradient));
            d = length(v_gradient);
            if isempty(obj.m_g_history) %initialize
                obj.m_g_history = repmat(v_gradient, [1 my_degree+1]);
            end
            if isempty(obj.m_stepsizes)
                obj.m_stepsizes = zeros(length(v_gradient),  my_degree);
                obj.m_stepsizes(:, 1) = obj.eta_0;
            end

            assert(my_degree == size(obj.m_g_history,2)-1);
            assert (size(obj.m_stepsizes,2)==my_degree);
            assert(d==size(obj.m_g_history, 1));
            
            % update gradient history (buffer)
            obj.m_g_history(:, 1:end-1) = obj.m_g_history(:, 2:end);
            obj.m_g_history(:, end) = v_gradient;
            
            if my_degree==3
                obj.m_stepsizes(:, end-1) = obj.m_stepsizes(:, end-1) + ...
                    obj.m_stepsizes(:, end-2).*obj.m_g_history(:,end-1).*...
                obj.m_g_history(:,end-2).^2.* obj.m_g_history(:,end-3);
                %alpha_t= alpha_t-1 + beta_t .*nabla_t-1.* (nabla_t-2)^2
                % .* nabla_t-3
            end
            if my_degree>=2
                obj.m_stepsizes(:, end) = obj.m_stepsizes(:, end) + ...
                    obj.m_stepsizes(:, end-1).*obj.m_g_history(:,end-1).*...
                    obj.m_g_history(:,end-2);
            else
                error 'my_degree must be greater or equal to 2'
            end
            % eta_t = eta_t-1 + alpha_t .* nabla_t-1 .* nabla_t-2
            
            obj.m_stepsizes = max(0, obj.m_stepsizes);
            
            v_eta_out = obj.m_stepsizes(:, end);
        end
    end
end
    