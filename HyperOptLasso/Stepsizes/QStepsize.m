classdef QStepsize < StepsizePolicy
    % Proposed by Luismi on Aug 10, 2019
    % Variance control with the Q function
    
    % This stepsize policy tries to keep the probability of the gradient
    % changing sign around a given level 0.5 - a small "delta". Under the
    % assumption that the gradient is Gaussian distributed, the qfunc is
    % calculated using the running estimates of the mean and variance of
    % the gradient.
    
    % TODO: several stepsize policies follow the same idea of keeping the
    % probability of the gradient inverting sign. They should be
    % children of a superclass that we can call
    % "VarianceControlStepsizePolicy"
    
    properties
        epsilon = 1e-6
        beta_1;

        v_v = 0;
        v_u = 0 
        nu          % "Stepsize of the stepsize"
        dqg = 0.01; % delta q goal: we want v_q \in 0.5 +- dqg
        v_q
        v_sigma
    end
    
    methods
        function v_eta_out = update_stepsize(obj, v_g, ~)
            obj.k = obj.k+1;
            if isempty(obj.v_eta)
                obj.v_eta = obj.eta_0; % it will normally enter 
                % the "if" only upon initialization
            end

            %running estimates of:
            obj.v_v = obj.beta_2*obj.v_v ...
                + (1-obj.beta_2)*v_g.^2;   % mean square gradient
            obj.v_u = obj.beta_1*obj.v_u ...
                + (1-obj.beta_1)*v_g;      % average gradient
            v_sigma2 = obj.v_v-obj.v_u.^2; % variance of gradient
            if v_sigma2<0 
                warning 'Obtained a negative variance estimate'
                obj.v_q = 0.5+obj.dqg; % in the unlikely case of a negative 
                % variance estimate, we set v_q to so the increase factor is 1
            else
                obj.v_sigma = sqrt(v_sigma2); %std of gradient
                obj.v_q = qfunc(obj.v_u./obj.v_sigma); % probability that 
                % the stochastic gradient takes a value with a sign 
                % opposite to that of the average gradient
            end
            
            eta_ftml = (1-obj.beta_1)./(1-obj.beta_1.^obj.k) ...
                .* obj.eta_0./...
                (sqrt(obj.v_v/(1-obj.beta_2^obj.k))+obj.epsilon); 
            % the stepsize that FTML would produce is used as an upper cap 
            % to the stepsize of this variance control algorithm
            increase_factor = 1./(1-obj.nu.*...
                (obj.v_q-0.5+obj.dqg).*(obj.v_q-0.5-obj.dqg));
            obj.v_eta = min(eta_ftml, obj.v_eta.*increase_factor);
            % TODO: the increase factor property will probably be common
            % to all variance control stepsize policies
            v_eta_out = obj.v_eta;
        end
    end
end