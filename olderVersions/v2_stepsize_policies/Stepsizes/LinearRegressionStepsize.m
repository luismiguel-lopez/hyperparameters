classdef LinearRegressionStepsize < StepsizePolicy
    properties
        nu = 0.001;
        kappa = 3/200;
        gamma = 0.01;
        N % number of samples per epoch
        
        p_hat = 0
        x_0 = 0
        v_x_history
        r %rls estimator object
        sigma_hat
        zcr
        
        version= 4;
        law = 'sqrt';
        inv_law = @(x)x.^2;
        desiredRate = 0.5;
        v_corrs
        desired_corr = 0.9;
    end
    methods
        function v_eta_out = update_stepsize(obj, v_g, v_x_prev)
            assert(isscalar(v_g) && isscalar(v_x_prev), ...
                'not supported for vectors yet')
            if(isempty(obj.r))
                obj.r= rls(obj.beta_2);
                obj.r.set_nWeights(2);
                obj.v_eta = obj.eta_0;
                obj.x_0 = v_x_prev*obj.beta_2;
            end
            obj.k = obj.k+1;
            obj.v_x_history = [obj.v_x_history; v_x_prev];
            
            ww_in = [obj.p_hat; obj.x_0];
            err = v_x_prev -[obj.k 1]*ww_in;
            ww_out = obj.r.update_weights(ww_in, [obj.k;1], err);
            obj.p_hat = ww_out(1);
            obj.x_0   = ww_out(2);
            weights = obj.beta_2.^(obj.k-1-(1:obj.k-1));
            residuals = obj.v_x_history-[1:obj.k-1; ones(1, obj.k-1)]'*ww_out;
            sigma2_hat = weights/sum(weights)*residuals.^2;
            if obj.k > 2*obj.N
                m_corr = corrcoef(residuals(end-obj.N+1:end), ...
                    residuals((end-obj.N*2+1):(end-obj.N)));
                obj.v_corrs = [obj.v_corrs m_corr(2)];
            else
                obj.v_corrs = [obj.v_corrs obj.desired_corr];
            end
            obj.sigma_hat = sqrt(sigma2_hat);
            q = abs(obj.p_hat/obj.sigma_hat);
            p_hat_previous = ww_in(1);
            if isempty(obj.zcr)
                obj.zcr = obj.desiredRate./obj.N;
            end
            b_zeroCrossing= obj.p_hat*p_hat_previous < 0;
            obj.zcr = ((obj.k-1)*obj.zcr + b_zeroCrossing)/obj.k;
            
            switch obj.version
                case 1
                    obj.v_eta = obj.v_eta*(q./obj.kappa)^obj.nu;
                case 2
                    obj.kappa = max(1./obj.N^2, obj.kappa+...
                        obj.gamma.*(obj.zcr-obj.desiredRate./(obj.N)));
                    v_xi = 1/feval(obj.inv_law,obj.v_eta);
                    v_new_xi = v_xi - obj.nu*(q-obj.kappa);
                    if v_new_xi < v_xi/1.01 %risk of exploding
                        if abs(err) < obj.sigma_hat/5
                            v_new_xi = v_xi/1.01;
                        else
                            v_new_xi = v_xi;
                        end
                    end
                    obj.v_eta = 1/feval(obj.law, v_new_xi);
                case 3 % update kappa based on the zero crossing rate of p_hat
                    obj.kappa = obj.kappa./(1-obj.gamma.*(obj.zcr-obj.desiredRate./(obj.N)));
                    obj.v_eta = obj.v_eta.*(1-obj.nu*(q-obj.kappa));
                case 4 % update kappa based on the correlation coefficient
                       % between the prediction residuals during the last two epochs
                    obj.kappa = obj.kappa./(1+obj.gamma.*(obj.v_corrs(end)-obj.desired_corr));
                    obj.v_eta = obj.v_eta.*(1-obj.nu*(q-obj.kappa));
                case 5 % update the stepsize trying for the x_0 and the x_t|t to be at sigma_hat distance from each other
                    obj.v_eta = obj.v_eta./(1+obj.nu.*( ...
                        obj.k.*abs(obj.p_hat)-obj.kappa.*obj.sigma_hat ));
                case 6 % same as case 5, but different kind of update
                    v_xi = 1/feval(obj.inv_law,obj.v_eta);
                    v_new_xi = v_xi - obj.nu*(...
                        obj.k.*abs(obj.p_hat)-obj.kappa.*obj.sigma_hat );
                    % TODO: check if it makes sense with sqrt(obj.k)
                    v_new_xi = max(v_new_xi,v_xi/1.01);
%                     if v_new_xi < v_xi/1.01 %risk of exploding
%                         if abs(err) < obj.sigma_hat/5
%                             v_new_xi = v_xi/1.01;
%                         else
%                             v_new_xi = v_xi;
%                         end
%                     end
                    obj.v_eta = 1/feval(obj.law, v_new_xi);
                case 7 %quadratic function (probably does not make sense)
                    % TODO: use method give_me_params
                    
                    % These lines were taken from the old version of
                    % OHO_Lasso
                    q = abs(p_hat/sigma_hat);
                        factor = lrag_a*q^2 + lrag_b*q + obj.lrag_rmax;
                        v_eta(k) = v_eta(k-1)*factor;
            end
            v_eta_out = obj.v_eta;
        end
        
        function plot_state(obj)
            cla;
            plot(obj.v_x_history);
            hold on
            ax = axis;
            ww = [obj.p_hat; obj.x_0];
            plot(1:obj.k, [1:obj.k; ones(1, obj.k)]'*ww, 'g');
            plot(1:obj.k, [1:obj.k; ones(1, obj.k)]'*ww + obj.sigma_hat, 'r');
            plot(1:obj.k, [1:obj.k; ones(1, obj.k)]'*ww - obj.sigma_hat, 'r');
            axis(ax);
        end
    end
    
    methods (Static)
        function [a_out, b_out, c_out] = give_me_params(k,l, u)
            
            c_out = u;
            syms x a b
            assume(b<0)
            f(x) = a*x^2 + b*x + u;
            x_star = solve(diff(f, x)==0,x);
            assume(x_star < k)
            [a_out,b_out] = solve(f(k) == 1, f(x_star) == l, a, b);
        end
    end  
end