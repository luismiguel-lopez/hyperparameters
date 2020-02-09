classdef HyperLasso2
    properties
        max_iter_outer = 20000;
        max_iter_inner = 10;
        beta = 1000;
        b_mem = 1;
        gamma = 0;
        b_shuffle = 0;
        b_implement_24 = 0;
    end
    
    methods
        function [v_w_j, lambda, lambda_24] = solve(obj, m_X, v_y)
            % Solves y = X*w using Lasso with hyperparameter optimization
            max_iter = obj.max_iter_outer;
            assert(iscolumn(v_y))
            N = length(v_y);
            assert(size(m_X, 1)==N);
            P = size(m_X, 2);
            m_Phi = m_X'*m_X;
            v_r   = m_X'*v_y;
            alpha = 1/trace(m_Phi);
            lambda_max = max(abs(v_r));
            lambda = zeros(max_iter, 1);
            lambda_24 = zeros(max_iter, 1);
            lambda(1) = lambda_max/100;
            lambda_avg = 0;
            if obj.b_mem
                v_w = zeros(P, 1);
                for k_inner = 1:obj.max_iter_outer
                    w_f = v_w - alpha*(m_Phi*v_w - v_r);
                    v_w = obj.soft_thresholding(w_f, alpha*lambda(1));
                    % TODO: check convergence and exit loop
                end
                m_W = repmat(v_w, [1 N]);
            else
                v_w_j = zeros(P, 1);
            end
            m_S = zeros(P, N);
            beta_seq = obj.beta./sqrt(1:max_iter)';
            if obj.b_shuffle
                v_j = randi(N, max_iter);
            else
                v_j = mod(0:max_iter-1, N)+1;
            end
            for k = 2:max_iter
                %j = mod(k-1, N)+1;
                j = v_j(k);
                v_x_j = m_X(j,:)';
                m_Phi_j = m_Phi - v_x_j * v_x_j';
                v_r_j   = v_r   - v_y(j)* v_x_j;
                % obtain w(k) via ISTA
                if obj.b_mem
                    v_w_j = m_W(:,j);
                end
                for k_inner = 1:obj.max_iter_inner
                    w_f = v_w_j - alpha*(m_Phi_j*v_w_j - v_r_j);
                    v_w_j = obj.soft_thresholding(w_f, alpha*lambda_avg);
                    % TODO: check convergence and exit loop
                end
                m_W(:,j) = v_w_j;
                v_s_j = max(-1, min(1, 1/(alpha*lambda(k-1))*w_f));
                m_S(:,j) = v_s_j;
                d = alpha*v_x_j'*v_s_j*(v_y(j)-v_x_j'*(v_w_j - alpha*(...
                    m_Phi_j*v_w_j - v_r_j + lambda(k-1)*v_s_j)));
                paren = (v_y(j) - v_x_j'*v_w_j)/alpha + ...
                    v_x_j'*(m_Phi_j*v_w_j - v_r_j);
                num_summands(j) = -v_x_j'*v_s_j*paren;
                den_summands(j) = (v_x_j'*v_s_j)^2;
                if d*beta_seq(k) < -0.9
                    warning 'negative d times beta too large: rate of increase capped at 10'
                    % beta(k) = beta./abs(2*beta(k)*d); % this is the old line
                    lambda(k) = lambda(k-1)*10;
                else
                    lambda(k) = lambda(k-1)/(1+beta_seq(k)*d);
                end
                lambda_avg = (1-obj.gamma)*lambda(k) + obj.gamma*lambda_avg;
%                 if obj.b_mem
%                     lambda_24(k) = obj.eval_eq_24(...
%                         m_X, m_S, m_W, v_y, m_Phi, v_r, alpha);
%                 end
% %         The previous 4 lines are too slow
                lambda_24(k) = sum(num_summands)/sum(den_summands);
                if obj.b_implement_24
                    assert(obj.b_mem>0, 'b_mem must be active to implement eq. 24');
                    if abs(lambda_24(k)-lambda_avg) < 0.25*lambda_avg
                        lambda_avg = lambda_24(k);
                    end
                end
            end
            v_w_j = mean(m_W, 2);
            %figure(2); imagesc(m_W)
%             if obj.b_mem
%                 figure(3); plot([lambda_24, lambda_24_alt'])
%             end
        end
        
    end
    methods (Static)
        function lambda_out = eval_eq_24(m_X, m_S, m_W, v_y, m_Phi, v_r, alpha) 
            % optimization of lambda without any proximal term, eq. 24
            assert(iscolumn(v_y))
            N = length(v_y);
            assert(size(m_X, 1)==N);
            
            num = 0;
            den = 0;
            for j = 1:N
                v_x_j = m_X(j,:)';
                v_s_j = m_S(:,j);
                v_w_j = m_W(:,j);
                m_Phi_j = m_Phi - v_x_j * v_x_j';
                v_r_j = v_r   - v_y(j)* v_x_j;
                paren = (v_y(j) - v_x_j'*v_w_j)/alpha + ...
                    v_x_j'*(m_Phi_j*v_w_j - v_r_j);
                num = num - v_x_j'*v_s_j*paren;
                den = den + (v_x_j'*v_s_j)^2;
            end
            lambda_out = num/den;
        end
        
        function w_out = soft_thresholding(w_in, rho)
            v_factors = max(0, 1-rho./abs(w_in));
            w_out = w_in.*v_factors;
        end
    end
    
end