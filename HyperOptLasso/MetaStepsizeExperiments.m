classdef MetaStepsizeExperiments
    %UNTITLED5 Summary of this class goes here
    %   Detailed explanation goes here
    
    
    methods
        function experiment_1(obj)
            d = 30;
            niter = 10000;
            seed = 1;
            n_eta = 20;

            epsilon = 1e-3;
            
            rng(seed)
            m_B_asym = randn(d);
            m_B_sym = m_B_asym + m_B_asym';
            [V, m_D] = eig(m_B_sym);
            L = max(diag(m_D));
            mu = min(diag(m_D));
            D_badCond = diag(epsilon +diag((L*(m_D-mu))./(L-mu)));
            m_A = V*D_badCond*V';
            true_x = randn(d,1);
            v_b = m_A*true_x;
            
            % minimize 1/2||Ax-b||^2
            m_x = zeros(d,niter);
            m_g = m_x;
            m_eta = m_x;
            v_cost = zeros(niter, 1);
            gradient = @(v_x) m_A'*(m_A*v_x - v_b);
            v_eta = logspace(-21, -15, n_eta);
            converged = 0;
            figure(1); clf
            for kk = 1:n_eta
                ms = MetaStepsize();
                ms.degree = 3;
                ms.eta_0 = v_eta(kk);
                for k = 1:niter
                    m_g(:,k) = gradient(m_x(:,k));
                    m_eta(:,k) = ms.update_stepsize(m_g(:,k));
                    v_cost(k) = 1/2*norm(m_A*m_x(:,k)-v_b).^2;
                    if converged || k==niter
                        break
                    end
                    m_x(:,k+1) = m_x(:,k) - m_eta(:,k).*m_g(:,k);
                end
                v_nmse = 2*v_cost./norm(v_b).^2;
                if max(v_nmse)>10
                    break
                end
                semilogy(v_nmse); hold on
                c_legend{kk} = string(ms.eta_0);
            end
            legend(c_legend)
        end
        
        function experiment_2(obj)
            d = 30;
            niter = 10000;
            seed = 1;
            n_eta = 20;

            epsilon = 1e-3;
            
            rng(seed)
            m_B_asym = randn(d);
            m_B_sym = m_B_asym + m_B_asym';
            [V, m_D] = eig(m_B_sym);
            L = max(diag(m_D));
            mu = min(diag(m_D));
            D_badCond = diag(epsilon +diag((L*(m_D-mu))./(L-mu)));
            m_A = V*D_badCond*V';
            true_x = randn(d,1);
            v_b = m_A*true_x;
            
            % minimize 1/2||Ax-b||^2
            m_x = zeros(d,niter);
            m_g = m_x;
            m_eta = m_x;
            v_cost = zeros(niter, 1);
            gradient = @(v_x) m_A'*(m_A*v_x - v_b);
            v_eta = logspace(-9, -6, n_eta);
            converged = 0;
            figure(2); clf
            for kk = 1:n_eta
                ms = MetaStepsize();
                ms.degree = 2;
                ms.eta_0 = v_eta(kk);
                for k = 1:niter
                    m_g(:,k) = gradient(m_x(:,k));
                    m_eta(:,k) = ms.update_stepsize(m_g(:,k));
                    v_cost(k) = 1/2*norm(m_A*m_x(:,k)-v_b).^2;
                    if converged || k==niter
                        break
                    end
                    m_x(:,k+1) = m_x(:,k) - m_eta(:,k).*m_g(:,k);
                end
                v_nmse = 2*v_cost./norm(v_b).^2;
                if max(v_nmse)>10
                    break
                end
                semilogy(v_nmse); hold on
                c_legend{kk} = string(ms.eta_0);
            end
            legend(c_legend)
        end
    end
end

