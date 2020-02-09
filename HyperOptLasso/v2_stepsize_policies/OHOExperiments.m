classdef OHOExperiments
    properties
        n_train = 200;
        n_test  = 2000;
        p       = 100;
        seed    = 3;
    end
      
methods
    function [A_train, A_test, y_train, y_test, true_x] = set_up_data(obj, ntrain, ntest)
        rng(3);

        A_train = randn(obj.n_train, obj.p); % sensing matrix
        A_test = randn(obj.n_test, obj.p);
        true_x = double(randn(obj.p, 1) > 1); % sparse coefficients
        epsilon = randn(obj.n_train, 1) * 0.3; % noise
        epsilon_test = randn(obj.n_test, 1) * 0.3;
        y_train = A_train * true_x + epsilon;
        y_test  = A_test * true_x + epsilon_test;
    end
end
    
methods
    
    function F = experiment_1(obj)
        % Approximate gradient vs mirror (not online)
        [A_train, A_test, y_train, y_test, true_x] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.beta = 50;
        hl.mirror_type = 'grad';        
        lambda_grad = hl.solve_approx_mirror(A_train, y_train);
        hl.mirror_type = 'log';
        hl.beta = 5;
        lambda_log = hl.solve_approx_mirror(A_train, y_train);
        figure(1); clf
        plot([lambda_grad lambda_log]);
        %plot(lambda_onlinemirror)
        legend({'Grad', 'Mirror (log)'})
        F = 0;
    end
    
    function F = experiment_2(obj)
        % Approximate mirror vs Online mirror
        [A_train, A_test, y_train, y_test, true_x] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'log';
        lambda_log = hl.solve_approx_mirror(A_train, y_train);
        hl.b_online = 1;
        hl.max_iter_outer= 10000;
        hl.beta = 500;
        lambda_onlinemirror = hl.solve_approx_mirror(A_train, y_train);
        
        figure(1); clf
        plot([lambda_log lambda_onlinemirror]);
        legend({'Batch', 'Online'})
        F = 0;
    end
    
    function F = experiment_11(obj)
        % Online mirror with diminishing stepsize
        % testing object-based stepsize policies
        [A_train, A_test, y_train, y_test, true_x] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'log';
        hl.b_online = 1;
        hl.max_iter_outer= 1000;
        hl.stepsize_policy_object.eta_0 = 500;
        v_lambda = hl.solve_approx_mirror(A_train, y_train);
        F = 0;
    end
    
    function F = experiment_12(obj)
        % Online mirror with AdaGrad stepsize
        % testing object-based stepsize policies
        [A_train, A_test, y_train, y_test, true_x] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'log';
        hl.b_online = 1;
        hl.max_iter_outer= 10000;
        hl.stepsize_policy = 'adagrad';
        hl.stepsize_policy_object = AdagradStepsize;
        hl.stepsize_policy_object.eta_0 = 0.05;
        hl.stepsize_policy_object.epsilon = 1/500;
        v_lambda = hl.solve_approx_mirror(A_train, y_train);
        F = 0;
    end
    
    function F = experiment_13(obj)
        % Online mirror with RmsProp stepsize
        % testing object-based stepsize policies
        [A_train, A_test, y_train, y_test, true_x] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'log';
        hl.b_online = 1;
        hl.max_iter_outer= 1000;
        hl.stepsize_policy = 'rmsprop';
        hl.stepsize_policy_object = RmsPropStepsize;
        hl.stepsize_policy_object.eta_0 = 0.003;
        hl.stepsize_policy_object.epsilon = 1/500;
        v_lambda = hl.solve_approx_mirror(A_train, y_train);
        % todo: maybe if we reduce the beta_2 param to 0.9 we get faster
        % convergence?
        F = 0;
    end

    function F = experiment_14(obj)
        % Online mirror with LR-based stepsize
        % testing object-based stepsize policies
        obj.n_train = 200;
        [A_train, A_test, y_train, y_test, true_x] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'log';
        hl.b_online = 1;
        hl.max_iter_outer= 10000;
        hl.stepsize_policy = 'linear_regression';
        hl.lrag_rmin = 0;
        hl.stepsize_policy_object = LinearRegressionStepsize;
        hl.stepsize_policy_object.eta_0 = 10;%! 500;
        hl.stepsize_policy_object.beta_2 = 1-1/obj.n_train;
        hl.stepsize_policy_object.nu = 0.1;
        hl.stepsize_policy_object.kappa = 1/obj.n_train;
        hl.stepsize_policy_object.gamma = 0.;
        hl.stepsize_policy_object.N = obj.n_train;
        hl.stepsize_policy_object.law = @(x)x;
        hl.stepsize_policy_object.inv_law = @(x)x;
        v_lambda = hl.solve_approx_mirror(A_train, y_train);
        F = 0;
    end

    
    function F = experiment_20(obj)
        % Online mirror with ADAGrad stepsize
        [A_train, A_test, y_train, y_test, true_x] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'log';
        hl.b_online = 1;
        hl.max_iter_outer= 40000;
        hl.stepsize_policy = 'linear_regression';
        lambda_adagrad= hl.solve_approx_mirror(A_train, y_train);
        
        figure(1); clf
        plot(lambda_adagrad);
    end
    
    function F = experiment_21(obj)
        % Online mirror with Adam-like adapted stepsize
        [A_train, A_test, y_train, y_test, true_x] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.max_iter_outer= 10000;
        hl.stepsize_policy = 'adam';
        spo_adam = AdamStepsize;
        spo_adam.eta_0 = 0.001;
        spo_adam.beta_2= 0.99;
        hl.stepsize_policy_object = spo_adam;       
        lambda_adagrad= hl.solve_approx_mirror(A_train, y_train);
        
        figure(1); clf
        plot(lambda_adagrad);
    end
    
    function F = experiment_22(obj)
        % Online mirror with U-FTML adapted stepsize
        [A_train, A_test, y_train, y_test, true_x] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.max_iter_outer= 10000;
        hl.stepsize_policy = 'uftml';
        spo_uftml = UftmlStepsize;
        spo_uftml.eta_policy = DiminishingStepsize;
        %spo_uftml.eta_policy.law = @(x)x; %1/k stepsize
        spo_uftml.eta_policy.eta_0 = 200;
        spo_uftml.beta_2= 0.99;
        hl.stepsize_policy_object = spo_uftml;       
        lambda_adagrad= hl.solve_approx_mirror(A_train, y_train);
        
        figure(1); clf
        plot(lambda_adagrad);
    end
    
    function F = experiment_31(obj)
        % Online mirror with LR-based stepsize
        % testing object-based stepsize policies
        obj.n_train = 2000;
        [A_train, A_test, y_train, y_test, true_x] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'log';
        hl.b_online = 1;
        hl.max_iter_outer= 10000;
        hl.stepsize_policy = 'linear_regression';
        hl.lrag_rmin = 0;
        hl.stepsize_policy_object = LinearRegressionStepsize;
        hl.stepsize_policy_object.eta_0 = 10;%! 500;
        hl.stepsize_policy_object.beta_2 = 1-1/obj.n_train;
        hl.stepsize_policy_object.nu = 0.01;
        hl.stepsize_policy_object.kappa = 5;
        hl.stepsize_policy_object.version = 6;
        v_lambda = hl.solve_approx_mirror(A_train, y_train);
        F = 0;
    end

    
    
    function F = experiment_1001(obj)
        [A_train, A_test, y_train, y_test, true_x] = obj.set_up_data();

        hl_mem = HyperLasso2; %memory
        hl_nomem = HyperLasso2;
        hl_nomem.b_mem = 0; %memoryless
        hl_mem_shuffle = HyperLasso2;
        hl_mem_shuffle.b_shuffle = 1; %memory, shuffle
        hl_nomem_shuffle = HyperLasso2;
        hl_nomem_shuffle.b_shuffle = 1;
        hl_nomem_shuffle.b_mem = 0; %memoryless, shuffle
        hl_mem_impl24 = HyperLasso2;
        hl_mem_impl24.b_implement_24 = 1;
        
        [beta_nomem, lambda_nomem, lambda_24_nomem] = hl_nomem.solve(A_train, y_train);
        
        [beta_mem, lambda_mem, lambda_24_mem] = hl_mem.solve(A_train, y_train);
        
        [beta_nomem_shuffle, lambda_nomem_shuffle, lambda_24_nomem_shuffle] = ...
            hl_nomem_shuffle.solve(A_train, y_train);
        
        [beta_mem_shuffle, lambda_mem_shuffle, lambda_24_mem_shuffle] = ...
            hl_mem_shuffle.solve(A_train, y_train);
        
        [beta_mem_impl24, lambda_mem_impl24, lambda_24_mem_impl24] = ...
            hl_mem_impl24.solve(A_train, y_train);
        
        
        %%
        figure(1); clf
        plot([lambda_nomem lambda_24_nomem])
        hold on
        plot([lambda_mem lambda_24_mem])
        plot([lambda_nomem_shuffle lambda_24_nomem_shuffle])
        plot([lambda_mem_shuffle lambda_24_mem_shuffle])
        plot([lambda_mem_impl24 lambda_24_mem_impl24], '--');
        
        my_cap = max(lambda_mem(ceil(9*length(lambda_mem)/10)));
        ylim(my_cap*[1/2,2])
        legend('Stoch. memoryless', 'Eq. 24, memoryless', ...
            'Stoch. memory',  'Eq. 24, memory', ...
            'Stoch. ml+shuffle', 'Eq. 24, ml+shuffle', ...
            'Stoch. memory+shuffle',  'Eq. 24, memory+shuffle', ...
            'Trust region', 'Eq.24, trust region')
        
        %TODO:try a non-stochastic proximal algorithm that minimizes the linear
        %approximation of the cost function minimized by (24) plus a proximal term,
        %that may be quadratic (gradient descent) or not (mirror descent).
        
        %%
        figure(2); clf
        [beta, fitinfo] = lasso(A_train, y_train, 'lambda', ...
            linspace(1, 10, 100)/n_train);
        test_mse = zeros(size(beta, 2), 1);
        for k = 1:size(beta, 2)
            yhat_test = A_test * beta(:, k);
            test_mse(k) = mean((y_test-yhat_test).^2);
        end
        plot(fitinfo.Lambda*n_train, test_mse);
        
        % TODO: compare with AloCV
        % TODO: compare with exact Leave-One-Out
        
    end
end

end