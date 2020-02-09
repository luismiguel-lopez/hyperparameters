classdef OHOExperiments
    properties
        n_train = 200;
        n_test  = 2000;
        p       = 100;
        seed    = 3;
    end
      
methods
    function [A_train, A_test, y_train, y_test, true_x] = ...
            set_up_data(obj)
        
        rng(obj.seed);
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
        hl.stepsize_policy = DiminishingStepsize;
        hl.stepsize_policy.eta_0 = 50;
        hl.mirror_type = 'grad';        
        lambda_grad = hl.solve_approx_mirror(A_train, y_train);
        hl.mirror_type = 'log';
        hl.stepsize_policy.eta_0 = 5;
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
        hl.stepsize_policy = DiminishingStepsize;
        hl.stepsize_policy.eta_0 = 5;
        lambda_log = hl.solve_approx_mirror(A_train, y_train);
        
        hl.b_online = 1;
        hl.max_iter_outer= 10000;
        hl.stepsize_policy.eta_0 = 500;
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
        hl.stepsize_policy.eta_0 = 500;
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
        hl.stepsize_policy = AdagradStepsize;
        hl.stepsize_policy.eta_0 = 0.1;
        hl.stepsize_policy.epsilon = 1e-5;
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
        hl.max_iter_outer= 10000;
        hl.stepsize_policy = RmsPropStepsize;
        hl.stepsize_policy.beta_2 = 0.9;
        hl.stepsize_policy.eta_0 = 0.01;
        hl.stepsize_policy.epsilon = 1e-6;
        
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
        hl.stepsize_policy = LinearRegressionStepsize;
        hl.stepsize_policy.version = 6;
        hl.stepsize_policy.eta_0 = 10;%! 500;
        hl.stepsize_policy.beta_2 = 1-1/obj.n_train;
        hl.stepsize_policy.nu = 0.1;
        hl.stepsize_policy.kappa = 1/obj.n_train;
        hl.stepsize_policy.gamma = 0.;
        hl.stepsize_policy.N = obj.n_train;
        hl.stepsize_policy.law = @(x)x;
        hl.stepsize_policy.inv_law = @(x)x;
        v_lambda = hl.solve_approx_mirror(A_train, y_train);
        F = 0;
    end
    
    function F = experiment_15(obj)
        % Online mirror with LR-based stepsize
        % testing object-based stepsize policies
        obj.n_train = 200;
        [A_train, A_test, y_train, y_test, true_x] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'log';
        hl.b_online = 1;
        hl.max_iter_outer= 10000;
        hl.stepsize_policy = LinearRegressionStepsize;
        hl.stepsize_policy.version = 6;
        hl.stepsize_policy.eta_0 = 50;%! 500;
        hl.stepsize_policy.beta_2 = 1-1/obj.n_train;
        hl.stepsize_policy.nu = 0.01;
        hl.stepsize_policy.kappa = 6;
        v_lambda = hl.solve_approx_mirror(A_train, y_train);
        F = 0;
    end

    
    function F = experiment_21(obj)
        % Online mirror with Adam-like adapted stepsize
        [A_train, A_test, y_train, y_test, true_x] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.max_iter_outer= 10000;
        spo_adam = AdamStepsize;
        spo_adam.eta_0 = 0.03;
        spo_adam.beta_2= 0.99;
        hl.stepsize_policy = spo_adam;       
        lambda_adagrad= hl.solve_approx_mirror(A_train, y_train);
        
        figure(1); clf
        plot(lambda_adagrad);
    end
    
    function F = experiment_22(obj)
        % Online mirror with U-FTML adapted stepsize
        obj.n_train = 200;
        [A_train, A_test, y_train, y_test, true_x] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.max_iter_outer= 10000;
        spo_uftml = UftmlStepsize;
        spo_uftml.eta_policy = DiminishingStepsize;
        spo_uftml.eta_policy.law = @(x)x; %1/k stepsize
        spo_uftml.eta_policy.eta_0 = 500;
        spo_uftml.beta_2= 1-1/obj.n_train;
        spo_uftml.beta_1 = 0.9;
        hl.stepsize_policy = spo_uftml;       
        lambda_adagrad= hl.solve_approx_mirror(A_train, y_train);
        
        figure(1); clf
        plot(lambda_adagrad);
    end
    
    function F = experiment_23(obj)
        % Online mirror with U-FTML adapted stepsize
        % Now with 10 times more samples!
        obj.n_train = 2000; %! 2000
        [A_train, A_test, y_train, y_test, true_x] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.max_iter_outer= 10000;
        spo_uftml = UftmlStepsize;
        spo_uftml.eta_policy = ConstantStepsize;
        %spo_uftml.eta_policy.law = @(x)x;%^(1/3); %1/k stepsize
        spo_uftml.eta_policy.eta_0 = 10;
        spo_uftml.beta_2 = 0.995;   %1-1/(obj.n_train);
        spo_uftml.beta_1 = 0.99;  %1-10/obj.n_train;
        hl.stepsize_policy = spo_uftml;       
        lambda_adagrad= hl.solve_approx_mirror(A_train, y_train);
        
        figure(1); clf
        plot(lambda_adagrad);
    end
    
    function F = experiment_24(obj)
        % Online mirror with Luismi's Q stepsize
        obj.n_train = 1000; %! 2000
        obj.seed = 4;
        [A_train, A_test, y_train, y_test, true_x] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.max_iter_outer= 20000;
        spoq = QStepsize;
        %spo_uftml.eta_policy.law = @(x)x;%^(1/3); %1/k stepsize
        spoq.eta_0 = 100;
        spoq.nu = 5;
        spoq.beta_2 = 1-1/(obj.n_train);
        spoq.beta_1 = 1-1/(obj.n_train);
        spoq.N = obj.n_train;
        hl.stepsize_policy = spoq;       
        lambda_adagrad= hl.solve_approx_mirror(A_train, y_train);
        
        figure(1); clf
        plot(lambda_adagrad);
    end
    
    function F = experiment_25(obj)
        % Online gradient descent with Luismi's Q stepsize
        obj.n_train = 1000; %! 2000
        obj.seed = 4;
        [A_train, A_test, y_train, y_test, true_x] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.max_iter_outer= 20000;
        spoq = QStepsize;
        %spo_uftml.eta_policy.law = @(x)x;%^(1/3); %1/k stepsize
        spoq.eta_0 = 20;
        spoq.nu = 10;
        spoq.beta_2 = 1-1/(obj.n_train);
        spoq.beta_1 = 1-1/(obj.n_train);
        hl.stepsize_policy = spoq;       
        lambda_adagrad= hl.solve_approx_mirror(A_train, y_train);
        
        figure(25); clf
        plot(lambda_adagrad);
    end
    
    function F = experiment_31(obj)
        % Memoryless vs. alg. with memory
        obj.n_train = 4000;
        obj.p = 100;
        obj.seed = 2;
        [A_train, A_test, y_train, y_test, true_x] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.max_iter_outer= 20000;
        memory_factor= 1;
        spoq = QStepsize;
        spoq.eta_0 = 200;
        spoq.nu = 40;
        spoq.beta_2 = 1-1/(memory_factor*obj.n_train); % 10*
        spoq.beta_1 = 1-1/(memory_factor*obj.n_train); % 10*
        hl.stepsize_policy = spoq;     
        hl.debug= 0;
        hl.normalized_lambda_0 = 1/obj.n_train;
        [lambda_memory, count_memory]    ...
            = hl.solve_approx_mirror(A_train, y_train);
        hl.b_memory = 0;
        spoq2 = QStepsize;
        spoq2.eta_0 = 200;
        spoq2.nu = 40;
        spoq2.beta_2 = 1-1/(memory_factor*obj.n_train); % 10*
        spoq2.beta_1 = 1-1/(memory_factor*obj.n_train); % 10*
        hl.stepsize_policy = spoq2;
        [lambda_memoryless, count_memoryless] ...
            = hl.solve_approx_mirror(A_train, y_train);             
        
        figure(31); clf
        plot([cumsum(count_memoryless) cumsum(count_memory)], ...
            [lambda_memoryless lambda_memory]);
        legend('Memoryless','With Memory')
    end
    
    function F = experiment_32(obj)
        % Memoryless, comparison between exact (let ista converge) and
        % inexact (few iterations of ista for each iteration of lambda)
        
        obj.n_train = 4000;
        obj.p = 400;
        obj.seed = 3;
        [A_train, A_test, y_train, y_test, true_x] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.max_iter_outer= 20000;%20000
        memory_factor= 1;
        spoq = QStepsize;
        spoq.eta_0 = 200;
        spoq.nu = 40;
        spoq.beta_2 = 1-1/(memory_factor*obj.n_train); % 10*
        spoq.beta_1 = 1-1/(memory_factor*obj.n_train); % 10*
        hl.stepsize_policy = spoq;     
        hl.debug= 0;
        hl.normalized_lambda_0 = 1/obj.n_train;
%         [lambda_memory, count_memory]    ...
%             = hl.solve_approx_mirror(A_train, y_train);
        hl.b_memory = 0;
        hl.tol = 1e-3;
        [lambda_exact, count_exact, v_w_exact] ...
            = hl.solve_approx_mirror(A_train, y_train);
        
        spoq2 = QStepsize;
        spoq2.eta_0 = 200;
        spoq2.nu = 40;
        spoq2.beta_2 = 1-1/(memory_factor*obj.n_train); % 10*
        spoq2.beta_1 = 1-1/(memory_factor*obj.n_train); % 10*
        hl.stepsize_policy = spoq2;
        hl.tol = 1e1;
        [lambda_inexact, count_inexact, v_w_inexact] ...
            = hl.solve_approx_mirror(A_train, y_train);
        figure(31); clf
        plot([cumsum(count_exact) cumsum(count_inexact)], ...
            [lambda_exact lambda_inexact]);
        legend('Exact','Inexact')
        
        % Evaluate test error
        final_lambda_inexact = lambda_inexact(find(count_inexact>0,1, 'last'));
        final_lambda_exact= lambda_exact(find(count_exact>0,1, 'last'));
        v_testLambdas = final_lambda_inexact*[0.01 0.1 0.3 0.5 0.8 0.9 1.1 1.25 2 3 10];
        v_testErrors = zeros(size(v_testLambdas));
        
        m_Phi = A_train'*A_train;
        v_r = A_train'*y_train;
        my_alpha = 10/trace(m_Phi);
        for k = 1:length(v_testLambdas)
            v_w_test = hl.ista(v_w_inexact, m_Phi, v_r, my_alpha, v_testLambdas(k));
            v_testErrors(k) = mean((y_test - A_test*v_w_test).^2);
        end
        
        test_error_exact = mean((y_test-A_test*v_w_exact).^2);
        test_error_inexact = mean((y_test-A_test*v_w_inexact).^2);
%         test_error_high = mean((y_test - A_test*v_w_high).^2);
%         test_error_low = mean((y_test - A_test*v_w_low).^2);
        
        figure(131); clf
        semilogx(v_testLambdas,  v_testErrors);hold on
        plot(final_lambda_inexact, test_error_inexact, 'xb')
        plot(final_lambda_exact, test_error_exact, 'xr'); 
        legend ('Test error', 'inexact', 'exact')
    end
     
    function F = experiment_33(obj)
        % Memoryless, inexact 
        % (few iterations of ista for each iteration of lambda)
        
        obj.n_train = 4000;
        obj.p = 400;
        obj.seed = 3;
        [A_train, A_test, y_train, y_test, true_x] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.max_iter_outer= 20000;
        memory_factor= 1;
        spoq = QStepsize;
        spoq.eta_0 = 200;
        spoq.nu = 40;
        spoq.beta_2 = 1-1/(memory_factor*obj.n_train);
        spoq.beta_1 = 1-1/(memory_factor*obj.n_train);
        hl.stepsize_policy = spoq;     
        hl.debug= 0;
        hl.normalized_lambda_0 = 1/obj.n_train;
        hl.b_memory = 0;
        hl.tol_g = 1e-3 %!
%         [lambda_exact, count_exact, v_w_exact] ...
%             = hl.solve_approx_mirror(A_train, y_train);
%         
%         spoq2 = QStepsize;
%         spoq2.eta_0 = 200;
%         spoq2.nu = 40;
%         spoq2.beta_2 = 1-1/(memory_factor*obj.n_train); % 10*
%         spoq2.beta_1 = 1-1/(memory_factor*obj.n_train); % 10*
%         hl.stepsize_policy = spoq2;
        hl.tol = 1e1;
        [lambda_inexact, count_inexact, v_w_inexact] ...
            = hl.solve_approx_mirror(A_train, y_train);
        figure(31); clf
        plot([cumsum(count_inexact)], ...
            [lambda_inexact]);
%         plot([cumsum(count_exact) cumsum(count_inexact)], ...
%             [lambda_exact lambda_inexact]);
        legend('Inexact') %,'exact'
        
        % Evaluate test error
        final_lambda_inexact = lambda_inexact(find(count_inexact>0,1, 'last'));
        %final_lambda_exact= lambda_exact(find(count_exact>0,1, 'last'));
        v_testLambdas = final_lambda_inexact*[0.01 0.1 0.3 0.5 0.8 0.9 1.1 1.25 2 3 10];
        v_testErrors = zeros(size(v_testLambdas));
        
        m_Phi = A_train'*A_train;
        v_r = A_train'*y_train;
        my_alpha = 10/trace(m_Phi);
        for k = 1:length(v_testLambdas)
            v_w_test = hl.ista(v_w_inexact, m_Phi, v_r, my_alpha, v_testLambdas(k));
            v_testErrors(k) = mean((y_test - A_test*v_w_test).^2);
        end
        
        %test_error_exact = mean((y_test-A_test*v_w_exact).^2);
        test_error_inexact = mean((y_test-A_test*v_w_inexact).^2);
%         test_error_high = mean((y_test - A_test*v_w_high).^2);
%         test_error_low = mean((y_test - A_test*v_w_low).^2);
        
        figure(131); clf
        semilogx(v_testLambdas,  v_testErrors);hold on
        plot(final_lambda_inexact, test_error_inexact, 'xb')
        %plot(final_lambda_exact, test_error_exact, 'xr'); 
        legend ('Test error', 'inexact') %, 'exact'
        xlabel \lambda
        ylabel MSE
    end    
    
    function F = experiment_34(obj)
        % Memoryless, exact
        % comparison with exact Leave-One-Out
        
        obj.n_train = 1000;
        obj.p = 100;
        obj.seed = 4;
        [A_train, A_test, y_train, y_test, true_x] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.max_iter_outer= 20000;
        memory_factor= 1;
        spoq = QStepsize;
        spoq.eta_0 = 200;
        spoq.nu = 20;
        spoq.beta_2 = 1-1/(memory_factor*obj.n_train);
        spoq.beta_1 = 1-1/(memory_factor*obj.n_train);
        hl.stepsize_policy = spoq;     
        hl.debug= 1;
        hl.normalized_lambda_0 = 1/sqrt(obj.n_train);
        hl.b_memory = 0;
        hl.tol = 1e-4;
        [lambda_exact, count_exact, v_w_exact] ...
            = hl.solve_approx_mirror(A_train, y_train);
        figure(34); clf
        plot(cumsum(count_exact), lambda_exact);
        xlabel '# ISTA iterations'
        ylabel '\lambda'
        legend('exact')
        drawnow
        
        % Evaluate test error
        final_lambda_exact = lambda_exact(find(count_exact>0,1, 'last'));
        v_testLambdas = final_lambda_exact*logspace(-2, 2);
        v_looLambdas = final_lambda_exact*linspace(0.8, 1.2, 7);
        v_testErrors = zeros(size(v_testLambdas));
        v_valErrors  = v_testErrors;
        v_looErrors  = zeros(size(v_looLambdas));
        
        disp 'Computing test errors'
        for k = 1:length(v_testLambdas)
            v_testErrors(k) = obj.estimate_outOfSample_error(A_train, ...
                y_train, A_test, y_test, v_testLambdas(k), v_w_exact);
            v_valErrors(k) = obj.estimate_outOfSample_error(A_test, ...
                y_test, A_train, y_train, v_testLambdas(k), v_w_exact);
        end
        test_error_exact = mean((y_test-A_test*v_w_exact).^2)./mean(y_test.^2);
        val_error_exact  = obj.estimate_outOfSample_error(A_test, ...
                y_test, A_train, y_train, final_lambda_exact, v_w_exact);
        disp 'Computing loo errors'
        for kl = 1:length(v_looLambdas)
            v_looErrors(kl) = obj.exact_loo(...
                A_train, y_train, v_looLambdas(kl), v_w_exact);
        end
        ind_final_lambda = find(v_looLambdas==final_lambda_exact, 1,'first');
        if isempty(ind_final_lambda)
            loo_error_exact = obj.exact_loo(A_train, y_train, ...
                final_lambda_exact, v_w_exact);
        else
            loo_error_exact = v_looErrors(ind_final_lambda);
        end
        figure(134); clf
        loglog(v_testLambdas, v_testErrors);hold on
        loglog(v_testLambdas, v_valErrors, 'r')
        loglog(v_looLambdas, v_looErrors,'g')
        plot(final_lambda_exact, test_error_exact, 'xb')
        plot(final_lambda_exact, val_error_exact, 'xr')
        plot(final_lambda_exact, loo_error_exact, 'xg')        
         
        legend ('Out-of-sample test error', ...
            'Out-of-sample validation error', 'Leave-one-out error', ...
            'exact')
        xlabel \lambda
        ylabel MSE
        
        save SavedResults_34
    end
    
    function F = experiment_35(obj)
        % Exact with memory vs. memoryless inexact
        % comparison with exact Leave-One-Out and Alocv
        % (approximate leave-one-out, Wang et al 2018)
        
        obj.n_train = 2000;
        obj.p = 200;
        obj.seed = 6;
        [A_train, A_test, y_train, y_test, true_x] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.max_iter_outer= 20000;
        memory_factor= 1;
        spoq = QStepsize;
        spoq.eta_0 = 200;
        spoq.nu = 20;
        spoq.beta_2 = 1-1/(memory_factor*obj.n_train);
        spoq.beta_1 = 1-1/(memory_factor*obj.n_train);
        hl.stepsize_policy = spoq;     
        hl.debug= 1;
        hl.normalized_lambda_0 = 1/sqrt(obj.n_train);
        hl.tol = 1e-4;
        hl.b_memory = 1;
        [lambda_exact, count_exact, v_w_exact] ...
            = hl.solve_approx_mirror(A_train, y_train);
        
        hl.tol = 1e0;
        hl.b_memory = 0;
        [lambda_inexact, count_inexact, v_w_inexact] ...
            = hl.solve_approx_mirror(A_train, y_train);

        figure(35); clf
        plot([cumsum(count_exact) cumsum(count_inexact)], ...
            [lambda_exact lambda_inexact]);
        xlabel '# ISTA iterations'
        ylabel '\lambda'
        legend('Exact with memory','Inexact, memoryless')
        drawnow
        
        %%
        % Evaluate test error
        final_lambda_exact = lambda_exact(find(count_exact>0, 1, 'last'));
        final_lambda_inexact = lambda_inexact(find(count_inexact>0, 1, 'last'));
        v_testLambdas = final_lambda_exact*logspace(-1, 0.5);
        v_looLambdas = final_lambda_exact*linspace(0.7, 1.3, 7);
        v_aloLambdas = final_lambda_exact*linspace(0.5, 1.5, 200);
        v_testErrors = zeros(size(v_testLambdas));
        v_valErrors  = v_testErrors;
        v_looErrors  = zeros(size(v_looLambdas));
        
        disp 'Computing test errors'
        for k = 1:length(v_testLambdas)
            v_testErrors(k) = obj.estimate_outOfSample_error(A_train, ...
                y_train, A_test, y_test, v_testLambdas(k), v_w_exact);
            v_valErrors(k) = obj.estimate_outOfSample_error(A_test, ...
                y_test, A_train, y_train, v_testLambdas(k), v_w_exact);
        end
        test_error_exact = mean((y_test-A_test*v_w_exact).^2)./mean(y_test.^2);
        test_error_inexact = mean((y_test-A_test*v_w_inexact).^2)./mean(y_test.^2);
        
        val_error_exact  = obj.estimate_outOfSample_error(A_test, ...
            y_test, A_train, y_train, final_lambda_exact, v_w_exact);
        val_error_inexact  = obj.estimate_outOfSample_error(A_test, ...
            y_test, A_train, y_train, final_lambda_inexact, v_w_inexact);

        disp 'Computing loo errors'
        for kl = 1:length(v_looLambdas)
            v_looErrors(kl) = obj.exact_loo(...
                A_train, y_train, v_looLambdas(kl), v_w_exact);
        end  
        ind_final_lambda = find(v_looLambdas==final_lambda_exact, 1,'first');
        if isempty(ind_final_lambda)
            loo_error_exact = obj.exact_loo(A_train, y_train, ...
                final_lambda_exact, v_w_exact);
        else
            loo_error_exact = v_looErrors(ind_final_lambda);
        end
        loo_error_inexact = obj.exact_loo(A_train, y_train, ...
            final_lambda_exact, v_w_inexact);
        
        disp 'Computing approximate loo errors'
        v_aloErrors = obj.estimate_alo_error(A_train, y_train, v_aloLambdas, ...
            v_w_exact);
        %% 
        figure(135); clf
        loglog(v_testLambdas, v_testErrors);hold on
        loglog(v_testLambdas, v_valErrors, 'r')
        loglog(v_looLambdas, v_looErrors,'g')
        loglog(v_aloLambdas, v_aloErrors, 'm')
        plot(final_lambda_exact, test_error_exact, 'ob')
        plot(final_lambda_exact, val_error_exact, 'or')
        plot(final_lambda_exact, loo_error_exact, 'og')
        plot(final_lambda_inexact, test_error_inexact, 'xb')
        plot(final_lambda_inexact, val_error_inexact, 'xr')
        plot(final_lambda_inexact, loo_error_inexact, 'xg')
         
        legend ('Out-of-sample test error', ...
            'Out-of-sample validation error', 'Leave-one-out error', ...
            'Approximate Loo (Wang, 2018)', ...
            'Exact', '', '', 'Inexact, memoryless', '', '')
        xlabel \lambda
        ylabel MSE
        
        %%
        save SavedResults_35
    end
    function F = experiment_36(obj)
        % Memoryless, inexact
        % comparison with Alocv (approximate leave-one-out, Wang et al 2018)
        % more unknowns than samples
        
        obj.n_train = 800;
        obj.p = 1000;
        obj.seed = 6;
        [A_train, A_test, y_train, y_test, true_x] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.max_iter_outer= 20000;
        memory_factor= 1;
        spoq = QStepsize;
        spoq.eta_0 = 200;
        spoq.nu = 20;
        spoq.beta_2 = 1-1/(memory_factor*obj.n_train);
        spoq.beta_1 = 1-1/(memory_factor*obj.n_train);
        hl.stepsize_policy = spoq;     
        hl.debug= 1;
        hl.normalized_lambda_0 = 1/sqrt(obj.n_train);
        hl.tol = 1e0;
        hl.b_memory = 0;
        [lambda_inexact, count_inexact, v_w_inexact] ...
            = hl.solve_approx_mirror(A_train, y_train);

        figure(36); clf
        plot(cumsum(count_inexact), lambda_inexact);
        xlabel '# ISTA iterations'
        ylabel '\lambda'
        legend('Inexact, memoryless')
        drawnow
        
        %%
        % Evaluate test error
        final_lambda_inexact = lambda_inexact(find(count_inexact>0, 1, 'last'));
        v_testLambdas = final_lambda_inexact*logspace(-1, 0.5);
        v_aloLambdas = final_lambda_inexact*linspace(0.5, 1.5, 200);
        v_testErrors = zeros(size(v_testLambdas));
        v_valErrors  = v_testErrors;
        
        disp 'Computing test errors'
        for k = 1:length(v_testLambdas)
            v_testErrors(k) = obj.estimate_outOfSample_error(A_train, ...
                y_train, A_test, y_test, v_testLambdas(k), v_w_inexact);
            v_valErrors(k) = obj.estimate_outOfSample_error(A_test, ...
                y_test, A_train, y_train, v_testLambdas(k), v_w_inexact);
        end
        test_error_inexact = mean((y_test-A_test*v_w_inexact).^2)./mean(y_test.^2);
        
        val_error_inexact  = obj.estimate_outOfSample_error(A_test, ...
            y_test, A_train, y_train, final_lambda_inexact, v_w_inexact);
        
        disp 'Computing approximate loo errors'
        v_aloErrors = obj.estimate_alo_error(A_train, y_train, v_aloLambdas, ...
            v_w_inexact);
        alo_error_inexact = obj.estimate_alo_error(A_train, y_train,...
            final_lambda_inexact, v_w_inexact);
        %% 
        figure(136); clf
        loglog(v_testLambdas, v_testErrors);hold on
        loglog(v_testLambdas, v_valErrors, 'r')
        loglog(v_aloLambdas, v_aloErrors, 'm')
        plot(final_lambda_inexact, test_error_inexact, 'xb')
        plot(final_lambda_inexact, val_error_inexact, 'xr')
        plot(final_lambda_inexact, alo_error_inexact, 'xg')
         
        legend ('Out-of-sample test error', ...
            'Out-of-sample validation error', 'Approximate Loo (Wang, 2018)', ...
            'Inexact, memoryless', '', '')
        xlabel \lambda
        ylabel MSE
        
        %%
        save SavedResults_36
    end

end
    methods (Static)
        
        function alo_nmse = estimate_alo_error(m_X_train, v_y_train, v_lambdas, v_w_0)
            N = length(v_y_train); assert(size(m_X_train, 1)==N);
            P = size(m_X_train, 2);
            m_Phi = m_X_train'*m_X_train;
            v_r = m_X_train'*v_y_train;
            my_alpha = 10/trace(m_Phi);
            hl = OHO_Lasso;
            hl.tol = 1e-3;
            K = length(v_lambdas);
            m_W = zeros(P,K);
            for k = 1:K
                m_W(:,k) = hl.ista(v_w_0, m_Phi, v_r, my_alpha, v_lambdas(k));
            end
            alo_nmse = alo_lasso_mex(m_X_train, v_y_train, m_W, 1e-5)...
                ./mean(v_y_train.^2);
        end
        
        function oos_error = estimate_outOfSample_error(m_X_train, v_y_train,...
                m_X_test, v_y_test, lambda, v_w_0)
            N = length(v_y_train); assert(size(m_X_train, 1)==N);
            m_Phi = m_X_train'*m_X_train;
            v_r = m_X_train'*v_y_train;
            my_alpha = 10/trace(m_Phi);
            hl = OHO_Lasso;
            hl.tol = 1e-3;
            v_w_test = hl.ista(v_w_0, m_Phi, v_r, my_alpha, lambda);
            oos_error = mean((v_y_test - m_X_test*v_w_test).^2)./mean(v_y_test.^2);
        end
%         function val_error = compute_validation_error(...
%                 m_X_trained, v_y_trained, m_X_val, v_y_val, lambda, v_w_0)
%             N = length(v_y_val); assert(size(m_X_val, 1)==N);
%             m_Phi = m_X_val'*m_X_val;
%             v_r = m_X_val'*v_y_val;
%             my_alpha = 10/trace(m_Phi);
%             hl = OHO_Lasso;
%             hl.tol = 1e-3;
%             v_w_val = hl.ista(v_w_0, m_Phi, v_r, my_alpha, lambda);
%             val_error = mean((v_y_trained - m_X_trained*v_w_val).^2);            
%         end
        
        function loo_error = exact_loo(m_X, v_y, lambda, v_w_0)
            
            N = length(v_y); assert(size(m_X, 1)==N);
            m_Phi = m_X'*m_X;
            v_r = m_X'*v_y;
            my_alpha = 10/trace(m_Phi);
            hl = OHO_Lasso;
            hl.tol = 1e-4;
            v_looErrors = zeros(N,1);
            ltc = LoopTimeControl(N);
            for j =1:N
                v_x_j = m_X(j,:)';
                m_Phi_j = m_Phi - v_x_j * v_x_j';
                v_r_j   = v_r   - v_y(j)* v_x_j;
                v_w_j = hl.ista(v_w_0, m_Phi_j, v_r_j, my_alpha, lambda);
                v_looErrors(j) = v_y(j)-m_X(j,:)*v_w_j;
                ltc.go(j);
            end
            loo_error = mean(v_looErrors.^2)./mean(v_y.^2);
        end
    end

end