classdef OHOExperiments
    properties
        n_train = 200;     % Number of train samples
        n_test  = 2000;    % Number of test  samples
        p       = 100;     % Dimensionality of dataset
        seed    = 3;       % Random seed
        sigma   = 0.3;     % Variance of the noise
        sparsity = 0.8;    % proportion of entries of true_x that are > 0
        
        b_colinear = 0;    %introduce colinear variables when creating 
        % synthetic data (see set_up_data method)
    end
      
methods % Constructor and synthetic-data creating procedure
    
    function obj = OHOExperiments() %The constructor sets the path
        addpath('Stepsizes/')
        addpath('utilities/')
        addpath('competitors/')
    end
    
    function [A_train, A_test, y_train, y_test, true_x] = ...
            set_up_data(obj)        
        rng(obj.seed);
        
        if obj.b_colinear % create correlated variables           
            block_size = 10;
            block_rank = 8; assert(block_rank<block_size);
            n_blocks = obj.p/block_size; assert(n_blocks==uint8(n_blocks))
            t_block = cell(n_blocks, 1);
            for b = 1:n_blocks
                m_rectangular = randn(block_size, block_rank);
                t_block{b} = m_rectangular*m_rectangular';
            end
            D = blkdiag(t_block{:});
        else
            D = eye(obj.p); %!
        end
        A_train = randn(obj.n_train, obj.p)*D; % sensing matrix
        A_test = randn(obj.n_test, obj.p)*D;
        true_x = double(rand(obj.p, 1) < obj.sparsity); % sparse coefficients
        epsilon = randn(obj.n_train, 1) * obj.sigma; % noise
        epsilon_test = randn(obj.n_test, 1) * obj.sigma;
        y_train = A_train * true_x + epsilon;
        y_test  = A_test * true_x + epsilon_test;
    end
end
    
methods %experiments
    
    function F = experiment_1(obj)
        % Approximate gradient vs mirror (not online)
        [A_train, ~, y_train, ~, ~] = obj.set_up_data();
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
        [A_train, ~, y_train, ~, ~] = obj.set_up_data();
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
        [A_train, ~, y_train, ~, ~] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'log';
        hl.b_online = 1;
        hl.max_iter_outer= 1000;
        hl.stepsize_policy.eta_0 = 500; %default policy is Diminishing
        
        v_lambda = hl.solve_approx_mirror(A_train, y_train);
        
        figure(1); clf
        plot(v_lambda)
        legend ('Mirror')
        F = 0;
    end
    
    function F = experiment_12(obj)
        % Online mirror with AdaGrad stepsize
        % testing object-based stepsize policies
        [A_train, ~, y_train, ~, ~] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'log';
        hl.b_online = 1;
        hl.max_iter_outer= 10000;
        hl.stepsize_policy = AdagradStepsize;
        hl.stepsize_policy.eta_0 = 0.1;
        hl.stepsize_policy.epsilon = 1e-5;
       
        v_lambda = hl.solve_approx_mirror(A_train, y_train);
        
        figure(1); clf
        plot(v_lambda)
        legend ('Mirror, Adagrad')
        F = 0;
    end
    
    function F = experiment_13(obj)
        % Online mirror with RmsProp stepsize
        % testing object-based stepsize policies
        [A_train, ~, y_train, ~, ~] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'log';
        hl.b_online = 1;
        hl.max_iter_outer= 10000;
        hl.stepsize_policy = RmsPropStepsize;
        hl.stepsize_policy.beta_2 = 0.9;
        hl.stepsize_policy.eta_0 = 0.01;
        hl.stepsize_policy.epsilon = 1e-6;
   
        v_lambda = hl.solve_approx_mirror(A_train, y_train);
        
        figure(1); clf
        plot(v_lambda)
        legend ('Mirror, RMSProp')
        F = 0;
        % todo: maybe if we reduce the beta_2 param to 0.9 we get faster
        % convergence?
    end

    function F = experiment_14(obj)
        % Online mirror with LR-based stepsize
        % testing object-based stepsize policies

        [A_train, ~, y_train, ~, ~] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'log';
        hl.b_online = 1;
        hl.max_iter_outer= 10000;
        
        sp = LinearRegressionStepsize; %Stepsize Policy: Linear regression
        sp.version = 6;
        sp.eta_0 = 10;%! 500;
        sp.beta_2 = 1-1/obj.n_train;
        sp.nu = 0.1;
        sp.kappa = 1/obj.n_train;
        sp.gamma = 0.;
        sp.N = obj.n_train;
        sp.law = @(x)x;
        sp.inv_law = @(x)x;
        
        hl.stepsize_policy = sp;
        v_lambda = hl.solve_approx_mirror(A_train, y_train);
        
        figure(1); clf
        plot(v_lambda)
        legend ('Mirror, LinearRegressionStepsize')
        F = 0;
    end
    
    function F = experiment_15(obj)
        % Online mirror with LR-based stepsize
        % testing object-based stepsize policies

        [A_train, ~, y_train, ~, ~] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'log';
        hl.b_online = 1;
        hl.max_iter_outer= 10000;
        
        sp = LinearRegressionStepsize; %Stepsize Policy: Linear regression
        sp.version = 6;
        sp.eta_0 = 50;%! 500;
        sp.beta_2 = 1-1/obj.n_train;
        sp.nu = 0.01;
        sp.kappa = 6;
        
        hl.stepsize_policy = sp;
        v_lambda = hl.solve_approx_mirror(A_train, y_train);
        
        figure(1); clf
        plot(v_lambda)
        legend ('Mirror, LinearRegressionStepsize')
        F = 0;    
    end
    
    function F = experiment_21(obj)
        % Online gradient with Adam-like adapted stepsize
        
        [A_train, ~, y_train, ~, ~] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.max_iter_outer= 10000;
        
        spo_adam = AdamStepsize;
        spo_adam.eta_0 = 0.03;
        spo_adam.beta_2= 0.99;
        
        hl.stepsize_policy = spo_adam;       
        lambda_adam= hl.solve_approx_mirror(A_train, y_train);
        
        figure(1); clf
        plot(lambda_adam);
        legend ('Grad, ADAM-like adapted stepsize')
        F = 0;
    end
    
    function F = experiment_22(obj)
        % Online gradient with Unconstrained FTML (U-FTML) adapted stepsize

        [A_train, ~, y_train, ~, ~] = obj.set_up_data();
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
        lambda_uftml= hl.solve_approx_mirror(A_train, y_train);
        
        figure(1); clf
        plot(lambda_uftml);
        legend('Grad, U-FTML stepsize')
        F = 0;
    end
    
    function F = experiment_23(obj)
        % Online gradient with U-FTML adapted stepsize
        % Now with 10 times more samples!
        obj.n_train = 2000;
        
        [A_train, ~, y_train, ~, ~] = obj.set_up_data();
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
        lambda_uftml= hl.solve_approx_mirror(A_train, y_train);
        
        figure(1); clf
        plot(lambda_uftml);
        title '2000 training samples'
        legend('Grad, U-FTML stepsize')
        F = 0;
    end
    
    function F = experiment_24(obj)
        % Online mirror with Luismi's Q stepsize
        obj.n_train = 2000;
        
        %obj.seed = 4;     
        [A_train, ~, y_train, ~, ~] = obj.set_up_data();
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
        %spoq.N = obj.n_train; %this was a property of the previous version
        % of QStepsize (now renamed as QStepsize_old)
        
        hl.stepsize_policy = spoq;       
        lambda_QS= hl.solve_approx_mirror(A_train, y_train);
        
        figure(1); clf
        plot(lambda_QS);
        title(sprintf('%d training samples', obj.n_train))
        legend('Grad, Q-step')
        F = 0;
    end
    
    function F = experiment_25(obj)
        % Online gradient descent with Luismi's Q stepsize
        obj.n_train = 1000; %! 2000
        
        %obj.seed = 4;
        [A_train, ~, y_train, ~, ~] = obj.set_up_data();
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
        lambda_QS= hl.solve_approx_mirror(A_train, y_train);
        
        figure(25); clf
        plot(lambda_QS);
        title(sprintf('%d training samples', obj.n_train))
        legend('Grad, Q-step')
        F = 0;
    end
    
    function F = experiment_31(obj)
        % Memoryless vs. alg. with memory
        obj.n_train = 4000;
        obj.p = 100;
        obj.seed = 2;
        [A_train, ~, y_train, ~, ~] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.max_iter_outer= 20000;
        
        spoq = QStepsize;
        memory_factor= 1;
        spoq.eta_0 = 200;
        spoq.nu = 40;
        spoq.beta_2 = 1-1/(memory_factor*obj.n_train); % 10*
        spoq.beta_1 = 1-1/(memory_factor*obj.n_train); % 10*
        
        hl.stepsize_policy = spoq;     
        hl.debug= 0;
        hl.normalized_lambda_0 = 1/obj.n_train;
        %By default the OHO_Lasso goes with memory
        [lambda_memory, count_memory]    ...
            = hl.solve_approx_mirror(A_train, y_train);
        
        spoq2 = QStepsize;
        spoq2.eta_0 = 200;
        spoq2.nu = 40;
        spoq2.beta_2 = 1-1/(memory_factor*obj.n_train); % 10*
        spoq2.beta_1 = 1-1/(memory_factor*obj.n_train); % 10*
        hl.stepsize_policy = spoq2;
        hl.b_memory = 0; %Now we experiment with the Memoryless option
        [lambda_memoryless, count_memoryless] ...
            = hl.solve_approx_mirror(A_train, y_train);             
        
        figure(31); clf
        plot([cumsum(count_memoryless) cumsum(count_memory)], ...
            [lambda_memoryless lambda_memory]);
        legend('Memoryless','With Memory')
        title(sprintf('%d training samples', obj.n_train))
        F = 0;
    end
    
    function F = experiment_32(obj) %TODO: SHORTEN EXPERIMENT CODE
        % Memoryless, comparison between exact (let ISTA converge) and
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
        % RStepsize instead of QStepsize
        
        obj.n_train = 2000;
        obj.p = 200;
        obj.seed = 6;
        [A_train, A_test, y_train, y_test, true_x] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.max_iter_outer= 20000;
        memory_factor= 1;
        spoq = RStepsize;
        spoq.eta_0 = 200;
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
    
    function F = experiment_37(obj)
        % Soft vs. hard approximation of gradient
        % Inexact, Memoryless
        % Params copied from experiment 35
        
        obj.n_train = 2000;
        obj.p = 300;
        obj.seed = 10;
        [A_train, A_test, y_train, y_test, true_x] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.max_iter_outer= 20000;
        hl.approx_type = 'soft'; %!! soft
        memory_factor= 1;
        spoq = QStepsize;
        spoq.eta_0 = 200;
        spoq.nu = 2;
        spoq.beta_2 = 1-1/(memory_factor*obj.n_train);
        spoq.beta_1 = 1-1/(memory_factor*obj.n_train);
        hl.stepsize_policy = spoq;     
        hl.debug= 1;
        hl.normalized_lambda_0 = 1/sqrt(obj.n_train);        
        hl.tol = 1e0;
        hl.b_memory = 0;
        obj.seed = 7;
        [lambda_soft, count_soft, v_w_soft] ...
            = hl.solve_approx_mirror(A_train, y_train);
        hl.approx_type = 'hard';
        spoq2 = QStepsize;
        spoq2.eta_0 = 200;
        spoq2.nu = 2;
        spoq2.beta_2 = 1-1/(memory_factor*obj.n_train);
        spoq2.beta_1 = 1-1/(memory_factor*obj.n_train);
        hl.stepsize_policy = spoq2;
        obj.seed = 7;
        [lambda_hard, count_hard, v_w_hard] ...
            = hl.solve_approx_mirror(A_train, y_train);

        figure(35); clf
        plot([cumsum(count_hard) cumsum(count_soft)], ...
            [lambda_hard lambda_soft]);
        xlabel '# ISTA iterations'
        ylabel '\lambda'
        legend('Hard approximation','Soft approximation')
        drawnow
        
        %%
        % Evaluate test error
        final_lambda_hard = lambda_hard(find(count_hard>0, 1, 'last'));
        final_lambda_soft = lambda_soft(find(count_soft>0, 1, 'last'));
        v_testLambdas = final_lambda_hard*logspace(-1, 0.5);
        %v_looLambdas = final_lambda_hard*linspace(0.85, 1.15, 9);
        v_looLambdas = linspace(...
            1.1*final_lambda_hard-0.1*final_lambda_soft, ...
            0.1*final_lambda_hard+0.9*final_lambda_soft, 7);
        v_aloLambdas = final_lambda_hard*linspace(0.5, 1.5, 200);
        v_testErrors = zeros(size(v_testLambdas));
        v_valErrors  = v_testErrors;
        v_looErrors  = zeros(size(v_looLambdas));
        
        disp 'Computing test errors'
        for k = 1:length(v_testLambdas)
            v_testErrors(k) = obj.estimate_outOfSample_error(A_train, ...
                y_train, A_test, y_test, v_testLambdas(k), v_w_hard);
            v_valErrors(k) = obj.estimate_outOfSample_error(A_test, ...
                y_test, A_train, y_train, v_testLambdas(k), v_w_hard);
        end
        test_error_hard = mean((y_test-A_test*v_w_hard).^2)./mean(y_test.^2);
        test_error_soft = mean((y_test-A_test*v_w_soft).^2)./mean(y_test.^2);
        
        val_error_hard  = obj.estimate_outOfSample_error(A_test, ...
            y_test, A_train, y_train, final_lambda_hard, v_w_hard);
        val_error_soft  = obj.estimate_outOfSample_error(A_test, ...
            y_test, A_train, y_train, final_lambda_soft, v_w_soft);

        disp 'Computing loo errors'
        for kl = 1:length(v_looLambdas)
            v_looErrors(kl) = obj.exact_loo(...
                A_train, y_train, v_looLambdas(kl), v_w_hard);
        end  
        ind_final_lambda = find(v_looLambdas==final_lambda_hard, 1,'first');
        if isempty(ind_final_lambda)
            loo_error_hard = obj.exact_loo(A_train, y_train, ...
                final_lambda_hard, v_w_hard);
        else
            loo_error_hard = v_looErrors(ind_final_lambda);
        end
        loo_error_soft = obj.exact_loo(A_train, y_train, ...
            final_lambda_hard, v_w_soft);
        
        disp 'Computing approximate loo errors'
        v_aloErrors = obj.estimate_alo_error(A_train, y_train, v_aloLambdas, ...
            v_w_hard);
        %% 
        figure(137); clf
        loglog(v_testLambdas, v_testErrors);hold on
        loglog(v_testLambdas, v_valErrors, 'r')
        loglog(v_looLambdas, v_looErrors,'g')
        loglog(v_aloLambdas, v_aloErrors, 'm')
        plot(final_lambda_hard, test_error_hard, 'ob')
        plot(final_lambda_hard, val_error_hard, 'or')
        plot(final_lambda_hard, loo_error_hard, 'og')
        plot(final_lambda_soft, test_error_soft, 'xb')
        plot(final_lambda_soft, val_error_soft, 'xr')
        plot(final_lambda_soft, loo_error_soft, 'xg')
         
        legend ('Out-of-sample test error', ...
            'Out-of-sample validation error', 'Leave-one-out error', ...
            'Approximate Loo (Wang, 2018)', ...
            'Hard', '', '', 'Soft', '', '')
        xlabel \lambda
        ylabel MSE
        
        %%
        save SavedResults_37
    end

    function F = experiment_40(obj)
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
        memory_factor = 1;
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
        save SavedResults_40
    end

    function F = experiment_41(obj)
        % Soft vs. hard approximation of gradient
        % More unknown than samples
        % Inexact, Memoryless
        % Simulation copied from 37;
        % parameters taken for 40
        
        obj.n_train = 200;
        obj.p = 300;
        obj.seed = 6;
        [A_train, A_test, y_train, y_test, true_x] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.max_iter_outer= 20000;
        hl.approx_type = 'soft';
        memory_factor= 1;
        spoq = QStepsize;
        spoq.eta_0 = 2;
        spoq.nu = 20;
        spoq.beta_2 = 1-1/(memory_factor*obj.n_train);
        spoq.beta_1 = 1-1/(memory_factor*obj.n_train);
        hl.stepsize_policy = spoq;     
        hl.debug= 1;
        hl.normalized_lambda_0 = 1/obj.n_train;        
        hl.tol = 1e0;
        hl.b_memory = 0;
        [lambda_soft, count_soft, v_w_soft] ...
            = hl.solve_approx_mirror(A_train, y_train);
        hl.approx_type = 'hard';
        [lambda_hard, count_hard, v_w_hard] ...
            = hl.solve_approx_mirror(A_train, y_train);

        figure(35); clf
        plot([cumsum(count_hard) cumsum(count_soft)], ...
            [lambda_hard lambda_soft]);
        xlabel '# ISTA iterations'
        ylabel '\lambda'
        legend('Hard approximation','Soft approximation')
        drawnow
        
        %%
        % Evaluate test error
        final_lambda_hard = lambda_hard(find(count_hard>0, 1, 'last'));
        final_lambda_soft = lambda_soft(find(count_soft>0, 1, 'last'));
        v_testLambdas = final_lambda_hard*logspace(-1, 0.5);
        v_looLambdas = final_lambda_hard*linspace(0.7, 1.3, 7);
        v_aloLambdas = final_lambda_hard*linspace(0.5, 1.5, 200);
        v_testErrors = zeros(size(v_testLambdas));
        v_valErrors  = v_testErrors;
        v_looErrors  = zeros(size(v_looLambdas));
        
        disp 'Computing test errors'
        for k = 1:length(v_testLambdas)
            v_testErrors(k) = obj.estimate_outOfSample_error(A_train, ...
                y_train, A_test, y_test, v_testLambdas(k), v_w_hard);
            v_valErrors(k) = obj.estimate_outOfSample_error(A_test, ...
                y_test, A_train, y_train, v_testLambdas(k), v_w_hard);
        end
        test_error_hard = mean((y_test-A_test*v_w_hard).^2)./mean(y_test.^2);
        test_error_soft = mean((y_test-A_test*v_w_soft).^2)./mean(y_test.^2);
        
        val_error_hard  = obj.estimate_outOfSample_error(A_test, ...
            y_test, A_train, y_train, final_lambda_hard, v_w_hard);
        val_error_soft  = obj.estimate_outOfSample_error(A_test, ...
            y_test, A_train, y_train, final_lambda_soft, v_w_soft);

        disp 'Computing loo errors'
        for kl = 1:length(v_looLambdas)
            v_looErrors(kl) = obj.exact_loo(...
                A_train, y_train, v_looLambdas(kl), v_w_hard);
        end  
        ind_final_lambda = find(v_looLambdas==final_lambda_hard, 1,'first');
        if isempty(ind_final_lambda)
            loo_error_hard = obj.exact_loo(A_train, y_train, ...
                final_lambda_hard, v_w_hard);
        else
            loo_error_hard = v_looErrors(ind_final_lambda);
        end
        loo_error_soft = obj.exact_loo(A_train, y_train, ...
            final_lambda_hard, v_w_soft);
        
        disp 'Computing approximate loo errors'
        v_aloErrors = obj.estimate_alo_error(A_train, y_train, v_aloLambdas, ...
            v_w_hard);
        %% 
        figure(135); clf
        loglog(v_testLambdas, v_testErrors);hold on
        loglog(v_testLambdas, v_valErrors, 'r')
        loglog(v_looLambdas, v_looErrors,'g')
        loglog(v_aloLambdas, v_aloErrors, 'm')
        plot(final_lambda_hard, test_error_hard, 'ob')
        plot(final_lambda_hard, val_error_hard, 'or')
        plot(final_lambda_hard, loo_error_hard, 'og')
        plot(final_lambda_soft, test_error_soft, 'xb')
        plot(final_lambda_soft, val_error_soft, 'xr')
        plot(final_lambda_soft, loo_error_soft, 'xg')
         
        legend ('Out-of-sample test error', ...
            'Out-of-sample validation error', 'Leave-one-out error', ...
            'Approximate Loo (Wang, 2018)', ...
            'Hard', '', '', 'Soft', '', '')
        xlabel \lambda
        ylabel MSE
        
        %%
        save SavedResults_41
    end
        
    function F = experiment_50(obj)
        % Elastic Net ( rest of params copied from Experiment_36)
        
        obj.n_train = 200;
        obj.p = 100;
        obj.b_colinear = 1;
        obj.sigma = 10;
        obj.seed = 9;
        [A_train, A_test, y_train, y_test, true_x] = obj.set_up_data();
        hen = OHO_ElasticNet;
        hen.mirror_type = 'log';
        hen.b_online = 1;
        hen.max_iter_outer= 2000; %!
%        memory_factor= 10;
%         spoq = QStepsize;
%         spoq.eta_0 = 10;
%         spoq.beta_2 = 1-1/(memory_factor*obj.n_train);
%         spoq.beta_1 = 1-1/(memory_factor*obj.n_train);
%         spoq.nu = 20;
        spoq = ConstantStepsize;
        spoq.eta_0 = 50;
        hen.stepsize_policy = spoq;     
        hen.debug= 1;
        hen.normalized_lambda_0 = 1/obj.n_train;
%         hl.normalized_lambda_0 = 1/sqrt(obj.n_train);
%         hl.tol = 1e-4;
%         hl.b_memory = 1;
%         [lambda_exact, count_exact, v_w_exact] ...
%             = hl.solve_approx_mirror(A_train, y_train);
        
        hen.tol = 1e0;
        hen.b_memory = 0;
        [lambda_inexact, count_inexact, v_w_final] ...
            = hen.solve_approx_mirror(A_train, y_train);

        figure(35); clf
        plot(cumsum(count_inexact), lambda_inexact);        
        xlabel '# ISTA iterations'
        ylabel '\lambda'
        legend('Exact with memory','Inexact, memoryless')
        drawnow
        
        %%
        % Evaluate test error

%         final_lambda_exact = lambda_exact(find(count_exact>0, 1, 'last'));
        final_niter = find(count_inexact>0, 1, 'last');
        final_lambda = lambda_inexact(final_niter, 1);
        final_rho    = mean(lambda_inexact(ceil(final_niter/2):final_niter, 2));
        v_testLambdas = final_lambda*logspace(-1, 0.5);
        v_looLambdas = final_lambda*linspace(0.3, 3, 15);
        v_aloLambdas = final_lambda*linspace(0.3, 3, 200);
        v_testErrors = zeros(size(v_testLambdas));
        v_valErrors  = v_testErrors;
        v_looErrors  = zeros(size(v_looLambdas));
        
        disp 'Computing test errors'
        for k = 1:length(v_testLambdas)
            v_testErrors(k) = obj.estimate_outOfSample_error(A_train, ...
                y_train, A_test, y_test, v_testLambdas(k), v_w_final, final_rho);
            v_valErrors(k) = obj.estimate_outOfSample_error(A_test, ...
                y_test, A_train, y_train, v_testLambdas(k), v_w_final, final_rho);
        end
        test_error = mean((y_test-A_test*v_w_final).^2)./mean(y_test.^2);
        
        val_error  = obj.estimate_outOfSample_error(A_test, ...
            y_test, A_train, y_train, final_lambda, v_w_final, final_rho);

        disp 'Computing loo errors'
        for kl = 1:length(v_looLambdas)
            v_looErrors(kl) = obj.exact_loo(...
                A_train, y_train, v_looLambdas(kl), v_w_final, final_rho);
        end  
        ind_final_lambda = find(v_looLambdas==final_lambda, 1,'first');
        if isempty(ind_final_lambda)
            loo_error = obj.exact_loo(A_train, y_train, ...
                final_lambda, v_w_final, final_rho);
        else
            loo_error = v_looErrors(ind_final_lambda);
        end
        
        disp 'Computing approximate loo errors'
        v_aloErrors = obj.estimate_alo_error(A_train, y_train, ...
            [v_aloLambdas' ones(size(v_aloLambdas))'*final_rho], ...
            v_w_final);
        %% 
        figure(135); clf
        loglog(v_testLambdas, v_testErrors);hold on
        loglog(v_testLambdas, v_valErrors, 'r')
        loglog(v_looLambdas, v_looErrors,'g')
        loglog(v_aloLambdas, v_aloErrors, 'm')
        plot(final_lambda, test_error, 'xb')
        plot(final_lambda, val_error, 'xr')
        plot(final_lambda, loo_error, 'xg')
         
        legend ('Out-of-sample test error', ...
            'Out-of-sample validation error', 'Leave-one-out error', ...
            'Approximate Loo (Wang, 2018)', ...
            'Inexact, memoryless', '', '')
        xlabel \lambda
        ylabel MSE
        
        %%
        save SavedResults_51
    end
    
    function F = experiment_52(obj)
        % Comparing Elastic Net and Lasso
        % optimal params for Elastic net should work better than
        % those for Lasso...
        
        obj.n_train = 200;
        obj.p = 100;
        obj.seed = 9;
        obj.sigma = 10;
        my_debug = 1;
        
        [A_train, A_test, y_train, y_test, true_x] = obj.set_up_data();
        hl  = OHO_Lasso;
        hl.max_iter_outer = 10000;
        hl.mirror_type = 'grad';
        memory_factor= 1;
        spoq = QStepsize;
        spoq.eta_0 = 1000;
        spoq.beta_2 = 1-1/(memory_factor*obj.n_train);
        spoq.beta_1 = 1-1/(memory_factor*obj.n_train);
        spoq.nu = 20;
        hl.stepsize_policy = spoq;
        hl.debug = my_debug;
        hl.tol = 1e0;
        hl.b_online = 1;
        hl.b_memory = 0;
        hl.normalized_lambda_0 = 1/10000;
        hl.tol_g = 1e-7;

        hen = OHO_ElasticNet;
        hen.max_iter_outer = 10000;
        hen.mirror_type = 'log';
        sp_constant = DiminishingStepsize;
        N = hen.max_iter_outer;
        sp_constant.law = @(k)(N/(N-k));
        sp_constant.eta_0 = 20;
        hen.stepsize_policy = sp_constant;
        hen.debug = my_debug;
        hen.tol = 1e0;
        hen.b_online = 1;
        hen.b_memory = 0;
        hen.normalized_lambda_0 = 1/10000;
        
        [lambda_lasso, count_lasso, v_w_lasso] = ...
            hl.solve_approx_mirror(A_train, y_train);
        [lambda_en, count_en, v_w_en] = ...
            hen.solve_approx_mirror(A_train, y_train);
        
        figure(52); clf
        plot(cumsum(count_lasso), lambda_lasso);
        hold on
        plot(cumsum(count_en), lambda_en);
        xlabel '# ISTA iterations'
        ylabel '\lambda'
        legend('\lambda_{Lasso}', '\lambda_{EN}', '\rho_{EN}')
        drawnow
        
        save SavedResults_52;
        
        %% 
        load SavedResults_52;
        niter_lasso = find(count_lasso>0, 1, 'last');
        lambda_lasso_final = lambda_lasso(niter_lasso, 1);
        niter_en = find(count_en>0, 1, 'last');
        lambda_en_final    = lambda_en(niter_en, 1);
        rho_final          = lambda_en(niter_en, 2);
        
        v_looLambdas = lambda_lasso_final*linspace(0.85, 1.15, 15);
        disp 'Computing loo errors'
        for kl = 1:length(v_looLambdas)
            v_loo_lasso(kl) = obj.exact_loo(...
                A_train, y_train, v_looLambdas(kl), v_w_lasso, 0);
            v_loo_en(kl) = obj.exact_loo(...
                A_train, y_train, v_looLambdas(kl), v_w_lasso, rho_final);
        end
            figure(152); clf
            loglog(v_looLambdas, v_loo_lasso); hold on
            loglog(v_looLambdas, v_loo_en);
            legend('Lasso', 'Elastic Net')
        
        
        
    end
    

end
    methods (Static)
        
        function alo_nmse = estimate_alo_error(m_X_train, v_y_train, ...
                m_hyperparams, v_w_0)
            N = length(v_y_train); assert(size(m_X_train, 1)==N);
            P = size(m_X_train, 2);
            m_Phi = m_X_train'*m_X_train;
            v_r = m_X_train'*v_y_train;
            my_alpha = 10/trace(m_Phi);
            hl = OHO_Lasso;
            hl.tol = 1e-3;
            if size(m_hyperparams, 2) == 1
                % LASSO
                v_lambdas = m_hyperparams(:,1);
                v_rhos    = zeros(size(v_lambdas));
                K = size(m_hyperparams, 1);

            elseif size(m_hyperparams, 2) == 2
                % Elastic net
                v_lambdas = m_hyperparams(:,1);
                v_rhos    = m_hyperparams(:,2);
                K = size(m_hyperparams, 1);
            elseif size(m_hyperparams, 1) ==1
                v_lambdas = m_hyperparams';
                v_rhos    = zeros(size(v_lambdas));
                K = size(m_hyperparams, 2);
            end
            m_W = zeros(P,K);
            disp('Training for each value of lambda')
            ltc = LoopTimeControl(K);
            for k = 1:K
                m_W(:,k) = hl.ista(v_w_0, m_Phi+v_rhos(k)*eye(P), v_r, ...
                    my_alpha, v_lambdas(k));
                ltc.go(k);
            end
            disp 'Computing approximate LOO for trained coefs'
            alo_nmse = alo_lasso_mex(m_X_train, v_y_train, m_W, 1e-5)...
                ./mean(v_y_train.^2);
        end
        
        function oos_error = estimate_outOfSample_error(m_X_train, v_y_train,...
                m_X_test, v_y_test, lambda, v_w_0, rho)
            N = length(v_y_train); assert(size(m_X_train, 1)==N);
            P = size(m_X_train, 2);
            if exist('rho', 'var')
                m_Phi = m_X_train'*m_X_train + rho*eye(P);
            else
                m_Phi = m_X_train'*m_X_train;
            end
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
        
        function loo_error = exact_loo(m_X, v_y, lambda, v_w_0, rho)
            
            N = length(v_y); assert(size(m_X, 1)==N);
            P = size(m_X, 2);
            
            if exist('rho', 'var')
                m_Phi = m_X'*m_X + rho*eye(P);
            else 
                m_Phi = m_X'*m_X ;
            end
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