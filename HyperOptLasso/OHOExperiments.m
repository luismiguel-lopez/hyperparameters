classdef OHOExperiments
    properties
        n_train = 200;     % Number of train samples
        n_test  = 2000;    % Number of test  samples
        p       = 100;     % Dimensionality of dataset
        seed    = 3;       % Random seed
        SNR     = 0.3;     % Signal to noise ratio  (natural units)
        sparsity= 0.2;     % proportion of entries of true_x that are > 0
        
        b_colinear = 0;    %introduce colinear variables when creating 
        % synthetic data (see set_up_data method)
    end
      
methods % Constructor and synthetic-data creating procedure
    
    function obj = OHOExperiments() %The constructor sets the path
        addpath('Stepsizes/')
        addpath('utilities/')
        addpath('competitors/')
    end
    
    function [m_A_train, m_A_test, v_y_train, v_y_test, true_w] = ...
            set_up_data(obj)        
        % create train and test data
        rng(obj.seed);
        
        if obj.b_colinear % create correlated variables           
            block_size = 10;
            block_rank = 5; assert(block_rank<block_size);
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
        
        true_w        = double(rand(obj.p, 1) < obj.sparsity); % sparse coefficients
        m_A_train     = randn(obj.n_train, obj.p)*D; % sensing matrix
        m_A_test      = randn(obj.n_test, obj.p)*D;   % sensing matrix (test)
        v_y_noiseless = true_w'*m_A_train';
        signal_power  = mean(v_y_noiseless.^2);
        noise_power   = signal_power/obj.SNR;
        epsilon       = randn(obj.n_train, 1) * sqrt(noise_power); % noise
        epsilon_test  = randn(obj.n_test, 1) * sqrt(noise_power);
        v_y_train     = m_A_train * true_w + epsilon;
        v_y_test      = m_A_test  * true_w + epsilon_test;
    end
end
    
methods % Experiments
            
    % Approximate gradient vs mirror (not online)
    function F = experiment_1(obj)
        [A_train, ~, y_train, ~, ~] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.stepsize_policy = ConstantStepsize;
        hl.stepsize_policy.eta_0 = 0.01;
        hl.max_iter_outer = 500;
        hl.mirror_type = 'grad';        
        lambda_grad = hl.solve_approx_mirror(A_train, y_train);
        hl.mirror_type = 'log';
        hl.stepsize_policy.eta_0 = 0.01;
        lambda_log = hl.solve_approx_mirror(A_train, y_train);
        
        exp_no = obj.determine_experiment_number();
        %%
        figure(exp_no*100+1); clf
        obj.show_lambda_iterates([lambda_grad; lambda_log]);
        legend({'Grad', 'Mirror (log)'})
        
        %% Figures
        ch_resultsFile = sprintf('results_OHO_%d', exp_no);
        save(ch_resultsFile);
        F = 0;
    end
    
    % Approximate gradient, compare effect of different fixed stepsizes
    function F = experiment_2(obj)
        obj.p = 100;
        obj.n_train = 500;
        [A_train, ~, y_train, ~, ~] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.b_online = 1;
        hl.b_memory = 0;
        hl.mirror_type = 'grad';
        hl.approx_type = 'hard';
        hl.max_iter_outer = 3000;
        hl.tol_g = 0; %inexact
        
        n_betas = 3;
        v_betas = logspace(-3.5, -2.5, n_betas);
        m_lambdas = zeros(n_betas, hl.max_iter_outer);
        for k = 1:n_betas
            sp = ConstantStepsize;
            sp.eta_0 = v_betas(k);
            hl.stepsize_policy = sp;
            m_lambdas(k, :) = hl.solve_approx_mirror(A_train, y_train);
        end
        
        exp_no = obj.determine_experiment_number();
        %%
        figure(exp_no*100+1); clf
        obj.show_lambda_iterates(m_lambdas);
        c_legend = cell(n_betas, 1);
        for nb = 1:n_betas
           c_legend{nb} = sprintf('\\beta = %g', v_betas(nb)); 
        end
        legend(c_legend, 'location', 'southeast');
        
        %% Figures
        ch_resultsFile = sprintf('results_OHO_%d', exp_no);
        save(ch_resultsFile);
        F = 0;
    end
    
    % Approximate mirror vs Online mirror
    function F = experiment_3(obj)
        
        [A_train, ~, y_train, ~, ~] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'log';
        hl.stepsize_policy = ConstantStepsize;
        hl.stepsize_policy.eta_0 = 0.1;
        hl.max_iter_outer = 1000;
        [lambda_log, count_log] = hl.solve_approx_mirror(A_train, y_train);
        
        hl.b_online = 1;
        hl.max_iter_outer= 10000;
        hl.stepsize_policy.eta_0 = 0.1;
        [lambda_onlinemirror, count_onlinemirror] ...
            = hl.solve_approx_mirror(A_train, y_train);
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        figure(exp_no*100+1); clf
        obj.show_lambda_vs_ista_iterates({lambda_log; lambda_onlinemirror},...
            {count_log;count_onlinemirror})                
        legend({'Batch', 'Online'})

        %%
        ch_resultsFile = sprintf('results_OHO_%d', exp_no);
        save(ch_resultsFile);
        F = 0;
    end
    
    %trying with dual averaging
    function F = experiment_4(obj)
        
        obj.n_train = 5000;
        [A_train, ~, y_train, ~, ~] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'log';
        hl.stepsize_policy = IncreasingStepsize;
        hl.max_iter_outer = 100000; %!
        hl.b_online = 1;
        hl.b_memory = 0;
        hl.tol_w = 1e-1;
        hl.b_DA = 1;
        hl.stepsize_policy.eta_0 = 30;
        
        [lambda_onlinemirror, count_onlinemirror] ...
            = hl.solve_approx_mirror(A_train, y_train);
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        figure(exp_no*100+1); clf
        obj.show_lambda_vs_ista_iterates({lambda_onlinemirror},...
            {count_onlinemirror})                
        legend({'Batch', 'Online'})

        %%
        ch_resultsFile = sprintf('results_OHO_%d', exp_no);
        save(ch_resultsFile);
        F = 0;
    end
    
    % Trying with the HyperGradientLasso, which uses Franceschi
    function F = experiment_5(obj)
        obj.n_train = 200;
        [A_train, ~, y_train, ~, ~] = obj.set_up_data();
        
        hgl = HyperGradientLasso;
        hgl.stepsize_lambda = 1e-3;
        hgl.tol_w = 1e-3;
        hgl.tol_g_lambda = 1e-4;
        hgl.max_iter_outer = 400;
        
        [m_W, v_lambda, v_it_count] = hgl.solve_gradient(A_train', y_train);
        final_lambda = v_lambda(find(v_it_count>0,1, 'last'));
        average_w = mean(m_W, 2);
        % Evaluate loo errors
        disp 'Computing loo errors'
     
        v_looLambdas = final_lambda*sort([1 linspace(0.9, 1.2, 21)]);
        %v_looLambdas = final_lambda_exact*sort([1 linspace(0.8, 1.5, 15)]);

        st_hyperparams_loo = struct; 
        for k =1:length(v_looLambdas)
            st_hyperparams_loo(k).lambda = v_looLambdas(k);
        end
        v_looErrors = obj.compute_loo_errors(st_hyperparams_loo, average_w, A_train, y_train);
        ind_final_lambda = find(v_looLambdas==final_lambda, 1,'first');
        loo_error_exact = v_looErrors(ind_final_lambda);
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        figure(exp_no*100+1); clf
        obj.show_lambda_vs_ista_iterates({v_lambda},...
            {v_it_count})                
        legend({'Batch'})
        
        figure(exp_no*100+2); clf
        plot(v_looLambdas, v_looErrors)
        xlabel '\lambda'
        ylabel 'LOO error'
        hold on
        plot(final_lambda, loo_error_exact, 'xr')

        %%
        ch_resultsFile = sprintf('results_OHO_%d', exp_no);
        save(ch_resultsFile);
        F = 0;


    end
    
    
    % Online mirror with diminishing stepsize
    % testing object-based stepsize policies
    function F = experiment_11(obj)
        
        [A_train, ~, y_train, ~, ~] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'log';
        hl.b_online = 1;
        hl.max_iter_outer= 10000;
        hl.stepsize_policy.eta_0 = 10; %default policy is Diminishing
        
        v_lambda = hl.solve_approx_mirror(A_train, y_train);
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        figure(exp_no*100+1); clf
        plot(v_lambda)
        legend ('Mirror')
        
        %%
        ch_resultsFile = sprintf('results_OHO_%d', exp_no);
        save(ch_resultsFile);
        F = 0;
    end
    
    % Online mirror with AdaGrad stepsize
    % testing object-based stepsize policies
    function F = experiment_12(obj)
        [A_train, ~, y_train, ~, ~] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'log';
        hl.b_online = 1;
        hl.max_iter_outer= 10000;
        hl.stepsize_policy = AdagradStepsize;
        hl.stepsize_policy.eta_0 = 1;
        hl.stepsize_policy.epsilon = 1e-5;
       
        v_lambda = hl.solve_approx_mirror(A_train, y_train);
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        figure(exp_no*100+1); clf
        plot(v_lambda)
        legend ('Mirror, Adagrad')
        
        %%
        ch_resultsFile = sprintf('results_OHO_%d', exp_no);
        save(ch_resultsFile);
        F = 0;
    end
    
    % Online mirror with RmsProp stepsize
    % testing object-based stepsize policies
    function F = experiment_13(obj)
       
        [A_train, ~, y_train, ~, ~] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'log';
        hl.b_online = 1;
        hl.max_iter_outer= 10000;
        hl.stepsize_policy = RmsPropStepsize;
        hl.stepsize_policy.beta_2 = 0.99; % other values?
        hl.stepsize_policy.eta_0 = 0.01;
        hl.stepsize_policy.epsilon = 1e-6;
   
        v_lambda = hl.solve_approx_mirror(A_train, y_train);
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        figure(exp_no*100+1); clf
        plot(v_lambda)
        legend ('Mirror, RMSProp')
        
        %%
        ch_resultsFile = sprintf('results_OHO_%d', exp_no);
        save(ch_resultsFile);
        F = 0;
    end

    % Online mirror with LR-based stepsize
    % testing object-based stepsize policies
    function F = experiment_14(obj)

        [A_train, ~, y_train, ~, ~] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'log';
        hl.b_online = 1;
        hl.max_iter_outer= 10000;
        
        sp = LinearRegressionStepsize; %Stepsize Policy: Linear regression
        sp.version = 6;
        sp.eta_0 = 0.1;
        sp.beta_2 = 1-1/obj.n_train;
        sp.nu = 0.1;
        sp.kappa = 1/obj.n_train;
        sp.gamma = 0.;
        sp.N = obj.n_train;
        sp.law = @(x)x;
        sp.inv_law = @(x)x;
        
        hl.stepsize_policy = sp;
        v_lambda = hl.solve_approx_mirror(A_train, y_train);
        
        exp_no = obj.determine_experiment_number();        
        %% Figures
        figure(exp_no*100+1); clf
        plot(v_lambda)
        legend ('Mirror, LinearRegressionStepsize')
        
        %%
        ch_resultsFile = sprintf('results_OHO_%d', exp_no);
        save(ch_resultsFile);
        F = 0;
    end
    
    % Online mirror with LR-based stepsize
    % testing object-based stepsize policies
    function F = experiment_15(obj)

        [A_train, ~, y_train, ~, ~] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'log';
        hl.b_online = 1;
        hl.max_iter_outer= 10000;
        
        sp = LinearRegressionStepsize; %Stepsize Policy: Linear regression
        sp.version = 6;
        sp.eta_0 = 0.1;
        sp.beta_2 = 1-1/obj.n_train;
        sp.nu = 0.01;
        sp.kappa = 6;
        
        hl.stepsize_policy = sp;
        v_lambda = hl.solve_approx_mirror(A_train, y_train);
        
        exp_no = obj.determine_experiment_number();        
        %% Figures
        figure(exp_no*100+1); clf
        plot(v_lambda)
        legend ('Mirror, LinearRegressionStepsize')
        %%
        ch_resultsFile = sprintf('results_OHO_%d', exp_no);
        save(ch_resultsFile);
        F = 0;
    end
    
    % Online gradient with Adam-like adapted stepsize
    function F = experiment_21(obj)
        
        [A_train, ~, y_train, ~, ~] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.max_iter_outer= 10000;
        
        spo_adam = AdamStepsize;
        spo_adam.eta_0 = 0.0003;
        spo_adam.beta_2= 0.99;
        
        hl.stepsize_policy = spo_adam;
        %TODO: check, because we are getting negative etas (strange)
        lambda_adam= hl.solve_approx_mirror(A_train, y_train);
        
        exp_no = obj.determine_experiment_number();        
        %% Figures
        figure(exp_no*100+1); clf
        plot(lambda_adam);
        legend ('Grad, ADAM-like adapted stepsize')
        %%
        ch_resultsFile = sprintf('results_OHO_%d', exp_no);
        save(ch_resultsFile);
        F = 0;
    end
    
    % Online gradient with Unconstrained FTML (U-FTML) adapted stepsize
    function F = experiment_22(obj)

        [A_train, ~, y_train, ~, ~] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.max_iter_outer= 10000;
        
        spo_uftml = UftmlStepsize;
        spo_uftml.eta_policy = DiminishingStepsize;
        spo_uftml.eta_policy.law = @(x)x; %1/k stepsize
        spo_uftml.eta_policy.eta_0 = 10;
        spo_uftml.beta_2= 1-1/obj.n_train;
        spo_uftml.beta_1 = 0.9;
        
        hl.stepsize_policy = spo_uftml;       
        lambda_uftml= hl.solve_approx_mirror(A_train, y_train);
        
        exp_no = obj.determine_experiment_number();        
        %% Figures
        figure(exp_no*100+1); clf
        plot(lambda_uftml);
        legend('Grad, U-FTML stepsize')
        %%
        ch_resultsFile = sprintf('results_OHO_%d', exp_no);
        save(ch_resultsFile);
        F = 0;
    end
    
    % Online gradient with U-FTML adapted stepsize
    % Now with 10 times more samples!
    function F = experiment_23(obj)
        obj.n_train = 2000;
        
        [A_train, ~, y_train, ~, ~] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.max_iter_outer= 10000;
        spo_uftml = UftmlStepsize;
        spo_uftml.eta_policy = ConstantStepsize;
        %spo_uftml.eta_policy.law = @(x)x;%^(1/3); %1/k stepsize
%         spo_uftml.eta_policy.eta_0 = 1;
%         spo_uftml.beta_2 = 0.995;   %1-1/(obj.n_train);
%         spo_uftml.beta_1 = 0.99;  %1-10/obj.n_train;
        spo_uftml.eta_policy.eta_0 = 0.01;
        spo_uftml.beta_2= 1-1/obj.n_train;
        spo_uftml.beta_1 = 0.99;
        hl.stepsize_policy = spo_uftml;       
        lambda_uftml= hl.solve_approx_mirror(A_train, y_train);
        
        exp_no = obj.determine_experiment_number();        
        %% Figures
        figure(exp_no*100+1); clf
        plot(lambda_uftml);
        title '2000 training samples'
        legend('Grad, U-FTML stepsize')
        %%
        ch_resultsFile = sprintf('results_OHO_%d', exp_no);
        save(ch_resultsFile);
        F = 0;
    end
    
    % Online mirror with Luismi's Q stepsize
    function F = experiment_24(obj)
        
        obj.n_train = 2000;
        
        %obj.seed = 4;     
        [A_train, ~, y_train, ~, ~] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.max_iter_outer= 20000;
        
        spoq = QStepsize;
        %spo_uftml.eta_policy.law = @(x)x;%^(1/3); %1/k stepsize
        spoq.eta_0 = 1;
        spoq.nu = 1;
        spoq.beta_2 = 1-1/(obj.n_train);
        spoq.beta_1 = 1-1/(obj.n_train);
        %spoq.N = obj.n_train; %this was a property of the previous version
        % of QStepsize (now renamed as QStepsize_old)
        
        hl.stepsize_policy = spoq;       
        lambda_QS= hl.solve_approx_mirror(A_train, y_train);
        
        exp_no = obj.determine_experiment_number();        
        %% Figures
        figure(exp_no*100+1); clf
        plot(lambda_QS);
        title(sprintf('%d training samples', obj.n_train))
        legend('Grad, Q-step')
        %%
        ch_resultsFile = sprintf('results_OHO_%d', exp_no);
        save(ch_resultsFile);
        F = 0;
    end
    
    % Online gradient descent with Luismi's Q stepsize
    function F = experiment_25(obj)
        
        obj.n_train =  2000;
        
        [A_train, ~, y_train, ~, ~] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.max_iter_outer= 20000;
        
        spoq = QStepsize;
        %spo_uftml.eta_policy.law = @(x)x;%^(1/3); %1/k stepsize
        spoq.eta_0 = 1;
        spoq.nu = 5;
        spoq.beta_2 = 1-1/(obj.n_train);
        spoq.beta_1 = 1-1/(obj.n_train);
        
        hl.stepsize_policy = spoq;       
        lambda_QS= hl.solve_approx_mirror(A_train, y_train);

        exp_no = obj.determine_experiment_number();        
        %% Figures
        figure(exp_no*100+1); clf
        plot(lambda_QS);
        title(sprintf('%d training samples', obj.n_train))
        legend('Grad, Q-step')
        %%
        ch_resultsFile = sprintf('results_OHO_%d', exp_no);
        save(ch_resultsFile);
        F = 0;
    end
    
    % Memoryless vs. alg. with memory
    function F = experiment_31(obj)
        
        obj.n_train = 4000;
        obj.p = 100;
        obj.seed = 2;
        [A_train, ~, y_train, ~, ~] = obj.set_up_data();
        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.max_iter_outer= 10000;
        
        spoq = QStepsize;
        memory_factor= 1;
        spoq.eta_0 = 0.1;
        spoq.nu = 10;
        spoq.beta_2 = 1-1/(memory_factor*obj.n_train); % 10*
        spoq.beta_1 = 1-1/(memory_factor*obj.n_train); % 10*
        
        hl.stepsize_policy = spoq;     
        hl.b_debug= 1;
        %hl.normalized_lambda_0 = 1/obj.n_train;
        %By default the OHO_Lasso goes with memory
        [lambda_memory, count_memory]    ...
            = hl.solve_approx_mirror(A_train, y_train);
        
        spoq2 = QStepsize;
        spoq2.eta_0 = 0.1;
        spoq2.nu = 10;
        spoq2.beta_2 = 1-1/(memory_factor*obj.n_train); % 10*
        spoq2.beta_1 = 1-1/(memory_factor*obj.n_train); % 10*
        hl.stepsize_policy = spoq2;
        hl.b_memory = 0; %Now we experiment with the Memoryless option
        [lambda_memoryless, count_memoryless] ...
            = hl.solve_approx_mirror(A_train, y_train);             
        
        exp_no = obj.determine_experiment_number();        
        %% Figures
        figure(exp_no*100+1); clf
        obj.show_lambda_vs_ista_iterates({lambda_memoryless; lambda_memory}, ...
            {count_memoryless;count_memory})        
        legend('Memoryless','With Memory')
        title(sprintf('%d training samples', obj.n_train))
        %%
        ch_resultsFile = sprintf('results_OHO_%d', exp_no);
        save(ch_resultsFile);
        F = 0;
    end
    
    % Memoryless, comparison between exact (let ISTA converge) and
    % inexact (few iterations of ista for each iteration of lambda)
    %%
    % TODO: SHORTEN EXPERIMENT CODE
    function F = experiment_32(obj) 
        
        obj.n_train = 4000;
        obj.p = 400;
        obj.seed = 3;
        [A_train, A_test, y_train, y_test, ~] = obj.set_up_data();
                
        memory_factor= 1;
        spoq = QStepsize; %stepsize policy object
        spoq.eta_0 = 1;
        spoq.nu = 10;
        spoq.beta_2 = 1-1/(memory_factor*obj.n_train); % 10*
        spoq.beta_1 = 1-1/(memory_factor*obj.n_train); % 10*
        
        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.max_iter_outer= 20000;       
        hl.stepsize_policy = spoq;     
        hl.b_debug= 1;
        hl.b_memory = 0;
        hl.tol_w = 1e-3;
        [lambda_exact, count_exact, v_w_exact] ...
            = hl.solve_approx_mirror(A_train, y_train);
        final_lambda_exact= lambda_exact(find(count_exact>0,1, 'last'));
        
        hl.stepsize_policy.reset()
        hl.tol_w = 1e1; %inexact
        [lambda_inexact, count_inexact, v_w_inexact] ...
            = hl.solve_approx_mirror(A_train, y_train);
        final_lambda_inexact = lambda_inexact(find(count_inexact>0,1, 'last'));
        
        % Evaluate test error             
        v_testLambdas = final_lambda_inexact*[0.01 0.1 0.3 0.5 0.8 0.9 1.1 1.25 2 3 10];
        st_hyperparams = struct; 
        for k =1:length(v_testLambdas)
            st_hyperparams(k).lambda = v_testLambdas(k);
        end
        v_testErrors = obj.heldOut_validation_errors(A_train, y_train, A_test, y_test, ...
            st_hyperparams, v_w_inexact);
                
        test_error_exact = mean((y_test-A_test*v_w_exact).^2)/mean(y_test.^2);
        test_error_inexact = mean((y_test-A_test*v_w_inexact).^2)/mean(y_test.^2);
        
        exp_no = obj.determine_experiment_number();        
        %% Figures
        
        figure(exp_no*100+1); clf        
        obj.show_lambda_vs_ista_iterates({lambda_exact; lambda_inexact}, ...
            {count_exact;count_inexact})
        legend('Exact','Inexact')

        figure(exp_no*100+2); clf
        semilogx(v_testLambdas, v_testErrors);hold on
        plot(final_lambda_inexact, test_error_inexact, 'xb')
        plot(final_lambda_exact, test_error_exact, 'xr'); 
        legend ('Test error', 'inexact', 'exact')
        xlabel \lambda
        ylabel NMSE

        %%
        ch_resultsFile = sprintf('results_OHO_%d', exp_no);
        save(ch_resultsFile);
        F = 0;

    end
    
    % Memoryless, inexact
    % (few iterations of ista for each iteration of lambda) 
    % Diminishing stepsize (similar convergence to that of exp 32)
    % run and time
    function F = experiment_33(obj)
        
        obj.n_train = 4000;
        obj.p = 400;
        obj.seed = 4;
        [A_train, A_test, y_train, y_test, ~] = obj.set_up_data();
        
%         memory_factor= 1;
%         sp = QStepsize;
%         sp.eta_0 = 1;
%         sp.nu = 1;
%         sp.beta_2 = 1-1/(memory_factor*obj.n_train);
%         sp.beta_1 = 1-1/(memory_factor*obj.n_train);
        sp = DiminishingStepsize;
        sp.eta_0 = 1;
        sp.law = @(k)sqrt(1000*k);

        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.max_iter_outer= 20000;
        hl.stepsize_policy = sp;     
        hl.b_debug= 0;
        hl.b_memory = 0;
        hl.tol_g = 1e-3; %less demanding convergence criterion
        hl.tol_w = 1e1;
        [lambda_inexact, count_inexact, v_w_inexact, v_eta_inexact] ...
            = hl.solve_approx_mirror(A_train, y_train);
        final_lambda_inexact = lambda_inexact(find(count_inexact>0,1, 'last'));

        % Evaluate test error
        %final_lambda_exact= lambda_exact(find(count_exact>0,1, 'last'));
        v_testLambdas = final_lambda_inexact*[0.01 0.1 0.3 0.5 0.8 0.9 1.1 1.25 2 3 10];
        st_hyperparams = struct; 
        for k =1:length(v_testLambdas)
            st_hyperparams(k).lambda = v_testLambdas(k);
        end
        v_testErrors = obj.heldOut_validation_errors(A_train, y_train, A_test, y_test, ...
            st_hyperparams, v_w_inexact);
        
        test_error_inexact = mean((y_test-A_test*v_w_inexact).^2)/mean(y_test.^2);
        
        exp_no = obj.determine_experiment_number();        
        %% Figures    
        figure(exp_no*100+1); clf        
        obj.show_lambda_vs_ista_iterates({lambda_inexact}, ...
            {count_inexact})
        legend('Inexact')

        figure(exp_no*100+2); clf
        semilogx(v_testLambdas,  v_testErrors);hold on
        plot(final_lambda_inexact, test_error_inexact, 'xb')
        %plot(final_lambda_exact, test_error_exact, 'xr'); 
        legend ('Test error', 'inexact') %, 'exact'
        xlabel \lambda
        ylabel NMSE
        
        figure(exp_no*100+3); clf
        semilogy(v_eta_inexact);
        xlabel 'time'
        ylabel 'stepsize \eta'
        
        %%
        ch_resultsFile = sprintf('results_OHO_%d', exp_no);
        save(ch_resultsFile);
        F = 0;

    end    
    
    % Memoryless, exact
    % comparison with exact Leave-One-Out
    function F = experiment_34(obj)
        
        obj.n_train = 2000; %!
        obj.p = 100;
        obj.seed = 4;
        [A_train, A_test, y_train, y_test, ~] = obj.set_up_data();
            
        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.max_iter_outer= 10000; %!20000
        hl.approx_type = 'hard';
%         memory_factor= 1;
%         spoq = QStepsize;
%         spoq.eta_0 = 10;
%         %spoq.dqg = 0.1;
%         spoq.nu = 0.01;
%         spoq.beta_2 = 1-1/(memory_factor*obj.n_train);
%         spoq.beta_1 = 1-1/(memory_factor*obj.n_train);
        spoq = ConstantStepsize;
        spoq.eta_0 = 0.001;
        hl.stepsize_policy = spoq;     
        hl.b_debug= 1;
        hl.normalized_lambda_0 = obj.n_train^(-1/4);
        hl.b_memory = 0; %! 1;
        hl.tol_w = 1; %! 1e-4;
        [v_lambda_exact, v_count_exact, v_w_exact] ...
            = hl.solve_approx_mirror(A_train, y_train);
        final_lambda_exact = v_lambda_exact(find(v_count_exact>0,1, 'last'));
        
        hl_gd = OHO_Lasso;
        hl_gd.stepsize_policy = ConstantStepsize;
        hl_gd.stepsize_policy.eta_0 = .1/obj.n_train;
        hl_gd.max_iter_outer = 50;
        hl_gd.mirror_type = 'grad';
        hl_gd.approx_type = 'hard';
        %hl_gd.b_memory = 0; %!
        %hl_gd.tol_w = 1;    %!
        [v_lambda_gd, v_count_gd, v_w_gd] = hl_gd.solve_approx_mirror(A_train, y_train, final_lambda_exact);
        final_lambda_gd   = v_lambda_gd(find(v_count_gd>0, 1, 'last'));

        
        % Evaluate test error
        v_testLambdas = final_lambda_exact*logspace(-2, 2);
        st_hyperparams = struct; 
        for k =1:length(v_testLambdas)
            st_hyperparams(k).lambda = v_testLambdas(k);
        end
        v_testErrors = obj.heldOut_validation_errors(A_train, y_train, A_test, y_test, ...
            st_hyperparams, v_w_exact);
        test_error_exact = mean((y_test-A_test*v_w_exact).^2)./mean(y_test.^2);
        test_error_gd    = mean((y_test-A_test*v_w_gd).^2)./mean(y_test.^2);
        
        % Evaluate loo errors
        disp 'Computing loo errors'
     
        v_looLambdas = final_lambda_exact*sort([1 linspace(0.4, 2.5, 15)]);
        %v_looLambdas = final_lambda_exact*sort([1 linspace(0.8, 1.5, 15)]);

        st_hyperparams_loo = struct; 
        for k =1:length(v_looLambdas)
            st_hyperparams_loo(k).lambda = v_looLambdas(k);
        end
        v_looErrors = obj.compute_loo_errors(st_hyperparams_loo, v_w_exact, A_train, y_train);
        ind_final_lambda = find(v_looLambdas==final_lambda_exact, 1,'first');
        loo_error_exact = v_looErrors(ind_final_lambda);
        
        exp_no = obj.determine_experiment_number();        
        %% Figures    
        figure(exp_no*100+1); clf  
        obj.show_lambda_vs_ista_iterates({v_lambda_exact, v_lambda_gd}, ...
            {v_count_exact, v_count_gd})
        legend('Exact', 'GD')
        
        figure(exp_no*100+2); clf
        loglog(v_testLambdas, v_testErrors);hold on
        loglog(v_looLambdas, v_looErrors,'r')
        plot(final_lambda_exact, test_error_exact, 'xb')
        plot(final_lambda_exact, loo_error_exact,  'xr') 
        plot(final_lambda_gd,    test_error_exact, 'ob')
         
        legend ('Held-out validation error', ...
             'Leave-one-out error', ...
            'exact, validation error', 'exact, LOO error')
        xlabel \lambda
        ylabel MSE
                
        %%
        ch_resultsFile = sprintf('results_OHO_%d', exp_no);
        save(ch_resultsFile);
        F = 0;

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
        hl.b_debug= 1;
        hl.normalized_lambda_0 = 1/sqrt(obj.n_train);
        hl.tol_w = 1e-4;
        hl.b_memory = 1;
        [lambda_exact, count_exact, v_w_exact] ...
            = hl.solve_approx_mirror(A_train, y_train);
        
        hl.tol_w = 1e0;
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
        hl.b_debug= 1;
        hl.normalized_lambda_0 = 1/sqrt(obj.n_train);
        hl.tol_w = 1e-4;
        hl.b_memory = 1;
        [lambda_exact, count_exact, v_w_exact] ...
            = hl.solve_approx_mirror(A_train, y_train);
        
        hl.tol_w = 1e0;
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
        hl.b_debug= 1;
        hl.normalized_lambda_0 = 1/sqrt(obj.n_train);        
        hl.tol_w = 1e0;
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
        hl.b_debug= 1;
        hl.normalized_lambda_0 = 1/sqrt(obj.n_train);
        hl.tol_w = 1e0;
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
        hl.b_debug= 1;
        hl.normalized_lambda_0 = 1/obj.n_train;        
        hl.tol_w = 1e0;
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
        % Elastic Net ( rest of params were copied from Experiment_36)
        % with memory (TODO: check taht OHO_Elastic net actually uses
        % memory)
        
        obj.n_train = 200;
        obj.p = 100;
        obj.b_colinear = 1;
        obj.sigma = 10;
        obj.seed = 9;
        [A_train, ~, y_train, ~, ~] = obj.set_up_data();
        hen = OHO_ElasticNet;
        hen.mirror_type = 'grad'; %! it was log before
        hen.b_online = 1;
        hen.max_iter_outer= 5000; %!
        
%        memory_factor= 10;
%         spoq = QStepsize;
%         spoq.eta_0 = 10;
%         spoq.beta_2 = 1-1/(memory_factor*obj.n_train);
%         spoq.beta_1 = 1-1/(memory_factor*obj.n_train);
%         spoq.nu = 20;
        sp = DiminishingStepsize;
        %sp.law = @(x)x;
        sp.eta_0 = 10000;
        
        hen.stepsize_policy = sp;     
        hen.b_debug= 1;
        hen.normalized_lambda_0 = 0.1/obj.n_train; %! it was 1/n_train
%         hl.normalized_lambda_0 = 1/sqrt(obj.n_train);
%         hl.tol_w = 1e-4;
%         hl.b_memory = 1;
%         [lambda_exact, count_exact, v_w_exact] ...
%             = hl.solve_approx_mirror(A_train, y_train);
        
        hen.tol_w = 1e0;
        hen.b_memory = 0;
        [lambda_inexact, count_inexact, v_w_final] ...
            = hen.solve_approx_mirror(A_train, y_train);

        figure(50); clf
        plot(cumsum(count_inexact), lambda_inexact);  
        title 'Elastic net'
        xlabel '# ISTA iterations'
        ylabel '\lambda'
        legend('\lambda','\rho')
        drawnow
        
        %%
        % Evaluate test error

%         final_lambda_exact = lambda_exact(find(count_exact>0, 1, 'last'));
        final_niter = find(count_inexact>0, 1, 'last');
        final_lambda = lambda_inexact(1, final_niter);
        final_rho    = mean(lambda_inexact(2, ceil(final_niter/2):final_niter));
        v_testLambdas = final_lambda*logspace(-1, 0.5);
        v_looLambdas = final_lambda*linspace(0.3, 3, 15);
        v_aloLambdas = final_lambda*linspace(0.3, 3, 200);
        v_testErrors = zeros(size(v_testLambdas));
        v_valErrors  = v_testErrors;
        v_looErrors  = zeros(size(v_looLambdas));
        
        compute_loo = 0;
        compute_alo = 0;
        if compute_loo
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
            
            figure(135); clf
            loglog(v_testLambdas, v_testErrors);hold on
            loglog(v_testLambdas, v_valErrors, 'r')
            loglog(v_looLambdas, v_looErrors,'g')
        
            plot(final_lambda, test_error, 'xb')
            plot(final_lambda, val_error, 'xr')
            plot(final_lambda, loo_error, 'xg')
            
            my_legend = {'Out-of-sample test error', ...
            'Out-of-sample validation error', 'Leave-one-out error',...
            'Inexact, memoryless', '', '' };
        
            xlabel \lambda
            ylabel MSE
            if compute_alo
                disp 'Computing approximate loo errors'
                v_aloErrors = obj.estimate_alo_error(A_train, y_train, ...
                    [v_aloLambdas' ones(size(v_aloLambdas))'*final_rho], ...
                    v_w_final);
                loglog(v_aloLambdas, v_aloErrors, 'm')
                my_legend = {my_legend(:), 'Approximate Loo (Wang, 2018)'};
            end
            legend(my_legend)
        end
        
        %%
        save SavedResults_50
    end
    
    function F = experiment_51(obj)
        % Elastic Net ( copied from exp 50 and now we use QStepsize)
        % memoryless
        % inexact
        obj.n_train = 200;
        obj.p = 100;
        obj.b_colinear = 1;
        obj.sigma = 10;
        obj.seed = 9;
        [A_train, A_test, y_train, y_test, true_x] = obj.set_up_data();
        hen = OHO_ElasticNet;
        hen.mirror_type = 'grad'; %! it was log before
        hen.b_online = 1;
        hen.b_memory = 0;
        hen.tol_w = 1e0;
        hen.max_iter_outer= 5000; %!
        
        memory_factor= 10;
        spoq = QStepsize_v;
        spoq.eta_0 = 2000;
        spoq.beta_2 = 1-1/(memory_factor*obj.n_train);
        spoq.beta_1 = 1-1/(memory_factor*obj.n_train);
        spoq.nu = 100;
        spoq.dqg = 0.01;
        
        hen.stepsize_policy = spoq;     
        hen.b_debug= 1;
        hen.normalized_lambda_0 = 1/obj.n_train; %! it was 1/n_train
%         hl.normalized_lambda_0 = 1/sqrt(obj.n_train);
%         hl.tol_w = 1e-4;
%         hl.b_memory = 1;
%         [lambda_exact, count_exact, v_w_exact] ...
%             = hl.solve_approx_mirror(A_train, y_train);
        
        hen.tol_w = 1e0;
        hen.b_memory = 0;
        [lambda_inexact, count_inexact, v_w_final] ...
            = hen.solve_approx_mirror(A_train, y_train);

        figure(51); clf
        plot(cumsum(count_inexact), lambda_inexact);  
        title 'Elastic net'
        xlabel '# ISTA iterations'
        ylabel '\lambda'
        legend('\lambda','\rho')
        drawnow
        
        %%
        % Evaluate test error

%         final_lambda_exact = lambda_exact(find(count_exact>0, 1, 'last'));
        final_niter = find(count_inexact>0, 1, 'last');
        final_lambda = lambda_inexact(1, final_niter);
        final_rho    = mean(lambda_inexact(ceil(final_niter/2):final_niter, 2));
        v_testLambdas = final_lambda*logspace(-1, 0.5);
        v_looLambdas = final_lambda*linspace(0.3, 3, 15);
        v_aloLambdas = final_lambda*linspace(0.3, 3, 200);
        v_testErrors = zeros(size(v_testLambdas));
        v_valErrors  = v_testErrors;
        v_looErrors  = zeros(size(v_looLambdas));
        
        compute_loo = 0;
        compute_alo = 0;
        if compute_loo
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
            
            figure(135); clf
            loglog(v_testLambdas, v_testErrors);hold on
            loglog(v_testLambdas, v_valErrors, 'r')
            loglog(v_looLambdas, v_looErrors,'g')
        
            plot(final_lambda, test_error, 'xb')
            plot(final_lambda, val_error, 'xr')
            plot(final_lambda, loo_error, 'xg')
            
            my_legend = {'Out-of-sample test error', ...
            'Out-of-sample validation error', 'Leave-one-out error',...
            'Inexact, memoryless', '', '' };
        
            xlabel \lambda
            ylabel MSE
            if compute_alo
                disp 'Computing approximate loo errors'
                v_aloErrors = obj.estimate_alo_error(A_train, y_train, ...
                    [v_aloLambdas' ones(size(v_aloLambdas))'*final_rho], ...
                    v_w_final);
                loglog(v_aloLambdas, v_aloErrors, 'm')
                my_legend = {my_legend(:), 'Approximate Loo (Wang, 2018)'};
            end
            legend(my_legend)
        end
        
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
        hl.b_debug = my_debug;
        hl.tol_w = 1e0;
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
        hen.b_debug = my_debug;
        hen.tol_w = 1e0;
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

methods (Static) % Routines such as out-of-sample error calculations 
        
    function alo_nmse = estimate_alo_error(m_X_train, v_y_train, ...
            m_hyperparams, v_w_0)
        N = length(v_y_train); assert(size(m_X_train, 1)==N);
        P = size(m_X_train, 2);
        m_Phi = m_X_train'*m_X_train;
        v_r = m_X_train'*v_y_train;
        my_alpha = 10/trace(m_Phi);
        hl = OHO_Lasso;
        hl.tol_w = 1e-3;
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
    
    function v_oos_errors = heldOut_validation_errors(m_X_train, v_y_train,...
            m_X_test, v_y_test, st_hyperparams, v_w_0)
        N = length(v_y_train); assert(size(m_X_train, 1)==N);
        P = size(m_X_train, 2);
        v_oos_errors = nan(length(st_hyperparams), 1);
        for k = 1:length(st_hyperparams)
            if isfield(st_hyperparams(k) ,'rho')
                m_Phi = m_X_train'*m_X_train/N + st_hyperparams(k).rho*eye(P);
            else
                m_Phi = m_X_train'*m_X_train/N;
            end
            v_r = m_X_train'*v_y_train/N;
            my_alpha = 10/trace(m_Phi);
            hl = OHO_Lasso;
            hl.tol_w = 1e-3;
            v_w_0 = hl.ista(v_w_0, m_Phi, v_r, my_alpha, st_hyperparams(k).lambda);
            v_oos_errors(k) = mean((v_y_test - m_X_test*v_w_0).^2)./mean(v_y_test.^2);
        end
    end
    function oos_error = estimate_outOfSample_error(m_X_train, v_y_train,...
            m_X_test, v_y_test, lambda, v_w_0, rho)
%         N = length(v_y_train); assert(size(m_X_train, 1)==N);
%         P = size(m_X_train, 2);
%         if exist('rho', 'var')
%             m_Phi = m_X_train'*m_X_train + rho*eye(P);
%         else
%             m_Phi = m_X_train'*m_X_train;
%         end
%         v_r = m_X_train'*v_y_train;
%         my_alpha = 10/trace(m_Phi);
%         hl = OHO_Lasso;
%         hl.tol_w = 1e-3;
%         v_w_test = hl.ista(v_w_0, m_Phi, v_r, my_alpha, lambda);
%         oos_error = mean((v_y_test - m_X_test*v_w_test).^2)./mean(v_y_test.^2);
    end
        
    function [loo_error, v_w_j] = exact_loo(m_X, v_y, st_hyperparams, v_w_0)
        
        N = length(v_y); assert(size(m_X, 1)==N);
        P = size(m_X, 2);
        assert(isscalar(st_hyperparams))
        lambda = st_hyperparams.lambda;
        
        if isfield(st_hyperparams, 'rho')
            m_Phi = m_X'*m_X/N + st_hyperparams.rho*eye(P);
        else
            m_Phi = m_X'*m_X/N ;
        end
        v_r = m_X'*v_y/N;
        my_alpha = 10/trace(m_Phi);
        hl = OHO_Lasso;
        hl.tol_w = 1e-4;
        v_looErrors = zeros(N,1);
        ltc = LoopTimeControl(N);
        for j =1:N
            v_x_j = m_X(j,:)';
            m_Phi_j = m_Phi - v_x_j * v_x_j'/N;
            v_r_j   = v_r   - v_y(j)* v_x_j/N;
            v_w_j = hl.ista(v_w_0, m_Phi_j, v_r_j, my_alpha, lambda);
            v_looErrors(j) = v_y(j)-m_X(j,:)*v_w_j;
            ltc.go(j);
        end
        loo_error = mean(v_looErrors.^2)./mean(v_y.^2);
    end
    
    function v_looErrors = compute_loo_errors(st_hyperparams, v_w_0, m_A, v_y)
        
        v_looErrors = nan(size(st_hyperparams));
        for k = 1:length(st_hyperparams)
            v_looErrors(k) = OHOExperiments.exact_loo(m_A, v_y, st_hyperparams(k), v_w_0);
        end
    end
    
    function n = determine_experiment_number
        % returns a string with the experiment number.
        % Requires that the experiment functions are named exactly
        % 'experiment_%d', where %d is the experiment number.
        s = dbstack;
        s_file = s(2).file; assert(s_file(end)=='m')        
        ch_theNumber = replace(s(2).name, [s_file(1:end-1) 'experiment_'], '');
        n = sscanf(ch_theNumber, '%d');
    end

end

methods (Static) % Plotting
        
    function show_lambda_iterates(m_lambdas)
        plot(m_lambdas')
        title '\lambda iterates'
        xlabel('time $t$', 'interpreter', 'latex')
        ylabel '\lambda'
    end
    
    function show_lambda_vs_ista_iterates(c_lambdas, c_itcounts)
        for k = 1:size(c_lambdas, 1)
            plot(cumsum(c_itcounts{k}), c_lambdas{k}); 
            hold on
        end
        xlabel('# ISTA iterations')
        ylabel '\lambda'
    end
end

end