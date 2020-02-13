classdef DynamicExperiments
properties
    T         = 1000;    % number of time instants
    P         = 100;     % dimensionality of x vectors
    sigma     = 10;  % Variance of the noise
    sparsity  = 0.1;     % proportion of entries of true_w that are > 0
    seed      = 1        % Random seed
end

methods % Constructor and pseudo-streaming data creating procedure
    function obj = DynamicExperiments() %The constructor sets the path
        addpath('utilities/')
    end
    
    function [m_X, v_y, true_w, m_X_test, v_y_test] ...
            = generate_pseudo_streaming_data(obj)
        %create train and test data
        rng(obj.seed);
        true_w = double(rand(obj.P, 1) < obj.sparsity); % sparse coefs
        m_X          = randn(obj.P, obj.T); % time series of x vectors
        m_X_test     = randn(obj.P, obj.T); % time series of x vectors
        epsilon      = randn(1, obj.T)*obj.sigma; % noise
        epsilon_test = randn(1, obj.T)*obj.sigma;
        v_y          = true_w'*m_X      + epsilon;
        v_y_test     = true_w'*m_X_test + epsilon_test;
    end
    
    function [m_W, v_lambda, v_loss] = drill_online_learning(obj, ...
            estimator, m_X, v_y, initial_lambda)
        %allocate memory
        m_W      = zeros(obj.P, obj.T);
        v_lambda = zeros(1, obj.T);
        v_lambda(1) = initial_lambda;
        v_loss   = zeros(1, obj.T);
        
        m_Phi = zeros(obj.P);
        v_r   = zeros(obj.P, 1);
        L = nan;
        
        %main loop
        ltc = LoopTimeControl(obj.T-1);
        for t = 1:obj.T-1
            switch class(estimator)
                case 'DynamicLassoHyper'
                    [m_W(:, t+1), v_lambda(t+1), v_loss(t+1), L] =  ...
                        estimator.update( m_W(:,t), v_lambda(t), ...
                        m_X(:,t), v_y(t), m_X(:, t+1), v_y(t+1), L, t);
                case 'DynamicRecursiveLassoHyper'
                    [m_W(:, t+1), v_lambda(t+1), v_loss(t+1), m_Phi, v_r] =  ...
                        estimator.update( m_W(:,t), v_lambda(t), ...
                        m_X(:,t), v_y(t), m_X(:, t+1), v_y(t+1), m_Phi, v_r);
            end
            ltc.go(t);
        end
    end
    
    function v_avgLoss = grid_search_lambda(obj, estimator, m_X, v_y, v_lambda_grid)
        estimator.stepsize_lambda = 0;
        n_lambdas = numel(v_lambda_grid);
        v_avgLoss = zeros(size(v_lambda_grid));
        for li = 1:n_lambdas
            [~, ~, v_loss] = drill_online_learning(obj, estimator, m_X,...
                v_y, v_lambda_grid(li));
            v_avgLoss(li) = mean(v_loss);
        end
    end
end

methods %experiments
    
        %First experiment, stationary data
    function F = experiment_1(obj) 
        
        obj.T =  100000;
        %generate data
        [m_X, v_y, true_w] = generate_pseudo_streaming_data(obj);
               
        %create estimators
        dlh = DynamicLassoHyper;
        dlh.stepsize_w = 1e-4;
        dlh.stepsize_lambda = 4e-2;
        dlh.approx_type = 'hard';
        
        dlh_fixedLambda = DynamicLassoHyper;
        dlh_fixedLambda.stepsize_w = 1e-4;
        dlh_fixedLambda.stepsize_lambda = 0;
        lambda_fixed = 0.2;
        
        %run the online estimators
        [m_W, v_lambda, v_loss] = obj.drill_online_learning(dlh, m_X, v_y, 0);
        [m_W_ls, ~, v_loss_ls]  = obj.drill_online_learning(dlh_fixedLambda, m_X, v_y, 0);
        [m_W_selector, ~, v_loss_selector] = ...
            obj.drill_online_learning(dlh_fixedLambda, m_X, v_y, lambda_fixed);
        
        %% Figures
        exp_no = obj.determine_experiment_number();
        ch_lambdaLabel = sprintf('\\lambda = %g', lambda_fixed);
        
        figure(exp_no*100+1); clf
        subplot(1,2,1);
        plot(v_lambda)
        title '\lambda'
        subplot(1,2,2)
        plot(cumsum(v_loss)./(1:obj.T))
        hold on
        plot(cumsum(v_loss_ls)./(1:obj.T));
        plot(cumsum(v_loss_selector)./(1:obj.T));
        legend('OHO', 'ls', ch_lambdaLabel)
        title 'average loss'
        
        figure(exp_no*100+2); clf
        thick_true_w = repmat(true_w, [1 1000]);
        imagesc(abs([thick_true_w m_W m_W_ls m_W_selector]))
        title(['w true, Dynamic H, LS, ' ch_lambdaLabel])
        
        ch_resultsFile = sprintf('results_DE_%d', exp_no);
        save(ch_resultsFile);
        F = 0;
    end
    
    % Instead of DynamicLassoHyper, we try DynamicRecursiveLassoHyper
    % stationary data
    function F = experiment_2(obj) 
        
        
        obj.T = 200000;
        obj.sigma = 20;
        %generate data
        [m_X, v_y, true_w] = generate_pseudo_streaming_data(obj);
               
        %create estimators
        dlh = DynamicRecursiveLassoHyper;
        dlh.stepsize_w = 1e-4;
        dlh.stepsize_lambda = 3e-2;
        dlh.approx_type = 'hard';
        dlh.forgettingFactor = 0.99;
        
        dlh_fixedLambda = DynamicRecursiveLassoHyper;
        dlh_fixedLambda.stepsize_w = 1e-4;
        dlh_fixedLambda.stepsize_lambda = 0;
        dlh_fixedLambda.forgettingFactor = 0.99;
        lambda_fixed = 0.2;
        
        %run the online estimators
        [m_W, v_lambda, v_loss] = obj.drill_online_learning(dlh, m_X, v_y, 0);
        [m_W_ls, ~, v_loss_ls]  = obj.drill_online_learning(dlh_fixedLambda, m_X, v_y, 0);
        [m_W_selector, ~, v_loss_selector] = ...
            obj.drill_online_learning(dlh_fixedLambda, m_X, v_y, lambda_fixed);
        
        %% Figures
        exp_no = obj.determine_experiment_number();
        ch_lambdaLabel = sprintf('\\lambda = %g', lambda_fixed);
        
        figure(exp_no*100+1); clf
        subplot(1,2,1);
        plot(v_lambda)
        title '\lambda'
        subplot(1,2,2)
        plot(cumsum(v_loss)./(1:obj.T))
        hold on
        plot(cumsum(v_loss_ls)./(1:obj.T));
        plot(cumsum(v_loss_selector)./(1:obj.T));
        legend('OHO', 'ls', ch_lambdaLabel)
        title 'average loss'
        
        figure(exp_no*100+2); clf
        thick_true_w = repmat(true_w, [1 obj.T/10]);
        imagesc(abs([thick_true_w m_W m_W_ls m_W_selector]))
        title(['w true, Dynamic H, LS, ' ch_lambdaLabel])
        
        ch_resultsFile = sprintf('results_DE_%d', exp_no);
        save(ch_resultsFile);
        F = 0;
    end
    
    % Hard vs soft
    % stationary data
    function F = experiment_3(obj) 
        
        
        obj.T = 200000;
        obj.sigma = 20;
        %generate data
        [m_X, v_y, true_w] = generate_pseudo_streaming_data(obj);
               
        %create estimators
        dlh_hard = DynamicRecursiveLassoHyper;
        dlh_hard.stepsize_w = 1e-4;
        dlh_hard.stepsize_lambda = 3e-2;
        dlh_hard.approx_type = 'hard';
        dlh_hard.forgettingFactor = 0.99;
        
        dlh_soft = dlh_hard;
        dlh_soft.approx_type = 'soft';
        
        dlh_fixedLambda = DynamicRecursiveLassoHyper;
        dlh_fixedLambda.stepsize_w = 1e-4;
        dlh_fixedLambda.stepsize_lambda = 0;
        dlh_fixedLambda.forgettingFactor = 0.99;
        %lambda_fixed = 0.2;
        
        %run the online estimators
        [m_W_hard, v_lambda_hard, v_loss_hard] = obj.drill_online_learning(dlh_hard, m_X, v_y, 0);
        [m_W_soft, v_lambda_soft, v_loss_soft] = obj.drill_online_learning(dlh_soft, m_X, v_y, 0);
       
        [m_W_ls, ~, v_loss_ls]  = obj.drill_online_learning(dlh_fixedLambda, m_X, v_y, 0);
%         [m_W_selector, ~, v_loss_selector] = ...
%             obj.drill_online_learning(dlh_fixedLambda, m_X, v_y, lambda_fixed);
        
        %% Figures
        exp_no = obj.determine_experiment_number();
        %ch_lambdaLabel = sprintf('\\lambda = %g', lambda_fixed);
        
        figure(exp_no*100+1); clf
        subplot(1,2,1);
        plot(v_lambda_hard)
        hold on
        plot(v_lambda_soft)
        legend('hard', 'soft');
        title '\lambda'
        subplot(1,2,2)
        plot(cumsum(v_loss_hard)./(1:obj.T))
        hold on
        plot(cumsum(v_loss_soft)./(1:obj.T));
        plot(cumsum(v_loss_ls)./(1:obj.T));
        legend('hard', 'soft', 'ls')
        title 'average loss'
        
        figure(exp_no*100+2); clf
        thick_true_w = repmat(true_w, [1 obj.T/10]);
        imagesc(abs([thick_true_w m_W_hard m_W_soft m_W_ls]))
        title('w true, hard, soft, LS')
        
        ch_resultsFile = sprintf('results_DE_%d', exp_no);
        save(ch_resultsFile);
        F = 0;
    end

    % Adaptive stepsize, and comparison with the cross-validated lambda
    % stationary data
    function F = experiment_4(obj) 
        
        obj.T = 30000;
        %generate data
        [m_X_train, v_y_train, true_w, m_X_test, v_y_test] = generate_pseudo_streaming_data(obj);
               
        %create estimators
        dlh = DynamicLassoHyper;
        dlh.stepsize_w = @(L, t) 1/(L*sqrt(t));
        dlh.stepsize_lambda = 2e-1;
        dlh.approx_type = 'hard';
        
        %do a grid search over lambda
        v_lambda_grid = linspace(0.07, 0.2, 10);
        v_avgLoss = obj.grid_search_lambda(dlh, m_X_train, v_y_train, v_lambda_grid);
        [~, best_lambda_idx] = min(v_avgLoss);
        
        dlh_fixedLambda = DynamicLassoHyper;
        dlh_fixedLambda.stepsize_w = @(L, t) 1/(L*sqrt(t));
        dlh_fixedLambda.stepsize_lambda = 0;
        lambda_fixed = v_lambda_grid(best_lambda_idx);
        
        %run the online estimators
        [m_W, v_lambda, v_loss] = obj.drill_online_learning(dlh, m_X_test, v_y_test, 0);
        [m_W_bestLambda, ~, v_loss_bestLambda] = ...
            obj.drill_online_learning(dlh_fixedLambda, m_X_test, v_y_test, lambda_fixed);
        v_loss_genie = ((v_y_test - true_w'*m_X_test).^2);
        v_loss_zero  = (v_y_test).^2;
        
        %% Figures
        exp_no = obj.determine_experiment_number();
        ch_lambdaLabel = sprintf('\\lambda = %g', lambda_fixed);
        
        figure(exp_no*100+1); clf
        subplot(1,2,1);
        plot(v_lambda)
        title '\lambda'
        subplot(1,2,2)
        obj.plot_normalizedLosses(...
            [v_loss; v_loss_bestLambda; v_loss_genie], v_loss_zero);
        legend({'OHO', ch_lambdaLabel, 'genie'})
        title 'average loss'
                
        figure(exp_no*100+2); clf
        thick_true_w = repmat(true_w, [1 1000]);
        imagesc(abs([thick_true_w m_W m_W_bestLambda]))
        title(['w true, Dynamic H, Grid search, ' ch_lambdaLabel])
        
        figure(exp_no*100+3); clf
        plot(v_lambda_grid, v_avgLoss)
        xlabel \lambda
        ylabel 'train loss'
        
        ch_resultsFile = sprintf('results_DE_%d', exp_no);
        save(ch_resultsFile);
        F = 0;
        
        
    end

    % Trying different stepsizes of lambda, lets see if all of them converge 
    % to the same neighborhood
    % stationary data
    function F = experiment_5(obj) 
        
        obj.T = 3000; %! 30000
        obj.sigma = 1;
        %generate data
        [m_X, v_y, true_w] = generate_pseudo_streaming_data(obj);
               
        %create estimators
        dlh = DynamicLassoHyper;
        dlh.stepsize_w = @(L, t) 1/(L*sqrt(t));
        dlh.approx_type = 'hard';
        
        n_betas = 10;
        v_grid_beta = logspace(-3, -1, n_betas);
        t_W = zeros(obj.P, obj.T, n_betas);
        m_lambdas = zeros(n_betas ,obj.T);
        m_losses = zeros(n_betas, obj.T);
        v_avgLoss  = zeros(n_betas, 1);
        %run the online estimators
        for ib = 1:n_betas
            dlh.stepsize_lambda = v_grid_beta(ib);
            [t_W(:,:,ib), m_lambdas(ib,:), m_losses(ib,:)] = ...
                obj.drill_online_learning(dlh, m_X, v_y, 0);
            v_avgLoss(ib) = mean(m_losses(ib,:));
        end
        v_loss_genie = ((v_y - true_w'*m_X).^2);
        v_loss_zero  = (v_y).^2;
        
        %% Figures
        exp_no = obj.determine_experiment_number();
        %ch_lambdaLabel = sprintf('\\lambda = %g', lambda_fixed);
        
        figure(exp_no*100+1); clf
        subplot(1,2,1);
        plot(m_lambdas')
        title '\lambda'
        legend('TODO!')
        subplot(1,2,2)
        obj.plot_normalizedLosses(...
            [m_losses; v_loss_genie], v_loss_zero);
        legend({'TODO!', 'genie'})
        title 'average loss'
                
        figure(exp_no*100+2); clf
        thick_true_w = repmat(true_w, [1 1000]);
        imagesc(abs([thick_true_w reshape( t_W, [obj.P, obj.T*size(t_W,3)] )]));
        title(['w true, TODO! ' ]) %! ch_lambdaLabels
        
        figure(exp_no*100+3); clf
        plot(v_grid_beta, v_avgLoss)
        xlabel \lambda
        ylabel 'train loss'

        ch_resultsFile = sprintf('results_DE_%d', exp_no);
        save(ch_resultsFile);
        F = 0;
        
    end

end

methods (Static)
    
    function plot_normalizedLosses(m_loss, v_reference)
        cla;
        T = size(m_loss, 2);
        assert(length(v_reference)==T, 'length of v_reference must match')
        for k = 1:size(m_loss, 1)
            semilogy(cumsum(m_loss(k, :))./cumsum(v_reference)); hold on
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

end
        