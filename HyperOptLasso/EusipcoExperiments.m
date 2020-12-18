classdef EusipcoExperiments < HyperGradientExperiments
      
methods % Constructor and data-generating procedures
    
    function obj = EusipcoExperiments() %The constructor sets the path
        addpath('Stepsizes/')
        addpath('utilities/')
        addpath('competitors/')
        addpath('OfflineEstimators/')
    end
end

methods
    
    function v_y = generate_y_data(obj, m_X, v_true_w)
        v_y_noiseless = v_true_w'*m_X;
        signal_power  = mean(v_y_noiseless.^2);
        noise_power   = signal_power/obj.SNR;
        epsilon       = randn(1, size(m_X, 2))*sqrt(noise_power); % noise
        v_y           = v_y_noiseless    + epsilon;
    end

    function [m_X, v_y, v_true_w, m_X_test, v_y_test] ...
            = generate_pseudo_streaming_data(obj)
        %create train and test data
        rng(obj.seed);
        v_true_w      = randn(obj.P, 1).*(rand(obj.P, 1) < obj.sparsity); % sparse coefs
        m_X           = randn(obj.P, obj.n_train); % time series of x vectors
        m_X_test      = randn(obj.P, obj.n_test); % time series of x vectors

        v_y           = obj.generate_y_data(m_X, v_true_w);
        v_y_test      = obj.generate_y_data(m_X_test, v_true_w);
    end
end
    
methods % Experiments

    % Original experiment generating figure 1
    % This function compares Offline and Online,  
    % computing m_Z by linsolve   
    % sweep over beta to get the fastest convergence
    function F = experiment_101(obj)
        obj.P = 100; 
        obj.n_train = 200; %! 110
        obj.seed = 4;
        [A_train, y_train] = obj.generate_pseudo_streaming_data();
        
        my_tol_w    = 1e-3;
        my_tol_g    = 0;
        my_beta     = 1e-3;
        my_momentum = 0.9;
        
        %%
        hsgd_it = HyperSubGradientDescent;
        hsgd_it.s_estimator = 'lasso';
        hsgd_it.tol_w = my_tol_w;
        hsgd_it.tol_g_lambda = my_tol_g;
        hsgd_it.stepsize_lambda = my_beta;
        hsgd_it.momentum_lambda = my_momentum;
        hsgd_it.s_method_Z = 'interleave';

        ohsgd_it = hsgd_it;
        ohsgd_it.b_online = 1;
        ohsgd_it.max_iter_outer = 10000;
        ohsgd_it.stepsize_lambda = my_beta/sqrt(obj.P);
        
        hsgd_lin  = hsgd_it;   hsgd_lin.s_method_Z = 'linsolve';
        ohsgd_lin = ohsgd_it; ohsgd_lin.s_method_Z = 'linsolve';
        
        c_estimators = { hsgd_lin, ohsgd_lin};
        n_estimators = length(c_estimators);
        
        n_tol = 7;
        v_betas      = logspace(-5,  -3, n_tol);

                       
        %%
        t_average_w = zeros(obj.P, n_tol, n_estimators);
        t_final_lambda = zeros(1, n_tol, n_estimators);
        for k_e = 1:n_estimators
            for k_t = 1:n_tol
                estimator_now = c_estimators{k_e};
                %estimator_now.tol_w = v_tolerances(k_t);
                estimator_now.stepsize_lambda = v_betas(k_t);
                [c_W{k_t, k_e}, c_lambda{k_t, k_e}, c_it_count{k_t, k_e}] = ...
                    estimator_now.solve_gradient(A_train, y_train);
                t_final_lambda(:, k_t, k_e) = mean( ...
                    c_lambda{k_t, k_e}(find(c_it_count{k_t, k_e}>0,1, 'last')), 2);
                t_average_w(:,k_t, k_e) = mean(c_W{k_t, k_e}, 2);
            end
        end
        m_final_lambda = reshape(t_final_lambda, size(t_final_lambda, 1), ...
            numel(t_final_lambda)/ size(t_final_lambda, 1));
        
        c_lambda_filtered = cell(size(c_lambda));
        filter_length = obj.n_train;
        for k =1:numel(c_lambda)
            c_lambda_filtered{k} = filter(ones(1, filter_length), ...
                filter_length, c_lambda{k});
        end
        
        %%        
        disp 'Computing loo errors' 
        [v_looLambdas, v_looErrors, v_loo_error_final_lambdas] = obj.compute_loo_curve(...
            m_final_lambda, linspace(0.4, 2.5, 11), ...
            mean(mean(t_average_w, 2), 3), A_train, y_train);
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        c_legend = cell(n_tol, n_estimators);
        for k_e = 1:n_estimators
            for k_t = 1:n_tol
                c_legend{k_t, k_e} = sprintf('\\beta = %g', ...
                    v_betas(k_t));
            end
        end
        c_legend{1}     = [c_legend{1},    ', Offline'];
        c_legend{k_t+1} = [c_legend{k_t+1}, ', Online'];
        figure(exp_no*100+1); clf
        obj.show_lambda_vs_ista_iterates(vec(c_lambda), vec(c_it_count));
        legend(c_legend(:))
        
        figure(exp_no*100+2); clf
        plot(v_looLambdas, v_looErrors)
        xlabel '\lambda'
        ylabel 'LOO error'
        hold on
        plot(m_final_lambda, v_loo_error_final_lambdas, 'xr')
        ylim([0, 1])
        xlim([0, max(v_looLambdas)])
        
        figure(exp_no*100+4); clf
            obj.show_lambda_vs_ista_iterates(vec(c_lambda_filtered), vec(c_it_count));
            legend(c_legend(:))

        %% Save data
        ch_resultsFile = sprintf('results_%s_%d', obj.ch_prefix, exp_no);
        save(ch_resultsFile);
        F = 0;
    end

    % Copied from 101, 
    % momentum set to 0.
    % added parfor to parallelize evaluation
    function F = experiment_102(obj)
        obj.P = 200; 
        obj.n_train = 400; %! 110
        obj.SNR = 0.7;
        obj.seed = 5;
        [A_train, y_train] = obj.generate_pseudo_streaming_data();
        
        my_tol_w    = 1e-3;
        my_tol_g    = 0;
        my_beta     = 1e-3;
        my_momentum = 0;
        
        %%
        hsgd_it = HyperSubGradientDescent;
        hsgd_it.s_estimator = 'lasso';
        hsgd_it.tol_w = my_tol_w;
        hsgd_it.tol_g_lambda = my_tol_g;
        hsgd_it.stepsize_lambda = my_beta;
        hsgd_it.momentum_lambda = my_momentum;
        hsgd_it.max_iter_outer = 25; %! %originally left in default value
        hsgd_it.b_w_sparse = 0;

        ohsgd_it = hsgd_it;
        ohsgd_it.b_online = 1;
        ohsgd_it.max_iter_outer = 6000;
        ohsgd_it.stepsize_lambda = my_beta/sqrt(obj.P);
        
        hsgd_lin  = hsgd_it;   hsgd_lin.s_method_Z = 'linsolve';
        ohsgd_lin = ohsgd_it; ohsgd_lin.s_method_Z = 'linsolve';
        
        c_estimators = {hsgd_lin, ohsgd_lin };
        n_estimators = length(c_estimators);
        
        n_tol = 4;
        v_betasOnline  = logspace(-5,  -4, n_tol);
        v_betas        = logspace(-3,  -2, n_tol);

                       
        %%
        t_average_w = zeros(obj.P, n_tol, n_estimators);
        t_final_lambda = zeros(1, n_tol, n_estimators);
        [c_W, c_lambda, c_it_count, c_inv_count] = deal(cell(n_tol, n_estimators));
        for k_e = 1:n_estimators
            for k_t = 1:n_tol %! parfor
                estimator_now = c_estimators{k_e};
                %estimator_now.tol_w = v_tolerances(k_t);
                if k_e == 1
                    estimator_now.stepsize_lambda = v_betas(k_t);
                elseif k_e == 2
                    estimator_now.stepsize_lambda = v_betasOnline(k_t);
                end
                [c_W{k_t, k_e}, c_lambda{k_t, k_e}, ...
                    c_it_count{k_t, k_e}, c_inv_count{k_t, k_e}] = ...
                    estimator_now.solve_gradient(A_train, y_train);
                t_final_lambda(:, k_t, k_e) = mean( ...
                    c_lambda{k_t, k_e}(find(c_it_count{k_t, k_e}>0,1, 'last')), 2);
                t_average_w(:,k_t, k_e) = mean(c_W{k_t, k_e}, 2);
            end
        end
        m_final_lambda = reshape(t_final_lambda, size(t_final_lambda, 1), ...
            numel(t_final_lambda)/ size(t_final_lambda, 1));
        
        c_lambda_filtered = cell(size(c_lambda));
        filter_length = obj.n_train;
        for k =1:numel(c_lambda)
            c_lambda_filtered{k} = filter(ones(1, filter_length), ...
                filter_length, c_lambda{k});
        end
        
        %%        
        disp 'Computing loo errors' 
        [v_looLambdas, v_looErrors, v_loo_error_final_lambdas] = obj.compute_loo_curve(...
            m_final_lambda, linspace(0.4, 2.5, 11), ...
            mean(mean(t_average_w, 2), 3), A_train, y_train);
        %%
        [v_looLambdas, v_looErrors, v_loo_error_final_lambdas] = obj.compute_loo_curve_sweep(...
            m_final_lambda, linspace(0.2, 1, 100), ...
            mean(mean(t_average_w, 2), 3), A_train, y_train, 1);

        %%
        exp_no = obj.determine_experiment_number();
        %% Figures
        c_legend = cell(n_tol, n_estimators);
        k_e = 1;
        for k_tol = 1:n_tol
            c_legend{k_tol, k_e} = sprintf('HSGD, \\beta = %5.3f', ...
                v_betas(k_tol));
        end
        k_e = 2;
        for k_tol = 1:n_tol
            c_legend{k_tol, k_e} = sprintf('OHSGD, \\beta = %5.3f', ...
                v_betas(k_tol));
        end
%         c_legend{1}     = [c_legend{1},    ', Offline'];
%         c_legend{n_tol+1} = [c_legend{n_tol+1}, ', Online'];
        figure(exp_no*100+1); clf
        obj.show_lambda_vs_matrix_inversions(vec(c_lambda), vec(c_inv_count));
        ax = gca;
        for k_l = 1:n_tol
            ax.Children(k_l).LineStyle = '--';
        end
        legend(c_legend(:))
        
        figure(exp_no*100+2); clf
        plot(v_looLambdas, v_looErrors)
        xlabel '\lambda'
        ylabel 'LOO error'
        hold on
        plot(m_final_lambda, v_loo_error_final_lambdas, 'xr')
        ylim([0, 1])
        xlim([0, max(v_looLambdas)])
        
        figure(exp_no*100+3); clf
            obj.show_lambda_vs_matrix_inversions(vec(c_lambda_filtered), vec(c_inv_count));
            legend(c_legend(:))

        figure(exp_no*100+4); clf
        c_markers = {'square', 'square', 'square', 'square', ...
                     'none',   'none',   'none',   'none'   };
        obj.doublePlot(c_lambda, c_inv_count, c_legend, v_looErrors, ...
            v_looLambdas, m_final_lambda, v_loo_error_final_lambdas, 4, c_markers);

        %% Save data
        ch_resultsFile = sprintf('results_%s_%d', obj.ch_prefix, exp_no);
        save(ch_resultsFile);
        F = 0;
    end

    %Original experiment generating figure 2
    function F = experiment_201(obj)
        obj.P = 200;
        obj.n_train = 400;
        obj.seed = 5;
        [A_train, y_train, ~, A_test, y_test] = obj.generate_pseudo_streaming_data();
        
        my_tol_w    = 1e-3;
        my_tol_g    = 0;
        my_beta     = 1e-4/7;
        my_momentum = 0.9;
        
        %%
        hsgd_it = HyperSubGradientDescent;
        hsgd_it.s_estimator = 'lasso';
        hsgd_it.tol_w = my_tol_w;
        hsgd_it.tol_g_lambda = my_tol_g;
        hsgd_it.stepsize_lambda = my_beta;
        hsgd_it.momentum_lambda = my_momentum;
        hsgd_it.s_method_Z = 'interleave';

        ohsgd_it = hsgd_it;
        ohsgd_it.b_online = 1;
        ohsgd_it.max_iter_outer = 10000; 
        
        hsgd_lin  = hsgd_it;   hsgd_lin.s_method_Z = 'linsolve';
        ohsgd_lin = ohsgd_it; ohsgd_lin.s_method_Z = 'linsolve';
        
        c_estimators = {ohsgd_lin};
        n_estimators = length(c_estimators);
        
        n_tol = 7;
        v_tolerances = logspace(-4,  -1, n_tol);
        v_betas      = logspace(-5,  -3, n_tol);
        
        %v_tolerances(:) = 1e-2;
        v_betas(:) = 6e-5;
                       
        %%
        t_average_w = zeros(obj.P, n_tol, n_estimators);
        t_final_lambda = zeros(1, n_tol, n_estimators);
        for k_e = 1:n_estimators
            for k_t = 1:n_tol
                estimator_now = c_estimators{k_e};
                estimator_now.tol_w = v_tolerances(k_t);
                estimator_now.stepsize_lambda = v_betas(k_t);
                [c_W{k_t, k_e}, c_lambda{k_t, k_e}, c_it_count{k_t, k_e}] = ...
                    estimator_now.solve_gradient(A_train, y_train);
                t_final_lambda(:, k_t, k_e) = mean( ...
                    c_lambda{k_t, k_e}(find(c_it_count{k_t, k_e}>0,1, 'last')), 2);
                t_average_w(:,k_t, k_e) = mean(c_W{k_t, k_e}, 2);
            end
        end
        m_final_lambda = reshape(t_final_lambda, size(t_final_lambda, 1), ...
            numel(t_final_lambda)/ size(t_final_lambda, 1));
        
        c_lambda_filtered = cell(size(c_lambda));
        filter_length = obj.n_train;
        for k =1:numel(c_lambda)
            c_lambda_filtered{k} = filter(ones(1, filter_length), ...
                filter_length, c_lambda{k});
        end
            
        
        
        %%        
        disp 'Computing loo errors' 
        [v_looLambdas, v_looErrors, v_loo_error_final_lambdas, st_hyperparams] = ...
            obj.compute_loo_curve(...
            m_final_lambda, linspace(0.4, 2.5, 21), ...
            mean(mean(t_average_w, 2), 3), A_train, y_train);
        %
        disp 'Computing test errors'
        v_test_errors = obj.heldOut_validation_errors(A_train, y_train,...
            A_test, y_test, st_hyperparams, mean(mean(t_average_w, 2), 3));

        exp_no = obj.determine_experiment_number();
        %% Figures
        c_legend = cell(n_tol, n_estimators);
        for k_e = 1:n_estimators
            for k_t = 1:n_tol
                c_legend{k_t, k_e} = sprintf('tol = %g', ...
                    v_tolerances(k_t));
            end
        end
        c_legend{1}     = [c_legend{1}]; %, ', linsolve'];
        %c_legend{k_t+1} = [c_legend{k_t+1}, ', iterative'];
        
        figure(exp_no*100+1); clf
            obj.show_lambda_vs_ista_iterates(vec(c_lambda), vec(c_it_count));
            legend(c_legend(:))
        
        figure(exp_no*100+2); clf
            plot(v_looLambdas, v_looErrors)
            xlabel '\lambda'
            ylabel 'Validation error'
            hold on
            plot(m_final_lambda, v_loo_error_final_lambdas, 'xr')
            plot(v_looLambdas, v_test_errors, 'm')
            legend('Leave-one-out', 'HSGD','Test')
        
        figure(exp_no*100+3); clf
            filter_length = obj.n_train;
            for k_e = 1:numel(c_estimators)
                for k_t = 1:n_tol
                    plot(cumsum(c_it_count{k_t,k_e}), filter(...
                        ones(1, filter_length), filter_length, c_lambda{k_t, k_e}))
                    hold on
                end
            end
            legend(c_legend(:))
        
        figure(exp_no*100+4); clf
            obj.show_lambda_vs_ista_iterates(vec(c_lambda_filtered), vec(c_it_count));
            legend(c_legend(:))
            
        %% Save data
        ch_resultsFile = sprintf('results_%s_%d', obj.ch_prefix, exp_no);
        save(ch_resultsFile);
        F = 0;
    end

    function F = experiment_202(obj)
        obj.P = 200;
        obj.n_train = 400;
        obj.seed = 5; %! 4
        [A_train, y_train, ~, A_test, y_test] = obj.generate_pseudo_streaming_data();
        
        my_tol_w    = 1e-3;
        my_tol_g    = 0;
        my_beta     = 1e-4/7;
        my_momentum = 0.1;
        
        %%
        hsgd_it = HyperSubGradientDescent;
        hsgd_it.s_estimator = 'lasso';
        hsgd_it.tol_w = my_tol_w;
        hsgd_it.tol_g_lambda = my_tol_g;
        hsgd_it.stepsize_lambda = my_beta;
        hsgd_it.momentum_lambda = my_momentum;
        hsgd_it.s_method_Z = 'interleave';
        hsgd_it.b_w_sparse = 0;

        ohsgd_it = hsgd_it;
        ohsgd_it.b_online = 1;
        ohsgd_it.max_iter_outer = 1000; %!10000 
        
        hsgd_lin  = hsgd_it;   hsgd_lin.s_method_Z = 'linsolve';
        ohsgd_lin = ohsgd_it; ohsgd_lin.s_method_Z = 'linsolve';
        
        c_estimators = {ohsgd_lin};
        n_estimators = length(c_estimators);
        
        n_tol = 5;
        v_tolerances = logspace(-1,1,  n_tol);
        v_betas      = logspace(-5, -3, n_tol);
        
        v_tolerances(:) = .01;
        %v_betas(:) = [1e-4, 3e-4, 1e-4, 3e-4];%!
                       
        %%
        t_average_w = zeros(obj.P, n_tol, n_estimators);
        t_final_lambda = zeros(1, n_tol, n_estimators);
        for k_e = 1:n_estimators
            for k_t = 1:n_tol %! parfor
                estimator_now = c_estimators{k_e};
                estimator_now.tol_w = v_tolerances(k_t);
                estimator_now.stepsize_lambda = v_betas(k_t);
                [c_W{k_t, k_e},  c_lambda{k_t, k_e}, ...
                    c_it_count{k_t, k_e},  c_inv_count{k_t, k_e}] = ...
                    estimator_now.solve_gradient(A_train, y_train);
                t_final_lambda(:, k_t, k_e) = mean( ...
                    c_lambda{k_t, k_e}(find(c_it_count{k_t, k_e}>0,1, 'last')), 2);
                t_average_w(:,k_t, k_e) = mean(c_W{k_t, k_e}, 2);
            end
        end
        m_final_lambda = reshape(t_final_lambda, size(t_final_lambda, 1), ...
            numel(t_final_lambda)/ size(t_final_lambda, 1));
        
        c_lambda_filtered = cell(size(c_lambda));
        filter_length = obj.n_train;
        for k =1:numel(c_lambda)
            c_lambda_filtered{k} = filter(ones(1, filter_length), ...
                filter_length, c_lambda{k});
        end
            
        
        
        %%        
        disp 'Computing loo errors' 
        [v_looLambdas, v_looErrors, v_loo_error_final_lambdas, st_hyperparams] = ...
            obj.compute_loo_curve(...
            m_final_lambda, linspace(0.4, 2.5, 21), ...
            mean(mean(t_average_w, 2), 3), A_train, y_train);
        %
        disp 'Computing test errors'
        v_test_errors = obj.heldOut_validation_errors(A_train, y_train,...
            A_test, y_test, st_hyperparams, mean(mean(t_average_w, 2), 3));

        exp_no = obj.determine_experiment_number();
        %% Figures
        c_legend = cell(n_tol, n_estimators);
        for k_e = 1:n_estimators
            for k_t = 1:n_tol
                c_legend{k_t, k_e} = sprintf('tol = %g, \\beta = %g', ...
                    v_tolerances(k_t), v_betas(k_t));
            end
        end
        c_legend{1}     = [c_legend{1}]; %, ', linsolve'];
        %c_legend{k_t+1} = [c_legend{k_t+1}, ', iterative'];
        
        figure(exp_no*100+1); clf
            obj.show_lambda_vs_matrix_inversions(vec(c_lambda), vec(c_inv_count));
            legend(c_legend(:))
        
        figure(exp_no*100+2); clf
            plot(v_looLambdas, v_looErrors)
            xlabel '\lambda'
            ylabel 'Validation error'
            hold on
            plot(m_final_lambda, v_loo_error_final_lambdas, 'xr')
            plot(v_looLambdas, v_test_errors, 'm')
            legend('Leave-one-out', 'HSGD','Test')
        
        figure(exp_no*100+3); clf
%             filter_length = obj.n_train;
%             for k_e = 1:numel(c_estimators)
%                 for k_t = 1:n_tol
%                     plot(cumsum(c_inv_count{k_t,k_e}), filter(...
%                         ones(1, filter_length), filter_length, c_lambda{k_t, k_e}))
%                     hold on
%                 end
%             end
%             legend(c_legend(:))
%         
%         figure(exp_no*100+4); clf
            obj.show_lambda_vs_ista_iterates(vec(c_lambda_filtered), vec(c_inv_count));
            legend(c_legend(:))
            
        figure(exp_no*100+4); clf
            obj.doublePlot(c_lambda, c_inv_count, c_legend, v_looErrors, ...
                v_looLambdas, m_final_lambda, v_loo_error_final_lambdas);
            
        %% Save data
        ch_resultsFile = sprintf('results_%s_%d', obj.ch_prefix, exp_no);
        save(ch_resultsFile);
        F = 0;
    end
    
    % OHSGD, comparing iterates vs. time with different tolerance
    % parameters
    function F = experiment_203(obj)
        obj.P = 200; %!
        obj.n_train = 4000; %!
        obj.seed = 5; %! 4
        [A_train, y_train, ~, A_test, y_test] = obj.generate_pseudo_streaming_data();
        
        my_tol_w    = 1e-3;
        my_tol_g    = 0;
        my_beta     = 1e-4/7;
        my_momentum = 0;
        
        %%
        hsgd_it = HyperSubGradientDescent;
        hsgd_it.s_estimator = 'lasso';
        hsgd_it.tol_w = my_tol_w;
        hsgd_it.tol_g_lambda = my_tol_g;
        hsgd_it.stepsize_lambda = my_beta;
        hsgd_it.momentum_lambda = my_momentum;
        hsgd_it.s_method_Z = 'interleave';
        hsgd_it.b_w_sparse = 0;

        ohsgd_it = hsgd_it;
        ohsgd_it.b_online = 1;
        ohsgd_it.max_iter_outer = 1000; %!10000 
        
        hsgd_lin  = hsgd_it;   hsgd_lin.s_method_Z = 'linsolve';
        ohsgd_lin = ohsgd_it; ohsgd_lin.s_method_Z = 'linsolve';
        
        c_estimators = {ohsgd_lin};
        n_estimators = length(c_estimators);
        
        n_tol = 5;
        v_tolerances = logspace(-3,-1,  n_tol);
        v_betas      = logspace(-5, -3, n_tol);
        
        %v_tolerances(:) = .01;
        v_betas(:) =  0.00001; %![1e-4, 3e-4, 1e-4, 3e-4];%!
                       
        %%
        t_average_w = zeros(obj.P, n_tol, n_estimators);
        t_final_lambda = zeros(1, n_tol, n_estimators);
        for k_e = 1:n_estimators
            for k_t = 1:n_tol %! parfor
                estimator_now = c_estimators{k_e};
                estimator_now.tol_w = v_tolerances(k_t);
                estimator_now.stepsize_lambda = v_betas(k_t);
                [c_W{k_t, k_e},  c_lambda{k_t, k_e}, ...
                    c_it_count{k_t, k_e},  c_inv_count{k_t, k_e}, v_elapsedTimes] = ...
                    estimator_now.solve_gradient(A_train, y_train);
                c_timings{k_t, k_e} = [0 diff(v_elapsedTimes)];
                t_final_lambda(:, k_t, k_e) = mean( ...
                    c_lambda{k_t, k_e}(find(c_it_count{k_t, k_e}>0,1, 'last')), 2);
                t_average_w(:,k_t, k_e) = mean(c_W{k_t, k_e}, 2);
            end
        end
        m_final_lambda = reshape(t_final_lambda, size(t_final_lambda, 1), ...
            numel(t_final_lambda)/ size(t_final_lambda, 1));
        
        c_lambda_filtered = cell(size(c_lambda));
        filter_length = obj.n_train;
        for k =1:numel(c_lambda)
            c_lambda_filtered{k} = filter(ones(1, filter_length), ...
                filter_length, c_lambda{k});
        end
            
        
        
%         %%        
        disp 'Computing loo errors' 
        [v_looLambdas, v_looErrors, v_loo_error_final_lambdas, st_hyperparams] = ...
            obj.compute_loo_curve(...
            m_final_lambda, linspace(0.4, 2.5, 21), ...
            mean(mean(t_average_w, 2), 3), A_train, y_train);
%         %
%         disp 'Computing test errors'
%         v_test_errors = obj.heldOut_validation_errors(A_train, y_train,...
%             A_test, y_test, st_hyperparams, mean(mean(t_average_w, 2), 3));
% 
        exp_no = obj.determine_experiment_number();
        %% Figures
        c_legend = cell(n_tol, n_estimators);
        for k_e = 1:n_estimators
            for k_t = 1:n_tol
                c_legend{k_t, k_e} = sprintf('tol = %g, \\beta = %g', ...
                    v_tolerances(k_t), v_betas(k_t));
            end
        end
        c_legend{1}     = [c_legend{1}]; %, ', linsolve'];
        %c_legend{k_t+1} = [c_legend{k_t+1}, ', iterative'];
        
        figure(exp_no*100+1); clf
            obj.show_lambda_vs_matrix_inversions(vec(c_lambda), vec(c_timings));
            legend(c_legend(:))
        
        figure(exp_no*100+2); clf
            plot(v_looLambdas, v_looErrors)
            xlabel '\lambda'
            ylabel 'Validation error'
            hold on
            plot(m_final_lambda, v_loo_error_final_lambdas, 'xr')
%             plot(v_looLambdas, v_test_errors, 'm')
%             legend('Leave-one-out', 'HSGD','Test')
        
%         figure(exp_no*100+3); clf
% %             filter_length = obj.n_train;
% %             for k_e = 1:numel(c_estimators)
% %                 for k_t = 1:n_tol
% %                     plot(cumsum(c_inv_count{k_t,k_e}), filter(...
% %                         ones(1, filter_length), filter_length, c_lambda{k_t, k_e}))
% %                     hold on
% %                 end
% %             end
% %             legend(c_legend(:))
% %         
% %         figure(exp_no*100+4); clf
%             obj.show_lambda_vs_ista_iterates(vec(c_lambda_filtered), vec(c_inv_count));
%             legend(c_legend(:))
%             
%         figure(exp_no*100+4); clf
%             obj.doublePlot(c_lambda, c_inv_count, c_legend, v_looErrors, ...
%                 v_looLambdas, m_final_lambda, v_loo_error_final_lambdas);
            
        %% Save data
        ch_resultsFile = sprintf('results_%s_%d', obj.ch_prefix, exp_no);
        save(ch_resultsFile);
        F = 0;
    end

    
    % Repeating success in exp 202, with Group Lasso
    % This is TODO
    
    function F = experiment_401(obj)
        load data/housingPrices.mat Xtrain Xtest y
        obj.P = 100;
        obj.n_train = 200; %! 110
        obj.seed = 4;
        [A_train_s, y_train_s] = obj.generate_pseudo_streaming_data();

        my_nTrain = size(Xtrain,1)-1; %400;
        A_train = normalize(Xtrain(2:my_nTrain+1,:)'); %!
        obj.P = size(A_train, 1);
        obj.n_train = size(A_train, 2); % obj.n_train = my_nTrain;
        y_train = normalize(y(1:my_nTrain)');
        % todo: apply same normalization factor to Xtest

        
        my_tol_w    = 1e-3;
        my_tol_g    = 0;
        my_beta     = 1e-3;
        my_momentum = 0;
        
        %%
        hsgd_it = HyperSubGradientDescent;
        hsgd_it.s_estimator = 'lasso';
        hsgd_it.tol_w = my_tol_w;
        hsgd_it.tol_g_lambda = my_tol_g;
        hsgd_it.stepsize_lambda = my_beta;
        hsgd_it.momentum_lambda = my_momentum;
        hsgd_it.max_iter_outer = 25; %! %originally left in default value
        hsgd_it.b_w_sparse = 0;

        ohsgd_it = hsgd_it;
        ohsgd_it.b_online = 1;
        ohsgd_it.max_iter_outer = 6000;
        ohsgd_it.stepsize_lambda = my_beta/sqrt(obj.P);
        
        hsgd_lin  = hsgd_it;   hsgd_lin.s_method_Z = 'linsolve';
        ohsgd_lin = ohsgd_it; ohsgd_lin.s_method_Z = 'lsqminnorm';
        
        c_estimators = {ohsgd_lin};%! {hsgd_lin, ohsgd_lin };
        n_estimators = length(c_estimators);
        
        n_tol = 2;%!4;
        v_betasOnline  = logspace(-5,  -4, n_tol);
        v_betas        = logspace(-8,  -7, n_tol);
                       
        %%
        t_average_w = zeros(obj.P, n_tol, n_estimators);
        t_final_lambda = zeros(1, n_tol, n_estimators);
        [c_W, c_lambda, c_it_count, c_inv_count] = deal(cell(n_tol, n_estimators));
        for k_e = 1:n_estimators
            parfor k_t = 1:n_tol %! parfor
                estimator_now = c_estimators{k_e};
                %estimator_now.tol_w = v_tolerances(k_t);
                if k_e == 1
                    estimator_now.stepsize_lambda = v_betas(k_t);
                elseif k_e == 2
                    estimator_now.stepsize_lambda = v_betasOnline(k_t);
                end
                [c_W{k_t, k_e}, c_lambda{k_t, k_e}, ...
                    c_it_count{k_t, k_e}, c_inv_count{k_t, k_e}] = ...
                    estimator_now.solve_gradient(A_train, y_train);
                t_final_lambda(:, k_t, k_e) = mean( ...
                    c_lambda{k_t, k_e}(find(c_it_count{k_t, k_e}>0,1, 'last')), 2);
                t_average_w(:,k_t, k_e) = mean(c_W{k_t, k_e}, 2);
            end
        end
        m_final_lambda = reshape(t_final_lambda, size(t_final_lambda, 1), ...
            numel(t_final_lambda)/ size(t_final_lambda, 1));
        
        c_lambda_filtered = cell(size(c_lambda));
        filter_length = obj.n_train;
        for k =1:numel(c_lambda)
            c_lambda_filtered{k} = filter(ones(1, filter_length), ...
                filter_length, c_lambda{k});
        end
        
        %%        
        disp 'Computing loo errors' 
%         [v_looLambdas, v_looErrors, v_loo_error_final_lambdas] = obj.compute_loo_curve(...
%             m_final_lambda, linspace(0.4, 2.5, 11), ...
%             mean(mean(t_average_w, 2), 3), A_train, y_train);
%         %%
        [v_looLambdas, v_looErrors, v_loo_error_final_lambdas] = obj.compute_loo_curve_sweep(...
            m_final_lambda, linspace(0, 5e-3, 10), ...
            mean(mean(t_average_w, 2), 3), A_train, y_train, 1);

        %%
        exp_no = obj.determine_experiment_number();
        %% Figures
        c_legend = cell(n_tol, n_estimators);
        k_e = 1;
        for k_tol = 1:n_tol
            c_legend{k_tol, k_e} = sprintf('HSGD, \\beta = %5.3f', ...
                v_betas(k_tol));
        end
        k_e = 2;
        for k_tol = 1:n_tol
            c_legend{k_tol, k_e} = sprintf('OHSGD, \\beta = %5.3f', ...
                v_betas(k_tol));
        end
%         c_legend{1}     = [c_legend{1},    ', Offline'];
%         c_legend{n_tol+1} = [c_legend{n_tol+1}, ', Online'];
        figure(exp_no*100+1); clf
        obj.show_lambda_vs_matrix_inversions(vec(c_lambda), vec(c_inv_count));
        ax = gca;
        for k_l = 1:n_tol
            ax.Children(k_l).LineStyle = '--';
        end
        legend(c_legend(:))
        
        figure(exp_no*100+2); clf
        plot(v_looLambdas, v_looErrors)
        xlabel '\lambda'
        ylabel 'LOO error'
        hold on
        plot(m_final_lambda, v_loo_error_final_lambdas, 'xr')
        ylim([0, 1])
        xlim([0, max(v_looLambdas)])
        
        figure(exp_no*100+3); clf
            obj.show_lambda_vs_matrix_inversions(vec(c_lambda_filtered), vec(c_inv_count));
            legend(c_legend(:))

        figure(exp_no*100+4); clf
        c_markers = {'square', 'square', 'square', 'square', ...
                     'none',   'none',   'none',   'none'   };
        obj.doublePlot(c_lambda, c_inv_count, c_legend, v_looErrors, ...
            v_looLambdas, m_final_lambda, v_loo_error_final_lambdas, 4, c_markers);

        %% Save data
        ch_resultsFile = sprintf('results_%s_%d', obj.ch_prefix, exp_no);
        save(ch_resultsFile);
        F = 0;
    end

    function F = experiment_901(obj)
        obj.P = 200; 
        obj.n_train = 1800; %! 110
        obj.SNR = 1;
        obj.seed = 10;
        [A_train, y_train, v_true_w, A_noTrain, y_noTrain] = ...
            obj.generate_pseudo_streaming_data();
        A_val = A_noTrain(:,floor(obj.n_train)+1:end);
        y_val = y_noTrain(floor(obj.n_train)+1:end);
        A_test = A_noTrain(:,1:floor(obj.n_train));
        y_test = y_noTrain(1:floor(obj.n_train));
        [m_W, m_lambda, v_it_count, v_timing, v_valCurve, v_lambdasCurve, v_valSequence, v_trainSequence] = ...
            holdout_validated_lasso(A_train, y_train, A_val, y_val);
        v_testSequence = mean((m_W'*A_test - y_test)'.^2);
        v_NMSD = mean((m_W - v_true_w).^2);
        figure(901); clf; 
        ax1 = subplot(2,4,1);
           plot(m_lambda')
           xlabel 'outer iteration'
           ylabel \lambda
        ax2 = subplot(2,4,2);
           plot(v_valCurve, v_lambdasCurve);
           xlabel 'validation error'
           ylabel '\lambda Lasso'
        linkaxes([ax1 ax2], 'y');
        subplot(2,4,3);
           plot(cumsum(v_it_count))
           xlabel 'outer iteration'
           ylabel 'cumulative computation'
        subplot(2,4,4);
            stem(m_W(:,end));
            hold on
            stem(v_true_w);
            legend ('estimated w', 'true w')
        subplot(2,4,5);
            plot(v_NMSD)
            xlabel 'outer iteration'
            legend MSD
        subplot(2,4,6);
            plot(v_trainSequence);
            hold on
            plot(v_valSequence);
            plot(v_testSequence);
            xlabel 'outer iteration'
            legend('train', 'val', 'test');
        subplot(2,4,7);
            scatter(v_true_w, m_W(:,end));
            xlabel 'true w'
            ylabel 'estimated w'
        subplot(2,4,8);
            scatter3(m_W(:,end), m_lambda(:,end), v_true_w);
            xlabel w
            ylabel \lambda
    end
    
    function F = experiment_902(obj)
        obj.P = 200; 
        obj.n_train = 1800; %! 110
        obj.SNR = 1;
        obj.seed = 10;
        [A_train, y_train, v_true_w, A_v, y_noTrain] = ...
            obj.generate_pseudo_streaming_data();
        A_val = A_noTrain(:,floor(obj.n_train)+1:end);
        y_val = y_noTrain(floor(obj.n_train)+1:end);
        A_test = A_noTrain(:,1:floor(obj.n_train));
        y_test = y_noTrain(1:floor(obj.n_train));
        [m_W, m_lambda, v_it_count, v_timing] = ...
            loo_validated_lasso(A_train, y_train);
        v_trainSequence = mean((m_W'*A_train - y_train)'.^2);
        v_testSequence  = mean((m_W'*A_test - y_test)'.^2);
        v_NMSD = mean((m_W - v_true_w).^2);
        figure(901); clf; 
        ax1 = subplot(2,4,1);
           plot(m_lambda')
           xlabel 'outer iteration'
           ylabel \lambda
%         ax2 = subplot(2,4,2);
%            plot(v_valCurve, v_lambdasCurve);
%            xlabel 'validation error'
%            ylabel '\lambda Lasso'
%         linkaxes([ax1 ax2], 'y');
        subplot(2,4,3);
           plot(cumsum(v_it_count))
           xlabel 'outer iteration'
           ylabel 'cumulative computation'
        subplot(2,4,4);
            stem(m_W(:,end));
            hold on
            stem(v_true_w);
            legend ('estimated w', 'true w')
        subplot(2,4,5);
            plot(v_NMSD)
            xlabel 'outer iteration'
            legend MSD
        subplot(2,4,6);
            plot(v_trainSequence);
            hold on
            %plot(v_valSequence);
            plot(v_testSequence);
            xlabel 'outer iteration'
            legend('train', 'test');
        subplot(2,4,7);
            scatter(v_true_w, m_W(:,end));
            xlabel 'true w'
            ylabel 'estimated w'
%         subplot(2,4,8);
%             scatter3(m_W(:,end), m_lambda(:,end), v_true_w);
%             xlabel w
%             ylabel \lambda
    end
    function F = experiment_903(obj)
        % do not change, create a new one
        obj.P = 200;
        obj.n_train = 800; %! 8000
        obj.SNR = 3;
        obj.seed = 13;
        [A_train, y_train, v_true_w, A_test, y_test] = ...
            obj.generate_pseudo_streaming_data();
        cvl = CrossValidatedLasso;
        cvl.b_adaptive_lasso = 1;
        cvl.stepsize_lambda = .1;
        cvl.minifold_size = 1; % 1 = LOO
        cvl.max_iter_outer = 15000; %! 30000
        cvl.factor_initial_lambda = 0.05;
        cvl.tol_w = 1e-4;
        cvl.b_efficient = 0;
        cvl.crazy_param = 1;
        % IDEA = initialize adaptive Lasso at the optimum of Lasso
        [m_W, m_lambda, v_it_count, v_timing, mb_valFold, ...
            v_valSequence, v_maxDev] = ...
            cvl.solve(A_train, y_train);
        %v_w_final % = estimate using all samples
        v_trainSequence = mean((m_W'*A_train - y_train)'.^2);
        v_testSequence  = mean((m_W'*A_test - y_test)'.^2);
        v_NMSD = mean((m_W - v_true_w).^2);
        v_pfa  = mean(and(m_W, not(v_true_w)))./mean(not(v_true_w));
        v_pmd  = mean(and(not(m_W), v_true_w))./mean(logical(v_true_w));
        v_cier = mean(xor(m_W, v_true_w));
        
        %% compute validation curve(s)
        if not(cvl.b_adaptive_lasso)
            % compute cv loss only for trajectory of lambda vectors
        end
        
        %% plotting
        figure(904); clf; 
        ax1 = subplot(2,4,1);
           plot(m_lambda')
           xlabel 'outer iteration'
           ylabel \lambda
        ax2 = subplot(2,4,2);
           plot(v_maxDev);
           title 'max deviation eff comp'
       %        if not(cvl.b_adaptive_lasso)        
%            plot(v_valCurve, v_lambdasCurve);
%            xlabel 'validation error'
%            ylabel '\lambda Lasso'
%        end
%        linkaxes([ax1 ax2], 'y');
        subplot(2,4,3);
           plot(cumsum(v_it_count))
           xlabel 'outer iteration'
           ylabel 'cumulative computation'
        subplot(2,4,4);
            v_support = m_W(:, end)~= 0;
            stem(find(v_support), m_W(v_support,end));
            hold on
            stem(find(v_true_w), v_true_w(v_true_w~=0));
            legend ('estimated w', 'true w')
        subplot(2,4,5);
            plot(v_NMSD)
            hold on
            plot(v_pfa);
            plot(v_pmd);
            plot(v_cier);
            xlabel 'outer iteration'
            legend('MSD', 'P_{FA}', 'P_{MD}', 'CIER')
        subplot(2,4,6);
            plot(v_trainSequence);
            hold on
            plot(v_valSequence);
            plot(v_testSequence);
            xlabel 'outer iteration'
            legend('train', 'val', 'test');
        subplot(2,4,7);
            cc = categorical(logical(m_W(:,end))-logical(v_true_w));
            cc = renamecats(cc, {'-1' '0' '1'}, {'miss', 'hit', 'fPos'});
            pie(cc)
        subplot(2,4,8);
            scatter(v_true_w, m_W(:,end)); 
            scatter3(m_W(:,end), m_lambda(:,end), v_true_w);
            xlabel 'estimated w'
            ylabel \lambda
            zlabel 'true w'
    end

    function F = experiment_904(obj)
        % do not change, create a new one
        obj.P = 200;
        obj.n_train = 800; %! 8000
        obj.SNR = 3;
        obj.seed = 1;
        [A_train, y_train, v_true_w, A_test, y_test] = ...
            obj.generate_pseudo_streaming_data();
        cvl = CrossValidatedLasso;
        cvl.b_adaptive_lasso = 1;
        cvl.stepsize_lambda = .15;
        cvl.minifold_size = 1; % 1 = LOO
        cvl.max_iter_outer = 50000; %! 30000
        cvl.factor_initial_lambda = 0.05; %!0.1
        cvl.tol_w = 1e-4;
        cvl.b_efficient = 0;
        cvl.crazy_param = 1;
        cvl.decay_factor  = 1e-3;
        [m_W_rew, m_lambda_rew] = ...
            cvl.solve_reweighted(A_train,y_train, 0.015);
        % IDEA = initialize adaptive Lasso at the optimum of Lasso
        [m_W, m_lambda, v_it_count, v_timing, mb_valFold, ...
            v_valSequence, v_maxDev, v_w_final] = ...
            cvl.solve(A_train, y_train, m_lambda_rew(:,100));
        %v_w_final % = estimate using all samples
        v_trainSequence = mean((m_W'*A_train - y_train)'.^2);
        v_testSequence  = mean((m_W'*A_test - y_test)'.^2);
        v_NMSD = mean((m_W - v_true_w).^2);
        v_pfa  = mean(and(m_W, not(v_true_w)))./mean(not(v_true_w));
        v_pmd  = mean(and(not(m_W), v_true_w))./mean(logical(v_true_w));
        v_cier = mean(xor(m_W, v_true_w));
        
        %% compute validation curve(s)
        if not(cvl.b_adaptive_lasso)
            % compute cv loss only for trajectory of lambda vectors
        end
        
        %% plotting
        figure(904); clf; 
        ax1 = subplot(2,4,1);
           plot(m_lambda')
           xlabel 'outer iteration'
           ylabel \lambda
        ax2 = subplot(2,4,2);
           plot(v_maxDev);
           title 'max deviation eff comp'
       %        if not(cvl.b_adaptive_lasso)        
%            plot(v_valCurve, v_lambdasCurve);
%            xlabel 'validation error'
%            ylabel '\lambda Lasso'
%        end
%        linkaxes([ax1 ax2], 'y');
        subplot(2,4,3);
           plot(cumsum(v_it_count))
           xlabel 'outer iteration'
           ylabel 'cumulative computation'
        subplot(2,4,4);
            %v_support = m_W(:, end)~= 0;
            v_support = v_w_final ~= 0;
            %stem(find(v_support), m_W(v_support,end));
            stem(find(v_support), v_w_final(v_support));
            hold on
            stem(find(v_true_w), v_true_w(v_true_w~=0));
            legend ('estimated w', 'true w')
        subplot(2,4,5);
            plot(v_NMSD)
            hold on
            plot(v_pfa);
            plot(v_pmd);
            plot(v_cier);
            xlabel 'outer iteration'
            legend('MSD', 'P_{FA}', 'P_{MD}', 'CIER')
        subplot(2,4,6);
            plot(v_trainSequence);
            hold on
            plot(v_valSequence);
            plot(v_testSequence);
            xlabel 'outer iteration'
            legend('train', 'val', 'test');
        subplot(2,4,7);
            %v_w_final = m_W(:,end);
            cc = categorical([logical(v_w_final)-logical(v_true_w);0; 1; -1]);
            cc = cc(1:end-3);
            cc = renamecats(cc, {'-1' '0' '1'}, {'miss', 'hit', 'fPos'});
            pie(cc)
        subplot(2,4,8);
            scatter(v_true_w, v_w_final); 
            scatter3(v_w_final, m_lambda(:,end), v_true_w);
            xlabel 'estimated w'
            ylabel \lambda
            zlabel 'true w'
    end

end

methods % more plotting
    function [ax1, ax2] = doublePlot(obj, c_lambda, c_inv_count, c_legend, ...
            v_looErrors, v_looLambdas, m_final_lambda, v_loo_error_final_lambdas, ...
            varargin)
        clf
        ax1 = subplot(1, 2, 1);
        obj.show_lambda_vs_matrix_inversions(vec(c_lambda), vec(c_inv_count), varargin{:})
        legend(c_legend(:));
        
        ax2 = subplot(1,2,2);
        plot(v_looErrors, v_looLambdas)
        xlabel 'LOO NMSE'
        nmse_min = min(xlim);
        xlim([nmse_min 1])
        hold on
        for i_lam = 1:length(m_final_lambda)
            plot([0; v_loo_error_final_lambdas(i_lam)], ...
                [1 1]*m_final_lambda(i_lam), '--k');
        end
        linkaxes([ax1 ax2], 'y');
        
    end
end


end