classdef HyperGradientExperiments
    properties
        n_train = 200;     % Number of train samples
        n_test  = 2000;    % Number of test  samples
        P       = 100;     % Dimensionality of dataset
        seed    = 1;       % Random seed
        SNR     = 0.3;     % Signal to noise ratio  (natural units)
        sparsity= 0.2;     % proportion of entries of true_x that are > 0
        
        b_colinear = 0;    %introduce colinear variables when creating 
        % synthetic data (see set_up_data method)
        
        ch_prefix = 'HG'
    end
      
methods % Constructor and data-generating procedures
    
    function obj = HyperGradientExperiments() %The constructor sets the path
        addpath('Stepsizes/')
        addpath('utilities/')
        addpath('competitors/')
        addpath('OfflineEstimators/')
    end
    
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
        v_true_w        = double(rand(obj.P, 1) < obj.sparsity); % sparse coefs
        m_X           = randn(obj.P, obj.n_train); % time series of x vectors
        m_X_test      = randn(obj.P, obj.n_test); % time series of x vectors

        v_y           = obj.generate_y_data(m_X, v_true_w);
        v_y_test      = obj.generate_y_data(m_X_test, v_true_w);
    end
    
    function [m_X, v_y, v_true_w, m_X_test, v_y_test, v_group_structure] ...
            = generate_pseudo_streaming_data_grouped(obj, n_groups)
        %create train and test data
        rng(obj.seed);
        v_group_structure = ceil(linspace(eps, n_groups, obj.P));
        validateGroupStructure(v_group_structure);
        v_active_groups = double(rand(n_groups, 1) < obj.sparsity);
        v_true_w        = v_active_groups(v_group_structure); % group-sparse coefs
        m_X           = randn(obj.P, obj.T); % time series of x vectors
        m_X_test      = randn(obj.P, obj.T); % time series of x vectors
        
        v_y           = obj.generate_y_data(m_X, v_true_w);
        v_y_test      = obj.generate_y_data(m_X_test, v_true_w);
    end
    
    function m_X = generate_colinear_X(obj, v_group_structure)
        validateGroupStructure(v_group_structure);
        n_blocks = max(v_group_structure);
        c_blocks = cell(n_blocks, 1);
        for b = 1:n_blocks
            v_indices = find(v_group_structure==b);
            block_size = length(v_indices);
            block_rank = floor(block_size*(1-obj.colinearity));
            m_rectangular = randn(block_size, block_rank);
            m_ortho_basis = orth(m_rectangular);
            c_blocks{b} = m_ortho_basis*m_ortho_basis';
        end
        m_blockDiagonal = blkdiag(c_blocks{:});
        m_X = m_blockDiagonal*randn(obj.P, obj.T);
    end
    
    function [m_X, v_y, v_true_w, m_X_test, v_y_test, v_group_structure] ...
            = generate_pseudo_streaming_data_colinear(obj, n_groups)
        %create train and test data
        rng(obj.seed);
        v_group_structure = ceil(linspace(eps, n_groups, obj.P)); %in this function,
        validateGroupStructure(v_group_structure);
        % the group structure is used to create groups of colinear X        
        v_true_w        = double(rand(obj.P, 1) < obj.sparsity); % sparse coefs
        
        m_X           = generate_colinear_X(obj, v_group_structure);
        m_X_test      = generate_colinear_X(obj, v_group_structure);
        v_y           = obj.generate_y_data(m_X, v_true_w);
        v_y_test      = obj.generate_y_data(m_X_test, v_true_w);
    end
end

methods % Dynamic learning-related routines: drill, and grid search
    
    function [m_W, v_lambda, v_loss] = drill_dynamic(obj, estimator, m_X, v_y, initial_lambda, T)
        [P_dim, N] = size(m_X); assert(length(v_y)==N);
        
        m_W      = zeros(P_dim, T);
        v_loss   = zeros(1, T);
        
        m_Phi = m_X*m_X'/N;
        v_r   = m_X*v_y(:)/N;
        v_wf_t = zeros(P_dim, 1);
        v_c_t  = zeros(P_dim, 1);
        if isempty(estimator.stepsize_w)
           estimator.stepsize_w = 0.5/trace(m_Phi);
        end
        
        v_lambda = zeros(1, T);
        v_lambda(1) = initial_lambda;

        ltc = LoopTimeControl(T-1);
        for t=1:T-1
            n = 1+mod(t-1, N);
            [m_W(:, t+1), v_lambda(t+1), v_loss(t+1), v_wf_t, m_Phi, ...
                v_r, v_c_t] =  estimator.update( v_lambda(t), ...
                m_X(:, n), v_y(n), v_wf_t, m_Phi, v_r, v_c_t);
            ltc.go(t);
        end
    end
    
    function v_avgLoss = grid_search_dynamic_lambda(obj, estimator, ...
            m_X, v_y, v_lambda_grid, T, T_begin)
        if ~exist('T_begin', 'var'), T_begin = floor(T/2); end
        
        estimator.stepsize_lambda = 0;
        n_lambdas = numel(v_lambda_grid);
        v_avgLoss = zeros(size(v_lambda_grid));
        for li = 1:n_lambdas
            [~, ~, v_loss] = obj.drill_dynamic(estimator, m_X,...
                v_y, v_lambda_grid(li), T);
            v_avgLoss(li) = mean(v_loss(T_begin:end))./mean(v_y.^2);
        end
    end
    
end
    
methods % Experiments
            
    % Gradient descent, comparison with grid-search Leave one out
    function F = experiment_10(obj)
        obj.P = 100;
        obj.n_train = 200;
        [A_train, y_train] = obj.generate_pseudo_streaming_data();
        
        hgl = HyperGradientLasso;
        hgl.stepsize_lambda = 3e-4;
        hgl.tol_w = 1e-3;
        hgl.tol_g_lambda = 1e-4;
        hgl.max_iter_outer = 400;
        
        [m_W, v_lambda, v_it_count] = hgl.solve_gradient(A_train, y_train);
        final_lambda = v_lambda(find(v_it_count>0,1, 'last'));
        average_w = mean(m_W, 2);
       
        disp 'Computing loo errors' 
        [v_looLambdas, v_looErrors, loo_error_final_lambda] = ...
            obj.compute_loo_curve(final_lambda, linspace(0.9, 1.2, 21), ...
            average_w, A_train, y_train);
        
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
        plot(final_lambda, loo_error_final_lambda, 'xr')

        %% Save data
        ch_resultsFile = sprintf('results_%s_%d', obj.ch_prefix, exp_no);
        save(ch_resultsFile);
        F = 0;
    end
    
    % Hyper Subgradient Descent (HSGD) as described in the paper, 
    % comparison with the code used in experiment 10
    function F = experiment_11(obj)
        obj.P = 100;        %! 20
        obj.n_train = 200;  %! 40
        obj.seed = 30;
        [A_train, y_train] = obj.generate_pseudo_streaming_data();
        
        hgl = HyperGradientLasso;
        hgl.stepsize_lambda = 1e-3;
        hgl.momentum_lambda = 0.9;
        hgl.tol_w = 1e-2;
        hgl.tol_g_lambda = 1e-4;
        hgl.max_iter_outer = 400;  %! 40
        
        hsgd = HyperSubGradientDescent;
        hsgd.s_estimator = 'lasso';
        hsgd.stepsize_lambda = 1e-3;
        hsgd.momentum_lambda = 0.9;
        hsgd.tol_w = 1e-2;
        hsgd.tol_g_lambda = 1e-4;
        hsgd.max_iter_outer = 400; %! 40,  100
        hsgd.s_method_Z = 'linsolve';
        
        [m_W_paper, v_lambda_paper, v_it_count_paper] = ...
            hsgd.solve_gradient(A_train, y_train);
        [m_W, v_lambda, v_it_count] = ...
            hgl.solve_gradient(A_train, y_train);
        final_lambda = v_lambda(find(v_it_count>0,1, 'last'));
        final_lambda_paper = v_lambda_paper(find(v_it_count_paper>0, 1, 'last'));
        average_w = mean(m_W, 2);
       
        disp 'Computing loo errors' 
        [v_looLambdas, v_looErrors, v_loo_error_final_lambda] = ...
            obj.compute_loo_curve([final_lambda final_lambda_paper], ...
            linspace(0.98, 1.02, 21), average_w, A_train, y_train);
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        figure(exp_no*100+1); clf
        obj.show_lambda_vs_ista_iterates({v_lambda, v_lambda_paper},...
            {v_it_count, v_it_count_paper})                
        legend({'Interleave', 'Solve linear system'})
        
        figure(exp_no*100+2); clf
        plot(v_looLambdas, v_looErrors)
        xlabel '\lambda'
        ylabel 'LOO error'
        hold on
        plot([final_lambda final_lambda_paper], v_loo_error_final_lambda, 'xr')

        %% Save data
        ch_resultsFile = sprintf('results_%s_%d', obj.ch_prefix, exp_no);
        save(ch_resultsFile);
        F = 0;
    end
    
    %%
    % After experiment 11, I wrote and executed script_test_solve_z and
    % concluded that in the offline HSGD the best iterative way to solve
    % for z is to initialize in the previous z and do the "natural
    % iterations" based on the fixed-point equation in the paper.
    
    % this experiment checks correct functioning for a given option
    function F = experiment_18(obj)
        obj.P = 10;       
        obj.n_train = 12;
        [A_train, y_train] = obj.generate_pseudo_streaming_data();
        
        n_groups = 2;
        group_size = 5;
        
        s_estimators = {'lasso', 'group-lasso', ...
            'weighted-lasso', 'weighted-group-lasso'};
        s_methods_Z  = { 'linsolve', ... 'iterate',
            'lsqminnorm', 'interleave', 'interleave-fast', 'none' };
        v_online = [0 1];
        v_memory = [1]; %  0
        
        hsgd = HyperSubGradientDescent;
        hsgd.stepsize_lambda = 1e-3;
        hsgd.momentum_lambda = 0.9;
        hsgd.tol_w = 1e-2;
        hsgd.tol_g_lambda = 1e-4;
        hsgd.max_iter_outer = 400; %! 40,  100
%         for k_e = 1:length(s_estimators)
%             for k_m = 1:length(s_methods_Z)
%                 for k_o = 1:length(v_online)
%                     for k_mem = 1:length(v_memory)
                        hsgd.s_estimator = 'group-lasso';
                        hsgd.v_group_structure = kron(1:n_groups, ones(1, group_size));
                        
                        [m_W, v_lambda, v_it_count] = ...
                            hsgd.solve_gradient(A_train, y_train);
%                     end
%                 end
%             end
%         end
        F = 0;
    end


    % This one is just to check that all options run correctly
    function F = experiment_19(obj)
        obj.P = 10;       
        obj.n_train = 20;
        [A_train, y_train] = obj.generate_pseudo_streaming_data();
        
        n_groups = 2;
        group_size = 5;
        
        s_estimators = {'lasso', 'group-lasso', ...
            'weighted-lasso', 'weighted-group-lasso'};
        s_methods_Z  = { 'linsolve', ... 'iterate',
            'lsqminnorm', 'interleave', 'interleave_fast', 'none' };
        v_online = [0 1];
        v_memory = [1]; %  0
        
        hsgd = HyperSubGradientDescent;
        hsgd.stepsize_lambda = 1e-3;
        hsgd.momentum_lambda = 0.9;
        hsgd.tol_w = 1e-2;
        hsgd.tol_g_lambda = 1e-4;
        hsgd.max_iter_outer = 400; %! 40,  100
        for k_e = 1:length(s_estimators)
            for k_m = 1:length(s_methods_Z)
                for k_o = 1:length(v_online)
                    for k_mem = 1:length(v_memory)
                        hsgd.s_estimator = s_estimators{k_e};
                        hsgd.s_method_Z = s_methods_Z{k_m};
                        hsgd.b_online = v_online(k_o);
                        hsgd.b_memory = v_memory(k_mem);
                        hsgd.v_group_structure = kron(1:n_groups, ones(1, group_size));
                        hsgd
                        
                        [m_W, v_lambda, v_it_count] = ...
                            hsgd.solve_gradient(A_train, y_train);
                    end
                end
            end
        end
        F = 0;
    end

    %%
    % Gradient Descent vs Online Gradient Descent
    % GD vs OGD
    function F = experiment_20(obj)
        obj.n_train = 200;
        [A_train, y_train] = obj.generate_pseudo_streaming_data();
        
        hgl_offline = HyperGradientLasso;
        hgl_offline.stepsize_lambda = 1e-3;
        hgl_offline.tol_w = 1e-3;
        hgl_offline.tol_g_lambda = 1e-4;
        hgl_offline.max_iter_outer = 400;
        
        hgl_online = hgl_offline;
        hgl_online.b_online = 1;
        hgl_online.stepsize_lambda = 3e-4;
        hgl_online.max_iter_outer = 10000;
        
        hgl_offline.max_iter_outer = 100;
        [m_W, v_lambda, v_it_count] = hgl_offline.solve_gradient(A_train, y_train);
        final_lambda = v_lambda(find(v_it_count>0,1, 'last'));
        average_w = mean(m_W, 2);
        
        
        [m_W_online, v_lambda_online, v_it_count_online] = ...
            hgl_online.solve_gradient(A_train, y_train);
        final_lambda_online = v_lambda_online(find(v_it_count_online>0,1, 'last'));
       
        disp 'Computing loo errors' 
        [v_looLambdas, v_looErrors, loo_error_final_lambda] = ...
            obj.compute_loo_curve(final_lambda, kron( ...
            [1 final_lambda_online/final_lambda], linspace(0.9, 1.2, 21) ), ...
            average_w, A_train, y_train);
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        figure(exp_no*100+1); clf
        obj.show_lambda_vs_ista_iterates({v_lambda, v_lambda_online},...
            {v_it_count, v_it_count_online})                
        legend({'Batch', 'Online'})
        
        figure(exp_no*100+2); clf
        plot(v_looLambdas, v_looErrors)
        xlabel '\lambda'
        ylabel 'LOO error'
        hold on
        plot(final_lambda, loo_error_final_lambda, 'xr')

        %% Save data
        ch_resultsFile = sprintf('results_%s_%d', obj.ch_prefix, exp_no);
        save(ch_resultsFile);
        F = 0;
    end
    
    % Online Hyper Subgradient Descent (OHSGD) as described in the paper,
    % comparison with the code used in experiment 20.
    % Online HyperGradientLasso vs OHSGD
    function F = experiment_21(obj)
        obj.P = 100;
        obj.n_train = 200;
        obj.seed = 4;
        [A_train, y_train] = obj.generate_pseudo_streaming_data();
        
        my_tol_w = 1e-3;
        my_tol_g = 0;
        
        %%
        hsgd = HyperSubGradientDescent;
        hsgd.s_estimator = 'lasso';
        hsgd.tol_w = my_tol_w;
        hsgd.tol_g_lambda = my_tol_g;

        ohsgd = hsgd;
        ohsgd.b_online = 1;
        ohsgd.stepsize_lambda = 3e-4;
        ohsgd.momentum_lambda = 0.9;
        ohsgd.max_iter_outer = 10000;
        
        %%
        hgl_offline = HyperGradientLasso;
        hgl_offline.tol_w = my_tol_w;
        hgl_offline.tol_g_lambda = my_tol_g;

        hgl_online = hgl_offline;
        hgl_online.b_online = 1;
        hgl_online.stepsize_lambda = 3e-4;
        hgl_online.momentum_lambda = 0.9;
        hgl_online.max_iter_outer = 10000;
        
        %%        
        [m_W_ohsgd, v_lambda_ohsgd, v_it_count_ohsgd] = ohsgd.solve_gradient(A_train, y_train);
        final_lambda_ohsgd = v_lambda_ohsgd(find(v_it_count_ohsgd>0,1, 'last'));
        average_w = mean(m_W_ohsgd, 2);
              
        [m_W_ohgl, v_lambda_ohgl, v_it_count_ohgl] = ...
            hgl_online.solve_gradient(A_train, y_train);
        final_lambda_ohgl = v_lambda_ohgl(find(v_it_count_ohgl>0,1, 'last'));
        
        v_the_final_lambdas = [final_lambda_ohsgd final_lambda_ohgl];
        
        %%        
        disp 'Computing loo errors' 
        [v_looLambdas, v_looErrors, v_loo_error_final_lambdas] = obj.compute_loo_curve(...
            v_the_final_lambdas, linspace(0.4, 2.5, 21), ...
            average_w, A_train, y_train);
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        figure(exp_no*100+1); clf
        obj.show_lambda_vs_ista_iterates({v_lambda_ohsgd, v_lambda_ohgl},...
            {v_it_count_ohsgd, v_it_count_ohgl})                
        legend({'OHSGD(paper)', 'Ohgl (previous ver)'})
        
        figure(exp_no*100+2); clf
        plot(v_looLambdas, v_looErrors)
        xlabel '\lambda'
        ylabel 'LOO error'
        hold on
        plot(v_the_final_lambdas, v_loo_error_final_lambdas, 'xr')

        %% Save data
        ch_resultsFile = sprintf('results_%s_%d', obj.ch_prefix, exp_no);
        save(ch_resultsFile);
        F = 0;
    end

    % This one is only to make sure that HyperSubGradientDescent with 
    % property method_Z set to 'interleave' does the same as
    % HyperGradientLasso
    function F = experiment_22(obj)
        obj.P = 100;
        obj.n_train = 200;
        obj.seed = 4;
        [A_train, y_train] = obj.generate_pseudo_streaming_data();
        
        my_tol_w = 1e-3;
        my_tol_g = 0;
        
        %%
        hsgd = HyperSubGradientDescent;
        hsgd.s_estimator = 'lasso';
        hsgd.tol_w = my_tol_w;
        hsgd.tol_g_lambda = my_tol_g;

        ohsgd = hsgd;
        ohsgd.b_online = 1;
        ohsgd.stepsize_lambda = 3e-4;
        ohsgd.momentum_lambda = 0.9;
        ohsgd.max_iter_outer = 10000;
        
        ohsgd.s_method_Z = 'interleave';
        
        %%
        hgl_offline = HyperGradientLasso;
        hgl_offline.tol_w = my_tol_w;
        hgl_offline.tol_g_lambda = my_tol_g;

        hgl_online = hgl_offline;
        hgl_online.b_online = 1;
        hgl_online.stepsize_lambda = 3e-4;
        hgl_online.momentum_lambda = 0.9;
        hgl_online.max_iter_outer = 10000;
        
        %%        
        [m_W_ohsgd, v_lambda_ohsgd, v_it_count_ohsgd] = ohsgd.solve_gradient(A_train, y_train);
        final_lambda_ohsgd = v_lambda_ohsgd(find(v_it_count_ohsgd>0,1, 'last'));
        average_w = mean(m_W_ohsgd, 2);
              
        [m_W_ohgl, v_lambda_ohgl, v_it_count_ohgl] = ...
            hgl_online.solve_gradient(A_train, y_train);
        final_lambda_ohgl = v_lambda_ohgl(find(v_it_count_ohgl>0,1, 'last'));
        
        v_the_final_lambdas = [final_lambda_ohsgd final_lambda_ohgl];
        
        %%        
        disp 'Computing loo errors' 
        [v_looLambdas, v_looErrors, v_loo_error_final_lambdas] = obj.compute_loo_curve(...
            v_the_final_lambdas, linspace(0.4, 2.5, 21), ...
            average_w, A_train, y_train);
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        figure(exp_no*100+1); clf
        obj.show_lambda_vs_ista_iterates({v_lambda_ohsgd, v_lambda_ohgl},...
            {v_it_count_ohsgd, v_it_count_ohgl})                
        legend({'OHSGD(paper)', 'Ohgl (previous ver)'})
        
        figure(exp_no*100+2); clf
        plot(v_looLambdas, v_looErrors)
        xlabel '\lambda'
        ylabel 'LOO error'
        hold on
        plot(v_the_final_lambdas, v_loo_error_final_lambdas, 'xr')

        %% Save data
        ch_resultsFile = sprintf('results_%s_%d', obj.ch_prefix, exp_no);
        save(ch_resultsFile);
        F = 0;
    end

    % This function compares 4 options of HSGD: 
    % Offline, iterative computation of m_Z
    % Online,  iterative computation of m_Z
    % Offline, compute m_Z by linsolve
    % Online,  compute m_Z by linsolve
    function F = experiment_23(obj)
        obj.P = 100;
        obj.n_train = 110;
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
        
        c_estimators = { hsgd_lin, ohsgd_lin, hsgd_it, ohsgd_it};
                       
        %%
        for k = 1:length(c_estimators)
            [c_W{k}, c_lambda{k}, c_it_count{k}] = c_estimators{k}.solve_gradient(A_train, y_train);
            m_final_lambda(:, k) = c_lambda{k}(find(c_it_count{k}>0,1, 'last'));
            m_average_w(:,k) = mean(c_W{k}, 2);
        end
        
        %%        
        disp 'Computing loo errors' 
        [v_looLambdas, v_looErrors, v_loo_error_final_lambdas] = obj.compute_loo_curve(...
            m_final_lambda, linspace(0.4, 2.5, 21), ...
            mean(m_average_w, 2), A_train, y_train);
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        c_legend = {'Offline, linsolve Z', 'Online, linsolve Z', ...
            'Offline, iterative Z', 'Online, iterative Z'};
        
        figure(exp_no*100+1); clf
        obj.show_lambda_vs_ista_iterates(c_lambda, c_it_count);
        legend(c_legend)
        
        figure(exp_no*100+2); clf
        plot(v_looLambdas, v_looErrors)
        xlabel '\lambda'
        ylabel 'LOO error'
        hold on
        plot(m_final_lambda, v_loo_error_final_lambdas, 'xr')

        %% Save data
        ch_resultsFile = sprintf('results_%s_%d', obj.ch_prefix, exp_no);
        save(ch_resultsFile);
        F = 0;
    end
    
    % This function compares Offline and Online,  
    % computing m_Z by linsolve   
    % sweep over beta to get the fastest convergence
    function F = experiment_24(obj)
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

    % Online Gradient Descent
    % Comparing effect of different tolerances of ISTA
    % (inexact approximation -- with memory)
    function F = experiment_30(obj)
        obj.n_train = 200;
        [A_train, y_train] = obj.generate_pseudo_streaming_data();
        
        hgl_online = HyperGradientLasso;
        hgl_online.b_online = 1;
        hgl_online.tol_g_lambda = 1e-4;        
        hgl_online.stepsize_lambda = 1e-4;
        hgl_online.max_iter_outer = 10000;
        
        n_lambdas = 4;
        %T = hgl_online.max_iter_outer;
        c_lambda = cell(n_lambdas, 1);
        c_it_count = c_lambda;
        v_tolerances = logspace(-3, 0, n_lambdas);
        v_final_lambda = zeros(n_lambdas, 1);
        m_average_w = zeros(obj.P, n_lambdas);
        
        for k_lambda = 1:n_lambdas
            hgl_online.tol_w = v_tolerances(k_lambda);
            [m_W, c_lambda{k_lambda}, c_it_count{k_lambda}] = ...
                hgl_online.solve_gradient(A_train, y_train);
            v_final_lambda(k_lambda) = c_lambda{k_lambda}( ...
                find(c_it_count{k_lambda}>0,1, 'last'));
            m_average_w(:, k_lambda) = mean(m_W, 2);
        end
               
        disp 'Computing loo errors' 
        [v_looLambdas, v_looErrors, loo_error_final_lambda] = ...
            obj.compute_loo_curve(v_final_lambda(1), ...
            linspace(0.9, 1.2, 21), m_average_w(:, 1), A_train, y_train);
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        figure(exp_no*100+1); clf
        obj.show_lambda_vs_ista_iterates(c_lambda, c_it_count)
        c_legend = cell(n_lambdas, 1);
        for k_lambda = 1:n_lambdas
            c_legend{k_lambda} = sprintf('tol = %g', v_tolerances(k_lambda));
        end
        legend(c_legend)
        
        figure(exp_no*100+2); clf
        plot(v_looLambdas, v_looErrors)
        xlabel '\lambda'
        ylabel 'LOO error'
        hold on
        plot(v_final_lambda(1), loo_error_final_lambda, 'xr')

        %% Save data
        ch_resultsFile = sprintf('results_%s_%d', obj.ch_prefix, exp_no);
        save(ch_resultsFile);
        F = 0;
    end
    
    % Online Hyper Subgradient Descent
    % linsolve vs. iterative Z
    % This experiment combines 23 and 30:
    % We compare the effect of different tolerances of ISTA
    % (inexact approximation -- with memory)
    function F = experiment_33(obj)
        obj.P = 100;
        obj.n_train = 90;
        obj.seed = 4;
        [A_train, y_train] = obj.generate_pseudo_streaming_data();
        
        my_tol_w    = 1e-3;
        my_tol_g    = 0;
        my_beta     = 1e-4;
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
        ohsgd_it.max_iter_outer = 10000; %! 1000
        
        hsgd_lin  = hsgd_it;   hsgd_lin.s_method_Z = 'linsolve';
        ohsgd_lin = ohsgd_it; ohsgd_lin.s_method_Z = 'linsolve';
        
        c_estimators = {ohsgd_lin,  ohsgd_it};
        n_estimators = length(c_estimators);
        
        n_tol = 7;
        v_tolerances = logspace(-3, 0, n_tol);
        
        m_betas = my_beta*ones(n_tol, n_estimators);
        m_betas(:, 1) = m_betas(:,1)/7;
                       
        %%
        t_average_w = zeros(obj.P, n_estimators, n_tol);
        for k_e = 1:n_estimators
            for k_t = 1:n_tol
                estimator_now = c_estimators{k_e};
                estimator_now.tol_w = v_tolerances(k_t);
                estimator_now.stepsize_lambda = m_betas(k_t, k_e);
                [c_W{k_t, k_e}, c_lambda{k_t, k_e}, c_it_count{k_t, k_e}] = ...
                    estimator_now.solve_gradient(A_train, y_train);
                    t_final_lambda(:, k_t, k_e) = c_lambda{k_t, k_e}(find(c_it_count{k_t, k_e}>0,1, 'last'));
                    t_average_w(:,k_t, k_e) = mean(c_W{k_t, k_e}, 2);
            end
        end
        m_final_lambda = reshape(t_final_lambda, size(t_final_lambda, 1), ...
            numel(t_final_lambda)/ size(t_final_lambda, 1));
        
        %%        
        disp 'Computing loo errors' 
        [v_looLambdas, v_looErrors, v_loo_error_final_lambdas] = obj.compute_loo_curve(...
            m_final_lambda, linspace(0.4, 2.5, 21), ...
            mean(mean(t_average_w, 2), 3), A_train, y_train);
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        c_legend = cell(n_tol, 2);
        for k_e = 1:numel(c_estimators)
            for k_t = 1:n_tol
                c_legend{k_t, k_e} = sprintf('tol = %g', v_tolerances(k_t));
            end
        end
        c_legend{1}     = [c_legend{1}, ', linsolve'];
        c_legend{k_t+1} = [c_legend{k_t+1}, ', iterative'];
        
        figure(exp_no*100+1); clf
        obj.show_lambda_vs_ista_iterates(vec(c_lambda), vec(c_it_count));
        legend(c_legend(:))
        
        figure(exp_no*100+2); clf
        plot(v_looLambdas, v_looErrors)
        xlabel '\lambda'
        ylabel 'LOO error'
        hold on
        plot(m_final_lambda, v_loo_error_final_lambdas, 'xr')

        %% Save data
        ch_resultsFile = sprintf('results_%s_%d', obj.ch_prefix, exp_no);
        save(ch_resultsFile);
        F = 0;
    end
    
    % Online Hyper Subgradient Descent
    % linsolve vs. iterative Z
    % %commented code is used to determine best stepsize beta
    % We compare the effect of different tolerances of ISTA
    % (inexact approximation -- with memory)
    function F = experiment_34(obj)
        obj.P = 100;
        obj.n_train = 200;
        obj.seed = 5;
        [A_train, y_train] = obj.generate_pseudo_streaming_data();
        
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
        ohsgd_it.max_iter_outer = 1000; %! 10000
        
        hsgd_lin  = hsgd_it;   hsgd_lin.s_method_Z = 'linsolve';
        ohsgd_lin = ohsgd_it; ohsgd_lin.s_method_Z = 'linsolve';
        
        c_estimators = {ohsgd_lin,  ohsgd_it};
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
        
        %%        
        disp 'Computing loo errors' 
        [v_looLambdas, v_looErrors, v_loo_error_final_lambdas] = obj.compute_loo_curve(...
            m_final_lambda, linspace(0.4, 2.5, 21), ...
            mean(mean(t_average_w, 2), 3), A_train, y_train);
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        c_legend = cell(n_tol, 2);
        for k_e = 1:numel(c_estimators)
            for k_t = 1:n_tol
                c_legend{k_t, k_e} = sprintf('tol = %g, \\beta = %g', ...
                    v_tolerances(k_t), v_betas(k_t));
            end
        end
        c_legend{1}     = [c_legend{1}, ', linsolve'];
        c_legend{k_t+1} = [c_legend{k_t+1}, ', iterative'];
        
        figure(exp_no*100+1); clf
        obj.show_lambda_vs_ista_iterates(vec(c_lambda), vec(c_it_count));
        legend(c_legend(:))
        
        figure(exp_no*100+2); clf
        plot(v_looLambdas, v_looErrors)
        xlabel '\lambda'
        ylabel 'LOO error'
        hold on
        plot(m_final_lambda, v_loo_error_final_lambdas, 'xr')
        
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

        %% Save data
        ch_resultsFile = sprintf('results_%s_%d', obj.ch_prefix, exp_no);
        save(ch_resultsFile);
        F = 0;
    end

    % Online Hyper Subgradient Descent
    % Linsolve
    % We compare the effect of different tolerances of ISTA
    % (inexact approximation -- with memory)
    % same as 34 but larger P, more samples
    function F = experiment_35(obj)
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

   
    % Online Gradient Descent
    % HyperGradientLasso (old version) and OHO (older version).
    % Memoryless fails if executed with very coarse tolerance.
    function F = experiment_40(obj)
        obj.n_train = 200;
        obj.seed = 10;
        [A_train, y_train] = obj.generate_pseudo_streaming_data();
        
        hgl_memory = HyperGradientLasso;
        hgl_memory.b_online = 1;
        hgl_memory.stepsize_lambda = 1e-4;
        hgl_memory.tol_w = 1e-2;
        hgl_memory.tol_g_lambda = 1e-4;
        hgl_memory.max_iter_outer = 10000;
        
        hgl_no_mem = hgl_memory;
        hgl_no_mem.max_iter_outer = 2000;
        hgl_no_mem.stepsize_lambda = 1e-4;
        hgl_no_mem.tol_w = 1e-2; % 1e-1 is too coarse. 1e-2 is ok.
        hgl_no_mem.b_memory = 0;
        
        [m_W_memory, v_lambda_memory, v_it_count_memory] = ...
            hgl_memory.solve_gradient(A_train, y_train);
        final_lambda_memory = v_lambda_memory(find(v_it_count_memory>0,1, 'last'));
        average_w = mean(m_W_memory, 2);
              
        [m_W_nomem, v_lambda_nomem, v_it_count_nomem] = ...
            hgl_no_mem.solve_gradient(A_train, y_train);
        final_lambda_nomem = v_lambda_nomem(find(v_it_count_nomem>0,1, 'last'));
        
%         dlh = FranceschiRecursiveLasso; % Dynamic
%         dlh.stepsize_w = 3e-4;
%         dlh.stepsize_lambda = 1e-4;
%         
%         niter_dynamic = 10000;
%         [m_W_dynamic, v_lambda_dynamic] = obj.drill_dynamic( ...
%             dlh, A_train, y_train, 0, niter_dynamic);
        
        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.b_memory = 0;
        hl.max_iter_outer= 2500;
        hl.stepsize_policy = ConstantStepsize;
        hl.tol_w = 1e0;
        hl.stepsize_policy.eta_0 = 1e-2;
        [v_lambda_OHO, v_count_OHO] ...
            = hl.solve_approx_mirror(A_train', y_train);
       
        disp 'Computing loo errors' 
        [v_looLambdas, v_looErrors, loo_error_final_lambda_nomem] = ...
            obj.compute_loo_curve(final_lambda_nomem, linspace(...
            0.2, ...1.1*final_lambda_nomem  -0.1*final_lambda_memory, ...
            0.7, ...1.1*final_lambda_memory -0.1*final_lambda_nomem,  ...
            21)./final_lambda_nomem, ...
            average_w, A_train, y_train);
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        figure(exp_no*100+1); clf
        obj.show_lambda_vs_ista_iterates({v_lambda_memory, v_lambda_nomem, v_lambda_OHO},...
            {v_it_count_memory, v_it_count_nomem, v_count_OHO})                
        legend({'With-memory', 'Memoryless', 'OHO'})
        
        figure(exp_no*100+2); clf
        plot(v_looLambdas, v_looErrors)
        xlabel '\lambda'
        ylabel 'LOO error'
        hold on
        plot(final_lambda_nomem, loo_error_final_lambda_nomem, 'xr')
        
        %% Save data
        ch_resultsFile = sprintf('results_%s_%d', obj.ch_prefix, exp_no);
        save(ch_resultsFile);
        F = 0;
    end

    % Online Gradient Descent
    % HyperGradientLasso (old version) and OHO (older version).
    % Memoryless fails if executed with very coarse tolerance.
    % TODO: currently this is a copy of experiment_40. I have to
    % rewrite it using the HyperSubGradientDescent class, confirm that it
    % does the same and play a bit to get to know what happens when 
    % we use online, inexact, memoryless (which is expected to be the
    % big-data-ready algorithm).
    function F = experiment_41(obj)
        obj.n_train = 200;
        obj.seed = 10;
        [A_train, y_train] = obj.generate_pseudo_streaming_data();
        
        hgl_memory = HyperGradientLasso;
        hgl_memory.b_online = 1;
        hgl_memory.stepsize_lambda = 1e-4;
        hgl_memory.tol_w = 1e-2;
        hgl_memory.tol_g_lambda = 1e-4;
        hgl_memory.max_iter_outer = 10000;
        
        hgl_no_mem = hgl_memory;
        hgl_no_mem.max_iter_outer = 2000;
        hgl_no_mem.stepsize_lambda = 1e-4;
        hgl_no_mem.tol_w = 1e-2; % 1e-1 is too coarse. 1e-2 is ok.
        hgl_no_mem.b_memory = 0;
        
        [m_W_memory, v_lambda_memory, v_it_count_memory] = ...
            hgl_memory.solve_gradient(A_train, y_train);
        final_lambda_memory = v_lambda_memory(find(v_it_count_memory>0,1, 'last'));
        average_w = mean(m_W_memory, 2);
              
        [m_W_nomem, v_lambda_nomem, v_it_count_nomem] = ...
            hgl_no_mem.solve_gradient(A_train, y_train);
        final_lambda_nomem = v_lambda_nomem(find(v_it_count_nomem>0,1, 'last'));
        
%         dlh = FranceschiRecursiveLasso; % Dynamic
%         dlh.stepsize_w = 3e-4;
%         dlh.stepsize_lambda = 1e-4;
%         
%         niter_dynamic = 10000;
%         [m_W_dynamic, v_lambda_dynamic] = obj.drill_dynamic( ...
%             dlh, A_train, y_train, 0, niter_dynamic);
        
        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.b_memory = 0;
        hl.max_iter_outer= 2500;
        hl.stepsize_policy = ConstantStepsize;
        hl.tol_w = 1e0;
        hl.stepsize_policy.eta_0 = 1e-2;
        [v_lambda_OHO, v_count_OHO] ...
            = hl.solve_approx_mirror(A_train', y_train);
       
        disp 'Computing loo errors' 
        [v_looLambdas, v_looErrors, loo_error_final_lambda_nomem] = ...
            obj.compute_loo_curve(final_lambda_nomem, linspace(...
            0.2, ...1.1*final_lambda_nomem  -0.1*final_lambda_memory, ...
            0.7, ...1.1*final_lambda_memory -0.1*final_lambda_nomem,  ...
            21)./final_lambda_nomem, ...
            average_w, A_train, y_train);
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        figure(exp_no*100+1); clf
        obj.show_lambda_vs_ista_iterates({v_lambda_memory, v_lambda_nomem, v_lambda_OHO},...
            {v_it_count_memory, v_it_count_nomem, v_count_OHO})                
        legend({'With-memory', 'Memoryless', 'OHO'})
        
        figure(exp_no*100+2); clf
        plot(v_looLambdas, v_looErrors)
        xlabel '\lambda'
        ylabel 'LOO error'
        hold on
        plot(final_lambda_nomem, loo_error_final_lambda_nomem, 'xr')
        
        %% Save data
        ch_resultsFile = sprintf('results_%s_%d', obj.ch_prefix, exp_no);
        save(ch_resultsFile);
        F = 0;
    end

    
    % Online Gradient Descent
    % With-memory, memoryless and OHO (old version)
    % An example where OHO works ok (see experiment_6 for an example where
    % OHO fails).
    function F = experiment_5(obj)
        obj.n_train = 200;
        obj.seed = 10;
        [A_train, y_train] = obj.generate_pseudo_streaming_data();
        
        hgl_memory = HyperGradientLasso;
        hgl_memory.b_online = 1;
        hgl_memory.stepsize_lambda = 1e-4;
        hgl_memory.tol_w = 1e-2;
        hgl_memory.tol_g_lambda = 1e-4;
        hgl_memory.max_iter_outer = 1000; %! 10000
        
        hgl_no_mem = hgl_memory;
        hgl_no_mem.max_iter_outer = 1000;
        hgl_no_mem.stepsize_lambda = 1e-4;
        hgl_no_mem.tol_w = 1e-3;
        hgl_no_mem.b_memory = 0;
        
        [m_W_memory, v_lambda_memory, v_it_count_memory] = ...
            hgl_memory.solve_gradient(A_train, y_train);
        final_lambda_memory = v_lambda_memory(find(v_it_count_memory>0,1, 'last'));
        average_w = mean(m_W_memory, 2);
        
        
        [m_W_nomem, v_lambda_nomem, v_it_count_nomem] = ...
            hgl_no_mem.solve_gradient(A_train, y_train);
        final_lambda_nomem = v_lambda_nomem(find(v_it_count_nomem>0,1, 'last'));
        
        dlh = FranceschiRecursiveLasso; % Dynamic
        dlh.stepsize_w = 1e-4;
        dlh.stepsize_lambda = 1e-4;
        
        niter_dynamic = 10000;
        [m_W_dynamic, v_lambda_dynamic] = obj.drill_dynamic( ...
            dlh, A_train, y_train, 0, niter_dynamic);
        
        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.b_memory = 0;
        hl.max_iter_outer= 10000;
        hl.stepsize_policy = ConstantStepsize;
        hl.tol_w = 1e-2;
        hl.stepsize_policy.eta_0 = 1e-3;
        hl.approx_type = 'hard';
        [v_lambda_OHO, v_count_OHO] ...
            = hl.solve_approx_mirror(A_train', y_train);
       
        disp 'Computing loo errors' 
        [v_looLambdas, v_looErrors, loo_error_final_lambda_nomem] = ...
            obj.compute_loo_curve(final_lambda_nomem, linspace(...
            0.35, ...1.1*final_lambda_nomem  -0.1*final_lambda_memory, ...
            0.9, ...1.1*final_lambda_memory -0.1*final_lambda_nomem,  ...
            21)./final_lambda_nomem, ...
            average_w, A_train, y_train);
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        figure(exp_no*100+1); clf
        obj.show_lambda_vs_ista_iterates({v_lambda_memory, v_lambda_nomem,v_lambda_dynamic, v_lambda_OHO},...
            {v_it_count_memory, v_it_count_nomem, ones(size(v_lambda_dynamic)), v_count_OHO})                
        legend({'With-memory', 'Memoryless', 'Dynamic', 'OHO'})
        
        figure(exp_no*100+2); clf
        plot(v_looLambdas, v_looErrors)
        xlabel '\lambda'
        ylabel 'LOO error'
        hold on
        plot(final_lambda_nomem, loo_error_final_lambda_nomem, 'xr')

        %% Save data
        ch_resultsFile = sprintf('results_%s_%d', obj.ch_prefix, exp_no);
        save(ch_resultsFile);
        F = 0;
    end
    
    % Online Gradient Descent
    % With-memory, memoryless, dynamic, and OHO (old version)
    % OHO fails.
    % Using the dynamic solver when the number of samples is very small
    % leads to a much smaller value of lambda.
    function F = experiment_6(obj)
        obj.n_train = 400;
        obj.n_test = 400;
        obj.seed = 10;
        [A_train, y_train, ~, A_test, y_test] = obj.generate_pseudo_streaming_data();
        
        hgl_memory = HyperGradientLasso;
        hgl_memory.b_online = 1;
        hgl_memory.stepsize_lambda = 1e-4;
        hgl_memory.tol_w = 1e-2;
        hgl_memory.tol_g_lambda = 1e-4;
        hgl_memory.max_iter_outer = 1000;
        
        hgl_no_mem = hgl_memory;
        hgl_no_mem.max_iter_outer = 1000;
        hgl_no_mem.stepsize_lambda = 1e-4;
        hgl_no_mem.tol_w = 1e-3;
        hgl_no_mem.b_memory = 0;
        
        [m_W_memory, v_lambda_memory, v_it_count_memory] = ...
            hgl_memory.solve_gradient(A_train, y_train);
        final_lambda_memory = v_lambda_memory(find(v_it_count_memory>0,1, 'last'));
        average_w = mean(m_W_memory, 2);
        
        
        [m_W_nomem, v_lambda_nomem, v_it_count_nomem] = ...
            hgl_no_mem.solve_gradient(A_train, y_train);
        final_lambda_nomem = v_lambda_nomem(find(v_it_count_nomem>0,1, 'last'));
        
        dlh = FranceschiRecursiveLasso; % Dynamic
        dlh.stepsize_w = []; %! 1e-5;
        dlh.stepsize_lambda = 1e-5;
        
        niter_dynamic = 100000;
        [m_W_dynamic, v_lambda_dynamic] = obj.drill_dynamic( ...
            dlh, A_test, y_test, 0, niter_dynamic);
        
        hl = OHO_Lasso;
        hl.mirror_type = 'grad';
        hl.b_online = 1;
        hl.b_memory = 0;
        hl.max_iter_outer= 2000;
        hl.stepsize_policy = ConstantStepsize;
        hl.tol_w = 1e-3;
        hl.stepsize_policy.eta_0 = 3e-3;
        hl.approx_type = 'hard';
        [v_lambda_OHO, v_count_OHO] ...
            = hl.solve_approx_mirror(A_train', y_train);
       
        disp 'Computing loo errors' 
        [v_looLambdas, v_looErrors, loo_error_final_lambda_nomem] = ...
            obj.compute_loo_curve(final_lambda_nomem, linspace(...
            0.15, ...1.1*final_lambda_nomem  -0.1*final_lambda_memory, ...
            0.5, ...1.1*final_lambda_memory -0.1*final_lambda_nomem,  ...
            21)./final_lambda_nomem, ... %! 21
            average_w, A_train, y_train);
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        figure(exp_no*100+1); clf
        obj.show_lambda_vs_ista_iterates({v_lambda_memory, v_lambda_nomem,v_lambda_dynamic, v_lambda_OHO},...
            {v_it_count_memory, v_it_count_nomem, ones(size(v_lambda_dynamic)), v_count_OHO})                
        legend({'With-memory', 'Memoryless', 'Dynamic', 'OHO'})
        
        figure(exp_no*100+2); clf
        plot(v_looLambdas, v_looErrors)
        xlabel '\lambda'
        ylabel 'LOO error'
        hold on
        plot(final_lambda_nomem, loo_error_final_lambda_nomem, 'xr')
        

        %% Save data
        ch_resultsFile = sprintf('results_%s_%d', obj.ch_prefix, exp_no);
        save(ch_resultsFile);
        F = 0;
    end

    % Online Gradient Descent
    % Memoryless executed with very coarse tolerance.
    % Number of samples increased to 5000, to see if it takes the
    % memoryless to the optimal lambda
    function F = experiment_7(obj)
        obj.n_train = 5000;
        obj.seed = 10;
        [A_train, y_train] = obj.generate_pseudo_streaming_data();
        
        hgl_memory = HyperGradientLasso;
        hgl_memory.b_online = 1;
        hgl_memory.stepsize_lambda = 1e-5;
        hgl_memory.tol_w = 1e-3;
        hgl_memory.tol_g_lambda = 1e-4;
        hgl_memory.max_iter_outer = 10000;
        
        hgl_no_mem = hgl_memory;
        hgl_no_mem.max_iter_outer = 20000;
        hgl_no_mem.stepsize_lambda = 3e-6;
        hgl_no_mem.tol_w = 1e-3; % 1e-1 is too coarse. 1e-2 is ok.
        hgl_no_mem.b_memory = 0;
        hgl_no_mem.max_iter_inner = 1;
        hgl_no_mem.param_c = 0;
        
        [m_W_memory, v_lambda_memory, v_it_count_memory] = ...
            hgl_memory.solve_gradient(A_train, y_train);
        final_lambda_memory = v_lambda_memory(find(v_it_count_memory>0,1, 'last'));
        average_w = mean(m_W_memory, 2);
              
        [m_W_nomem, v_lambda_nomem, v_it_count_nomem] = ...
            hgl_no_mem.solve_gradient(A_train, y_train);
        final_lambda_nomem = v_lambda_nomem(find(v_it_count_nomem>0,1, 'last'));
        
        dlh = FranceschiRecursiveLasso; % Dynamic
        dlh.stepsize_w = [];
        dlh.stepsize_lambda = 1e-5;
        
        niter_dynamic = 50000;
        [m_W_dynamic, v_lambda_dynamic] = obj.drill_dynamic( ...
            dlh, A_train, y_train, 0, niter_dynamic); %#ok<ASGLU>
        final_lambda_dynamic = v_lambda_dynamic(end);
        v_it_count_dynamic = ones(1, length(v_lambda_dynamic));
        
%         hl = OHO_Lasso;
%         hl.mirror_type = 'grad';
%         hl.b_online = 1;
%         hl.b_memory = 0;
%         hl.max_iter_outer= 250;
%         hl.stepsize_policy = ConstantStepsize;
%         hl.tol_w = 1e0;
%         hl.stepsize_policy.eta_0 = 1e-2;
%         [v_lambda_OHO, v_it_count_OHO] ...
%             = hl.solve_approx_mirror(A_train', y_train);
       
        disp 'Computing loo errors' 
        lambda_0_loo = [final_lambda_nomem, final_lambda_memory, final_lambda_dynamic];
        [v_looLambdas, v_looErrors, v_looError_lambda_0] = ...
            obj.compute_loo_curve(lambda_0_loo,...
            [], average_w, A_train, y_train, [0 1], 7);
        
        disp 'grid search over lambda for the dynamic setting'
        v_avgLoss_dynamic = obj.grid_search_dynamic_lambda(dlh, ...
            A_train, y_train, v_looLambdas, niter_dynamic);
               
        exp_no = obj.determine_experiment_number();
        %% Figures
        figure(exp_no*100+1); clf
        obj.show_lambda_vs_ista_iterates({v_lambda_memory, v_lambda_nomem, v_lambda_dynamic},...
            {v_it_count_memory, v_it_count_nomem, v_it_count_dynamic})                
        legend({'With-memory', 'Memoryless', 'Dynamic'})
        
        figure(exp_no*100+2); clf
        plot(v_looLambdas, v_looErrors)
        xlabel '\lambda'
        ylabel 'LOO error'
        hold on
        plot(lambda_0_loo, v_looError_lambda_0, 'xr')
        plot(v_looLambdas, v_avgLoss_dynamic)
        
        %% Save data
        ch_resultsFile = sprintf('results_%s_%d', obj.ch_prefix, exp_no);
        save(ch_resultsFile);
        F = 0;
    end

    % Online Gradient Descent
    % Memoryless executed with different values of param_c
    function F = experiment_8(obj)
        obj.n_train = 500;
        obj.seed = 10;
        [A_train, y_train] = obj.generate_pseudo_streaming_data();
        
        n_grid_c = 5;
        v_grid_c = zeros(1, n_grid_c);
        v_grid_beta = logspace(-4, -3, n_grid_c);
        c_lambda = cell(n_grid_c, 1);
        c_it_count = c_lambda;
        v_final_lambdas = zeros(1, n_grid_c);
        
        
        hgl_no_mem = HyperGradientLasso;
        hgl_no_mem.b_online = 1;        
        hgl_no_mem.tol_g_lambda = 1e-4;
        hgl_no_mem.max_iter_outer = 20000;
        hgl_no_mem.tol_w = 1e-2; % 1e-1 is too coarse. 1e-2 is ok.
        hgl_no_mem.b_memory = 0;
        hgl_no_mem.max_iter_inner = 1;
        
        for k = 1:n_grid_c
            hgl_no_mem.param_c = v_grid_c(k);
            hgl_no_mem.stepsize_lambda = v_grid_beta(k);
            [m_W, c_lambda{k}, c_it_count{k}] = ...
                hgl_no_mem.solve_gradient(A_train, y_train);
            v_final_lambdas(k) = c_lambda{k}(find(c_it_count{k}>0,1, 'last'));
        end
        average_w = mean(m_W, 2);
       
        disp 'Computing loo errors' 
        [v_looLambdas, v_looErrors, v_looError_lambda_0] = ...
            obj.compute_loo_curve(v_final_lambdas,...
            [], average_w, A_train, y_train, [-0.2 1.2], 7);
                       
        exp_no = obj.determine_experiment_number();
        %% Figures
        figure(exp_no*100+1); clf
        obj.show_lambda_vs_ista_iterates(c_lambda, c_it_count)   
        c_legend = cell(n_grid_c, 1);
        for k = 1:n_grid_c
            c_legend{k} = sprintf('C = %g, \beta = %g', v_grid_c(k), v_grid_beta(k));
        end
        legend(c_legend)
        
        figure(exp_no*100+2); clf
        plot(v_looLambdas, v_looErrors)
        xlabel '\lambda'
        ylabel 'LOO error'
        hold on
        plot(v_final_lambdas, v_looError_lambda_0, 'xr')
        
        %% Save data
        ch_resultsFile = sprintf('results_%s_%d', obj.ch_prefix, exp_no);
        save(ch_resultsFile);
        F = 0;
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
        N = length(v_y_train); assert(size(m_X_train, 2)==N);
        P = size(m_X_train, 1);
        v_oos_errors = nan(length(st_hyperparams), 1);
        for k = 1:length(st_hyperparams)
            if isfield(st_hyperparams(k) ,'rho')
                m_Phi = m_X_train*m_X_train'/N + st_hyperparams(k).rho*eye(P);
            else
                m_Phi = m_X_train*m_X_train'/N;
            end
            v_r = m_X_train*v_y_train(:)/N;
            my_alpha = 10/trace(m_Phi);
            hl = OHO_Lasso;
            hl.tol_w = 1e-3;
            v_w_0 = hl.ista(v_w_0, m_Phi, v_r, my_alpha, st_hyperparams(k).lambda);
            v_oos_errors(k) = mean((v_y_test(:) - m_X_test'*v_w_0).^2)./mean(v_y_test.^2);
        end
    end
        
    function [loo_error, v_w_j] = exact_loo(m_X, v_y, st_hyperparams, v_w_0)
        
        N = length(v_y); assert(size(m_X, 2)==N);
        P = size(m_X, 1);
        assert(isscalar(st_hyperparams))
        lambda = st_hyperparams.lambda;
        
        if isfield(st_hyperparams, 'rho')
            m_Phi = m_X*m_X'/N + st_hyperparams.rho*eye(P);
        else
            m_Phi = m_X*m_X'/N ;
        end
        v_r = m_X*v_y(:)/N;
        my_alpha = 1/eigs(m_Phi, 1);
        hl = OHO_Lasso;
        hl.tol_w = 1e-4;
        v_looErrors = zeros(N,1);
        ltc = LoopTimeControl(N);
        for j =1:N
            v_x_j = m_X(:, j);
            m_Phi_j = m_Phi - v_x_j * v_x_j'/N;
            v_r_j   = v_r   - v_y(j)* v_x_j/N;
            v_w_j = hl.ista(v_w_0, m_Phi_j, v_r_j, my_alpha, lambda);
            v_looErrors(j) = v_y(j)-v_x_j'*v_w_j;
            ltc.go(j);
        end
        loo_error = mean(v_looErrors.^2)./mean(v_y.^2);
    end
    
    function v_looErrors = compute_loo_errors(st_hyperparams, v_w_0, m_X, v_y)
        % CAN BE DELETED
        
        v_looErrors = nan(size(st_hyperparams));
        for k = 1:length(st_hyperparams)
            v_looErrors(k) = HyperGradientExperiments.exact_loo(m_X, v_y, st_hyperparams(k), v_w_0);
        end
    end
    
    function [v_looLambdas, v_looErrors, v_looErrors_lambda_0, st_hyperparams_loo] = ...
            compute_loo_curve(v_lambda_0, v_factors, v_w_in, m_X, v_y, v_spread, n_factors)
        
        if isempty(v_factors)
            assert(length(v_spread)==2, ['if v_factors is empty, then '...
                'a 6th argument v_spread of length 2 is required'])
            assert(length(v_lambda_0)>1, ['if the v_spread is used, then'...
                'length of v_lambda_0 should be larger than 1']);
            my_top = max(v_lambda_0); my_bottom = min(v_lambda_0);
            v_lambdas_sweep = linspace(my_bottom + v_spread(1)*(my_top-my_bottom), ...
                my_bottom + v_spread(2)*(my_top-my_bottom), n_factors);
        else
            v_lambdas_sweep = mean(v_lambda_0)*v_factors;
        end
        v_looLambdas = sort(unique([v_lambda_0 v_lambdas_sweep]));
        
        st_hyperparams_loo = struct;
        for k =1:length(v_looLambdas)
            st_hyperparams_loo(k).lambda = v_looLambdas(k);
        end
        
%         v_looErrors = HyperGradientExperiments.compute_loo_errors(...
%             st_hyperparams_loo, v_w_in, m_X, v_y);
        
        v_looErrors = nan(size(st_hyperparams_loo));
        for k = 1:length(st_hyperparams_loo) % main loop
            v_looErrors(k) = HyperGradientExperiments.exact_loo(m_X, v_y,...
                st_hyperparams_loo(k), v_w_in);
        end
        
        v_looErrors_lambda_0 = zeros(size(v_lambda_0));
        for k = 1:length(v_lambda_0)
            ind_lambda_0 = find(v_looLambdas==v_lambda_0(k), 1,'first');
            v_looErrors_lambda_0(k) = v_looErrors(ind_lambda_0);
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
        for k = 1:length(c_lambdas)
            plot(cumsum(c_itcounts{k}), c_lambdas{k}); 
            hold on
        end
        xlabel('# ISTA iterations')
        ylabel '\lambda'
    end
end

end