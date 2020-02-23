classdef DynamicExperiments
properties
    T         = 100000;  % number of time instants
    P         = 100;     % dimensionality of x vectors
    SNR       = 0.3;     % Signal to Noise ratio (natural units)
    sparsity  = 0.1;     % proportion of entries of true_w that are > 0
    seed      = 1        % Random seed
    betaSweep = logspace(-4, -2, 4);
    biasSweep = linspace(0, 0.012, 10);
    colinearity = 0;%0.2;
    
end

methods % Constructor and pseudo-streaming data creating procedure
    function obj = DynamicExperiments() %The constructor sets the path
        addpath('utilities/')
    end
    
    function v_y = generate_y_data(obj, m_X, v_true_w)
        v_y_noiseless = v_true_w'*m_X;
        signal_power  = mean(v_y_noiseless.^2);
        noise_power   = signal_power/obj.SNR;
        epsilon       = randn(1, obj.T)*sqrt(noise_power); % noise
        v_y           = v_y_noiseless    + epsilon;
    end

    function [m_X, v_y, v_true_w, m_X_test, v_y_test] ...
            = generate_pseudo_streaming_data(obj)
        %create train and test data
        rng(obj.seed);
        v_true_w        = double(rand(obj.P, 1) < obj.sparsity); % sparse coefs
        m_X           = randn(obj.P, obj.T); % time series of x vectors
        m_X_test      = randn(obj.P, obj.T); % time series of x vectors

        v_y           = obj.generate_y_data(m_X, v_true_w);
        v_y_test      = obj.generate_y_data(m_X_test, v_true_w);
    end
    
    function validateGroupStructure(~, v_groups)
        assert(all(v_groups==abs(round(v_groups))), ...
            'All entries in v_group_structure must be natural numbers')
        assert(min(v_groups)==1, 'Group labels must start by 1')
        for gr = 1:max(v_groups)
            assert(sum(v_groups==gr)>0, ...
                'Groups must be labeled with consecutive numbers')
        end
    end
    
    function [m_X, v_y, v_true_w, m_X_test, v_y_test, v_group_structure] ...
            = generate_pseudo_streaming_data_grouped(obj, n_groups)
        %create train and test data
        rng(obj.seed);
        v_group_structure = ceil(linspace(eps, n_groups, obj.P));
        obj.validateGroupStructure(v_group_structure);
        v_active_groups = double(rand(n_groups, 1) < obj.sparsity);
        v_true_w        = v_active_groups(v_group_structure); % group-sparse coefs
        m_X           = randn(obj.P, obj.T); % time series of x vectors
        m_X_test      = randn(obj.P, obj.T); % time series of x vectors
        
        v_y           = obj.generate_y_data(m_X, v_true_w);
        v_y_test      = obj.generate_y_data(m_X_test, v_true_w);
    end
    
    function m_X = generate_colinear_X(obj, v_group_structure)
        obj.validateGroupStructure(v_group_structure);
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
        obj.validateGroupStructure(v_group_structure);
        % the group structure is used to create groups of colinear X        
        v_true_w        = double(rand(obj.P, 1) < obj.sparsity); % sparse coefs
        
        m_X           = generate_colinear_X(obj, v_group_structure);
        m_X_test      = generate_colinear_X(obj, v_group_structure);
        v_y           = obj.generate_y_data(m_X, v_true_w);
        v_y_test      = obj.generate_y_data(m_X_test, v_true_w);
    end
    
end

methods % Routines
    function [m_W, m_hyper, v_loss, m_Wf] = drill_online_learning(obj, ...
            estimator, m_X, v_y, initial_lambda)
        %allocate memory
        m_W      = zeros(obj.P, obj.T);
        v_loss   = zeros(1, obj.T);
        
        m_Phi = zeros(obj.P);
        v_r   = zeros(obj.P, 1);
        L = nan;
        m_Wf = zeros(obj.P, obj.T);
        v_wfBar = zeros(obj.P, 1);
        v_grad = zeros(1, obj.T);
        v_wf_t = zeros(obj.P, 1);
        v_c_t = zeros(obj.P, 1);
        v_w_bar = zeros(obj.P, 1);
        
        if isa(estimator, 'FranceschiRecursiveElasticNet')
            m_lambda_rho = zeros(2, obj.T);
            m_lambda_rho(:,1) = initial_lambda;
            m_c_t = zeros(obj.P, 2);
        elseif isa(estimator, 'FranceschiRecursiveWeightedLasso')
            m_lambda = zeros(obj.P, obj.T);
            m_lambda(:,1) = initial_lambda;
            m_c_t = zeros(obj.P); %square matrix
        elseif isa(estimator, 'FranceschiRecursiveAdaptiveLasso')
            estimator_ols = estimator;
            v_wf_ols_t = zeros(obj.P, 1);
            v_lambda = zeros(1, obj.T);
            v_lambda(1) = initial_lambda;
        else
            v_lambda = zeros(1, obj.T);
            v_lambda(1) = initial_lambda;
        end
        
        %main loop
        nsa = 1; %number of samples ahead, by default 1 % CAN BE REMOVED
        ltc = LoopTimeControl(obj.T-nsa);
        for t = 1:obj.T-nsa
            switch class(estimator)
                case 'DynamicLassoHyper_alt'
%                     [m_W(:, t+1), v_lambda(t+1), v_loss(t+1), v_wf_t] =  ...
%                         estimator.update_modified( m_W(:,t), v_lambda(t), ...
%                         m_X(:,t), v_y(t), m_X(:, t+1), v_y(t+1), v_wf_t);

%                     [m_W(:, t+1), v_lambda(t+1), v_loss(t+1), v_wf_t] =  ...
%                         estimator.update_alt( v_lambda(t), ...
%                        m_X(:, t), v_y(t), v_wf_t, m_X(:, t+nsa), v_y(t+nsa));

%                     [m_W(:, t+1), v_lambda(t+1), v_loss(t+1)] =  ...
%                         estimator.update_twoStepish(  m_W(:,t), v_lambda(t), ...
%                         m_X(:,t), v_y(t), m_X(:, t+1), v_y(t+1));

                    [m_W(:, t+1), v_lambda(t+1), v_loss(t+1), m_Wf(:, t+1)] =  ...
                          estimator.update( v_lambda(t), ...
                         m_X(:, t), v_y(t), m_Wf(:, t));
                     
%                     [m_W(:, t+1), v_lambda(t+1), v_loss(t+1), v_wf_t, v_wfBar] =  ...
%                           estimator.update_wfBar( v_lambda(t), ...
%                          m_X(:, t), v_y(t), v_wf_t, v_wfBar);
                    

                case 'DynamicLassoHyper'
                    [m_W(:, t+1), v_lambda(t+1), v_loss(t+1), L] =  ...
                        estimator.update( m_W(:,t), v_lambda(t), ...
                        m_X(:,t), v_y(t), m_X(:, t+1), v_y(t+1), L, t);
                case 'DynamicRecursiveLassoHyper'
                    [m_W(:, t+1), v_lambda(t+1), v_loss(t+1), m_Phi, v_r] =  ...
                        estimator.update( m_W(:,t), v_lambda(t), ...
                        m_X(:,t), v_y(t), m_X(:, t+1), v_y(t+1), m_Phi, v_r, t);
                case 'DynamicRecursiveLassoHyper_alt'
%                     [m_W(:, t+1), v_lambda(t+1), v_loss(t+1), m_Phi, v_r] =  ...
%                         estimator.update_twoStepish( m_W(:,t), v_lambda(t), ...
%                         m_X(:,t), v_y(t), m_X(:, t+1), v_y(t+1), m_Phi, v_r, t);
                    [m_W(:, t+1), v_lambda(t+1), v_loss(t+1), v_wf_t, m_Phi, v_r] =  ...
                          estimator.update( v_lambda(t), ...
                         m_X(:, t), v_y(t), v_wf_t, m_Phi, v_r);
                case 'FranceschiRecursiveLasso'
                    [m_W(:, t+1), v_lambda(t+1), v_loss(t+1), v_wf_t, m_Phi, v_r, v_c_t] =  ...
                          estimator.update( v_lambda(t), ...
                         m_X(:, t), v_y(t), v_wf_t, m_Phi, v_r, v_c_t);
                case 'FranceschiRecursiveAdaptiveLasso'
                    [m_w_ols_t, ~, ~, v_wf_ols_t] = estimator_ols.update(...
                        0, m_X(:,t), v_y(t), v_wf_ols_t, m_Phi, v_r, 0, 0);
                    v_my_weights = 1./(estimator.epsilon+abs(m_w_ols_t).^estimator.exponent);
                    [m_W(:, t+1), v_lambda(t+1), v_loss(t+1), v_wf_t, m_Phi, v_r, v_c_t] =  ...
                          estimator.update( v_lambda(t), ...
                         m_X(:, t), v_y(t), v_wf_t, m_Phi, v_r, v_c_t, v_my_weights);
                case 'FranceschiRecursiveGroupLasso'
                    [m_W(:, t+1), v_lambda(t+1), v_loss(t+1), v_wf_t, m_Phi, v_r, v_c_t] =  ...
                          estimator.update( v_lambda(t), ...
                         m_X(:, t), v_y(t), v_wf_t, m_Phi, v_r, v_c_t);
                case 'FranceschiRecursiveElasticNet'
                    [m_W(:, t+1), m_lambda_rho(:,t+1), v_loss(t+1), v_wf_t, m_Phi, v_r, m_c_t] =  ...
                          estimator.update( m_lambda_rho(:,t), ...
                         m_X(:, t), v_y(t), v_wf_t, m_Phi, v_r, m_c_t);
                case 'FranceschiRecursiveWeightedLasso'
                    [m_W(:, t+1), m_lambda(:,t+1), v_loss(t+1), v_wf_t, m_Phi, v_r, m_c_t] =  ...
                          estimator.update( m_lambda(:,t), ...
                         m_X(:, t), v_y(t), v_wf_t, m_Phi, v_r, m_c_t);

            end
            ltc.go(t);
        end
        
        %assign outputs
        if isa(estimator, 'FranceschiRecursiveElasticNet')
            m_hyper = m_lambda_rho;
        elseif isa(estimator, 'FranceschiRecursiveWeightedLasso')
            m_hyper = m_lambda;
           
        else
            m_hyper = v_lambda;
        end
    end
    
    function v_avgLoss = grid_search_lambda(obj, estimator, m_X, v_y, v_lambda_grid, T_begin)
        if ~exist('T_begin', 'var'), T_begin = floor(obj.T/2); end
        
        estimator.stepsize_lambda = 0;
        n_lambdas = numel(v_lambda_grid);
        v_avgLoss = zeros(size(v_lambda_grid));
        for li = 1:n_lambdas
            [~, ~, v_loss] = drill_online_learning(obj, estimator, m_X,...
                v_y, v_lambda_grid(li));
            v_avgLoss(li) = mean(v_loss(T_begin:end))./mean(v_y(T_begin:end).^2);
        end
    end
    
    function v_loss = drill_fixed_m_Wf(obj, m_X, v_y, m_Wf, alpha_lambda) %#ok<INUSL>
        T = numel(v_y); %#ok<PROPLC>
        v_loss = zeros(size(v_y));
        ltc = LoopTimeControl(T);%#ok<PROPLC>
        for t = 1:T %#ok<PROPLC>
            v_w_t = DynamicLassoHyper.soft_thresholding(m_Wf(:,t), alpha_lambda);
            prediction_error = v_y(t) - v_w_t'*m_X(:,t);
            v_loss(t) = prediction_error.^2;
            ltc.go(t);
        end        
    end
    
    function v_avgLoss = grid_search_lambda_in_hindsight(obj, m_X, v_y, m_Wf, v_alpha_lambda_grid, T_begin)
        if ~exist('T_begin', 'var'), T_begin = floor(obj.T/2); end
        n_lambdas = numel(v_alpha_lambda_grid);
        v_avgLoss = zeros(size(v_alpha_lambda_grid));
        for li = 1:n_lambdas
            v_loss = drill_fixed_m_Wf(obj, m_X, v_y, m_Wf, v_alpha_lambda_grid(li));
            v_avgLoss(li) = mean(v_loss(T_begin:end))./mean(v_y(T_begin:end).^2);
        end
    end
end

methods % Experiments
    
    %First experiment, stationary data
    function F = experiment_10(obj) 
        
        %generate data
        [m_X, v_y, true_w] = generate_pseudo_streaming_data(obj);
               
        %create estimators
        dlh = DynamicLassoHyper;
        dlh.stepsize_w = 3e-4;
        dlh.stepsize_lambda = 3e-2;
        dlh.approx_type = 'hard';
        
        dlh_fixedLambda = DynamicLassoHyper;
        dlh_fixedLambda.stepsize_w = 3e-4;
        dlh_fixedLambda.stepsize_lambda = 0;
        lambda_fixed = 0.2;
        
        %run the online estimators
        [m_W, v_lambda, v_loss] = obj.drill_online_learning(dlh, m_X, v_y, 0);
        [m_W_ls, ~, v_loss_ls]  = obj.drill_online_learning(dlh_fixedLambda, m_X, v_y, 0);
        [m_W_selector, ~, v_loss_selector] = ...
            obj.drill_online_learning(dlh_fixedLambda, m_X, v_y, lambda_fixed);
        
        v_loss_genie = ((v_y - true_w'*m_X).^2);
        v_loss_zero  = (v_y).^2;
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        ch_lambdaLabel = sprintf('\\lambda = %g', lambda_fixed);
        c_legend = {'OHO', 'ls', ch_lambdaLabel, 'genie'};
        
        figure(exp_no*100+1); clf
        subplot(1,2,1);
            obj.show_lambda_iterates(v_lambda)
        subplot(1,2,2)
            obj.plot_normalizedLosses([v_loss; v_loss_ls; ...
                v_loss_selector; v_loss_genie], v_loss_zero);
            legend(c_legend)
        
        figure(exp_no*100+2); clf
            t_W = cat(3, m_W, m_W_ls, m_W_selector);
            obj.show_w(t_W, true_w, c_legend(1:end-1));
        
        %%
        ch_resultsFile = sprintf('results_DE_%d', exp_no);
        save(ch_resultsFile);
        F = 0;
    end
    
    % Comparison between DynamicLassoHyper and DynamicLassoHyper_alt
    % With a little bias difference, better performance. No idea why
    function F = experiment_11(obj) 
        
        obj.T = 100000;
        %generate data
        [m_X, v_y, true_w] = generate_pseudo_streaming_data(obj);
               
        %create estimators
        dlh = DynamicLassoHyper;
        dlh.stepsize_w = 3e-4;
        dlh.stepsize_lambda = 3e-2;
        dlh.approx_type = 'hard';
        
        dlh_alt = DynamicLassoHyper_alt;
        dlh_alt.stepsize_w = 3e-4;
        dlh_alt.stepsize_lambda = 3e-2;
        %dlh_alt.bias_correction = 0.0005;
        
        %run the online estimators
        [m_W, v_lambda, v_loss] = obj.drill_online_learning(dlh, m_X, v_y, 0);
        [m_W_alt, v_lambda_alt, v_loss_alt]  = obj.drill_online_learning(dlh_alt, m_X, v_y, 0);
        
        v_loss_genie = ((v_y - true_w'*m_X).^2);
        v_loss_zero  = (v_y).^2;
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        %ch_lambdaLabel = sprintf('\\lambda = %g', lambda_fixed);
        c_legend = {'old', 'alt', 'genie'};
        
        figure(exp_no*100+1); clf
        subplot(1,2,1);
            obj.show_lambda_iterates([v_lambda; v_lambda_alt])
        subplot(1,2,2)
            obj.plot_normalizedLosses([v_loss; v_loss_alt; ...
                v_loss_genie], v_loss_zero);
            legend(c_legend)
        
        if obj.T <= 100000
            figure(exp_no*100+2); clf
                t_W = cat(3, m_W, m_W_alt);
                obj.show_w(t_W, true_w, c_legend(1:end-1));
        end
        
        %%
        if obj.T <= 100000
            ch_resultsFile = sprintf('results_DE_%d', exp_no);
            save(ch_resultsFile);
            F = 0;
        end
    end
    
    % Comparison between DynamicRecursiveLassoHyper and
    % DynamicRecursiveLassoHyper_alt: two-stepish
    function F = experiment_12(obj)
        
        obj.T = 100000;
        %generate data
        [m_X, v_y, true_w] = generate_pseudo_streaming_data(obj);
               
        %create estimators
        dlh = DynamicRecursiveLassoHyper;
        dlh.stepsize_w = 3e-4;
        dlh.stepsize_lambda = 3e-3;
        dlh.approx_type = 'hard';
        
        dlh_alt = DynamicRecursiveLassoHyper_alt;
        dlh_alt.stepsize_w = 3e-4;
        dlh_alt.stepsize_lambda = 3e-3;
        dlh_alt.approx_type = 'hard';
        
        %run the online estimators
        [m_W, v_lambda, v_loss] = obj.drill_online_learning(dlh, m_X, v_y, 0);
        [m_W_alt, v_lambda_alt, v_loss_alt]  = obj.drill_online_learning(dlh_alt, m_X, v_y, 0);
        
        v_loss_genie = ((v_y - true_w'*m_X).^2);
        v_loss_zero  = (v_y).^2;
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        %ch_lambdaLabel = sprintf('\\lambda = %g', lambda_fixed);
        c_legend = {'old', 'alt', 'genie'};
        
        figure(exp_no*100+1); clf
        subplot(1,2,1);
            obj.show_lambda_iterates([v_lambda; v_lambda_alt])
        subplot(1,2,2)
            obj.plot_normalizedLosses([v_loss; v_loss_alt; ...
                v_loss_genie], v_loss_zero);
            legend(c_legend)
        
        if obj.T <= 100000
            figure(exp_no*100+2); clf
                t_W = cat(3, m_W, m_W_alt);
                obj.show_w(t_W, true_w, c_legend(1:end-1));
        end
        
        %%
        if obj.T <= 100000
            ch_resultsFile = sprintf('results_DE_%d', exp_no);
            save(ch_resultsFile);
            F = 0;
        end
    end

    % Comparison between DynamicRecursiveLassoHyper and DynamicRecursiveLassoHyper_alt
    % With a little bias difference, better performance. No idea why
    function F = experiment_13(obj) 
        
        obj.T = 100000;
        %generate data
        [m_X, v_y, true_w] = generate_pseudo_streaming_data(obj);
               
        %create estimators
        dlh = DynamicRecursiveLassoHyper;
        dlh.stepsize_w = 3e-4;
        dlh.stepsize_lambda = 3e-2;
        dlh.approx_type = 'hard';
        
        dlh_alt = DynamicRecursiveLassoHyper_alt;
        dlh_alt.stepsize_w = 3e-4;
        dlh_alt.stepsize_lambda = 3e-2;
        dlh_alt.bias_correction = 0.0005;
        
        %run the online estimators
        [m_W, v_lambda, v_loss] = obj.drill_online_learning(dlh, m_X, v_y, 0);
        [m_W_alt, v_lambda_alt, v_loss_alt]  = obj.drill_online_learning(dlh_alt, m_X, v_y, 0);
        
        v_loss_genie = ((v_y - true_w'*m_X).^2);
        v_loss_zero  = (v_y).^2;
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        %ch_lambdaLabel = sprintf('\\lambda = %g', lambda_fixed);
        c_legend = {'old', 'alt', 'genie'};
        
        figure(exp_no*100+1); clf
        subplot(1,2,1);
            obj.show_lambda_iterates([v_lambda; v_lambda_alt])
        subplot(1,2,2)
            obj.plot_normalizedLosses([v_loss; v_loss_alt; ...
                v_loss_genie], v_loss_zero);
            legend(c_legend)
        
        if obj.T <= 100000
            figure(exp_no*100+2); clf
                t_W = cat(3, m_W, m_W_alt);
                obj.show_w(t_W, true_w, c_legend(1:end-1));
        end
        
        %%
        if obj.T <= 100000
            ch_resultsFile = sprintf('results_DE_%d', exp_no);
            save(ch_resultsFile);
            F = 0;
        end
    end
    
    % Comparison between greedy approx. and 1-step-ahead approximation of
    % the gradient of the value function wrt lambda
    function F = experiment_14(obj)
        
        obj.T = 100000;
        %generate data
        [m_X, v_y, true_w] = generate_pseudo_streaming_data(obj);
               
        %create estimators
        dlh = DynamicRecursiveLassoHyper;
        dlh.stepsize_w = 3e-4;
        dlh.stepsize_lambda = 3e-2;
        dlh.approx_type = 'hard';
        
        dlh_alt = DynamicRecursiveLassoHyper_alt;
        dlh_alt.stepsize_w = 3e-4;
        dlh_alt.stepsize_lambda = 3e-2;
        dlh_alt.b_QL = 1;
        
        %run the online estimators
        [m_W, v_lambda, v_loss] = obj.drill_online_learning(dlh, m_X, v_y, 0);
        [m_W_alt, v_lambda_alt, v_loss_alt]  = obj.drill_online_learning(dlh_alt, m_X, v_y, 0);
        
        v_loss_genie = ((v_y - true_w'*m_X).^2);
        v_loss_zero  = (v_y).^2;
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        %ch_lambdaLabel = sprintf('\\lambda = %g', lambda_fixed);
        c_legend = {'old', 'alt', 'genie'};
        
        figure(exp_no*100+1); clf
        subplot(1,2,1);
            obj.show_lambda_iterates([v_lambda; v_lambda_alt])
        subplot(1,2,2)
            obj.plot_normalizedLosses([v_loss; v_loss_alt; ...
                v_loss_genie], v_loss_zero);
            legend(c_legend)
        
        if obj.T <= 100000
            figure(exp_no*100+2); clf
                t_W = cat(3, m_W, m_W_alt);
                obj.show_w(t_W, true_w, c_legend(1:end-1));
        end
        
        %%
        if obj.T <= 100000
            ch_resultsFile = sprintf('results_DE_%d', exp_no);
            save(ch_resultsFile);
            F = 0;
        end
    end
    
    function F = experiment_16(obj) 
        
        obj.T = 100000;
        %generate data
        [m_X, v_y, true_w] = generate_pseudo_streaming_data(obj);
               
        %create estimators
        dlh = DynamicLassoHyper_alt;
        dlh.stepsize_w = 3e-4;
        dlh.stepsize_lambda = 3e-3;
        dlh.approx_type = 'hard';
        
        dlh_fixedLambda = DynamicLassoHyper_alt;
        dlh_fixedLambda.stepsize_w = 3e-4;
        dlh_fixedLambda.stepsize_lambda = 0;
        lambda_fixed = 0.2;
        
        %run the online estimators
        [m_W, v_lambda, v_loss, m_Wf] = obj.drill_online_learning(dlh, m_X, v_y, 0);
        [m_W_ls, ~, v_loss_ls, m_Wf_ls]  = obj.drill_online_learning(dlh_fixedLambda, m_X, v_y, 0);
%         [m_W_selector, ~, v_loss_selector] = ...
%             obj.drill_online_learning(dlh_fixedLambda, m_X, v_y, lambda_fixed);
        
        v_loss_genie = ((v_y - true_w'*m_X).^2);
        v_loss_zero  = (v_y).^2;
        
        %% evaluate how good a fixed lambda is for the sequences m_Wf and m_Wf_ls
        lambda_final = v_lambda(end); %this is what DynamicLassoHyper thinks is the optimal lambda
        v_loss_fixed_m_Wf = obj.drill_fixed_m_Wf(m_X, v_y, m_Wf, dlh.stepsize_w*lambda_final);
        
        %sweep over different values of lambda
        v_lambda_sweep = linspace(0, 200*lambda_final, 5);
        %v_lambda_sweep = linspace(0, 1000*lambda_final, 3);
        v_avg_loss = obj.grid_search_lambda_in_hindsight(m_X, v_y, m_Wf, v_lambda_sweep*dlh.stepsize_w);
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        ch_lambdaLabel = sprintf('\\lambda = %g', lambda_final);
        c_legend = {'OHO', 'ls', ch_lambdaLabel, 'genie'};
        
        figure(exp_no*100+1); clf
        subplot(1,2,1);
            obj.show_lambda_iterates(v_lambda)
        subplot(1,2,2)
            obj.plot_normalizedLosses([v_loss; v_loss_ls; ... v_loss_selector;               
                v_loss_fixed_m_Wf; v_loss_genie], v_loss_zero);
            legend(c_legend)
        
        figure(exp_no*100+2); clf
            t_W = cat(3, m_W, m_W_ls, m_Wf);%, m_W_selector);
            obj.show_w(t_W, true_w, c_legend(1:end-1));
            
        figure(exp_no*100+3); clf
            plot(v_lambda_sweep, v_avg_loss);
            xlabel '\lambda'
            ylabel 'avg loss'
        
        %%
        ch_resultsFile = sprintf('results_DE_%d', exp_no);
        %save(ch_resultsFile);
        F = 0;
    end
       
%%
    % Instead of DynamicLassoHyper, we try DynamicRecursiveLassoHyper
    % stationary data
    function F = experiment_20(obj) 
             
        %generate data
        [m_X, v_y, true_w] = generate_pseudo_streaming_data(obj);
               
        %create estimators
        dlh = DynamicRecursiveLassoHyper;
        dlh.stepsize_w = 3e-4;
        dlh.stepsize_lambda = 3e-2;
        dlh.approx_type = 'hard';
        dlh.forgettingFactor = 0.99;
        
        dlh_fixedLambda = DynamicRecursiveLassoHyper;
        dlh_fixedLambda.stepsize_w = 3e-4;
        dlh_fixedLambda.stepsize_lambda = 0;
        dlh_fixedLambda.forgettingFactor = 0.99;
        lambda_fixed = 0.08;
        
        %run the online estimators
        [m_W, v_lambda, v_loss] = obj.drill_online_learning(dlh, m_X, v_y, 0);
        [m_W_ls, ~, v_loss_ls]  = obj.drill_online_learning(dlh_fixedLambda, m_X, v_y, 0);
        [m_W_selector, ~, v_loss_selector] = ...
            obj.drill_online_learning(dlh_fixedLambda, m_X, v_y, lambda_fixed);
        
        v_loss_genie = ((v_y - true_w'*m_X).^2);
        v_loss_zero  = (v_y).^2;

        exp_no = obj.determine_experiment_number();
        %% Figures
        ch_lambdaLabel = sprintf('\\lambda = %g', lambda_fixed);
        c_legend = {'OHO', 'ls', ch_lambdaLabel, 'genie'};
        
        figure(exp_no*100+1); clf
        subplot(1,2,1);
            obj.show_lambda_iterates(v_lambda)
        subplot(1,2,2)
            obj.plot_normalizedLosses([v_loss; v_loss_ls; ...
                v_loss_selector; v_loss_genie], v_loss_zero);
            legend(c_legend)
        
        figure(exp_no*100+2); clf
            t_W = cat(3, m_W, m_W_ls, m_W_selector);
            obj.show_w(t_W, true_w, c_legend(1:end-1));
        
        %%
        ch_resultsFile = sprintf('results_DE_%d', exp_no);
        save(ch_resultsFile);
        F = 0;    end
    
    % Hard vs soft
    % stationary data
    function F = experiment_30(obj) 
        
        %generate data
        [m_X, v_y, true_w] = generate_pseudo_streaming_data(obj);
               
        %create estimators
        dlh_hard = DynamicRecursiveLassoHyper;
        dlh_hard.stepsize_w = 3e-4;
        dlh_hard.stepsize_lambda = 3e-2;
        dlh_hard.approx_type = 'hard';
        dlh_hard.forgettingFactor = 0.99;
        
        dlh_soft = dlh_hard;
        dlh_soft.approx_type = 'soft';
        
        dlh_fixedLambda = DynamicRecursiveLassoHyper;
        dlh_fixedLambda.stepsize_w = 3e-4;
        dlh_fixedLambda.stepsize_lambda = 0;
        dlh_fixedLambda.forgettingFactor = 0.99;
        %lambda_fixed = 0.2;
        
        %run the online estimators
        [m_W_hard, v_lambda_hard, v_loss_hard] = obj.drill_online_learning(dlh_hard, m_X, v_y, 0);
        [m_W_soft, v_lambda_soft, v_loss_soft] = obj.drill_online_learning(dlh_soft, m_X, v_y, 0);
       
        [m_W_ls, ~, v_loss_ls]  = obj.drill_online_learning(dlh_fixedLambda, m_X, v_y, 0);
%         [m_W_selector, ~, v_loss_selector] = ...
%             obj.drill_online_learning(dlh_fixedLambda, m_X, v_y, lambda_fixed);
        
        m_lambdas = [v_lambda_hard; v_lambda_soft];
        m_losses  = [v_loss_hard; v_loss_soft; v_loss_ls];
        
        v_loss_genie = ((v_y - true_w'*m_X).^2);
        v_loss_zero  = (v_y).^2;
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        %ch_lambdaLabel = sprintf('\\lambda = %g', lambda_fixed);
        c_legend = {'Hard', 'Soft', 'ls', 'Genie'};

        figure(exp_no*100+1); clf
        subplot(1,2,1);
            obj.show_lambda_iterates(m_lambdas);
            legend(c_legend(1:2));
        subplot(1,2,2)
            obj.plot_normalizedLosses([m_losses; v_loss_genie], v_loss_zero);
            legend(c_legend)
        
        figure(exp_no*100+2); clf        
            t_W = cat(3, m_W_hard, m_W_soft, m_W_ls);
            obj.show_w(t_W, true_w, c_legend(1:end-1));
        
        %%
        ch_resultsFile = sprintf('results_DE_%d', exp_no);
        save(ch_resultsFile);
        F = 0;    end

    % Comparison with the held-out-validated lambda
    % stationary data
    function F = experiment_40(obj) 
        
        %generate data
        [m_X_train, v_y_train, true_w, m_X_test, v_y_test] = ...
            generate_pseudo_streaming_data(obj);
               
        %create estimators
        dlh = DynamicLassoHyper;
        dlh.stepsize_w = 1e-3;        
        dlh.stepsize_lambda = 5e-4;
        dlh.approx_type = 'hard';
        
        %do a grid search over lambda
        v_lambda_grid = linspace(0.05, 0.4, 9);
        v_avgLoss = obj.grid_search_lambda(dlh, m_X_train, v_y_train, v_lambda_grid);
        [~, best_lambda_idx] = min(v_avgLoss);
        
        dlh_fixedLambda = DynamicLassoHyper;
        dlh_fixedLambda.stepsize_w = 1e-3;
        dlh_fixedLambda.stepsize_lambda = 0;
        lambda_heldout = v_lambda_grid(best_lambda_idx);
        
        %run the online estimators
        [m_W, v_lambda, v_loss] = obj.drill_online_learning(dlh, m_X_test, v_y_test, 0);
        [m_W_bestLambda, ~, v_loss_bestLambda] = ...
            obj.drill_online_learning(dlh_fixedLambda, m_X_test, v_y_test, lambda_heldout);
        [m_W_ls, ~, v_loss_ls]  = obj.drill_online_learning(dlh_fixedLambda, m_X_test, v_y_test, 0);

        
        m_losses  = [v_loss; v_loss_bestLambda; v_loss_ls];
        
        v_loss_genie = ((v_y_test - true_w'*m_X_test).^2);
        v_loss_zero  = (v_y_test).^2;
                
        exp_no = obj.determine_experiment_number();
        %% Figures
        ch_lambdaLabel = sprintf('\\lambda = %g', lambda_heldout);
        c_legend = {'OHO', ['HeldOut ' ch_lambdaLabel], 'LS', 'Genie'};
        
        figure(exp_no*100+1); clf
        subplot(1,2,1);
            obj.show_lambda_iterates(v_lambda)       
        subplot(1,2,2)
            obj.plot_normalizedLosses([m_losses; v_loss_genie], v_loss_zero);
        legend(c_legend)
                
        figure(exp_no*100+2); clf
            t_W = cat(3, m_W, m_W_bestLambda, m_W_ls);
            obj.show_w(t_W, true_w, c_legend(1:end-1))
        
        figure(exp_no*100+3); clf
            plot(v_lambda_grid, v_avgLoss)
            xlabel \lambda
            ylabel 'train loss'
            title 'Cross-validated loss values'
            
        %% 
        ch_resultsFile = sprintf('results_DE_%d', exp_no);
        save(ch_resultsFile);
        F = 0;
        
        
    end
    
    % same as exp 40, but more samples
    function F = experiment_41(obj)
        obj.T = 500000;

        %generate data
        [m_X_train, v_y_train, true_w, m_X_test, v_y_test] = ...
            generate_pseudo_streaming_data(obj);
               
        %create estimators
        dlh = DynamicLassoHyper;
        dlh.stepsize_w = 1e-3;        
        dlh.stepsize_lambda = 5e-4;
        dlh.approx_type = 'hard';
        
        %do a grid search over lambda
        v_lambda_grid = linspace(0.05, 0.4, 9);
        v_avgLoss = obj.grid_search_lambda(dlh, m_X_train, v_y_train, v_lambda_grid);
        [~, best_lambda_idx] = min(v_avgLoss);
        
        dlh_fixedLambda = DynamicLassoHyper;
        dlh_fixedLambda.stepsize_w = 1e-3;
        dlh_fixedLambda.stepsize_lambda = 0;
        lambda_heldout = v_lambda_grid(best_lambda_idx);
        
        %run the online estimators
        [m_W, v_lambda, v_loss] = obj.drill_online_learning(dlh, m_X_test, v_y_test, 0);
        [m_W_bestLambda, ~, v_loss_bestLambda] = ...
            obj.drill_online_learning(dlh_fixedLambda, m_X_test, v_y_test, lambda_heldout);
        [m_W_ls, ~, v_loss_ls]  = obj.drill_online_learning(dlh_fixedLambda, m_X_test, v_y_test, 0);

        
        m_losses  = [v_loss; v_loss_bestLambda; v_loss_ls];
        
        v_loss_genie = ((v_y_test - true_w'*m_X_test).^2);
        v_loss_zero  = (v_y_test).^2;
                
        exp_no = obj.determine_experiment_number();
        %% Figures
        ch_lambdaLabel = sprintf('\\lambda = %g', lambda_heldout);
        c_legend = {'OHO', ['HeldOut ' ch_lambdaLabel], 'LS', 'Genie'};
        
        figure(exp_no*100+1); clf
        subplot(1,2,1);
            obj.show_lambda_iterates(v_lambda)       
        subplot(1,2,2)
            obj.plot_normalizedLosses([m_losses; v_loss_genie], v_loss_zero);
        legend(c_legend)
                
        figure(exp_no*100+2); clf
            t_W = cat(3, m_W, m_W_bestLambda, m_W_ls);
            obj.show_w(t_W, true_w, c_legend(1:end-1))
        
        figure(exp_no*100+3); clf
            plot(v_lambda_grid, v_avgLoss)
            xlabel \lambda
            ylabel 'train loss'
            title 'Cross-validated loss values'
            
        %% 
        ch_resultsFile = sprintf('results_DE_%d', exp_no);
        save(ch_resultsFile);
        F = 0;
        
        
    end
    
    % Comparison with the offline OHO_Lasso Lambda
    % stationary data
    function F = experiment_45(obj) 
        
        obj.T = 100000; %!
        %generate data
        [m_X_train, v_y_train, true_w, m_X_test, v_y_test] = ...
            generate_pseudo_streaming_data(obj);
               
        %create estimators
        dlh = DynamicLassoHyper;
        dlh.stepsize_w = 1e-3;        
        dlh.stepsize_lambda = 3e-3;
        dlh.approx_type = 'hard';
        
        %do a grid search over lambda
        v_lambda_grid = linspace(0.01, 0.4, 9);
        v_avgLoss = obj.grid_search_lambda(dlh, m_X_train, v_y_train, v_lambda_grid);
        [~, best_lambda_idx] = min(v_avgLoss);
        
%         % search the best lambda with OHO_Lasso.batch
%         hl = OHO_Lasso;
%         hl.stepsize_policy = ConstantStepsize;
%         hl.stepsize_policy.eta_0 = 50;
%         hl.mirror_type = 'grad';        
%         hl.b_online = 1;
%         hl.b_memory = 0;
%         hl.max_iter_outer = 1000;
%         v_lambda_grad = hl.solve_approx_mirror(m_X_train',...
%             v_y_train');
%         lambda_offline  = v_lambda_grad(end); 
        lambda_offline = 0.04;

        dlh_fixedLambda = DynamicLassoHyper;
        dlh_fixedLambda.stepsize_w = 1e-3;
        dlh_fixedLambda.stepsize_lambda = 0;
        lambda_heldout = v_lambda_grid(best_lambda_idx);

        %run the online estimators
        [m_W, v_lambda, v_loss] = ...
            obj.drill_online_learning(dlh, m_X_test, v_y_test, 0);
        [m_W_heldout, ~, v_loss_heldout] = ...
            obj.drill_online_learning(dlh_fixedLambda, m_X_test, v_y_test, lambda_heldout);
        [m_W_offline, ~, v_loss_offline] = ...
            obj.drill_online_learning(dlh_fixedLambda, m_X_test, v_y_test, lambda_offline);
        [m_W_ls, ~, v_loss_ls]  = ...
            obj.drill_online_learning(dlh_fixedLambda, m_X_test, v_y_test, 0);

        
        m_losses  = [v_loss; v_loss_heldout; v_loss_offline; v_loss_ls];
        
        v_loss_genie = ((v_y_test - true_w'*m_X_test).^2);
        v_loss_zero  = (v_y_test).^2;
                
        exp_no = obj.determine_experiment_number();
        %% Figures
        ch_lambdaLabel_heldout = sprintf('\\lambda = %g', lambda_heldout);
        ch_lambdaLabel_offline = sprintf('\\lambda = %g', lambda_offline);
        c_legend = {'OHO',  ['HeldOut ' ch_lambdaLabel_heldout], ...
            ['OfflineH ' ch_lambdaLabel_offline], 'LS', 'Genie'};
        
        figure(exp_no*100+1); clf
        subplot(1,2,1);
            obj.show_lambda_iterates([v_lambda])%; v_lambda_grad])        %!
        subplot(1,2,2)
            obj.plot_normalizedLosses([m_losses; v_loss_genie], v_loss_zero);
        legend(c_legend)
                
        figure(exp_no*100+2); clf
            t_W = cat(3, m_W, m_W_heldout, m_W_offline, m_W_ls);
            obj.show_w(t_W, true_w, c_legend(1:end-1))
        
        figure(exp_no*100+3); clf
            plot(v_lambda_grid, v_avgLoss)
            xlabel \lambda
            ylabel 'train loss'
            title 'Cross-validated loss values'
            
        %% 
        ch_resultsFile = sprintf('results_DE_%d', exp_no);
        save(ch_resultsFile);
        F = 0;
        
        
    end

    %%
    % Trying different stepsizes for lambda, lets see if all of them converge 
    % to the same neighborhood
    % stationary data
    function F = experiment_50(obj)
        
        %generate data
        [m_X, v_y, true_w] = generate_pseudo_streaming_data(obj);
               
        %create estimators
        dlh = DynamicLassoHyper;
        %dlh.stepsize_w = @(L, t) 10/(L*sqrt(t));
        dlh.stepsize_w = 1e-3;
        dlh.approx_type = 'hard';
        
        v_grid_beta = obj.betaSweep;
        n_betas = length(v_grid_beta);
        
        t_W = zeros(obj.P, obj.T, n_betas);
        m_lambdas = zeros(n_betas ,obj.T);
        m_losses = zeros(n_betas + 1, obj.T);
        v_avgLoss  = zeros(n_betas, 1);
        %run the online estimators
        for ib = 1:n_betas
            dlh.stepsize_lambda = v_grid_beta(ib);
            [t_W(:,:,ib), m_lambdas(ib,:), m_losses(ib,:)] = ...
                obj.drill_online_learning(dlh, m_X, v_y, 0);
            v_avgLoss(ib) = mean(m_losses(ib,:));
        end
        dlh.stepsize_lambda = 0; %Least-Squares
        [t_W(:,:,ib+1), ~, m_losses(ib+1, :)] = ...
            obj.drill_online_learning(dlh, m_X, v_y, 0);
        
        v_loss_genie = ((v_y - true_w'*m_X).^2);
        v_loss_zero  = (v_y).^2;
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        %ch_lambdaLabel = sprintf('\\lambda = %g', lambda_fixed);
        
        figure(exp_no*100+1); clf
        subplot(1,2,1);
            obj.show_lambda_iterates(m_lambdas)   
            c_legend = cell(1, n_betas);
            for ib = 1:n_betas
                c_legend{ib} = sprintf('\\beta = %g ', v_grid_beta(ib));
            end 
            legend(c_legend)
        subplot(1,2,2)
            obj.plot_normalizedLosses(...
                [m_losses; v_loss_genie], v_loss_zero);
            legend([c_legend 'LS' 'genie'])
                
        figure(exp_no*100+2); clf
            obj.show_w(t_W, true_w, [c_legend 'LS']);
        
        figure(exp_no*100+3); clf
            plot(v_grid_beta, v_avgLoss)
            xlabel \beta
            ylabel 'train loss'
            
        %%
        ch_resultsFile = sprintf('results_DE_%d', exp_no);
        save(ch_resultsFile);
        F = 0;
        
    end

    % Same as exp 50, but with Dynamic Recursive Lasso
    % Trying different stepsizes for lambda, lets see if all of them converge 
    % to the same neighborhood
    % stationary data
    function F = experiment_51(obj) 
        
        %generate data
        [m_X, v_y, true_w] = generate_pseudo_streaming_data(obj);
               
        %create estimators
        dlh = DynamicRecursiveLassoHyper;
        %dlh.stepsize_w = @(L, t) 0.1/L;
        dlh.stepsize_w  = 1e-3;
        dlh.approx_type = 'hard';
        
        v_grid_beta = obj.betaSweep;
        n_betas = length(v_grid_beta);

        t_W = zeros(obj.P, obj.T, n_betas);
        m_lambdas = zeros(n_betas ,obj.T);
        m_losses = zeros(n_betas + 1, obj.T);
        v_avgLoss  = zeros(n_betas, 1);
        %run the online estimators
        for ib = 1:n_betas
            dlh.stepsize_lambda = v_grid_beta(ib);
            [t_W(:,:,ib), m_lambdas(ib,:), m_losses(ib,:)] = ...
                obj.drill_online_learning(dlh, m_X, v_y, 0);
            v_avgLoss(ib) = mean(m_losses(ib,:));
        end
        dlh.stepsize_lambda = 0; %Least-Squares
        [t_W(:,:,ib+1), ~, m_losses(ib+1, :)] = ...
            obj.drill_online_learning(dlh, m_X, v_y, 0);
        
        v_loss_genie = ((v_y - true_w'*m_X).^2);
        v_loss_zero  = (v_y).^2;
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        %ch_lambdaLabel = sprintf('\\lambda = %g', lambda_fixed);
        
        figure(exp_no*100+1); clf
        subplot(1,2,1);
            obj.show_lambda_iterates(m_lambdas)   
            c_legend = cell(1, n_betas);
            for ib = 1:n_betas
                c_legend{ib} = sprintf('\\beta = %g ', v_grid_beta(ib));
            end 
            legend(c_legend)
        subplot(1,2,2)
            obj.plot_normalizedLosses(...
                [m_losses; v_loss_genie], v_loss_zero);
            legend([c_legend 'LS' 'genie'])
                
        figure(exp_no*100+2); clf
            obj.show_w(t_W, true_w, [c_legend 'LS']);
        
        figure(exp_no*100+3); clf
            plot(v_grid_beta, v_avgLoss)
            xlabel \beta
            ylabel 'train loss'
            
        %%
        ch_resultsFile = sprintf('results_DE_%d', exp_no);
        save(ch_resultsFile);
        F = 0;
        
    end

    % Same as exp 51, but smaller dimensionality and more iterates
    % stationary data
    function F = experiment_52(obj)
        
        obj.P = 30;
        obj.T = 200000;
        %generate data
        [m_X, v_y, true_w] = generate_pseudo_streaming_data(obj);
               
        %create estimators
        dlh = DynamicRecursiveLassoHyper;
        %dlh.stepsize_w = @(L, t) 0.1/L;
        dlh.stepsize_w  = 1e-3;
        dlh.approx_type = 'hard';
        
        v_grid_beta = obj.betaSweep;
        n_betas = length(v_grid_beta);

        t_W = zeros(obj.P, obj.T, n_betas);
        m_lambdas = zeros(n_betas ,obj.T);
        m_losses = zeros(n_betas + 1, obj.T);
        v_avgLoss  = zeros(n_betas, 1);
        %run the online estimators
        for ib = 1:n_betas
            dlh.stepsize_lambda = v_grid_beta(ib);
            [t_W(:,:,ib), m_lambdas(ib,:), m_losses(ib,:)] = ...
                obj.drill_online_learning(dlh, m_X, v_y, 0);
            v_avgLoss(ib) = mean(m_losses(ib,:));
        end
        dlh.stepsize_lambda = 0; %Least-Squares
        [t_W(:,:,ib+1), ~, m_losses(ib+1, :)] = ...
            obj.drill_online_learning(dlh, m_X, v_y, 0);
        
        v_loss_genie = ((v_y - true_w'*m_X).^2);
        v_loss_zero  = (v_y).^2;
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        %ch_lambdaLabel = sprintf('\\lambda = %g', lambda_fixed);
        
        figure(exp_no*100+1); clf
        subplot(1,2,1);
            obj.show_lambda_iterates(m_lambdas)   
            c_legend = cell(1, n_betas);
            for ib = 1:n_betas
                c_legend{ib} = sprintf('\\beta = %g ', v_grid_beta(ib));
            end 
            legend(c_legend)
        subplot(1,2,2)
            obj.plot_normalizedLosses(...
                [m_losses; v_loss_genie], v_loss_zero);
            legend([c_legend 'LS' 'genie'])
                
        figure(exp_no*100+2); clf
            obj.show_w(t_W, true_w, [c_legend 'LS']);
        
        figure(exp_no*100+3); clf
            plot(v_grid_beta, v_avgLoss)
            xlabel \beta
            ylabel 'train loss'
            
        %%
        ch_resultsFile = sprintf('results_DE_%d', exp_no);
        save(ch_resultsFile);
        F = 0;
        
    end


    % Comparison between DynamicRecursiveLassoHyper and FranceschiTirso, 
    % which combines the Dynamic Recursive Lasso with the forward
    % gradient-based hyperparameter optimization discussed in [Franceschi,
    % 2017]. SUCCESS!!
    function F = experiment_60(obj)
        
        obj.T = 190000; 
        %generate data
        [m_X, v_y, true_w] = generate_pseudo_streaming_data(obj);
               
        %create estimators
        dlh = FranceschiRecursiveLasso;
        dlh.stepsize_w = 3e-4;
        dlh.stepsize_lambda = 1e-5;

        dlh_ls = DynamicRecursiveLassoHyper;
        dlh_ls.stepsize_w = 3e-4;
        dlh_ls.stepsize_lambda = 0;
                
        %run the online estimators
        [m_W, v_lambda, v_loss] = obj.drill_online_learning(dlh, m_X, v_y, 0);
        [m_W_ls, v_lambda_ls, v_loss_ls]  = obj.drill_online_learning(dlh_ls, m_X, v_y, 0);
        
        final_lambda = mean(v_lambda(floor(obj.T/2):end));
        
        %do a grid search over lambda
        v_lambda_grid = linspace(0.9, 1.2, 5)*final_lambda;
        v_avgLoss = obj.grid_search_lambda(dlh, m_X, v_y, v_lambda_grid);
        [~, best_lambda_idx] = min(v_avgLoss);
        
        dlh_fixedLambda = DynamicRecursiveLassoHyper;
        dlh_fixedLambda.stepsize_w = 3e-4;
        dlh_fixedLambda.stepsize_lambda = 0;
        lambda_heldout = v_lambda_grid(best_lambda_idx);
        
        %run the online estimator
        [m_W_bestLambda, v_lambda_bestLambda, v_loss_bestLambda] = ...
            obj.drill_online_learning(dlh_fixedLambda, m_X, v_y, lambda_heldout);

        
        v_loss_genie = ((v_y - true_w'*m_X).^2);
        v_loss_zero  = (v_y).^2;
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        ch_lambdaLabel = sprintf('\\lambda = %g', lambda_heldout);
        c_legend = {'Franceschi', 'ls', ch_lambdaLabel, 'genie'};
        
        figure(exp_no*100+1); clf
        subplot(1,2,1);
            obj.show_lambda_iterates([v_lambda; v_lambda_ls; v_lambda_bestLambda])
        subplot(1,2,2)
            obj.plot_normalizedLosses([v_loss; v_loss_ls; v_loss_bestLambda; ...
                v_loss_genie], v_loss_zero);
            legend(c_legend)
            
        figure(exp_no*100+3); clf
            plot(v_lambda_grid, v_avgLoss)
            xlabel \lambda
            ylabel 'train loss'
            title 'Cross-validated loss values'
        
        
        if obj.T <= 200000
            figure(exp_no*100+2); clf
                t_W = cat(3, m_W, m_W_ls, m_W_bestLambda);
                obj.show_w(t_W, true_w, c_legend(1:end-1));
        end
        
        %%
        if obj.T <= 200000
            ch_resultsFile = sprintf('results_DE_%d', exp_no);
            save(ch_resultsFile);
            F = 0;
        end
    end
    
    % Same as 61 but Group Lasso - SUCCESS!!
    function F = experiment_61(obj)
        
        obj.T = 190000;
        obj.SNR = 0.05;
        %generate data
        [m_X, v_y, true_w, ~, ~, v_group_structure] = generate_pseudo_streaming_data_grouped(obj, 10);
               
        %create estimators
        dlh = FranceschiRecursiveGroupLasso;
        dlh.stepsize_w = 1e-3;
        dlh.stepsize_lambda = 3e-5;
        dlh.v_group_structure = v_group_structure;

        dlh_ls = FranceschiRecursiveGroupLasso;
        dlh_ls.stepsize_w = 1e-3;
        dlh_ls.stepsize_lambda = 0;
        dlh_ls.v_group_structure = v_group_structure;
               
        %run the online estimators
        [m_W, v_lambda, v_loss] = obj.drill_online_learning(dlh, m_X, v_y, 0);
        [m_W_ls, v_lambda_ls, v_loss_ls]  = obj.drill_online_learning(dlh_ls, m_X, v_y, 0);
        
        final_lambda = mean(v_lambda(floor(obj.T/2):end));
        
        %do a grid search over lambda
        v_lambda_grid = linspace(0.9, 1.2, 5)*final_lambda; %!
        v_avgLoss = obj.grid_search_lambda(dlh, m_X, v_y, v_lambda_grid);
        [~, best_lambda_idx] = min(v_avgLoss);
        
        dlh_fixedLambda = FranceschiRecursiveGroupLasso;
        dlh_fixedLambda.stepsize_w = 1e-3;
        dlh_fixedLambda.stepsize_lambda = 0;
        dlh_fixedLambda.v_group_structure = v_group_structure;
        lambda_heldout = v_lambda_grid(best_lambda_idx);
        
        %run the online estimator
        [m_W_bestLambda, v_lambda_bestLambda, v_loss_bestLambda] = ...
            obj.drill_online_learning(dlh_fixedLambda, m_X, v_y, lambda_heldout);

        
        v_loss_genie = ((v_y - true_w'*m_X).^2);
        v_loss_zero  = (v_y).^2;
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        ch_lambdaLabel = sprintf('\\lambda = %g', lambda_heldout);
        c_legend = {'Franceschi', 'ls', ch_lambdaLabel, 'genie'};
        
        figure(exp_no*100+1); clf
        subplot(1,2,1);
            obj.show_lambda_iterates([v_lambda; v_lambda_ls; v_lambda_bestLambda])
        subplot(1,2,2)
            obj.plot_normalizedLosses([v_loss; v_loss_ls; v_loss_bestLambda; ...
                v_loss_genie], v_loss_zero);
            legend(c_legend)
            
        figure(exp_no*100+3); clf
            plot(v_lambda_grid, v_avgLoss)
            xlabel \lambda
            ylabel 'train loss'
            title 'Cross-validated loss values'
        
        
        if obj.T <= 200000
            figure(exp_no*100+2); clf
                t_W = cat(3, m_W, m_W_ls, m_W_bestLambda);
                obj.show_w(t_W, true_w, c_legend(1:end-1));
        end
        
        %%
        if obj.T <= 200000
            ch_resultsFile = sprintf('results_DE_%d', exp_no);
            save(ch_resultsFile);
            F = 0;
        end
    end

    % Same as 60 and 61, but Elastic Net
    function F = experiment_62(obj)
        
        obj.T = 150000;
        obj.SNR = 2; %! 0.05
        obj.colinearity = 0.2;
        n_groups = sqrt(obj.P);
        %generate data
        [m_X, v_y, true_w] = ...
            generate_pseudo_streaming_data_colinear(obj, n_groups); % make colinear inputs
               
        alpha = 1/trace(m_X*m_X'/obj.T);
        beta = diag([1e-5 1e-6]);
        gamma = 0.99;
        
        %create estimators
        dlh = FranceschiRecursiveElasticNet;
        dlh.stepsize_w = alpha;
        dlh.stepsize_lambda = beta;
        dlh.forgettingFactor = gamma;
        
        dlh_lasso = FranceschiRecursiveLasso;
        dlh_lasso.stepsize_w = alpha;
        dlh_lasso.stepsize_lambda = beta(1);
        dlh_lasso.forgettingFactor = gamma;

        dlh_ls = FranceschiRecursiveElasticNet;
        dlh_ls.stepsize_w = alpha;
        dlh_ls.stepsize_lambda = 0;
        dlh_ls.forgettingFactor = gamma;
               
        %run the online estimators
        [m_W, m_lambda, v_loss] = obj.drill_online_learning(dlh, m_X, v_y, 0);
        [m_W_ls, m_lambda_ls, v_loss_ls]  = obj.drill_online_learning(dlh_ls, m_X, v_y, 0);
        [m_W_lasso, m_lambda_lasso, v_loss_lasso]  = obj.drill_online_learning(dlh_lasso, m_X, v_y, 0);
                
        final_lambda_rho = mean(m_lambda(:,floor(obj.T/2):end), 2);
        
%         %do a 2D grid search over lambda and rho
%         v_lambda_grid = linspace(0.9, 1.2, 5)*final_lambda; %!
%         v_avgLoss = obj.grid_search_lambda(dlh, m_X, v_y, v_lambda_grid);
%         [~, best_lambda_idx] = min(v_avgLoss);
%         lambda_heldout = v_lambda_grid(best_lambda_idx);

        dlh_fixedLambdaRho = FranceschiRecursiveElasticNet;
        dlh_fixedLambdaRho.stepsize_w = alpha;
        dlh_fixedLambdaRho.stepsize_lambda = 0;
        dlh_fixedLambdaRho.forgettingFactor = gamma;
        lambda_rho_heldout = final_lambda_rho;

        %run the online estimator
        [m_W_bestLambdaRho, v_lambda_bestLambdaRho, v_loss_bestLambdaRho] = ...
            obj.drill_online_learning(dlh_fixedLambdaRho, m_X, v_y, lambda_rho_heldout);

        
        v_loss_genie = ((v_y - true_w'*m_X).^2);
        v_loss_zero  = (v_y).^2;
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        ch_hyperLabel = sprintf('\\lambda = %g; \\rho  = %g', ...
            lambda_rho_heldout(1), lambda_rho_heldout(2));
        c_legend = {'Franceschi', 'ls', 'Lasso', ch_hyperLabel, 'genie'};
        
        figure(exp_no*100+1); clf
        subplot(1,2,1);
            obj.show_lambda_iterates([m_lambda; m_lambda_ls; m_lambda_lasso; v_lambda_bestLambdaRho])
        subplot(1,2,2)
            obj.plot_normalizedLosses([v_loss; v_loss_ls; v_loss_lasso; v_loss_bestLambdaRho; ...
                v_loss_genie], v_loss_zero);
            legend(c_legend)
            
%         figure(exp_no*100+3); clf
%             plot(v_lambda_grid, v_avgLoss)
%             xlabel \lambda
%             ylabel 'train loss'
%             title 'Cross-validated loss values'
        
        
        if obj.T <= 200000
            figure(exp_no*100+2); clf
                t_W = cat(3, m_W, m_W_ls, m_W_lasso, m_W_bestLambdaRho);
                obj.show_w(t_W, true_w, c_legend(1:end-1));
        end
        
        %%
        if obj.T <= 200000
            ch_resultsFile = sprintf('results_DE_%d', exp_no);
            save(ch_resultsFile);
            F = 0;
        end
    end

        % Same as 60, 61, and 52, but Weighted Lasso and Adaptive Lasso
    function F = experiment_63(obj)
        
        obj.T = 150000; %! 150000
        obj.SNR = 0.3;
        %obj.colinearity = 0.2;
        %n_groups = sqrt(obj.P);
        %generate data
%         [m_X, v_y, true_w] = ...
%             generate_pseudo_streaming_data_colinear(obj, n_groups); % make colinear inputs
        [m_X, v_y, true_w] = generate_pseudo_streaming_data(obj); 
        alpha = 1/trace(m_X*m_X'/obj.T);
        beta_lasso = 3e-5;
        beta_weights = 3e-4;
        beta_adaptive = 3e-6;
        gamma = 0.99;
        
        %create estimators
        dlh = FranceschiRecursiveWeightedLasso;
        dlh.stepsize_w = alpha;
        dlh.stepsize_lambda = beta_weights;
        dlh.forgettingFactor = gamma;
        
        dlh_lasso = FranceschiRecursiveLasso;
        dlh_lasso.stepsize_w = alpha;
        dlh_lasso.stepsize_lambda = beta_lasso;
        dlh_lasso.forgettingFactor = gamma;

        dlh_ls = FranceschiRecursiveLasso;
        dlh_ls.stepsize_w = alpha;
        dlh_ls.stepsize_lambda = 0;
        dlh_ls.forgettingFactor = gamma;
        
        dlh_adaptive = FranceschiRecursiveAdaptiveLasso;
        dlh_adaptive.stepsize_w = alpha;
        dlh_adaptive.stepsize_lambda = beta_adaptive;
        dlh_adaptive.forgettingFactor = gamma;
        
        v_initial_lambda = zeros(obj.P, 1);
               
        %run the online estimators
        [m_W, m_lambda, v_loss] = ...
            obj.drill_online_learning(dlh, m_X, v_y, v_initial_lambda);
        [m_W_ls, ~, v_loss_ls]  = ...
            obj.drill_online_learning(dlh_ls, m_X, v_y, 0);
        [m_W_lasso, m_lambda_lasso, v_loss_lasso]  = ...
            obj.drill_online_learning(dlh_lasso, m_X, v_y, 0);
        [m_W_adaptive, v_lambda_adaptive, v_loss_adaptive] = ...
            obj.drill_online_learning(dlh_adaptive, m_X, v_y, 0);
                
        final_v_lambda = mean(m_lambda(:,floor(obj.T/2):end), 2);
        
%         %do a 2D grid search over lambda and rho
%         v_lambda_grid = linspace(0.9, 1.2, 5)*final_lambda; %!
%         v_avgLoss = obj.grid_search_lambda(dlh, m_X, v_y, v_lambda_grid);
%         [~, best_lambda_idx] = min(v_avgLoss);
%         lambda_heldout = v_lambda_grid(best_lambda_idx);

        dlh_fixed_v_lambda = FranceschiRecursiveWeightedLasso;
        dlh_fixed_v_lambda.stepsize_w = alpha;
        dlh_fixed_v_lambda.stepsize_lambda = 0;
        dlh_fixed_v_lambda.forgettingFactor = gamma;
        v_lambda_heldout = final_v_lambda;

        %run the online estimator
        [m_W_fixed_v_lambda, ~, v_loss_fixed_v_lambda] = ...
            obj.drill_online_learning(dlh_fixed_v_lambda, m_X, v_y, v_lambda_heldout);

        
        v_loss_genie = ((v_y - true_w'*m_X).^2);
        v_loss_zero  = (v_y).^2;
        
        exp_no = obj.determine_experiment_number();
        %% Figures
%         ch_hyperLabel = sprintf('\\lambda = %g; \\rho  = %g', ...
%             lambda_rho_heldout(1), lambda_rho_heldout(2));
        c_legend = {'Weighted Lasso', 'ls', 'Lasso', 'fixed \lambda', ...
            'Adaptive Lasso', 'genie'};
        
        figure(exp_no*100+1); clf
        subplot(1,3,1);
            obj.show_lambda_iterates([m_lambda_lasso; v_lambda_adaptive])
            legend('Lasso', 'Adaptive Lasso');
        subplot(1, 3,2);
            imagesc(m_lambda);
            title('Weighted Lasso')
        subplot(1,3,3)
            obj.plot_normalizedLosses([v_loss; v_loss_ls; v_loss_lasso; ...
                v_loss_fixed_v_lambda; v_loss_adaptive; v_loss_genie], v_loss_zero);
            legend(c_legend)
            
%         figure(exp_no*100+3); clf
%             plot(v_lambda_grid, v_avgLoss)
%             xlabel \lambda
%             ylabel 'train loss'
%             title 'Cross-validated loss values'
        
        
        if obj.T <= 200000
            figure(exp_no*100+2); clf
                t_W = cat(3, m_W, m_W_ls, m_W_lasso, m_W_fixed_v_lambda, m_W_adaptive);
                obj.show_w(t_W, true_w, c_legend(1:end-1));
        end
        
        %%
        if obj.T <= 200000
            ch_resultsFile = sprintf('results_DE_%d', exp_no);
            save(ch_resultsFile);
            F = 0;
        end
    end

end

methods (Static) %Plotting, etc
    
    function plot_normalizedLosses(m_loss, v_reference)
        T = size(m_loss, 2);
        assert(length(v_reference)==T, 'length of v_reference must match')
        bottom = 1;
        for k = 1:size(m_loss, 1)
            normalizedLosses = cumsum(m_loss(k, :))./cumsum(v_reference);
            semilogy(normalizedLosses);
            hold on
            bottom = min(bottom, min(normalizedLosses(10:end)));
        end  
        ylim([0.99*bottom, 1.01])
        title 'Average normalized loss (NMSE)'
    end
    
    function show_lambda_iterates(m_lambdas)
        plot(m_lambdas')
        title '\lambda iterates'
        xlabel('time $t$', 'interpreter', 'latex')
        ylabel '\lambda'
    end
    
    % Show in an imagesc format several matrices containing sequences of 
    % estimates of the weight vector. True weights shown at the left
    function show_w(t_W, true_w, c_legend)
        T = size(t_W, 2);
        P = length(true_w);
        thick_true_w = repmat(true_w, [1 T/20]);
        imagesc(abs([thick_true_w reshape( t_W, ...
            [P, T*size(t_W,3)] )]));
        caxis([0 1])

        if exist('c_legend', 'var')
            c_legend_with_commas = repmat({', '}, ...
                [1 length(c_legend)*2-1]);
            c_legend_with_commas(1:2:end) = c_legend;
            title(['w true, ', cat(2, c_legend_with_commas{:}) ])
        end
    end
    
    % Call from within an experiment to get the experiment number, used
    % when saving data and generating figures with nonoverlapping numbers
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
        