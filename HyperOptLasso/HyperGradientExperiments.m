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

methods % Dynamic learning routine
    
    function [m_W, v_lambda, v_loss] = drill_dynamic(obj, estimator, m_X, v_y, initial_lambda, T)
        [P_dim, N] = size(m_X); assert(length(v_y)==N);
        
        m_W      = zeros(P_dim, T);
        v_loss   = zeros(1, T);
        
        m_Phi = m_X*m_X';
        v_r   = m_X*v_y(:);
        v_wf_t = zeros(P_dim, 1);
        v_c_t  = zeros(P_dim, 1);
        
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
    
end
    
methods % Experiments
            
    % Gradient descent, comparison with grid-search Leave one out
    function F = experiment_1(obj)
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
    
    % Gradient Descent vs Online Gradient Descent
    % GD vs OGD
    function F = experiment_2(obj)
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

    % Online Gradient Descent
    % Comparing effect of different tolerances of ISTA
    % (inexact approximation)
    function F = experiment_3(obj)
        obj.n_train = 200;
        [A_train, y_train] = obj.generate_pseudo_streaming_data();
        
        hgl_online = HyperGradientLasso;
        hgl_online.b_online = 1;
        hgl_online.tol_g_lambda = 1e-4;        
        hgl_online.stepsize_lambda = 3e-4;
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
   
    % Online Gradient Descent
    % With-memory vs memoryless
    function F = experiment_4(obj)
        obj.n_train = 200;
        obj.seed = 9;
        [A_train, y_train] = obj.generate_pseudo_streaming_data();
        
        hgl_memory = HyperGradientLasso;
        hgl_memory.b_online = 1;
        hgl_memory.stepsize_lambda = 3e-4;
        hgl_memory.tol_w = 1e-2;
        hgl_memory.tol_g_lambda = 1e-4;
        hgl_memory.max_iter_outer = 10; %10000
        
        hgl_no_mem = hgl_memory;
        hgl_no_mem.stepsize_lambda = 3e-5;
        hgl_no_mem.tol_w = 1e0;
        hgl_no_mem.b_memory = 0;
        
        [m_W_memory, v_lambda_memory, v_it_count_memory] = ...
            hgl_memory.solve_gradient(A_train, y_train);
        final_lambda_memory = v_lambda_memory(find(v_it_count_memory>0,1, 'last'));
        average_w = mean(m_W_memory, 2);
        
        
        [m_W_nomem, v_lambda_nomem, v_it_count_nomem] = ...
            hgl_no_mem.solve_gradient(A_train, y_train);
        final_lambda_nomem = v_lambda_nomem(find(v_it_count_nomem>0,1, 'last'));
        
        dlh = FranceschiRecursiveLasso; % Dynamic
        dlh.stepsize_w = 3e-4;
        dlh.stepsize_lambda = 1e-5;
        
        [m_W_dynamic, v_lambda_dynamic] = obj.drill_dynamic( ...
            dlh, A_train, y_train, 0.05, 10000);
       
        disp 'Computing loo errors' 
        [v_looLambdas, v_looErrors, loo_error_final_lambda] = ...
            obj.compute_loo_curve(final_lambda_memory, kron( ...
            [1 final_lambda_nomem/final_lambda_memory], linspace(0.9, 1.2, 3) ), ...
            average_w, A_train, y_train);
        
        exp_no = obj.determine_experiment_number();
        %% Figures
        figure(exp_no*100+1); clf
        obj.show_lambda_vs_ista_iterates({v_lambda_memory, v_lambda_nomem,v_lambda_dynamic},...
            {v_it_count_memory, v_it_count_nomem, ones(size(v_lambda_dynamic))})                
        legend({'With-memory', 'Memoryless'})
        
        figure(exp_no*100+2); clf
        plot(v_looLambdas, v_looErrors)
        xlabel '\lambda'
        ylabel 'LOO error'
        hold on
        plot(final_lambda_memory, loo_error_final_lambda, 'xr')

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
        my_alpha = 10/trace(m_Phi);
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
        
        v_looErrors = nan(size(st_hyperparams));
        for k = 1:length(st_hyperparams)
            v_looErrors(k) = HyperGradientExperiments.exact_loo(m_X, v_y, st_hyperparams(k), v_w_0);
        end
    end
    
    function [v_looLambdas, v_looErrors, loo_error_lambda_0] = ...
            compute_loo_curve(lambda_0, v_factors, v_w_in, m_X, v_y)
        
        v_looLambdas = lambda_0*sort([1 v_factors]);
        
        st_hyperparams_loo = struct;
        for k =1:length(v_looLambdas)
            st_hyperparams_loo(k).lambda = v_looLambdas(k);
        end
        
        v_looErrors = HyperGradientExperiments.compute_loo_errors(...
            st_hyperparams_loo, v_w_in, m_X, v_y);
        ind_lambda_0 = find(v_looLambdas==lambda_0, 1,'first');
        loo_error_lambda_0 = v_looErrors(ind_lambda_0);
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