% generate figure 1 for Eusipco paper

load results_HG_102.mat
F = figure(1);clf

c_lambda{8}(1718:end) = [];
c_inv_count{8}(1718:end) = [];
c_lambda{7}(2992:end) = [];
c_inv_count{7}(2992:end) = [];

[ax1, ax2]= obj.doublePlot(c_lambda, c_inv_count, c_legend, v_looErrors, ...
    v_looLambdas, m_final_lambda, v_loo_error_final_lambdas, 4, c_markers);

axes(ax1);
xlim([0, 6000]);
ylim([0.4, 0.9]);
axes(ax2);
xlim([0.87, 0.91])

ax1.Position =  [0.1300    0.1100    0.4879    0.8150];
ax2.Position =  [0.6804 0.1100 0.2246 0.8150];


