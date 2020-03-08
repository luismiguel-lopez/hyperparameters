%% Offline Estimators

% First version: OHO_Lasso, OHO_GroupLasso, OHO_ElasticNet
% These ones use the simplified gradient which I believed would correctly
% approximate the exact gradient.
%
% Second version: HyperGradientLasso and two backup files
% these ones contain the correct gradient but they are encompassed by 
% HyperSubGradientDescent with s_estimator set to 'lasso'. There are
% several experiments in HyperGradientExperiments that confirm that both
% codes do the same.
%
% Third version: HyperSubGradientDescent currently contains the last 
% version, where the same class will be used for Lasso, Group Lasso, and
% other extensions such as Weighted Lasso, Weighted Group Lasso, and
% possibly Elastic net.
%
% FUTURE VERSION: I am considering using the inheritance to ease
% development of extensions of this code to the diverse estimators.

%% Online Estimators

% First version: DynamicLassoHyper (TISO) and DynamicRecursiveLassoHyper
% (TIRSO), together with their _alt versions which contain several
% modifications that I did when I was trying to make it work -- I did not
% manage to make them work because the gradient approximation is biased
% and, therefore, incorrect.

% Second version: FranceschiRecursiveLasso (TIRSO), and several extensions,
% to Group Lasso, Elastic Net (probably incorrect, I have to check),
% Adaptive Lasso and Weighted Lasso. These ones use the forward gradient
% calculation proposed in [Franceschi2017forward].