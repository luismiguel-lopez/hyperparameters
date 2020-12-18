v_lambda = m_lambda(:,k);
v_w_star = obj.ista(v_w, m_F, v_r, v_lambda, stepsize_w);
%v_support = w_star ~= 0;
%v_alpha_star = v_w_star(v_support);
m_X_valFold = m_X(:, mb_valFold(f,:));
v_y_valFold = vec(v_y(mb_valFold(f,:)));
n_valFold   = nnz(mb_valFold(f,:));
n_trainFold = N-n_valFold;
m_Phi_fold = (N*m_F-m_X_valFold*m_X_valFold')/n_trainFold;
v_r_fold   = (N*v_r-m_X_valFold*v_y_valFold)/n_trainFold;
v_w_j = obj.ista(v_w_star, m_Phi_fold, v_r_fold, v_lambda, stepsize_w);

v_support_alt = v_w_j ~= 0;
m_S = N*m_F(v_support_alt, v_support_alt);
m_T_j = m_X_valFold(v_support_alt,:);
m_Psi_j = (1/n_trainFold) * (m_S - m_T_j*m_T_j');
v_alpha_j = v_w_j(v_support_alt);
m_V_j = -diag(sign(v_alpha_j));
%m_D_j = n_trainFold* ((m_S - m_T_j*m_T_j') \ m_V_j);
v_valErrors_alt = m_T_j'*v_alpha_j-v_y_valFold;
v_hypergradient_values_alt2 = m_V_j'*(m_Psi_j\m_T_j)*v_valErrors_alt;

%figure(209); clf
%scatter(v_hypergradient_values_alt, v_hypergradient_values);
