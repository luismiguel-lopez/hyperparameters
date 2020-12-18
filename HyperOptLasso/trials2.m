lambda = m_lambda(:,k);
v_x = m_X(:,1);
%y = v_y(1);
N_f = nnz(not(mb_valFold(f,:)));
m_F_j = (N*m_F - m_X(:,mb_valFold(f,:))*....
    m_X(:, mb_valFold(f,:))')/N_f;
v_r_j = (N*v_r - m_X(:,mb_valFold(f,:))*...
    vec(v_y(mb_valFold(f,:))))/N_f;

v_w_star = obj.ista(v_w, m_F, v_r, lambda, stepsize_w);
v_w_j = obj.ista(v_w_star, m_F_j, v_r_j, lambda, stepsize_w);
vb_support = v_w_star~=0;

m_T = m_X(vb_support,mb_valFold(f,:));
m_Psi = N*m_F(vb_support, vb_support);
m_Psi_inv_T = m_Psi\m_T;
v_v = -sign(v_w_star(vb_support));

v_alpha_star = v_w_star(vb_support);
v_alpha_j2 = (m_Psi-m_T*m_T')\(m_Psi*v_alpha_star-m_T*vec(v_y(vb_val))...
    -(lambda.*v_v));
v_alpha_j = (m_Psi-m_T*m_T')\(m_Psi*v_alpha_star-m_T*vec(v_y(vb_val))...
    -(N-N_f)*(lambda.*v_v));


figure(99); clf
stem(v_alpha_j);
hold on
stem(v_w_j(vb_support))
stem(v_w_star(vb_support))
legend('alpha j', 'w j', 'w star')


v_myValErrors = (m_T'*v_w_j(vb_support)-vec(v_y(vb_val)));
v_ave = (m_T'*v_w_star(vb_support) - vec(v_y(vb_val)));

v_alpha_star = v_w_star(vb_support);
v_eve = (m_T'*v_alpha_j - vec(v_y(vb_val)));
v_eve_efficient = (eye(s) - m_T'*m_Psi_inv_T)\(m_T'*v_alpha_star ...
    - vec(v_y(vb_val)) - (N-N_f)*m_Psi_inv_T'*(lambda.*v_v));

v_hv_efficient2 = 1/s*m_V'*m_Psi_inv_T * ...
                ( (eye(s)-m_T'*m_Psi_inv_T)\v_eve_efficient);

figure(201); clf
stem([v_myValErrors-v_ave])
hold on
stem(v_myValErrors-v_eve)
stem(v_myValErrors-v_eve_efficient)
legend('approx', 'exact', 'efficient')

figure(209); clf
stem(v_hv_efficient2)
hold on
%stem(v_hypergradient_values)
stem(v_hv_efficient)