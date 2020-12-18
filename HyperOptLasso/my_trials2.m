%%
lambda = 1;
v_x = m_X(:,1);
%y = v_y(1);
m_F_j = (N*m_F-v_x*v_x')/(N-1);
v_r_j = (N*v_r-v_x*v_y(1))/(N-1);

v_w_star = obj.ista(v_w, m_F, v_r, lambda, stepsize_w);
vb_support = v_w_star~=0;
v_w_j = obj.ista(v_w_star, m_F_j, v_r_j, lambda, stepsize_w);

figure(101); clf;
stem(v_w_star);
hold on
stem(v_w_j);

m_Psi = N*m_F(vb_support, vb_support);
v_t   = v_x(vb_support);

e = v_x'*v_w_j - v_y(1)
e2 = (1-v_t'*(m_Psi\v_t))\(v_x'*v_w_star - v_y(1))


%v_t'*m_psi\v_t
%(1-v_t'*(m_psi\v_t))\(v_x'*v_w_j - v_y(1))
v_s_plus_old = (m_Psi-v_t*v_t')\v_t*(v_x'*v_w_star-v_y(1));
v_v = -sign(v_w_star(vb_support));
v_s_plus  = (m_Psi-v_t*v_t')\(v_t*(v_x'*v_w_star-v_y(1))+lambda*v_v);
v_alpha_star = v_w_star(vb_support);
v_alpha_j = (m_Psi-v_t*v_t')\(m_Psi*v_alpha_star-v_t*v_y(1)-lambda*v_v);

figure(99); clf
stem(v_alpha_j);
hold on
stem(v_w_j(vb_support))
stem(v_w_star(vb_support))
legend('alpha j', 'w j', 'w star')