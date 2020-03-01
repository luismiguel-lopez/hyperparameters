alpha = obj.stepsize_w;

m_A_natural = m_A_tilde - alpha*m_A_tilde*m_F_j;
m_A = eye(P)-m_A_natural;

v_b = m_B_tilde;
v_z_true = v_c_paper;

% sloppy init
x_sloppy = (eye(P) - m_A_tilde + alpha*m_A_tilde.*m_F_j)\m_B_tilde;

niter = 1000;
m_x = nan(P, niter, 2, 2);
m_err = nan(P, 2, 2);
m_diff = nan(P, 2, 2);
m_x(:,1, 1, 1) = x_sloppy;
m_x(:,1, 1, 2) = x_sloppy;
m_x(:,1, 2, 1) = v_c_previous;
m_x(:,1, 2, 2) = v_c_previous;
eta = 1;
m_eta = inv((eye(P) - m_A_tilde + alpha*m_A_tilde.*m_F_j));

for k = 1:2
    for grad = 0:1
        for i = 2:niter
            if grad > 0
                m_x(:, i, k, 1) = m_x(:,i-1, k, 1) - eta*m_eta*m_A'*(m_A*m_x(:,i-1,k, 1)-v_b);
            else
                m_x(:, i, k, 2) = m_A_natural*m_x(:,i-1, k, 2) + v_b;
            end
            m_err(i,k,2-grad) = norm(v_z_true - m_x(:,i,k, 2-grad));
            m_diff(i, k, 2-grad) = norm(m_x(:,i, k, 2-grad)- m_x(:,i-1,k, 2-grad));
        end
    end
end

figure(1009); 
subplot(1, 2, 1);
semilogy(reshape(m_err, [ niter 4]));
legend ('grad, sloppy', 'grad, prev', 'nat, sloppy', 'nat, prev');
subplot(1, 2, 2);
semilogy(reshape(m_diff, [ niter 4]));
legend ('grad, sloppy', 'grad, prev', 'nat, sloppy', 'nat, prev');
hold on
