figure(1312);
clf
v_w_final = m_W_rew(:,31);
subplot(2, 2, 1);
v_support = v_w_final ~= 0;
%stem(find(v_support), m_W(v_support,end));
stem(find(v_support), v_w_final(v_support));
hold on
stem(find(v_true_w), v_true_w(v_true_w~=0));
legend ('estimated w', 'true w')

subplot(2, 2, 2);
plot(m_lambda_rew')

v_NMSD_rew = mean((m_W_rew - v_true_w).^2);

subplot(2, 2, 3);
plot(v_NMSD_rew)
xlabel 'outer it'
title 'NMSD'
            