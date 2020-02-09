nu = 20;
p = 0.49;

v_x = logspace(-10, -1, 50);
v_Q = qfunc(v_x);
v_k = 1./(1-nu.*(v_Q-p).*(v_Q-1+p));

figure(109);clf
plot(v_x, v_k)
xlabel 'u/\sigma'
ylabel 'increase factor'