// lkjcholesky correlation model (issue #224): d=2 multivariate normal with an LKJ
// Cholesky correlation prior (eta=2) and per-dimension log-normal scales. The
// generated `Omega` reproduces UncertainTea's packed lower-triangular correlation
// factor (column-major [L11, L21, L22]) so the cross-PPL gate aligns by name;
// Omega[1] == 1 is a structural constant the gate skips (zero variance).
data {
  int<lower=0> N;
  array[N] vector[2] y;
}
parameters {
  cholesky_factor_corr[2] L;
  vector<lower=0>[2] tau;
}
model {
  L ~ lkj_corr_cholesky(2.0);
  tau ~ lognormal(0.0, 0.5);
  matrix[2, 2] Ltril = diag_pre_multiply(tau, L);
  for (i in 1:N)
    y[i] ~ multi_normal_cholesky(rep_vector(0.0, 2), Ltril);
}
generated quantities {
  vector[3] Omega = [L[1, 1], L[2, 1], L[2, 2]]';
}
