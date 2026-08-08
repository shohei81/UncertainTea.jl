// J=32 hierarchical Gaussian (eight-schools shape, NONCENTERED, log-normal
// tau) — the P >= 24 model for the UncertainTea reverse-mode leg (issue #317;
// noncentered since issue #365, post-#326). log_tau is the sampled parameter
// (normal prior) in every framework and theta = mu + exp(log_tau) * theta_z is
// exported as a transformed parameter so the gate columns align by name.
data {
  int<lower=0> J;
  vector[J] y;
  vector<lower=0>[J] sigma;
}
parameters {
  real mu;
  real log_tau;
  vector[J] theta_z;
}
transformed parameters {
  vector[J] theta = mu + exp(log_tau) * theta_z;
}
model {
  mu ~ normal(0, 5);
  log_tau ~ normal(0, 1);
  theta_z ~ std_normal();
  y ~ normal(theta, sigma);
}
