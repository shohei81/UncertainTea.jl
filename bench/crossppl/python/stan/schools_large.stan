// J=32 hierarchical Gaussian (eight-schools shape, CENTERED, log-normal tau) —
// the P >= 24 model for the UncertainTea reverse-mode leg (issue #317).
// log_tau is the sampled parameter (normal prior) in every framework so the
// exported columns align by name.
data {
  int<lower=0> J;
  vector[J] y;
  vector<lower=0>[J] sigma;
}
parameters {
  real mu;
  real log_tau;
  vector[J] theta;
}
model {
  mu ~ normal(0, 5);
  log_tau ~ normal(0, 1);
  theta ~ normal(mu, exp(log_tau));
  y ~ normal(theta, sigma);
}
