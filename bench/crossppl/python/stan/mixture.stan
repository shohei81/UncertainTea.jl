// issue #224: 2-component Gaussian mixture, ordered means (mu2 = mu1 + exp(log_gap)),
// shared scale, fixed weights 0.4/0.6. Hand-marginalized over the component
// indicator with log_mix -- the reference the marginalize=:enumerate / mixture
// machinery is checked against. Same parameterization as the Julia/NumPyro twins.
data {
  int<lower=0> N;
  vector[N] y;
}
parameters {
  real mu1;
  real log_gap;
  real<lower=0> s;
}
model {
  mu1 ~ normal(0, 3);
  log_gap ~ normal(0, 1);
  s ~ gamma(2, 1);
  real mu2 = mu1 + exp(log_gap);
  for (i in 1:N)
    target += log_mix(0.4, normal_lpdf(y[i] | mu1, s), normal_lpdf(y[i] | mu2, s));
}
