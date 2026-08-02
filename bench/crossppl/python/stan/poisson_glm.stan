// Poisson GLM, one covariate, log link (issue #317). The vectorized
// formulation every framework states naturally: UncertainTea's broadcast
// `{:y} ~ poisson.(exp.(a .+ b .* x))`, NumPyro's plate over Poisson(exp(eta)).
data {
  int<lower=0> N;
  vector[N] x;
  array[N] int<lower=0> y;
}
parameters {
  real a;
  real b;
}
model {
  a ~ normal(0, 1);
  b ~ normal(0, 1);
  y ~ poisson_log(a + b * x);
}
