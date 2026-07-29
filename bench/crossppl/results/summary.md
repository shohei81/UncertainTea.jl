# Cross-PPL benchmark summary

Common diagnostics: ArviZ rank-normalized split R-hat and bulk/tail ESS computed identically for every framework. Correctness gate: R-hat < 1.01, mean/quantile agreement with the Stan reference within 4.0 combined MCSEs.

## eight_schools_centered

| framework | chains | draws/chain | precision | correct | min bulk ESS/s | min tail ESS/s | sampling s | warmup s | TTFX/compile s | div rate |
|---|---|---|---|---|---|---|---|---|---|---|
| numpyro-parallel | 4 | 1000 | float64 | FAIL | 185 ± 92 | 93.4 | 0.526 | 0.69 | 1.39 | 0.021 |
| stan | 4 | 1000 | float64 | FAIL | 5,404 ± 2,615 | 4,594 | 0.0297 | 0.0273 | 0.797 | 0.027 |
| uncertaintea-batched-cpu | 4 | 1000 | Float64 | FAIL | 110 ± 41 | 46 | 0.769 | 1.59 | 5.58 | 0.012 |
| uncertaintea-cpu | 4 | 1000 | Float64 | FAIL | 1,311 ± 6.4e+02 | 1,280 | 0.0562 | 0.0735 | 1.2 | 0.014 |

Correctness failures (timings above are reported for context but MUST NOT be quoted):
- numpyro-parallel__chains4: mu: rhat=1.0131
- numpyro-parallel__chains4: mu: rhat=1.0620
- numpyro-parallel__chains4: tau: rhat=1.0235
- numpyro-parallel__chains4: tau: rhat=1.0391
- numpyro-parallel__chains4: tau: rhat=1.0991
- numpyro-parallel__chains4: theta[1]: rhat=1.0186
- numpyro-parallel__chains4: theta[2]: rhat=1.0266
- numpyro-parallel__chains4: theta[3]: rhat=1.0339
- numpyro-parallel__chains4: theta[4]: rhat=1.0115
- numpyro-parallel__chains4: theta[4]: rhat=1.0324
- numpyro-parallel__chains4: theta[5]: q95 z=6.8 (11.895 vs ref 10.350)
- numpyro-parallel__chains4: theta[5]: rhat=1.0440
- numpyro-parallel__chains4: theta[6]: rhat=1.0351
- numpyro-parallel__chains4: theta[7]: rhat=1.0209
- numpyro-parallel__chains4: theta[8]: rhat=1.0248
- stan__chains4: mu: rhat=1.0118
- stan__chains4: tau: rhat=1.0131
- stan__chains4: tau: rhat=1.0159
- stan__chains4: tau: rhat=1.0168
- stan__chains4: theta[1]: rhat=1.0115
- stan__chains4: theta[7]: rhat=1.0109
- uncertaintea-batched-cpu__chains4: mu: rhat=1.0100
- uncertaintea-batched-cpu__chains4: mu: rhat=1.0300
- uncertaintea-batched-cpu__chains4: mu: rhat=1.0672
- uncertaintea-batched-cpu__chains4: tau: rhat=1.0209
- uncertaintea-batched-cpu__chains4: tau: rhat=1.0303
- uncertaintea-batched-cpu__chains4: tau: rhat=1.0897
- uncertaintea-batched-cpu__chains4: theta[1]: rhat=1.0168
- uncertaintea-batched-cpu__chains4: theta[1]: rhat=1.0334
- uncertaintea-batched-cpu__chains4: theta[2]: rhat=1.0132
- uncertaintea-batched-cpu__chains4: theta[2]: rhat=1.0378
- uncertaintea-batched-cpu__chains4: theta[3]: rhat=1.0190
- uncertaintea-batched-cpu__chains4: theta[3]: rhat=1.0347
- uncertaintea-batched-cpu__chains4: theta[4]: rhat=1.0204
- uncertaintea-batched-cpu__chains4: theta[4]: rhat=1.0337
- uncertaintea-batched-cpu__chains4: theta[5]: q95 z=5.8 (12.244 vs ref 10.350)
- uncertaintea-batched-cpu__chains4: theta[5]: rhat=1.0156
- uncertaintea-batched-cpu__chains4: theta[5]: rhat=1.0490
- uncertaintea-batched-cpu__chains4: theta[6]: rhat=1.0128
- uncertaintea-batched-cpu__chains4: theta[6]: rhat=1.0382
- uncertaintea-batched-cpu__chains4: theta[7]: rhat=1.0204
- uncertaintea-batched-cpu__chains4: theta[7]: rhat=1.0350
- uncertaintea-batched-cpu__chains4: theta[8]: rhat=1.0119
- uncertaintea-batched-cpu__chains4: theta[8]: rhat=1.0368
- uncertaintea-cpu__chains4: mu: rhat=1.0107
- uncertaintea-cpu__chains4: mu: rhat=1.0114
- uncertaintea-cpu__chains4: mu: rhat=1.0203
- uncertaintea-cpu__chains4: tau: rhat=1.0363
- uncertaintea-cpu__chains4: tau: rhat=1.0665
- uncertaintea-cpu__chains4: tau: rhat=1.1331
- uncertaintea-cpu__chains4: theta[1]: rhat=1.0182
- uncertaintea-cpu__chains4: theta[1]: rhat=1.0282
- uncertaintea-cpu__chains4: theta[2]: rhat=1.0183
- uncertaintea-cpu__chains4: theta[2]: rhat=1.0189
- uncertaintea-cpu__chains4: theta[3]: rhat=1.0220
- uncertaintea-cpu__chains4: theta[3]: rhat=1.0231
- uncertaintea-cpu__chains4: theta[4]: rhat=1.0187
- uncertaintea-cpu__chains4: theta[5]: rhat=1.0116
- uncertaintea-cpu__chains4: theta[5]: rhat=1.0157
- uncertaintea-cpu__chains4: theta[5]: rhat=1.0254
- uncertaintea-cpu__chains4: theta[6]: rhat=1.0154
- uncertaintea-cpu__chains4: theta[6]: rhat=1.0159
- uncertaintea-cpu__chains4: theta[7]: rhat=1.0240
- uncertaintea-cpu__chains4: theta[7]: rhat=1.0243
- uncertaintea-cpu__chains4: theta[8]: rhat=1.0280
- uncertaintea-cpu__chains4: theta[8]: rhat=1.0350

## eight_schools_noncentered

| framework | chains | draws/chain | precision | correct | min bulk ESS/s | min tail ESS/s | sampling s | warmup s | TTFX/compile s | div rate |
|---|---|---|---|---|---|---|---|---|---|---|
| numpyro-parallel | 4 | 1000 | float64 | PASS | 4,265 ± 4.4e+02 | 3,465 | 0.533 | 0.608 | 1.22 | 0.00017 |
| stan | 4 | 1000 | float64 | PASS | 112,647 ± 16,135 | 92,259 | 0.0193 | 0.0143 | 0.0273 | 0.00025 |
| uncertaintea-batched-cpu | 4 | 1000 | Float64 | PASS | 4,896 ± 3.7e+02 | 3,858 | 0.397 | 0.653 | 4.72 | 0.00058 |
| uncertaintea-cpu | 4 | 1000 | Float64 | PASS | 37,560 ± 5,798 | 28,998 | 0.0497 | 0.0631 | 1.61 | 0.00092 |

## gauss

| framework | chains | draws/chain | precision | correct | min bulk ESS/s | min tail ESS/s | sampling s | warmup s | TTFX/compile s | div rate |
|---|---|---|---|---|---|---|---|---|---|---|
| numpyro-parallel | 4 | 1000 | float64 | PASS | 7,346 ± 4.8e+02 | 5,590 | 0.434 | 0.508 | 0.99 | 0 |
| numpyro-vectorized | 64 | 500 | float32 | PASS | 23,024 ± 1,888 | 19,774 | 1.05 | 0.918 | 2.43 | 0 |
| numpyro-vectorized | 512 | 500 | float32 | PASS | 45,922 ± 2,797 | 39,558 | 4.37 | 2.82 | 7.21 | 0 |
| numpyro-vectorized | 4096 | 500 | float32 | PASS | 10,078 ± 0 | 8,810 | 156 | 70.1 | 176 | 0 |
| stan | 4 | 1000 | float64 | PASS | 278,037 ± 9.3e+02 | 190,540 | 0.0127 | 0.01 | 0.562 | 0 |
| uncertaintea-batched-cpu | 4 | 1000 | Float64 | PASS | 52,795 ± 24,342 | 56,940 | 0.0552 | 0.0329 | 2.52 | 0 |
| uncertaintea-batched-cpu | 64 | 500 | Float64 | PASS | 101,537 ± 2,101 | 105,114 | 0.165 | 0.0819 | 2.73 | 0 |
| uncertaintea-batched-cpu | 512 | 500 | Float64 | PASS | 60,481 ± 22,963 | 64,827 | 2.67 | 1.38 | 4.62 | 0 |
| uncertaintea-batched-cpu | 4096 | 500 | Float64 | PASS | 58,697 ± 0 | 64,436 | 17.1 | 7.9 | 26.4 | 0 |
| uncertaintea-batched-cpu-ka | 64 | 500 | Float64 | PASS | 6,277 ± 3.1e+02 | 6,509 | 2.64 | 1.77 | 7.69 | 0 |
| uncertaintea-batched-cpu-ka | 512 | 500 | Float64 | PASS | 6,532 ± 61 | 6,955 | 19.9 | 17.4 | 42.8 | 0 |
| uncertaintea-batched-metal | 64 | 500 | Float32 | PASS | 1,261 ± 11 | 1,297 | 13 | 7.08 | 34.5 | 0 |
| uncertaintea-batched-metal | 512 | 500 | Float32 | PASS | 6,195 ± 76 | 6,676 | 20.8 | 10.6 | 46.7 | 0 |
| uncertaintea-cpu | 4 | 1000 | Float64 | PASS | 546,365 ± 99,595 | 574,571 | 0.00369 | 0.00618 | 0.812 | 0 |

## logistic

| framework | chains | draws/chain | precision | correct | min bulk ESS/s | min tail ESS/s | sampling s | warmup s | TTFX/compile s | div rate |
|---|---|---|---|---|---|---|---|---|---|---|
| numpyro-parallel | 4 | 1000 | float64 | PASS | 10,386 ± 7e+02 | 5,600 | 0.494 | 0.571 | 1.16 | 0 |
| stan | 4 | 1000 | float64 | PASS | 61,078 ± 1,162 | 33,374 | 0.081 | 0.075 | 0.0281 | 0 |
| uncertaintea-batched-cpu | 4 | 1000 | Float64 | PASS | 11,067 ± 1,113 | 7,981 | 0.364 | 0.439 | 3.67 | 0 |
| uncertaintea-batched-metal | 64 | 500 | Float32 | PASS | 2,411 ± 1.2e+02 | 1,735 | 14.5 | 8.91 | 38.6 | 0 |
| uncertaintea-batched-metal | 512 | 500 | Float32 | PASS | 9,648 ± 1.8e+02 | 6,717 | 30 | 17.8 | 61.1 | 0 |
| uncertaintea-cpu | 4 | 1000 | Float64 | PASS | 8,869 ± 4.3e+02 | 5,859 | 0.47 | 0.491 | 1.95 | 0 |

## logistic_large

| framework | chains | draws/chain | precision | correct | min bulk ESS/s | min tail ESS/s | sampling s | warmup s | TTFX/compile s | div rate |
|---|---|---|---|---|---|---|---|---|---|---|
| numpyro-parallel | 4 | 1000 | float64 | PASS | 6,755 ± 5.6e+02 | 2,061 | 1.22 | 1.3 | 2.56 | 0 |
| stan | 4 | 1000 | float64 | PASS | 5,538 ± 5.3e+02 | 1,703 | 1.52 | 1.43 | 4.07 | 0 |
| uncertaintea-batched-cpu | 4 | 1000 | Float64 | PASS | 1,531 ± 79 | 526 | 5.03 | 7.52 | 14.2 | 0 |
| uncertaintea-batched-metal | 64 | 500 | Float32 | PASS | 2,277 ± 5.7e+02 | 765 | 32.5 | 29.3 | 68.3 | 0 |
| uncertaintea-cpu | 4 | 1000 | Float64 | PASS | 170 ± 12 | 53 | 48.2 | 48.3 | 75.1 | 0 |
