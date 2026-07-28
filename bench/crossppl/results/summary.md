# Cross-PPL benchmark summary

Common diagnostics: ArviZ rank-normalized split R-hat and bulk/tail ESS computed identically for every framework. Correctness gate: R-hat < 1.01, mean/quantile agreement with the Stan reference within 4.0 combined MCSEs.

## eight_schools_centered

| framework | chains | draws/chain | precision | correct | min bulk ESS/s | min tail ESS/s | sampling s | warmup s | TTFX/compile s | div rate |
|---|---|---|---|---|---|---|---|---|---|---|
| numpyro-parallel | 4 | 1000 | float64 | FAIL | 69.7 ± 37 | 33.5 | 1.4 | 1.58 | 3.24 | 0.021 |
| stan | 4 | 1000 | float64 | FAIL | 2,019 ± 8.7e+02 | 1,689 | 0.078 | 0.0647 | 8.08 | 0.027 |
| uncertaintea-batched-cpu | 4 | 1000 | Float64 | FAIL | 108 ± 71 | 78.7 | 1.89 | 3.88 | 17.7 | 0.014 |
| uncertaintea-cpu | 4 | 1000 | Float64 | FAIL | 408 ± 24 | 276 | 0.426 | 0.737 | 4.98 | 0.0095 |

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
- uncertaintea-batched-cpu__chains4: mu: rhat=1.0281
- uncertaintea-batched-cpu__chains4: tau: rhat=1.0174
- uncertaintea-batched-cpu__chains4: tau: rhat=1.0348
- uncertaintea-batched-cpu__chains4: tau: rhat=1.0563
- uncertaintea-batched-cpu__chains4: theta[1]: rhat=1.0344
- uncertaintea-batched-cpu__chains4: theta[2]: rhat=1.0108
- uncertaintea-batched-cpu__chains4: theta[2]: rhat=1.0421
- uncertaintea-batched-cpu__chains4: theta[3]: rhat=1.0328
- uncertaintea-batched-cpu__chains4: theta[4]: rhat=1.0528
- uncertaintea-batched-cpu__chains4: theta[5]: rhat=1.0126
- uncertaintea-batched-cpu__chains4: theta[6]: rhat=1.0290
- uncertaintea-batched-cpu__chains4: theta[7]: rhat=1.0201
- uncertaintea-batched-cpu__chains4: theta[7]: rhat=1.0348
- uncertaintea-batched-cpu__chains4: theta[8]: rhat=1.0303
- uncertaintea-cpu__chains4: mu: rhat=1.0110
- uncertaintea-cpu__chains4: tau: rhat=1.0127
- uncertaintea-cpu__chains4: tau: rhat=1.0253
- uncertaintea-cpu__chains4: tau: rhat=1.0326
- uncertaintea-cpu__chains4: theta[1]: rhat=1.0108

## eight_schools_noncentered

| framework | chains | draws/chain | precision | correct | min bulk ESS/s | min tail ESS/s | sampling s | warmup s | TTFX/compile s | div rate |
|---|---|---|---|---|---|---|---|---|---|---|
| numpyro-parallel | 4 | 1000 | float64 | PASS | 1,978 ± 1.2e+02 | 1,609 | 1.14 | 1.54 | 2.65 | 0.00017 |
| stan | 4 | 1000 | float64 | PASS | 52,567 ± 3,823 | 42,969 | 0.041 | 0.022 | 7.16 | 0.00025 |
| uncertaintea-batched-cpu | 4 | 1000 | Float64 | PASS | 1,640 ± 3.3e+02 | 1,468 | 1.28 | 1.5 | 15.8 | 0.00042 |
| uncertaintea-cpu | 4 | 1000 | Float64 | PASS | 9,554 ± 2,393 | 8,568 | 0.23 | 0.392 | 4.18 | 0.00033 |

## gauss

| framework | chains | draws/chain | precision | correct | min bulk ESS/s | min tail ESS/s | sampling s | warmup s | TTFX/compile s | div rate |
|---|---|---|---|---|---|---|---|---|---|---|
| numpyro-parallel | 4 | 1000 | float64 | PASS | 4,155 ± 2.2e+02 | 3,164 | 0.768 | 1.01 | 2.32 | 0 |
| numpyro-vectorized | 64 | 500 | float32 | PASS | 14,279 ± 1,205 | 12,260 | 1.7 | 1.63 | 4.99 | 0 |
| numpyro-vectorized | 512 | 500 | float32 | PASS | 28,378 ± 1,773 | 24,441 | 7.08 | 4.23 | 13 | 0 |
| numpyro-vectorized | 4096 | 500 | float32 | PASS | 7,687 ± 0 | 6,720 | 205 | 93.6 | 237 | 0 |
| stan | 4 | 1000 | float64 | PASS | 171,372 ± 32,742 | 115,900 | 0.021 | 0.0147 | 0.769 | 0 |
| uncertaintea-batched-cpu | 4 | 1000 | Float64 | PASS | 38,708 ± 2,955 | 43,713 | 0.05 | 0.0856 | 5.34 | 0 |
| uncertaintea-batched-cpu | 64 | 500 | Float64 | PASS | 57,519 ± 17,018 | 68,095 | 0.257 | 0.615 | 5.69 | 0 |
| uncertaintea-batched-cpu | 512 | 500 | Float64 | PASS | 53,430 ± 5,336 | 63,051 | 2.05 | 4.68 | 11.7 | 0 |
| uncertaintea-batched-cpu | 4096 | 500 | Float64 | PASS | 36,879 ± 0 | 44,204 | 23.2 | 38.7 | 71.4 | 0 |
| uncertaintea-batched-cpu-ka | 64 | 500 | Float64 | PASS | 5,259 ± 1.4e+02 | 5,445 | 3.2 | 2.86 | 9.88 | 0 |
| uncertaintea-batched-cpu-ka | 512 | 500 | Float64 | PASS | 2,916 ± 6.2e+02 | 3,100 | 46 | 33.5 | 67.6 | 0 |
| uncertaintea-batched-cpu-ka | 4096 | 500 | Float64 | PASS | 6,282 ± 0 | 6,902 | 159 | 178 | 412 | 0 |
| uncertaintea-batched-metal | 64 | 500 | Float32 | PASS | 648 ± 4.2 | 657 | 25.9 | 18.9 | 60.7 | 0 |
| uncertaintea-batched-metal | 512 | 500 | Float32 | PASS | 2,770 ± 53 | 2,947 | 46.9 | 37.9 | 98.9 | 0 |
| uncertaintea-batched-metal | 4096 | 500 | Float32 | PASS | 18,104 ± 0 | 19,924 | 55.3 | 72.3 | 142 | 0 |
| uncertaintea-cpu | 4 | 1000 | Float64 | PASS | 509 ± 37 | 515 | 4.05 | 4.12 | 9.74 | 0 |

## logistic

| framework | chains | draws/chain | precision | correct | min bulk ESS/s | min tail ESS/s | sampling s | warmup s | TTFX/compile s | div rate |
|---|---|---|---|---|---|---|---|---|---|---|
| numpyro-parallel | 4 | 1000 | float64 | PASS | 10,504 ± 6.7e+02 | 5,679 | 0.489 | 0.623 | 1.15 | 0 |
| numpyro-vectorized | 64 | 500 | float32 | PASS | 31,886 ± 1,242 | 20,501 | 1.21 | 1.44 | 3.6 | 0 |
| numpyro-vectorized | 512 | 500 | float32 | PASS | 54,197 ± 2,387 | 36,263 | 5.68 | 5.25 | 12.1 | 0 |
| stan | 4 | 1000 | float64 | PASS | 64,569 ± 2,514 | 35,243 | 0.0767 | 0.072 | 1.16 | 0 |
| uncertaintea-batched-cpu | 4 | 1000 | Float64 | PASS | 11,790 ± 7.9e+02 | 8,235 | 0.338 | 0.46 | 4.45 | 0 |
| uncertaintea-batched-cpu | 64 | 500 | Float64 | PASS | 21,563 ± 1,450 | 17,442 | 1.43 | 1.53 | 7.17 | 0 |
| uncertaintea-batched-cpu | 512 | 500 | Float64 | PASS | 14,458 ± 1,537 | 11,705 | 17.6 | 14 | 32.4 | 0 |
| uncertaintea-cpu | 4 | 1000 | Float64 | PASS | 1,078 ± 12 | 714 | 3.93 | 3.75 | 7.36 | 0 |
