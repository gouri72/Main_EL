# Results (last 100 frozen-policy evaluation episodes per seed; mean ± 95% CI over 8 seeds)

| experiment | immediate_success | post_transfer | transfer_gain | assist_spent | full_assist_rate | first_attempt_rate | takeover_rate | efficiency |
|---|---|---|---|---|---|---|---|---|
| A_novice_v2__loo_transfer | 0.354 ± 0.018 | 0.253 ± 0.004 | 0.003 ± 0.006 | 8.425 ± 0.623 | 0.026 ± 0.010 | 0.565 ± 0.037 | 0.017 ± 0.006 | 0.000 ± 0.001 |
| A_novice_v2__opblind_all | 0.374 ± 0.020 | 0.240 ± 0.008 | -0.015 ± 0.011 | 10.248 ± 0.748 | 0.020 ± 0.014 | 0.557 ± 0.040 | 0.013 ± 0.009 | -0.001 ± 0.001 |
| A_novice_v2__opblind_caps | 0.373 ± 0.017 | 0.237 ± 0.007 | -0.018 ± 0.011 | 9.747 ± 0.624 | 0.029 ± 0.009 | 0.590 ± 0.019 | 0.019 ± 0.006 | -0.002 ± 0.001 |
| A_novice_v2__opblind_state | 0.323 ± 0.026 | 0.286 ± 0.005 | 0.047 ± 0.005 | 6.079 ± 0.575 | 0.000 ± 0.001 | 0.598 ± 0.053 | 0.000 ± 0.001 | 0.008 ± 0.001 |
| A_novice_v2__performance_only | 0.426 ± 0.012 | 0.229 ± 0.014 | -0.030 ± 0.019 | 12.635 ± 0.739 | 0.084 ± 0.020 | 0.493 ± 0.038 | 0.056 ± 0.015 | -0.002 ± 0.001 |
| A_novice_v2__tabular_q | 0.357 ± 0.017 | 0.255 ± 0.009 | 0.005 ± 0.012 | 8.462 ± 0.623 | 0.023 ± 0.010 | 0.547 ± 0.034 | 0.016 ± 0.006 | 0.001 ± 0.001 |

# Paired differences vs A_novice_v2__tabular_q (experiment minus reference; paired by seed; p = paired t-test)

| experiment | Δ immediate_success | Δ transfer_gain | p | Δ post_transfer | Δ assist_spent |
|---|---|---|---|---|---|
| A_novice_v2__loo_transfer | -0.003 ± 0.025 | -0.002 ± 0.015 | 0.766 | -0.001 ± 0.011 | -0.038 ± 1.025 |
| A_novice_v2__opblind_all | 0.017 ± 0.032 | -0.020 ± 0.021 | 0.063 | -0.015 ± 0.016 | 1.785 ± 1.252 |
| A_novice_v2__opblind_caps | 0.016 ± 0.013 | -0.023 ± 0.014 | 0.007 | -0.017 ± 0.011 | 1.285 ± 0.737 |
| A_novice_v2__opblind_state | -0.034 ± 0.021 | 0.042 ± 0.014 | 0.000 | 0.031 ± 0.010 | -2.384 ± 0.865 |
| A_novice_v2__performance_only | 0.069 ± 0.017 | -0.035 ± 0.029 | 0.026 | -0.026 ± 0.022 | 4.172 ± 1.079 |
