# Results (last 100 frozen-policy evaluation episodes per seed; mean ± 95% CI over 8 seeds)

| experiment | immediate_success | post_transfer | transfer_gain | assist_spent | full_assist_rate | first_attempt_rate | takeover_rate | efficiency |
|---|---|---|---|---|---|---|---|---|
| A_novice_v2__always_full | 0.509 ± 0.010 | 0.107 ± 0.005 | -0.195 ± 0.007 | 22.538 ± 0.177 | 0.329 ± 0.010 | 0.000 ± 0.000 | 0.216 ± 0.006 | -0.009 ± 0.000 |
| A_novice_v2__always_none | 0.158 ± 0.009 | 0.274 ± 0.003 | 0.031 ± 0.002 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | - |
| A_novice_v2__fixed_human_first | 0.588 ± 0.009 | 0.154 ± 0.007 | -0.131 ± 0.010 | 18.621 ± 0.378 | 0.268 ± 0.012 | 1.000 ± 0.000 | 0.174 ± 0.010 | -0.007 ± 0.000 |
| A_novice_v2__linear_q | 0.406 ± 0.081 | 0.280 ± 0.016 | 0.039 ± 0.022 | 10.364 ± 3.545 | 0.000 ± 0.000 | 0.669 ± 0.341 | 0.000 ± 0.000 | 0.004 ± 0.002 |
| A_novice_v2__linucb | 0.478 ± 0.015 | 0.207 ± 0.005 | -0.059 ± 0.007 | 12.586 ± 0.230 | 0.000 ± 0.000 | 0.907 ± 0.039 | 0.000 ± 0.000 | -0.005 ± 0.001 |
| A_novice_v2__rule_adaptive | 0.511 ± 0.010 | 0.266 ± 0.005 | 0.019 ± 0.006 | 15.692 ± 0.232 | 0.059 ± 0.004 | 0.691 ± 0.006 | 0.043 ± 0.004 | 0.001 ± 0.000 |
| A_novice_v2__tabular_q | 0.357 ± 0.017 | 0.255 ± 0.009 | 0.005 ± 0.012 | 8.462 ± 0.623 | 0.023 ± 0.010 | 0.547 ± 0.034 | 0.016 ± 0.006 | 0.001 ± 0.001 |

# Paired differences vs A_novice_v2__tabular_q (experiment minus reference; paired by seed; p = paired t-test)

| experiment | Δ immediate_success | Δ transfer_gain | p | Δ post_transfer | Δ assist_spent |
|---|---|---|---|---|---|
| A_novice_v2__always_full | 0.152 ± 0.021 | -0.199 ± 0.014 | 0.000 | -0.148 ± 0.011 | 14.075 ± 0.553 |
| A_novice_v2__always_none | -0.199 ± 0.016 | 0.026 ± 0.012 | 0.001 | 0.020 ± 0.009 | -8.462 ± 0.623 |
| A_novice_v2__fixed_human_first | 0.231 ± 0.019 | -0.136 ± 0.016 | 0.000 | -0.101 ± 0.012 | 10.159 ± 0.662 |
| A_novice_v2__linear_q | 0.049 ± 0.078 | 0.034 ± 0.026 | 0.018 | 0.025 ± 0.020 | 1.901 ± 3.277 |
| A_novice_v2__linucb | 0.121 ± 0.027 | -0.064 ± 0.011 | 0.000 | -0.048 ± 0.009 | 4.124 ± 0.686 |
| A_novice_v2__rule_adaptive | 0.154 ± 0.025 | 0.014 ± 0.014 | 0.041 | 0.011 ± 0.010 | 7.230 ± 0.663 |
