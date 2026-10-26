# Results

Model: huber. Sources: rev, wiki, retail. 99 firms.

![Source values](source_values.png)

## Source values (Shapley, 95% CI)

| setting | source | value | low | high | verdict |
| --- | --- | --- | --- | --- | --- |
| Development, all quarters | rev | +0.056 | +0.034 | +0.081 | helps |
| Development, all quarters | wiki | -0.005 | -0.015 | -0.000 | hurts |
| Development, all quarters | retail | +0.030 | +0.010 | +0.044 | helps |
| Development, outside COVID | rev | +0.050 | +0.034 | +0.066 | helps |
| Development, outside COVID | wiki | -0.011 | -0.034 | +0.001 | no effect |
| Development, outside COVID | retail | -0.003 | -0.012 | +0.005 | no effect |
| Holdout, Jul 2024 on | rev | +0.035 | +0.005 | +0.063 | helps |
| Holdout, Jul 2024 on | wiki | -0.002 | -0.009 | +0.004 | no effect |
| Holdout, Jul 2024 on | retail | -0.002 | -0.009 | +0.004 | no effect |

## Accuracy, development period (to Jun 2024)

| sources | n | MAE | naive MAE | skill | skill ex-COVID | direction hit rate | DM t | DM p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (none) | 2809 | 0.123 | 0.123 | -0.001 | -0.005 | 0.506 | -1.037 | 0.300 |
| rev | 2809 | 0.115 | 0.123 | +0.061 | +0.045 | 0.637 | +2.829 | 0.005 |
| wiki | 2809 | 0.123 | 0.123 | -0.006 | -0.015 | 0.496 | -1.332 | 0.183 |
| retail | 2809 | 0.118 | 0.123 | +0.035 | -0.006 | 0.563 | +1.587 | 0.113 |
| rev + wiki + retail | 2809 | 0.113 | 0.123 | +0.079 | +0.032 | 0.630 | +2.235 | 0.025 |

## Accuracy, holdout (Jul 2024 on, scored once)

| sources | n | MAE | naive MAE | skill | skill ex-COVID | direction hit rate | DM t | DM p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (none) | 882 | 0.052 | 0.052 | -0.007 | -0.007 | 0.492 | -0.668 | 0.504 |
| rev | 882 | 0.050 | 0.052 | +0.030 | +0.030 | 0.610 | +2.533 | 0.011 |
| wiki | 882 | 0.052 | 0.052 | -0.005 | -0.005 | 0.490 | -1.240 | 0.215 |
| retail | 882 | 0.052 | 0.052 | -0.009 | -0.009 | 0.465 | -1.777 | 0.076 |
| rev + wiki + retail | 882 | 0.050 | 0.052 | +0.024 | +0.024 | 0.602 | +2.710 | 0.007 |
