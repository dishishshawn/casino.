# TSMOM Sensitivity Grid (Branch C step 1.5)

- Generated: 2026-05-07T21:36:43.780489+00:00
- Universe: SPY, QQQ, IWM, EFA, EEM, TLT, IEF, GLD, DBC, USO
- Cost: 12.0 bps round-trip
- Cells: 90 (single-lookback grid)
- DSR n_trials: 90

## Grid summary (Sharpe distribution)

- mean: 0.2876
- std:  0.1696
- min:  -0.1113
- max:  0.4701
- top-quartile cutoff (75th pct): 0.3743

## Chosen production config

- lookback: blend(1,3,6,12)  (21, 63, 126, 252 bdays)
- vol-target: 10.0%
- rebalance: monthly
- Sharpe: 0.3461
- Deflated Sharpe: -1.4038
- Max DD: -0.2070
- Ann return: 0.0332

## Verdict

- chosen_sharpe = 0.3461, grid_mean = 0.2876, top_quartile_cutoff = 0.3743, percentile = 0.5556
- VERDICT: **FAIL**
- Chosen config is below the top-quartile cutoff. Per PRD section 3 hard-restart rule, the TSMOM signal is overfit and downstream tasks (39, 40, 41, 42) are BLOCKED until Branch C restarts.

## Top 10 cells by Sharpe

| lookback_months | vol_target_pct | rebalance | sharpe | deflated_sharpe | deflated_p_value | max_drawdown | ann_return | total_return | ann_vol |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 8.0 | weekly | 0.4701 | -0.8304 | 0.79683 | -0.2025 | 0.0428 | 1.3457 | 0.0942 |
| 6 | 10.0 | weekly | 0.4697 | -0.833 | 0.79758 | -0.2025 | 0.0432 | 1.3627 | 0.0952 |
| 3 | 8.0 | biweekly | 0.4696 | -0.8292 | 0.796515 | -0.1756 | 0.0441 | 1.404 | 0.0974 |
| 3 | 10.0 | biweekly | 0.4695 | -0.8323 | 0.797389 | -0.1756 | 0.0446 | 1.4266 | 0.0986 |
| 3 | 12.0 | biweekly | 0.4661 | -0.8514 | 0.802728 | -0.1756 | 0.0446 | 1.4281 | 0.0996 |
| 6 | 12.0 | weekly | 0.466 | -0.8505 | 0.802489 | -0.2025 | 0.0431 | 1.3596 | 0.096 |
| 3 | 6.0 | biweekly | 0.4614 | -0.865 | 0.806484 | -0.1756 | 0.0426 | 1.3382 | 0.096 |
| 3 | 15.0 | biweekly | 0.4605 | -0.8849 | 0.811898 | -0.1756 | 0.0446 | 1.4273 | 0.1011 |
| 6 | 15.0 | weekly | 0.4605 | -0.8777 | 0.809941 | -0.2025 | 0.043 | 1.3535 | 0.0971 |
| 6 | 6.0 | weekly | 0.4596 | -0.8782 | 0.810077 | -0.2025 | 0.0412 | 1.2744 | 0.093 |

## Bottom 5 cells by Sharpe

| lookback_months | vol_target_pct | rebalance | sharpe | deflated_sharpe | deflated_p_value | max_drawdown | ann_return | total_return | ann_vol |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 12.0 | monthly | -0.1113 | -3.5151 | 0.99978 | -0.504 | -0.0167 | -0.2903 | 0.0995 |
| 1 | 10.0 | monthly | -0.1113 | -3.5146 | 0.99978 | -0.5004 | -0.0165 | -0.2867 | 0.0984 |
| 1 | 8.0 | monthly | -0.1091 | -3.5045 | 0.999771 | -0.4964 | -0.016 | -0.2796 | 0.0972 |
| 1 | 15.0 | monthly | -0.1064 | -3.4926 | 0.999761 | -0.5026 | -0.0166 | -0.2884 | 0.1012 |
| 1 | 6.0 | biweekly | -0.1042 | -3.482 | 0.999751 | -0.4764 | -0.0151 | -0.2668 | 0.0955 |
