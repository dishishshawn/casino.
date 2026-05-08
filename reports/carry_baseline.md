# Carry Baseline Gate (Branch C step 2)

- Generated: 2026-05-07T21:56:51.201907+00:00
- Universe: SPY, QQQ, IWM, EFA, EEM, TLT, IEF, GLD, DBC, USO
- Window: 2006-01-02 .. 2026-05-06
- Cost: 12.0 bps round-trip
- Mode: long_short

## Selection-bias caveat

Carry was chosen *after* the TSMOM sensitivity grid failed (memory: tsmom_killed_2026-05-07.md). The DSR n_trials=30 figure used here is the spec value, but post-TSMOM-failure the true trial count is closer to **30 + the 90-cell TSMOM grid ≈ 120**. If any of the three gate components is borderline (Sharpe ≈ 0.4, DSR ≈ 0, corr ≈ 0.4), re-run with `--n-trials 60` (or higher) before declaring PASS.

## Gate components

- Sharpe:           -0.1130  (target ≥ 0.4 → FAIL)
- Deflated Sharpe: -3.1310  (target > 0, n_trials=30 → FAIL)
  - p-value: 0.9991, n_obs: 5392
- corr(carry, TSMOM): -0.2602  (target < 0.4 → PASS)

## Verdict

- **FAIL**: Sharpe -0.11 < 0.4; deflated SR -3.13 <= 0 (p=0.9991, n_trials=30)

## Full metrics

- Sortino:           -0.1790
- Max Drawdown:      -0.3364
- Win rate:          0.4381
- Total return:      -0.1773
- Annualized return: -0.0095
- Annualized vol:    0.0631

## Per-asset average |weight|

- DBC: 0.0000
- EEM: 0.0757
- EFA: 0.1357
- GLD: 0.0000
- IEF: 0.2987
- IWM: 0.1017
- QQQ: 0.1274
- SPY: 0.2043
- TLT: 0.1364
- USO: 0.0000

## Per-year Sharpe / total return

| year | sharpe | total_return |
| --- | --- | --- |
| 2006 | nan | 0.0 |
| 2007 | 0.415 | 0.028 |
| 2008 | 0.602 | 0.0451 |
| 2009 | 0.864 | 0.0573 |
| 2010 | 0.293 | 0.0164 |
| 2011 | 0.121 | 0.006 |
| 2012 | 0.554 | 0.0312 |
| 2013 | 0.338 | 0.0144 |
| 2014 | -0.621 | -0.0151 |
| 2015 | -0.54 | -0.0219 |
| 2016 | -0.27 | -0.0136 |
| 2017 | 0.206 | 0.0043 |
| 2018 | 0.156 | 0.008 |
| 2019 | 0.724 | 0.03 |
| 2020 | -0.192 | -0.0292 |
| 2021 | 0.98 | 0.0321 |
| 2022 | -0.484 | -0.0416 |
| 2023 | -1.109 | -0.0729 |
| 2024 | -1.409 | -0.0859 |
| 2025 | -0.86 | -0.1018 |
| 2026 | -0.993 | -0.0652 |
