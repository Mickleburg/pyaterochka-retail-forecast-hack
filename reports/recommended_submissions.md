# Recommended Submissions

Current confirmed best: `submissions/test_baseline_last_month.csv` with score `95.86`.

New candidates were generated only when they did not look worse than `baseline_last_month` on both fold 10 and weighted_mape_127, except conservative ensemble aliases that are intentionally almost identical to the best baseline.

Recommended next submissions:

1. `submissions/test_last_month_mult_102.csv`
   - weighted_mape_127=6.546991, fold10=5.545709, max_delta_vs_best=0.020000
   - why: pred = rto_lag_1 * 1.020
2. `submissions/test_last_month_mult_1015.csv`
   - weighted_mape_127=6.567553, fold10=5.711058, max_delta_vs_best=0.015000
   - why: pred = rto_lag_1 * 1.015
3. `submissions/test_last_month_mult_101.csv`
   - weighted_mape_127=6.613995, fold10=5.903976, max_delta_vs_best=0.010000
   - why: pred = rto_lag_1 * 1.010

After every new LB result, record it:

```bash
python scripts/record_leaderboard_result.py --file <submission> --model <model> --lb-score <score> --verdict OK --comment "<note>"
```

If a new candidate is worse, restore the current best:

```bash
python scripts/restore_best_submission.py
```
