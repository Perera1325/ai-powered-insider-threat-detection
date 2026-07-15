# Day 3 Feature Ranking Sanity Check

Simple z-score-sum ranking over all 25 engineered features (not a real model - see Day 4). Out of 100 users, 6 are known malicious insiders.

## Where malicious users land in the ranking

```
 rank user_id          scenario  anomaly_score
    1   U0027  mass_file_access      42.600986
    2   U0079  usb_exfiltration      39.406829
    3   U0016 after_hours_login      34.056910
    4   U0092  data_exfil_email      32.830326
    5   U0038 after_hours_login      32.233782
    6   U0050   privilege_probe      23.345495
```

6/6 malicious users appear in the top 10 of 100 by this naive combined z-score - a real model (Day 4) with proper feature weighting/learning should do at least this well, likely better since it can learn non-linear combinations rather than a flat sum of z-scores.

## Top 10 ranked users overall

```
 rank user_id  anomaly_score  is_malicious          scenario
    1   U0027      42.600986          True  mass_file_access
    2   U0079      39.406829          True  usb_exfiltration
    3   U0016      34.056910          True after_hours_login
    4   U0092      32.830326          True  data_exfil_email
    5   U0038      32.233782          True after_hours_login
    6   U0050      23.345495          True   privilege_probe
    7   U0099      22.045177         False               NaN
    8   U0096      20.034149         False               NaN
    9   U0067      18.921268         False               NaN
   10   U0078      18.406862         False               NaN
```
