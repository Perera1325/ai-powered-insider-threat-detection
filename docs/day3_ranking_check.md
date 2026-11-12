# Day 3 Feature Ranking Sanity Check

Simple z-score-sum ranking over all 28 engineered features (not a real model - see Day 4). Out of 100 users, 6 are known malicious insiders.

## Where malicious users land in the ranking

```
 rank user_id          scenario  anomaly_score
    1   U0027  mass_file_access      44.559726
    2   U0079  usb_exfiltration      41.494764
    3   U0016 after_hours_login      37.016784
    4   U0092  data_exfil_email      34.194498
    5   U0038 after_hours_login      33.821569
    7   U0050   privilege_probe      24.844288
```

6/6 malicious users appear in the top 10 of 100 by this naive combined z-score - a real model (Day 4) with proper feature weighting/learning should do at least this well, likely better since it can learn non-linear combinations rather than a flat sum of z-scores.

## Top 10 ranked users overall

```
 rank user_id  anomaly_score  is_malicious          scenario
    1   U0027      44.559726          True  mass_file_access
    2   U0079      41.494764          True  usb_exfiltration
    3   U0016      37.016784          True after_hours_login
    4   U0092      34.194498          True  data_exfil_email
    5   U0038      33.821569          True after_hours_login
    6   U0099      27.559929         False               NaN
    7   U0050      24.844288          True   privilege_probe
    8   U0011      21.589885         False               NaN
    9   U0096      21.096210         False               NaN
   10   U0067      21.006968         False               NaN
```
