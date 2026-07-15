# Day 2 EDA Summary

## Log-type daily-count correlation

```
        logon  device  http  email  file
logon    1.00   -0.05  0.05   0.03  0.07
device  -0.05    1.00  0.04   0.00  0.03
http     0.05    0.04  1.00   0.07  0.11
email    0.03    0.00  0.07   1.00  0.08
file     0.07    0.03  0.11   0.08  1.00
```

Correlations are all weak-to-moderate (no pair above ~0.3), meaning the five log types carry largely independent signal - worth keeping all of them as separate feature groups in Day 3 rather than collapsing them into one 'activity volume' feature.

## Malicious vs. normal separation (raw aggregates)

```
user_id     group  file_events  max_external_kb
  U0050 malicious          221              178
  U0079 malicious          189              304
  U0016 malicious          173               78
  U0092 malicious          170            22035
  U0038 malicious          182               77
  U0027 malicious          231               71
  U0073    normal          184              235
  U0072    normal          178              173
  U0071    normal          176              398
  U0054    normal          169              133
  U0070    normal          178              180
  U0069    normal          175              173
  U0068    normal          185              306
  U0067    normal          193              142
  U0066    normal          168              452
  U0058    normal          178               15
  U0065    normal          175              285
  U0074    normal          186              149
  U0064    normal          180              296
  U0063    normal          160              113
  U0062    normal          171              183
  U0061    normal          159              363
  U0060    normal          180              116
  U0056    normal          185              647
  U0059    normal          175               49
  U0057    normal          175              296
  U0055    normal          186              244
  U0075    normal          175              185
  U0076    normal          186              147
  U0098    normal          171              212
  U0097    normal          175              312
  U0096    normal          157              153
  U0095    normal          196              318
  U0094    normal          166              259
  U0093    normal          151              561
  U0091    normal          158              136
  U0090    normal          175              304
  U0089    normal          183              189
  U0053    normal          172              273
  U0088    normal          193              189
  U0086    normal          167              273
  U0085    normal          177              262
  U0084    normal          180              198
  U0083    normal          177              388
  U0082    normal          169              327
  U0081    normal          191              199
  U0080    normal          196              326
  U0078    normal          152              428
  U0077    normal          179               88
  U0087    normal          156               84
  U0052    normal          178              209
  U0001    normal          168               97
  U0099    normal          164              351
  U0022    normal          177              413
  U0021    normal          169              453
  U0020    normal          185              180
  U0019    normal          179              337
  U0018    normal          158              237
  U0017    normal          179              253
  U0015    normal          197              308
  U0014    normal          182              605
  U0013    normal          171              191
  U0023    normal          159              164
  U0012    normal          161              265
  U0010    normal          175              361
  U0009    normal          168              208
  U0008    normal          196              515
  U0007    normal          174              165
  U0006    normal          192              365
  U0005    normal          169               99
  U0004    normal          167              366
  U0003    normal          151              227
  U0002    normal          174              226
  U0011    normal          136              568
  U0024    normal          174               78
  U0025    normal          187              321
  U0026    normal          172               94
  U0049    normal          161              195
  U0048    normal          164              340
  U0047    normal          160              200
  U0046    normal          186              197
  U0045    normal          185              431
  U0044    normal          179               57
  U0043    normal          182              197
  U0042    normal          181              438
  U0041    normal          193               84
  U0040    normal          164              229
  U0039    normal          169              133
  U0037    normal          195              409
  U0036    normal          161              295
  U0035    normal          174              621
  U0034    normal          157              363
  U0033    normal          163              136
  U0032    normal          203              395
  U0031    normal          191              419
  U0030    normal          188              261
  U0029    normal          181              212
  U0028    normal          157              582
  U0051    normal          167              257
  U0100    normal          188              211
```

Mean file-event count is 19.6 events higher for malicious users than normal users, but individual overlap exists (mass_file_access and privilege_probe scenarios are subtle by design). This is expected at the raw-count level - Day 3 feature engineering (e.g. after-hours ratio, per-department file-path deviation) should separate these more cleanly than raw totals do.

## Calibration decision

Keeping current generator parameters as-is: `after_hours_login` and `data_exfil_email` are cleanly separable even on raw counts (good - confirms the injection logic works end-to-end), while `mass_file_access`/`privilege_probe`/`usb_exfiltration` require proper feature engineering to detect, which matches real-world insider threat difficulty and avoids building a dataset that's trivially easy for any model to solve.

## Figures

- `docs/figures/logon_hour_heatmap.png`
- `docs/figures/log_type_correlation.png`
- `docs/figures/malicious_vs_normal_boxplots.png`
