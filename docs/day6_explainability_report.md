# Day 6 Explainability Report

## U0016 — scenario: `after_hours_login` (anomaly score=0.572)

**Top SHAP contributors:**
```
after_hours_logon_ratio    0.044
std_logon_hour             0.030
after_hours_logon_count    0.027
copy_ratio                 0.022
copy_count                 0.018
distinct_recipients        0.014
```
**Top LIME contributors:**
```
distinct_recipients    -0.009
copy_count              0.009
http_event_count       -0.007
external_email_count   -0.006
avg_recipients          0.005
email_count            -0.005
```

## U0027 — scenario: `mass_file_access` (anomaly score=0.602)

**Top SHAP contributors:**
```
copy_count                        0.037
max_daily_file_events             0.037
max_daily_shared_folder_events    0.032
copy_ratio                        0.029
file_event_count                  0.029
distinct_files                    0.018
```
**Top LIME contributors:**
```
total_logons           -0.008
distinct_recipients    -0.007
http_event_count        0.006
total_attachments      -0.005
avg_recipients         -0.004
max_external_size_kb    0.004
```

## U0038 — scenario: `after_hours_login` (anomaly score=0.555)

**Top SHAP contributors:**
```
after_hours_logon_ratio    0.041
after_hours_logon_count    0.028
std_logon_hour             0.022
external_email_count       0.014
external_email_ratio       0.012
delete_count               0.008
```
**Top LIME contributors:**
```
avg_session_min                  -0.009
copy_ratio                        0.009
external_email_count             -0.008
max_daily_shared_folder_events    0.007
std_logon_hour                    0.007
copy_count                        0.007
```

## U0050 — scenario: `privilege_probe` (anomaly score=0.514)

**Top SHAP contributors:**
```
max_daily_file_events    0.044
file_event_count         0.023
distinct_files           0.022
http_event_count         0.013
total_attachments        0.012
email_count              0.011
```
**Top LIME contributors:**
```
total_attachments        0.010
email_count              0.009
http_event_count         0.007
max_daily_file_events    0.006
distinct_files           0.006
avg_session_min         -0.005
```

## U0079 — scenario: `usb_exfiltration` (anomaly score=0.590)

**Top SHAP contributors:**
```
copy_count                  0.029
device_after_hours_count    0.028
copy_ratio                  0.026
device_connect_count        0.025
max_daily_file_events       0.025
external_email_count        0.015
```
**Top LIME contributors:**
```
external_email_count   -0.009
std_logon_hour         -0.007
avg_recipients          0.007
total_attachments      -0.006
avg_logon_hour          0.006
external_email_ratio   -0.006
```

## U0092 — scenario: `data_exfil_email` (anomaly score=0.511)

**Top SHAP contributors:**
```
avg_email_size_kb       0.040
max_external_size_kb    0.034
http_event_count        0.014
copy_ratio              0.009
std_session_min         0.008
delete_count            0.006
```
**Top LIME contributors:**
```
std_session_min      0.011
http_event_count    -0.009
copy_ratio           0.008
delete_count         0.007
copy_count           0.007
total_attachments    0.007
```

## U0074 — scenario: `normal_baseline` (anomaly score=0.433)

**Top SHAP contributors:**
```
http_event_count                  0.013
total_attachments                 0.011
max_daily_shared_folder_events    0.010
avg_recipients                    0.008
total_logons                      0.007
copy_ratio                        0.004
```
**Top LIME contributors:**
```
total_logons        -0.014
copy_ratio          -0.009
total_attachments   -0.009
http_event_count     0.008
avg_recipients       0.008
delete_count         0.007
```

