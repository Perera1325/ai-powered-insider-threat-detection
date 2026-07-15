# Day 6 Explainability Report

## U0016 — scenario: `after_hours_login` (anomaly score=0.557)

**Top SHAP contributors:**
```
after_hours_logon_ratio    0.050
after_hours_logon_count    0.040
std_logon_hour             0.025
copy_ratio                 0.022
copy_count                 0.016
email_count                0.009
```
**Top LIME contributors:**
```
copy_ratio           0.009
delete_count        -0.007
std_session_min     -0.006
copy_count           0.005
avg_session_min     -0.005
total_attachments   -0.005
```

## U0027 — scenario: `mass_file_access` (anomaly score=0.635)

**Top SHAP contributors:**
```
max_daily_file_events             0.056
max_daily_shared_folder_events    0.052
copy_count                        0.033
copy_ratio                        0.033
distinct_files                    0.030
file_event_count                  0.028
```
**Top LIME contributors:**
```
total_logons            -0.006
std_logon_hour           0.005
http_event_count         0.005
total_attachments       -0.005
max_daily_file_events    0.004
file_event_count         0.004
```

## U0038 — scenario: `after_hours_login` (anomaly score=0.582)

**Top SHAP contributors:**
```
after_hours_logon_ratio           0.046
after_hours_logon_count           0.040
std_logon_hour                    0.027
total_logons                      0.014
external_email_count              0.013
max_daily_shared_folder_events    0.013
```
**Top LIME contributors:**
```
avg_session_min                  -0.011
copy_count                        0.009
max_daily_shared_folder_events    0.008
external_email_ratio             -0.007
copy_ratio                        0.007
http_event_count                  0.007
```

## U0050 — scenario: `privilege_probe` (anomaly score=0.537)

**Top SHAP contributors:**
```
max_daily_file_events    0.051
distinct_files           0.029
file_event_count         0.026
http_event_count         0.019
delete_count             0.014
total_attachments        0.009
```
**Top LIME contributors:**
```
http_event_count         0.011
max_daily_file_events    0.009
file_event_count         0.007
distinct_files           0.006
email_count              0.006
std_logon_hour          -0.006
```

## U0079 — scenario: `usb_exfiltration` (anomaly score=0.596)

**Top SHAP contributors:**
```
device_after_hours_count    0.035
copy_ratio                  0.029
max_daily_file_events       0.028
copy_count                  0.026
device_connect_count        0.026
external_email_count        0.011
```
**Top LIME contributors:**
```
file_event_count     0.010
total_logons         0.007
std_logon_hour      -0.007
delete_count        -0.007
avg_email_size_kb   -0.007
copy_count           0.006
```

## U0092 — scenario: `data_exfil_email` (anomaly score=0.525)

**Top SHAP contributors:**
```
avg_email_size_kb       0.050
max_external_size_kb    0.034
copy_ratio              0.014
http_event_count        0.011
delete_count            0.010
external_email_count    0.008
```
**Top LIME contributors:**
```
copy_ratio              0.011
std_session_min         0.009
external_email_count    0.008
delete_count            0.008
copy_count              0.008
http_event_count       -0.007
```

## U0045 — scenario: `normal_baseline` (anomaly score=0.431)

**Top SHAP contributors:**
```
avg_logon_hour          0.015
avg_email_size_kb       0.010
total_attachments       0.009
email_count             0.007
device_connect_count    0.004
delete_count            0.004
```
**Top LIME contributors:**
```
avg_logon_hour      -0.012
avg_email_size_kb    0.012
total_attachments   -0.008
total_logons         0.008
distinct_files       0.008
email_count         -0.006
```

