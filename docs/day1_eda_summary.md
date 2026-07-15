# Day 1 EDA Summary

- Date range: 2026-06-01 07:43:55 to 2026-07-01 02:03:21.134000
- Users: 100  (malicious: 6)
- Logon events: 4228
- Device events: 268
- HTTP events: 13404
- Email events: 6641
- File events: 17595

## Sanity check: malicious users stand out on simple aggregates

File-access event count (top 10 users):
```
user_id
U0027    231
U0050    221
U0032    203
U0015    197
U0080    196
U0095    196
U0008    196
U0037    195
U0088    193
U0067    193
```
Malicious users: ['U0016', 'U0027', 'U0038', 'U0050', 'U0079', 'U0092']

Logon (not logoff) events outside 06:00-20:00 (should be ~0 for normal users):
```
user_id
U0016    1
U0038    1
```

Users with largest external email attachments (KB):
```
user_id  size_kb  attachments
  U0092    22035            3
  U0056      647            0
  U0035      621            0
  U0014      605            0
  U0028      582            0
  U0011      568            0
  U0093      561            0
  U0008      515            1
  U0021      453            0
  U0066      452            1
```
