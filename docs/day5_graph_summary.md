# Day 5 Graph Analysis Summary

Graph: 104 nodes (100 users, 4 shared resources), 5097 edges.

## Centrality features by group (mean)

```
              graph_degree  graph_weighted_degree  graph_betweenness  resource_diversity
is_malicious                                                                            
False              97.9468               439.9894             0.0004                 4.0
True               97.8333               457.5000             0.0004                 4.0
```

## Malicious users' graph features

```
user_id          scenario  graph_degree  graph_weighted_degree  graph_betweenness  resource_diversity
  U0016 after_hours_login            93                    400           0.000380                   4
  U0027  mass_file_access            94                    472           0.000406                   4
  U0038 after_hours_login            99                    453           0.000462                   4
  U0050   privilege_probe           102                    516           0.000472                   4
  U0079  usb_exfiltration           101                    464           0.000462                   4
  U0092  data_exfil_email            98                    440           0.000429                   4
```
