# Real-time (micro-batch) detection

## Architecture

```
event_producer.py  --appends JSON lines-->  events.jsonl  --tailed by-->  scorer.py  -->  alerts + risk history  -->  dashboard "Real-Time" tab
 (plays the org's log pipeline,             (user / event /                (byte-offset reads, 30-day
  one simulated working day per tick)        truth / tick lines)            sliding window, same features as training,
                                                                            trained model, absolute 0-100 risk)
```

* `src/realtime/event_producer.py` - separate process. First writes 30 working days of normal history (tick 0) so the
  window matches what the models were trained on, then one simulated day every `--interval` real seconds.
  Attacks are injected automatically (`--inject-after`) or on demand through `control.json` (the dashboard button).
* `src/realtime/scorer.py` - `RealTimeScorer.poll()` reads only NEW complete lines, scores after every tick marker,
  and raises an alert when risk >= 70 (re-armed when it falls below 50). Alert latency = wall-clock seconds from the
  producer writing the batch to the scorer raising the alert.
* Risk is **absolute**: `(score - min_train) / (max_train - min_train) * 100`, clipped, using the training population.
  (The older Live Demo tab scales within its 12-user cohort, so someone is always at 100.)

## Run it

Dashboard: `streamlit run app.py` -> **Real-Time** tab -> *Start stream* -> wait ~3 ticks -> *Inject attack now*.

Command line (no dashboard):
```bash
python src/realtime/event_producer.py --dir live --interval 1 --inject-user L03 --scenario mass_file_access --inject-after 8
```

## Measured results (Isolation Forest, 12 users, 3 seeds x 5 scenarios, attack on tick 8)

| Scenario | Detected (alert for the attacked user) | Tick of alert (attack = 8) |
|---|---|---|
| after_hours_login | 3 / 3 | 8, 9, 8 |
| mass_file_access | 3 / 3 | 8, 8, 8 |
| usb_exfiltration | 3 / 3 | 8, 9, 8 |
| data_exfil_email | 3 / 3 | 8, 9, 8 |
| privilege_probe | **1 / 3** | 8 (one seed only) |

Total 13 / 15 attacks detected; 1 false alert (a normal user) in the 15 runs (~11 normal users x ~14 ticks each).
Highest risk reached by normal users before the attack: 60-71 (threshold 70) - so the margin is thin for subtle attacks.
Wall-clock: an alert appears < 5 s after the logs are written (asserted in `tests/test_realtime.py`).

## Limitations (say these in the viva)
1. **Micro-batch, not true streaming.** Features are daily-granularity over a 30-day window; delay is seconds, not milliseconds.
   Not Kafka/Flink, not connected to a SIEM (out of scope in the proposal).
2. **`privilege_probe` is the weak spot** - a handful of out-of-department file accesses barely moves 28 whole-window features.
3. The threshold (70) and the training-population min/max set the false-alarm/miss trade-off; they were not tuned per scenario.
4. Events are simulated; the "truth" lines exist only for the demo - a real deployment has no ground truth.
