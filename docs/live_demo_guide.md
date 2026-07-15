# Live Demo Guide (for presenting to your lecturer)

The dashboard's **🔴 Live Demo** tab simulates a small "company" of 12
demo users generating fresh activity day by day, in real time, and
scores them against the same model trained on the Day 1-6 baseline
population - so instead of only showing a static, pre-computed result,
you can watch a risk score rise and an alert fire live, in front of an
audience.

## Why this exists

Everything else in this project (Days 1-7) runs on a fixed, already-
generated dataset - great for rigor and reproducibility, but not very
compelling to *watch*. This tab reuses the exact same generator
functions (`generate_logs.py`), injector functions (`inject_anomalies.py`),
and feature-engineering functions (`feature_engineering.py`) as the rest
of the project - it's provably the same pipeline, just run on a small
live cohort instead of the static 100-user dataset, one simulated day
at a time.

## How to run it

1. `streamlit run app.py`
2. Click the **🔴 Live Demo** tab
3. Pick:
   - **Demo user** - which of the 12 live users will be "the insider" tonight
   - **Attack scenario** - one of the 5 from Day 1 (`after_hours_login`,
     `mass_file_access`, `usb_exfiltration`, `data_exfil_email`,
     `privilege_probe`)
   - **Simulated days to run** and **inject attack on day** - e.g. run
     10 days, inject on day 6, so the audience sees a few days of
     normal (slightly noisy) activity before the attack happens
   - **Seconds per simulated day** - demo pacing; ~1 second per day
     keeps a 10-day run under 15 seconds, long enough to be visible,
     short enough not to drag
4. Click **▶ Start Live Feed** and narrate as it runs

## What to say while it's running

- **Days 1 to (injection day - 1):** "This is normal noise - the top
  alert bounces around day to day because with only a few days of
  history, small random differences look proportionally large. This is
  expected and is actually realistic: a monitoring system needs a
  baseline period before it's confident."
- **The injection day:** point at the risk chart - the demo user's line
  jumps straight to 100/100 and a red alert banner appears. Say what
  scenario you picked and why the system caught it (tie back to Day 6's
  SHAP findings - e.g. "for USB exfiltration, it's the device connect
  count and after-hours device usage that spike").
- **After the injection:** "Once flagged, the user stays flagged for
  every subsequent day - the system doesn't 'forget' a confirmed alert
  just because a few normal days followed."

## If your lecturer asks "is this really live?"

Yes and no, and it's worth being upfront about the distinction: the
*data generation* is genuinely happening in real time (each simulated
day's events are generated, injected if applicable, and re-scored live,
right there in the loop) - it's not a pre-recorded animation. What's
simulated is the "12 employees, day by day" premise itself, since there
isn't a real company feeding this system real logs. That's exactly the
same honest framing as the rest of the project: a synthetic dataset by
design (see the Ethics section in the main README), demonstrating a
pipeline that *would* work the same way against a real SIEM/log feed.

## Troubleshooting

If the live feed looks "stuck" on one risk ranking for several days
before settling, that's the small-sample noise described above -
increase "simulated days to run" for a longer baseline before the
injection day, or just narrate it as expected behavior (see docs/day5
and day4 progress notes - this project's pattern throughout has been to
explain *why* something looks a certain way, not just show a clean
number).
