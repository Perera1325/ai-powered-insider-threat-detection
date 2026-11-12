# Day 8 - closing the gaps against the proposal

## What was missing after Day 7
- Real scikit-learn / TensorFlow / shap / lime were only in an un-run Colab notebook.
- Every model scored 1.0 on one 100-user dataset (no evidence of quality).
- No PyVis graph (listed in the proposal's tools).
- No automated tests; the dashboard had never rendered.
- Proposal 5.3 asked for e-mail metadata features; only size/attachments/external were used.

## What was done
1. `src/models/real_models.py` - scikit-learn `IsolationForest`, `OneClassSVM`, Keras autoencoder, same
   `decision_function` interface (higher = more anomalous). All three are now dashboard choices.
2. `src/explain/real_explainers.py` - real `shap.KernelExplainer` and `lime.lime_tabular`; dashboard has an
   engine switch (library / scratch).
3. `src/robust_evaluation.py` - 8 unseen seeds x 200 users x 10 insiders, plus an attack-strength stress test.
4. `src/pyvis_graph.py` - interactive risk-coloured graph, embedded in the dashboard.
5. 3 new e-mail recipient features (28 features now).
6. 48 pytest tests incl. the real Streamlit app (`AppTest`), CI workflow, deterministic seeds/params on the generators.

## Findings worth putting in the report
- **Autoencoder memorisation:** AUC fell 0.95 -> 0.38 as training epochs rose 10 -> 100 on ~100 users. The few extreme
  insiders dominate the loss and are learned. Fix: winsorised training (clip inputs to +/-3 sigma, score raw):
  mean AUC 0.89 -> 0.97 on fresh seeds.
- **One-Class SVM default kernel too narrow:** `gamma="scale"` gave AUC 0.76; `gamma = 0.15 / n_features` gives ~0.94 across seeds.
- **Perfect scores were an artefact of one easy dataset:** on unseen seeds the best model is 0.998 AUC but precision@k is
  0.66-0.94 depending on the model, and every model degrades for low-and-slow attacks.
- Hyper-parameters were tuned on the original dataset and validated on unseen seeds; see `docs/final_test_report.md` limitations.

## Dashboard fixes after first real-browser run
- **Graph hairball:** the whole-org graph drew ~5,000 edges. Now only strong e-mail links are drawn (slider, default >= 7 emails),
  low-risk users are unlabelled, insiders have a black outline, shared folders are boxes, straight edges.
- **"Stuck" UI:** (1) all six models trained at start-up (~25 s) -> now trained lazily; (2) the from-scratch SHAP/LIME engine took
  13-47 s per click -> removed from the UI (real `shap`/`lime` take < 1 s; scratch remains an automatic fallback);
  (3) alert table, explanations, graph and neighbourhood plot were recomputed on every widget change -> cached.
- Measured (AppTest): first render 6.3 s, slider/model change ~1.3 s, first switch to the Keras autoencoder 3.5 s.
- The `ConnectionResetError [WinError 10054]` in the Git Bash console is a harmless Windows asyncio message printed when the browser tab reloads/closes.
