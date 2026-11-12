# Final test report

Environment: 3.13.18 sklearn 1.9.1 tf 2.21.0 shap 0.54.1 streamlit 1.67.0 pandas 3.1.1 numpy 2.5.4

**Result: 48 / 48 tests passed** (`python -m pytest -q tests`, ~1 min).

## What was tested

| Area | File | What it proves |
|---|---|---|
| Data integrity | `test_data.py` | all log files exist; 6 labelled insiders covering all 5 scenarios; **no ground-truth column leaks into the model inputs**; no NaNs; only simulated e-mail domains |
| Pipeline | `test_pipeline.py` | same seed -> identical data (reproducible); different seed -> different data; injection creates the right number of insiders; the after-hours injection moves the after-hours feature |
| Models | `test_models.py` | scikit-learn IF / One-Class SVM rank obvious outliers first; Keras autoencoder flags off-manifold points; every dashboard model reaches AUC > 0.85 on the committed dataset; metric helpers are correct |
| Explainability | `test_explain_graph.py` | real SHAP satisfies additivity (sum of values = f(x) - f(background)); SHAP puts logon features on top for the after-hours insider and e-mail features on top for the exfiltration insider; scratch and library engines agree on direction; LIME returns one weight per feature |
| Graph | `test_explain_graph.py` | all users + shared resources present; PyVis HTML is self-contained and marks insiders |
| Live demo | `test_live_demo.py` | injecting an attack makes the insider's risk jump above every pre-attack value and reach HIGH (>= 70) |
| Dashboard | `test_dashboard_app.py` | the **real** Streamlit app (Streamlit `AppTest`) renders with no exception; metrics, 4 tabs, figures; every model choice renders; clicking "Start Live Feed" runs the simulation and raises an ALERT |
| Responsiveness | `test_performance.py` | models train lazily (first render ~6 s instead of ~30 s); every model's SHAP/LIME explanation < 15 s; the whole-organisation PyVis graph stays under 900 edges (was 5,000+) and labels only risky users |

A real `streamlit run app.py --server.headless true` server was also started and
answered `/_stcore/health` = `ok` and `/` = HTTP 200.

## Individual results

| Test | Status |
|---|---|
| `test_app_renders_without_exception PASSED ` | PASS |
| `test_alert_metrics_present PASSED         ` | PASS |
| `test_four_tabs PASSED                     ` | PASS |
| `test_explanation_figure_rendered_for_selected_alert` | PASS |
| `test_every_model_choice_renders[Isolation Forest (sklearn)]` | PASS |
| `test_every_model_choice_renders[One-Class SVM (sklearn)]` | PASS |
| `test_every_model_choice_renders[Autoencoder (Keras)]` | PASS |
| `test_every_model_choice_renders[Isolation Forest]` | PASS |
| `test_every_model_choice_renders[Autoencoder]` | PASS |
| `test_every_model_choice_renders[Mahalanobis detector]` | PASS |
| `test_live_demo_button_runs_and_raises_alert` | PASS |
| `test_all_dataset_files_exist_and_nonempty PASSED   ` | PASS |
| `test_users_and_labels_cover_same_users PASSED      ` | PASS |
| `test_exactly_the_injected_insiders_are_labelled` | PASS |
| `test_feature_table_has_no_labels_or_nans PASSED    ` | PASS |
| `test_no_real_personal_data_markers PASSED          ` | PASS |
| `test_real_shap_is_additive PASSED         ` | PASS |
| `test_shap_explains_after_hours_insider_with_logon_features` | PASS |
| `test_shap_explains_exfil_email_insider_with_email_features` | PASS |
| `test_explain_engines_both_work_and_agree_on_sign_of_top_feature` | PASS |
| `test_lime_returns_one_weight_per_feature` | PASS |
| `test_graph_has_all_users_and_resources` | PASS |
| `test_pyvis_html_is_self_contained_and_marks_insiders` | PASS |
| `test_attack_injection_spikes_risk_score PASSED` | PASS |
| `test_keras_autoencoder_flags_off_manifold_points` | PASS |
| `test_real_models_rank_obvious_outliers_highest[SklearnIsolationForest]` | PASS |
| `test_real_models_rank_obvious_outliers_highest[SklearnOneClassSVM]` | PASS |
| `test_roc_auc_and_precision_at_k_helpers PASSED   ` | PASS |
| `test_every_dashboard_model_detects_insiders_on_committed_dataset[Isolation Forest (sklearn)]` | PASS |
| `test_every_dashboard_model_detects_insiders_on_committed_dataset[One-Class SVM (sklearn)]` | PASS |
| `test_every_dashboard_model_detects_insiders_on_committed_dataset[Autoencoder (Keras)]` | PASS |
| `test_every_dashboard_model_detects_insiders_on_committed_dataset[Isolation Forest]` | PASS |
| `test_every_dashboard_model_detects_insiders_on_committed_dataset[Autoencoder]` | PASS |
| `test_every_dashboard_model_detects_insiders_on_committed_dataset[Mahalanobis detector]` | PASS |
| `test_models_do_not_use_labels PASSED             ` | PASS |
| `test_models_train_lazily PASSED             ` | PASS |
| `test_explanation_is_interactive_speed[Isolation Forest (sklearn)]` | PASS |
| `test_explanation_is_interactive_speed[One-Class SVM (sklearn)]` | PASS |
| `test_explanation_is_interactive_speed[Autoencoder (Keras)]` | PASS |
| `test_explanation_is_interactive_speed[Isolation Forest]` | PASS |
| `test_explanation_is_interactive_speed[Autoencoder]` | PASS |
| `test_explanation_is_interactive_speed[Mahalanobis detector]` | PASS |
| `test_whole_org_graph_is_sparse_enough_to_read` | PASS |
| `test_whole_org_graph_labels_only_risky_users` | PASS |
| `test_pipeline_is_reproducible_for_a_seed PASSED` | PASS |
| `test_different_seeds_give_different_data PASSED` | PASS |
| `test_injection_creates_expected_number_of_insiders` | PASS |
| `test_after_hours_injection_changes_after_hours_feature` | PASS |

## Known limitations (be upfront about these in the viva / report)

1. **Synthetic data.** Results show the method works on simulated insiders, not that it
   catches real ones. The CERT r4.2 dataset (proposal 5.2, optional) is not included.
2. **Unsupervised, transductive scoring.** Models are fitted and scored on the same
   population (normal for anomaly detection). Hyper-parameters were chosen on the original
   100-user dataset, then validated on unseen seeds (101-108) and, for the autoencoder fix,
   on seeds 201-205.
3. **Low-and-slow attacks are harder.** Every detector degrades as the attack is weakened
   (see `docs/robust_evaluation.md`, experiment 2); the weakened attacks are produced by
   shrinking feature vectors toward the mean, not by re-simulating logs.
4. **E-mail "NLP" features are recipient metadata only** (count, max, distinct recipients) -
   the simulated e-mail log has no subject/body text.
5. **Per-user, whole-period features.** The risk score is per user over the whole window, not a
   streaming per-event detector (the Live Demo recomputes features day by day to approximate this).
6. **The scratch autoencoder is the weakest model** (AUC 0.805); the Keras one is the default.
