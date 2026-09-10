# Task 3 — final script placement

Generated from the layout on disk after the move, so this reflects what actually
happened (including four corrections found during verification).

**Core: 12** · **Experiments: 98**

## Corrections made during the move

| Script | Was | Now | Why |
|---|---|---|---|
| `multilag_estimator.py` | core | `research/experiments/ch2_connectivity/` | imports `wrapup_run` (an experiment) — core may not depend on experiments |
| `spectral_analysis.py` | core | `research/experiments/ch2_connectivity/` | loads a hardcoded thesis result path at module level |
| `network_stats.py` | core | `research/experiments/shared/` | imports `make_panel`, which you kept out of the core |
| `wrapup_run.py` | core | `research/experiments/shared/` | used by BOTH experiment groups — promoted to shared/ |

## Core — the installable tool

| Module | Import as |
|---|---|
| `src/calcinet/connectivity/analyze_run.py` | `calcinet.connectivity.analyze_run` |
| `src/calcinet/connectivity/decision_analysis.py` | `calcinet.connectivity.decision_analysis` |
| `src/calcinet/connectivity/metrics_report.py` | `calcinet.connectivity.metrics_report` |
| `src/calcinet/connectivity/separation_diagnostics.py` | `calcinet.connectivity.separation_diagnostics` |
| `src/calcinet/connectivity/solve_from_cached.py` | `calcinet.connectivity.solve_from_cached` |
| `src/calcinet/connectivity/solver_comparison.py` | `calcinet.connectivity.solver_comparison` |
| `src/calcinet/io/hawkes_to_moments.py` | `calcinet.io.hawkes_to_moments` |
| `src/calcinet/io/pool_chunks_and_solve.py` | `calcinet.io.pool_chunks_and_solve` |
| `src/calcinet/ledger_cli.py` | `calcinet.ledger_cli` |
| `src/calcinet/simulation/hawkes_ground_truth.py` | `calcinet.simulation.hawkes_ground_truth` |
| `src/calcinet/simulation/ou_linear_ground_truth.py` | `calcinet.simulation.ou_linear_ground_truth` |
| `src/calcinet/simulation/sanity_check.py` | `calcinet.simulation.sanity_check` |

## research/experiments/shared/ (7)

| Script | Docstring |
|---|---|
| `figstyle.py` | Shared figure style for the results-section figures. |
| `make_panel.py` | Compact diagnostic panel for one estimation. |
| `matrix_views.py` | Four views of the connectivity matrix (ground truth vs estimated) for on |
| `network_stats.py` | Network diagnostics for one dataset: firing rates + raster + calcium tra |
| `plot_sweeps.py` | Aggregate one parameter sweep into metric-vs-parameter curves (mean ± st |
| `r2_metrics.py` | R.2 connectivity metric set — shared by fig_r2_compute.py (score at swee |
| `wrapup_run.py` | Wrap-up: local run (small net) that produces the estimates the figures n |

## research/experiments/ch1_proxy/ (11)

| Script | Docstring |
|---|---|
| `compare_preprocessing.py` | Compare OLS on raw calcium vs OLS on preprocessed feed. |
| `fig_preproc_compute.py` | Effect of preprocessing — COMPUTE (refresh at scale, consistent metrics) |
| `fig_preproc_plot.py` | Effect of preprocessing — PLOT (local), consistent style. |
| `fig_r2_camera_longT_compare.py` | R.2 follow-up: overlay the original T=500k-ms camera-rate panel against  |
| `fig_r2_compute.py` | R.2 (calcium-observation section) — COMPUTE stage. |
| `fig_r2_longT.py` | R.2 follow-up: at a FIXED, realistic camera interval, does more recordin |
| `fig_r2_phase_avg.py` | R.2 follow-up: does averaging over camera START PHASE recover any of the |
| `fig_r2_plot.py` | R.2 (calcium-observation section) — PLOT stage, runs LOCALLY. |
| `fig_r2_rescore.py` | R.2 (calcium-observation section) — RESCORE stage. Pure numpy, runs LOCA |
| `fig_r2_smooth_compute.py` | R.2 follow-up — does TUNING the deconvolution's own smoothing window let |
| `fig_r2_smooth_plot.py` | R.2 follow-up — PLOT. Does tuning the deconvolution's own smoothing wind |

## research/experiments/ch2_connectivity/ (80)

| Script | Docstring |
|---|---|
| `check_cached_regime.py` | Measure rate / ISI-CV / synchrony from the ACTUAL cached ground-truth sp |
| `combine_test.py` | Regularize-then-rescale — does the order break the detection/magnitude t |
| `compare_solvers.py` | Compare chunked OLS (numpy normal equations) vs sklearn OLS (LinearRegre |
| `dale_candidates_test.py` | Dale-regularization candidates — head-to-head on the N=100 feed. |
| `dale_reg_test.py` | Dale-as-regularization test — enforce single-sign columns DURING the fit |
| `data_size_test.py` | Data-size test — is the ceiling caused by too little data (variance) or  |
| `fig_best_compute.py` | Best-results showcase — COMPUTE stage (cluster). |
| `fig_best_n1250r4_entries2.py` | Trimmed version of fig_best_plot.py's "sample neurons" panel: ONE excita |
| `fig_best_n1250r4_signrank.py` | Sign reliability vs. magnitude, for the OLS estimate on the N=1250 ladde |
| `fig_best_plot.py` | Best-results showcase — PLOT (local). |
| `fig_brunel_regime_comparison.py` | Memory-safe version of regime_probe.py's multi-config comparison figure. |
| `fig_chunking_cpu_vs_gpu.py` | Schematic — the INNER chunking axis: how one node streams its own T/K-ms |
| `fig_hawkes_validation.py` | Pernice et al. 2011 Figure 2-style validation panel for our Hawkes groun |
| `fig_linearity_overall_metrics.py` | Overall connectivity-recovery performance, as a figure -- the companion  |
| `fig_linearity_way2.py` | Way 2: does false-positive strength still grow with shared-driver exposu |
| `fig_linearity_way2_rate.py` | Same three-way (LIF/PIF/OU) linearity comparison as fig_linearity_way2.p |
| `fig_mcc_scaling_handoff.py` | STOPGAP script -- hand-transcribed from a companion analysis chat's data |
| `fig_mcc_table.py` | MCC across network size and dynamical regime — table + heatmap, local. |
| `fig_parallel_pooling_schematic.py` | Schematic — how the pipeline runs today (one node, one long recording) v |
| `fig_r1_compute.py` | R.1 (method-demonstration panel) — COMPUTE stage, runs ON THE CLUSTER. |
| `fig_r1_confusion_options.py` | R.1 confusion matrix — NORMALIZATION OPTIONS (exploratory, local). |
| `fig_r1_plot.py` | R.1 (method-demonstration panel) — PLOT stage, runs LOCALLY. |
| `fig_r3_compute.py` | R.3 (lag & delay) — COMPUTE stage. |
| `fig_r3_plot.py` | R.3 (lag & delay) — PLOT stage, local. |
| `fig_r4_ladder_stability.py` | R.4 ladder stability figure: rate / CV(ISI) / synchrony held (near-)cons |
| `fig_r4_plot.py` | R.4 (data length x network size) — the scaling law, plotted LOCALLY. |
| `fig_r4ci_plot.py` | R.4b — fixed connection-PROBABILITY ladder (eps=0.1, C_E grows with N) v |
| `fig_r5_compute.py` | R.5 (regularization vs data) — COMPUTE stage. |
| `fig_r5_confusion.py` | R.5 confusion matrices — COMPUTE stage. |
| `fig_r5_confusion_plot.py` | R.5 confusion matrices — PLOT stage, local. |
| `fig_r5_plot.py` | R.5 (regularization vs data) — PLOT stage, local. |
| `fig_r6_plot.py` | R.6 (dynamical regime) — PLOT stage, local. |
| `fig_r7_compute.py` | R.7 (hidden neurons / shared input) — COMPUTE stage. |
| `fig_r7_plot.py` | R.7 (hidden neurons / shared input) — PLOT stage, local. |
| `fig_r7_plot_bynet.py` | R.7 (hidden neurons / shared input) — PLOT stage, ALTERNATE layout. |
| `fig_r8_compute.py` | R.8 (shared input vs directionality) — COMPUTE stage, core. |
| `fig_r8_compute_regularized.py` | R.8 (shared input vs directionality) — REGULARIZED-estimator COMPUTE sta |
| `fig_r8_counts.py` | R.8 (shared input vs directionality) — COUNT-based PLOT stage, local. |
| `fig_r8_plot.py` | R.8 (shared input vs directionality) — PLOT stage, local. |
| `fig_r8_violin.py` | R.8 (shared input vs directionality) — violin-only PLOT stage, local. |
| `fig_r8_violin_fpsplit.py` | R.8 (shared input vs directionality) — false-positive SPLIT violin, loca |
| `fig_r8b_plot.py` | R.8 phase 2 (multiple lags) — PLOT, local. |
| `fig_r8b_signal_counts.py` | Timing signature: spikes vs calcium — COUNT-based PLOT (local). |
| `fig_r8b_signal_plot.py` | Timing signature: spikes vs calcium — PLOT (local). |
| `fig_r8b_signal_violin.py` | Timing signature: spikes vs calcium — VIOLIN, count-based (local). |
| `fig_regime2d_plot.py` | 2-D regime probe — PLOT stage, local. |
| `fig_regime2d_ridge.py` | 2-D regime probe — summary panel. |
| `fig_two_axis_parallelism.py` | Schematic — BOTH chunking axes together: K nodes (outer, across-node --  |
| `fig_way2.py` | Way 2 — the correlational test: fake strength vs HIDDEN vs OBSERVED shar |
| `fig_way2_motifs.py` | Way 2 extended — motif-based exposure variables, following the Pernice ( |
| `fig_way3.py` | Way 3 — the causal "add the hidden driver back" test (local, N=1250). |
| `linearity_confound_attribution.py` | What fraction of an arm's false positives are actually explained by shar |
| `linearity_overall_metrics.py` | Overall connectivity-recovery performance for every arm in the linearity |
| `method_comparison_viz.py` | Method-comparison dashboard — three views across the pipeline solutions. |
| `method_dashboards.py` | Full per-method dashboards for every pipeline solution. |
| `methods_overview.py` | Master methods overview — every regularization / post-processing method  |
| `multilag_estimator.py` | Joint multi-lag estimator — does conditioning on SEVERAL past lags at on |
| `oracle_ladder.py` | Oracle-ladder diagnostic — where does the connectivity signal die? |
| `pif_tau_probe.py` | Eta probe for the PIF-pilot tau_m ladder (professor's comment on |
| `postprocess_test.py` | Post-processing test — can rescaling undo the inhibition bias? |
| `r4_probe_2d.py` | 2-D regime probe: can we hit a realistic rate AND textbook-AI irregulari |
| `r4_tune_regime.py` | R.4 regime tuning: hold the SAME regime (AI) and the SAME rate across ne |
| `r8b_multilag.py` | R.8 phase 2 (multiple lags) — the TIMING axis, measured on its own. |
| `recovery_curve.py` | Data-length recovery curve for the clean-AI regime (n1250ai): does more |
| `regime_compare.py` | Network-state comparison of the two N=1250 regimes: |
| `regime_probe.py` | Compare ground-truth network configurations at full Brunel scale (N=1250 |
| `regime_scan.py` | Regime scan at N=1250: sweep (J_ex, eta) to find a clean asynchronous-ir |
| `regularization_test.py` | Regularization test — can Elastic Net clean up the exact connections (pr |
| `run_ladder.py` | Stage 2 — the regularization ladder at the operating point. |
| `run_lasso_sweep.py` | Stage 2b — pure-Lasso vs Elastic-Net L1/L2 sweep at the N=1250 operating |
| `run_n1250.py` | First scale-up run on the cluster: the recommended pipeline at N=1250. |
| `run_n1250_experiments.py` | Parameter sweeps at N=1250: EN -> Dale -> balance, every stage scored, m |
| `run_n1250_group3.py` | Group 3 at N=1250: full recommended pipeline, scored stage by stage. |
| `run_n1250_sweep.py` | Groups 1 & 2 solver/strength comparison at N=1250 (SLURM job array). |
| `scaling_curve.py` | Scaling result: recovery vs recording length at N=1250 and N=12500 (cano |
| `spectral_analysis.py` | Spectral analysis of the estimated adjacency matrix. |
| `sweep_lag.py` | Sweep over lag_ms to find the effect of AR lag on connectivity inference |
| `unsup_rescale_test.py` | Unsupervised rescale on the regularized matrix — can we drop the oracle? |
| `wrapup_figures.py` | Wrap-up figures. Reads the per-seed estimates written by experiments/ch2 |
| `wrapup_run_stream.py` | Streaming data-length sweep: simulate ONE long recording per seed and in |
