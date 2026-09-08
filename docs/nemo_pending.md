# NEMO2 — pending actions (deferred until the local/GitHub restructure is done)

Everything cluster-related is parked here so it does not get lost. Nothing in
this list has been done yet.

## Time-critical

- [ ] **Extend the workspace before it expires.** The `calcium_ar` workspace is
      due to be deleted (bwHPC workspaces last max ~90 days). Losing it means
      losing `$WS/results`, `$WS/env`, `$WS/env-gpu`, `$WS/gt_cache`, the
      checkout and the SLURM logs.
      ```bash
      ws_list                        # check remaining days
      ws_extend calcium_ar 30        # extend if still possible
      ```

## Bring down before the workspace disappears

- [ ] **SLURM logs** — wanted in the repo as evidence of the long runs. The local
      `slurm_logs/` is currently EMPTY; the logs only exist on the cluster.
      ```bash
      rsync -avhz fr_mj200@login1.nemo.uni-freiburg.de:'$(ws_find calcium_ar)/Calcium---AR/slurm_logs/' \
            slurm_logs/
      ```
      Then un-ignore them: `.gitignore` currently has `slurm_logs/*`, which must
      be relaxed so the logs are tracked.
- [ ] **Ledger files** from `$WS/results/*/ledger.csv` (small text, the run record).
- [ ] Anything else under `$WS/results` still needed for the thesis.

## After the restructure lands

- [ ] `git pull` in the cluster checkout. A plain pull is enough while the
      checkout folder keeps its current name — git renames folder *contents*,
      not the folder.
- [ ] Re-point the SLURM scripts if Task 3 moved the script paths
      (`scripts/foo.py` -> `experiments/...`).
- [ ] Run one job end to end to confirm nothing broke.

## Deliberately NOT changing

- **The workspace name `calcium_ar`** (`ws_find calcium_ar`). It is not stored in
  git, renaming it would mean rebuilding both conda environments (conda bakes in
  absolute paths) and copying all results. The workspace is being retired anyway.
