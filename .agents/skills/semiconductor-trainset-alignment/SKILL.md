---
name: semiconductor-trainset-alignment
description: Project-local skill for `/vepfs-mlp2/mlp-public/250259/naura` and GitHub repo `hiude957/naura_GRU_0318-0427`. Explain the background, data semantics, merge logic, aligned-TSV contract, cache preparation flow, target semantics, GRU training policy, evaluation rules, project code architecture, experiment-record workflow, uv environment constraints, and Git/GitHub management rules for this semiconductor APC/LiveData/log project. Use only for this project when answering what the project is, how APC and livedata are merged, how log actions are inserted, what evt/state mean, how source and source_code work, how prepare_dataset builds cache arrays and targets, how to organize code/scripts/configs/models/outputs, how to record tuning results in Markdown, how to train/evaluate the GRU model on the configured A100 80G environment, how uv cache/tmp paths must be set, and what files should or should not be committed.
---

# Semiconductor Trainset Alignment

Use this project-local skill only for the repository at `/vepfs-mlp2/mlp-public/250259/naura` and the GitHub project `hiude957/naura_GRU_0318-0427`. Do not use it for unrelated repositories or other semiconductor projects unless the user explicitly points to this repo.

Use this skill when the user wants a fast, accurate explanation of the semiconductor dataset-alignment project, the aligned TSV files, the repo's cache-preparation flow, the GRU training/evaluation policy, or the project environment and Git rules.

Default response order:

1. Summarize the project goal.
2. Explain the three input sources: `log`, `apc`, `livedata`.
3. Explain the merge flow: build `110ms` grid rows only inside `livedata` coverage for training/output, keep a hidden chronological action/state pass outside `livedata` coverage, then sensor projection/interpolation, action aggregation, and `evt_*` / `state_*`.
4. Explain the aligned TSV contract and what `upgrade_semiconductor_alignment.py` changes.
5. Explain how `sensor_proxy.prepare_dataset` turns aligned TSV into cache arrays, model inputs, and future targets.
6. Explain GRU training/evaluation policy when the user asks about model training, parameters, accuracy, or one-month rollout.
7. Explain project architecture and experiment Markdown workflow when the user asks where code, scripts, configs, outputs, models, or tuning records should live.
8. Explain uv/Git constraints when the user asks about dependency management, environment setup, commits, or GitHub upload.
9. Separate fixed merge rules from current adjustable policy rules.

Answer in Chinese by default. Keep field names, source names, and column names in their original English form.

Read references as needed:

- For project background, current environment examples, and dataset scale, read [references/project-background.md](references/project-background.md).
- For the fixed merge rules, read [references/merge-rules.md](references/merge-rules.md).
- For the current training-output policy and extensible `source` / `source_code`, read [references/training-policy.md](references/training-policy.md).
- For the current repo implementation from aligned TSV to cache arrays, read [references/repo-data-processing.md](references/repo-data-processing.md).
- For project code layout, script responsibilities, output locations, and Markdown experiment records, read [references/project-architecture.md](references/project-architecture.md).
- For GRU training recommendations, one-month closed-loop prediction, and accuracy rules, read [references/model-training-policy.md](references/model-training-policy.md).
- For uv cache/tmp paths, development hardware, and GitHub commit rules, read [references/environment-and-git.md](references/environment-and-git.md).

When answering:

- Start high-level, then add structure.
- Do not present `/home/lyh/naura/...` as a required deployment path on other servers. Treat it as the current environment example only.
- If asked "why", explain the modeling intent: each `110ms` grid row should represent current sensor values, grid-level actions, and all IO continuous states at the same time.
- If asked about days without `livedata`, no-sensor periods, or action outside `livedata` coverage, explain that final training/output rows are limited to `livedata` coverage. However, action outside the output range must still be processed first so `state_*` is correct when the next output row begins. These hidden action rows are dropped only after state inheritance is complete.
- If asked about `20260318`, state that it is the current dataset's boot/start day, but training/output starts from the first `livedata` coverage at `2026-03-18 11:14:23.321`. Before that point, initialize `state_*` from `action_default.xlsx` and process any hidden action updates if present; do not emit pre-`livedata` training rows.
- If asked about empty days, state that no log/sensor does not imply shutdown; without explicit reset evidence, `state_*` continues across that day, but a day without `livedata` emits no training/output rows.
- If asked which rules may still change, keep `110ms` grid spacing, sensor interpolation priority inside `livedata` coverage, and action/state aggregation as fixed rules. Present final row filtering, long-gap handling, and training-output sampling/compression policy as adjustable areas.
- If asked about `prepare_dataset`, `cache`, `target_delta`, `target_log_dt`, `valid_target`, or input features, anchor the answer in `references/repo-data-processing.md` and keep the explanation consistent with the current code.
- If asked about repository structure, script placement, output paths, model files, or tuning records, anchor the answer in `references/project-architecture.md`; keep core logic under `src/naura_gru/` and scripts as thin entrypoints.
- If asked about model training, state that the current target model is GRU and anchor parameter suggestions in `references/model-training-policy.md`.
- If asked about dependency installation, use `uv`, set `UV_CACHE_DIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-cache` and `TMPDIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-tmp`, and do not modify uv source or mirror settings.
- If asked about GitHub upload, keep data files, generated training data, caches, checkpoints, logs, and temporary outputs out of Git; commit only code, config, docs, and reproducibility files such as `uv.lock`.
