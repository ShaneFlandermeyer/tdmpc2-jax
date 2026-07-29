# Dense-RHS Experiment Decisions

This file is append-only campaign memory for `goal-experiment-steward`.

## Bootstrap State

- `1179` is the current clean no-RHS quadruped baseline: `924.65 ± 16.91`, runtime `01:15:58`.
- `1387` is the current clean Dense-RHS winner: `915.34 ± 24.33`, runtime about `01:29:59`, horizon path `2 -> 2 -> 3 -> 2 -> 3 -> 4`.
- `1600` credible transition with `p=.65,c=.06` stayed at `h=3` and underperformed; it was too conservative.
- `1618` credible transition with `p=.56,c=.03` moved to `h=4` but later switched to `h=5`, bucketed to `8`, and collapsed final eval despite good late training episodes.

## Open RFCs

- `expected_improvement_transition`: replace hard switch probability threshold with expected net benefit so the model can accept moderate but valuable transitions and reject high-risk late switches.
- `cap_or_uncertain_late_large_horizon`: avoid late large-horizon moves unless uncertainty-adjusted evidence is strong; this must be written as a general uncertainty rule, not a task-specific schedule.

## 2026-05-01 Expected-Improvement Transition RFC

Implemented `expected_improvement_transition` behind config flags:

- `dense_rhs.credible_transition_rule=expected_improvement`
- `dense_rhs.transition_risk_weight`
- `dense_rhs.transition_min_expected_net`

Decision equation:

`X_h = U(h) - U(h_current) - C(h, h_current)`

`EI_h = E[max(X_h, 0)]`

`EL_h = E[max(-X_h, 0)]`

`B_h = EI_h - transition_risk_weight * EL_h`

The proposed horizon is selected by `B_h` over the candidate set, and the switch is accepted when `B_h > transition_min_expected_net`. This keeps the transition rule Bayesian but avoids the brittle hard switch-probability threshold that made `1600` too conservative and `1618` too permissive late in training.

Added one smoke profile: `dense_rhs_ei_smoke_80k_vec8_s15_risk15_c003_start20k`. It must not launch until the code is committed/synced because this is not marked as a dirty smoke.

## 2026-05-01T10:15:53+00:00

Launched `dense_rhs_ei_smoke_80k_vec8_s15_risk15_c003_start20k` as job `1644`.

## 2026-05-01T14:29:12+00:00

Processed `dense_rhs_ei_smoke_80k_vec8_s15_risk15_c003_start20k` job `1644`: `ei_too_conservative`.

- SLURM state: `COMPLETED`
- Runtime: `00:39:23`
- Final eval: `241.28457641601562` ± `123.68336486816406`
- Best eval: `241.28457641601562` at step `80000`
- Horizon path: `3->3->3`
- Reason: EI smoke stayed at h=3 with non-positive expected net benefit at every query; lower risk/cost should be tested before a full run.
- Follow-up: Queued follow-up `dense_rhs_ei_smoke_80k_vec8_s15_risk05_c002_start20k`.

## 2026-05-01T14:30:11+00:00

Launched `dense_rhs_ei_smoke_80k_vec8_s15_risk05_c002_start20k` as job `1655`.

## 2026-05-01T15:18:52+00:00

Processed `dense_rhs_ei_smoke_80k_vec8_s15_risk05_c002_start20k` job `1655`: `weak_smoke`.

- SLURM state: `COMPLETED`
- Runtime: `00:41:16`
- Final eval: `286.9360656738281` ± `150.44711303710938`
- Best eval: `286.9360656738281` at step `80000`
- Horizon path: `2->2->2`
- Reason: EI smoke best eval is low at `286.9360656738281`; needs a bounded parameter iteration, not a full run.
- Follow-up: Queued follow-up `dense_rhs_ei_smoke_80k_vec8_s15_risk10_c002_min010_start20k`.

## 2026-05-01T15:19:24+00:00

Launched `dense_rhs_ei_smoke_80k_vec8_s15_risk10_c002_min010_start20k` as job `1658`.

## 2026-05-01T16:12:26+00:00

Processed `dense_rhs_ei_smoke_80k_vec8_s15_risk10_c002_min010_start20k` job `1658`: `weak_smoke`.

- SLURM state: `COMPLETED`
- Runtime: `00:43:00`
- Final eval: `463.6806945800781` ± `129.16029357910156`
- Best eval: `463.6806945800781` at step `80000`
- Horizon path: `3->3->3`
- Reason: EI smoke best eval is low at `463.6806945800781`; needs a bounded parameter iteration, not a full run.
- Follow-up: Follow-up `dense_rhs_ei_smoke_80k_vec8_s15_risk10_c002_min010_start20k` already exists.

## 2026-05-01 Sparse-HiFi Fallback RFC

The expected-improvement transition family has now failed the high-risk, low-risk, and midpoint 80k smokes. The failure pattern is not enough to reject Dense-RHS, because the existing Sparse-HiFi family already reached `915.34` at 300k. The next bounded hypothesis is therefore `dense_rhs_sparse_hifi_smoke_120k_vec8_s15_start20k`: disable EI/credible transition gating, keep local-window bucketed Sparse-HiFi search, and run a 120k smoke with eval every 40k. If this recovers the known learning trajectory, the campaign should use Sparse-HiFi as the base family for the next clean/chaos comparisons instead of continuing EI threshold tuning.

## 2026-05-01T16:17:59+00:00

Launched `dense_rhs_sparse_hifi_smoke_120k_vec8_s15_start20k` as job `1659`.

## 2026-05-01T17:22:26+00:00

Processed `dense_rhs_sparse_hifi_smoke_120k_vec8_s15_start20k` job `1659`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `00:56:30`
- Final eval: `640.4043579101562` ± `58.951446533203125`
- Best eval: `640.4043579101562` at step `120000`
- Horizon path: `2->2`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-04T15:10:00+00:00

Promoted MJX environment ports from passive `pending_port` markers to first-class steward gates.

- Quadruped-run remains `ready`; its single-seed clean and chaos no-RHS/Dense-RHS table cells are filled.
- The next campaign blocker is `fish-swim`, status `pending_implementation`.
- Each non-quadruped environment now requires a gate report, rollout parity check, chaos-semantics check, and short no-RHS learning smoke before training profiles can launch.
- `goalctl.py` now reports MJX gate state in `status` and refuses launches for environments whose MJX gate is not `ready` or `passed`.

## 2026-05-01 Sparse-HiFi Chaos Smoke RFC

Clean Dense-RHS does not need more EI transition tuning right now: run `1387` already satisfies the `0.98 * no_rhs` clean non-inferiority gate, and the new Sparse-HiFi fallback recovered to `640.40` by 120k after weak early evals. The next campaign step is therefore quadruped chaos. Queue `dense_rhs_sparse_hifi_chaos_smoke_120k_vec8_s15_start20k` using the chaos launcher, clean evaluation, and the same sparse high-fidelity query budget. If this smoke is non-degenerate, promote the same family to a 300k chaos comparison against the no-RHS chaos reference.

## 2026-05-01T17:23:49+00:00

Launched `dense_rhs_sparse_hifi_chaos_smoke_120k_vec8_s15_start20k` as job `1661`.

## 2026-05-01 Clean +2% Goal Correction

The formal campaign target is corrected from clean non-inferiority to clean improvement: Dense-RHS must beat the clean no-RHS reference by `2%`. With no-RHS run `1179` at `924.65`, the clean target is `943.14`. Current best Dense-RHS run `1387` reached `915.34`, so it remains the best-so-far but does not satisfy the goal. Chaos validation is postponed until this clean +2% gate is met.

Queued two clean 300k attempts to keep the max-2-GPU policy saturated:

- `dense_rhs_plus2_sparse_hifi_margin08_300k_vec8_s15_start20k`: Sparse-HiFi with stronger incumbent margin (`0.08`) to reduce low-evidence early downshifts.
- `dense_rhs_plus2_sparse_hifi_start70_300k_vec8_s15`: Sparse-HiFi with a moderate `70k` fixed-horizon warmup before free adaptive Dense-RHS.

## 2026-05-01T17:31:25+00:00

Processed `dense_rhs_sparse_hifi_chaos_smoke_120k_vec8_s15_start20k` job `1661`: `postponed_clean_plus2`.

- SLURM state: `CANCELLED by 1002`
- Runtime: `00:07:22`
- Final eval: `None` ± `None`
- Best eval: `None` at step `None`
- Horizon path: ``
- Reason: Chaos validation postponed until clean Dense-RHS beats the no-RHS baseline by `2%`.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-01T17:34:51+00:00

Launched `dense_rhs_plus2_sparse_hifi_margin08_300k_vec8_s15_start20k` as job `1662`.

## 2026-05-01T17:35:18+00:00

Launched `dense_rhs_plus2_sparse_hifi_start70_300k_vec8_s15` as job `1663`.

## 2026-05-01T19:10:57+00:00

Processed `dense_rhs_plus2_sparse_hifi_margin08_300k_vec8_s15_start20k` job `1662`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:31:16`
- Final eval: `901.0157470703125` ± `17.160640716552734`
- Best eval: `901.0157470703125` at step `300000`
- Horizon path: `2->2->3->4->4->4`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-01T19:10:59+00:00

Processed `dense_rhs_plus2_sparse_hifi_start70_300k_vec8_s15` job `1663`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:31:18`
- Final eval: `922.63330078125` ± `21.33823585510254`
- Best eval: `922.63330078125` at step `300000`
- Horizon path: `3->3->3->3->3`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-01 Clean +2% Follow-Up RFC

The two parallel clean +2% attempts completed without reaching the target `943.14`.

- `dense_rhs_plus2_sparse_hifi_margin08_300k_vec8_s15_start20k`: `901.02 ± 17.16`, horizon path `2->2->3->4->4->4`.
- `dense_rhs_plus2_sparse_hifi_start70_300k_vec8_s15`: `922.63 ± 21.34`, horizon path `3->3->3->3->3`.

Interpretation: early low-horizon adaptation still hurts, while the `70k` warmup profile recovers near the no-RHS baseline but does not exploit enough to pass the `+2%` target. Queue two clean attempts to keep the max-2-GPU policy saturated:

- `dense_rhs_plus2_sparse_hifi_start120_margin03_300k_vec8_s15`: later `120k` adaptation with small margin, testing whether safer late adaptation can improve over fixed `h=3`.
- `dense_rhs_plus2_sparse_hifi_start70_returnlite_300k_vec8_s15`: same `70k` warmup with mildly return-dominant deployment, testing whether the search can exploit a better local horizon without aggressive early switching.

## 2026-05-01 Clean +2% Follow-Up RFC E

`dense_rhs_plus2_sparse_hifi_start120_margin03_300k_vec8_s15` completed at `831.64`, so delaying the first query to `120k` is not enough and likely leaves too little useful adaptation budget. One GPU is free while `dense_rhs_plus2_sparse_hifi_start70_returnlite_300k_vec8_s15` continues.

Queue `dense_rhs_plus2_sparse_hifi_start70_returndom_r2b5_300k_vec8_s15`: keep the stronger `70k` warmup, widen the local candidate window to radius `2`, evaluate `5` candidates, and make deployment return-dominant by setting roughness and return-std deployment weights to `0`. This tests whether the stable start-70 profile can exploit better local horizons when the query has richer local evidence.

## 2026-05-01 Clean +2% Follow-Up RFC F

`dense_rhs_plus2_sparse_hifi_start70_returnlite_300k_vec8_s15` completed at `855.80`, so mild return-dominance with a narrow local window degraded performance rather than improving over the stable `h=3` path.

Queue `dense_rhs_plus2_sparse_hifi_start20_r2b5_margin05_300k_vec8_s15` as a different hypothesis from the pending return-dominant run: start adaptation early as in the best-so-far Sparse-HiFi family, but widen the high-fidelity local window to radius `2` and evaluate `5` candidates while keeping the standard geometric deployment score and a moderate incumbent margin. This tests whether the original early-adaptive family was under-informed rather than structurally wrong.

## 2026-05-01 Evidence-Quality Hypothesis Queue

The next campaign direction should improve the evidence Dense-RHS uses, not only the transition heuristic. Add two runnable evidence-quality full runs after the active `1672/1673` jobs:

- `dense_rhs_plus2_evidence_ultrahifi_start70_300k_vec8_s15`: safe start-70 profile with full training-planner query budget (`512/24/64/6`), `32` replicas, and `512` query eval steps.
- `dense_rhs_plus2_evidence_ultrahifi_returnlocal_300k_vec8_s15`: same ultra-HiFi query budget with radius-2 local evidence and mildly return-dominant deployment.

Also add non-launchable RFCs for the evidence mechanisms that need code before they can be tested:

- learner-aware virtual update query,
- shadow horizon training statistics,
- posterior over observed deployment utility,
- paired fixed-seed query evaluation verification.

The steward should exhaust runnable ultra-HiFi evidence profiles first. If they fail and no launch profile is pending, the next steward action should be a small RFC-style implementation patch rather than another transition-threshold sweep.

## 2026-05-01T19:12:23+00:00

Launched `dense_rhs_plus2_sparse_hifi_start120_margin03_300k_vec8_s15` as job `1670`.

## 2026-05-01T19:12:39+00:00

Launched `dense_rhs_plus2_sparse_hifi_start70_returnlite_300k_vec8_s15` as job `1671`.

## 2026-05-01T20:47:30+00:00

Processed `dense_rhs_plus2_sparse_hifi_start120_margin03_300k_vec8_s15` job `1670`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:34:51`
- Final eval: `831.63916015625` ± `21.24997901916504`
- Best eval: `831.63916015625` at step `300000`
- Horizon path: `2->2->3->3`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-01T20:48:56+00:00

Processed `dense_rhs_plus2_sparse_hifi_start70_returnlite_300k_vec8_s15` job `1671`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:35:10`
- Final eval: `855.7972412109375` ± `17.007858276367188`
- Best eval: `855.7972412109375` at step `300000`
- Horizon path: `3->2->3->2->3`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-01T20:49:50+00:00

Launched `dense_rhs_plus2_sparse_hifi_start70_returndom_r2b5_300k_vec8_s15` as job `1672`.

## 2026-05-01T20:50:15+00:00

Launched `dense_rhs_plus2_sparse_hifi_start20_r2b5_margin05_300k_vec8_s15` as job `1673`.

## 2026-05-01 Three-GPU Evidence Search

Cancelled active jobs `1672` and `1673` because they were still local-window/transition variants rather than the evidence-quality direction needed to beat the clean no-RHS baseline by `2%`.

Update the campaign cap from `2` to `3` active GPUs and run three clean evidence-quality attempts when capacity is free:

- `dense_rhs_plus2_evidence_ultrahifi_start70_300k_vec8_s15`: safe start-70 with ultra-HiFi query evidence.
- `dense_rhs_plus2_evidence_ultrahifi_returnlocal_300k_vec8_s15`: ultra-HiFi query evidence with wider local window and mildly return-dominant deployment.
- `dense_rhs_plus2_evidence_ultrahifi_start20_300k_vec8_s15`: early start-20 with ultra-HiFi evidence to test whether early adaptation failed because evidence was too noisy.

The clean +2% target remains unchanged at `943.143`; chaos remains postponed until this gate is passed.

## 2026-05-01T21:34:54+00:00

Processed `dense_rhs_plus2_sparse_hifi_start70_returndom_r2b5_300k_vec8_s15` job `1672`: `failed`.

- SLURM state: `CANCELLED by 1002`
- Runtime: `00:44:33`
- Final eval: `97.265380859375` ± `141.6887664794922`
- Best eval: `97.265380859375` at step `50000`
- Horizon path: `4`
- Reason: SLURM state `CANCELLED by 1002`.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-01T21:34:56+00:00

Processed `dense_rhs_plus2_sparse_hifi_start20_r2b5_margin05_300k_vec8_s15` job `1673`: `failed`.

- SLURM state: `CANCELLED by 1002`
- Runtime: `00:44:08`
- Final eval: `262.2226867675781` ± `118.21903228759766`
- Best eval: `262.2226867675781` at step `50000`
- Horizon path: `3->4`
- Reason: SLURM state `CANCELLED by 1002`.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-01T21:35:21+00:00

Launched `dense_rhs_plus2_evidence_ultrahifi_start70_300k_vec8_s15` as job `1674`.

## 2026-05-01T21:35:41+00:00

Launched `dense_rhs_plus2_evidence_ultrahifi_returnlocal_300k_vec8_s15` as job `1675`.

## 2026-05-01T21:36:01+00:00

Launched `dense_rhs_plus2_evidence_ultrahifi_start20_300k_vec8_s15` as job `1676`.

## 2026-05-01T23:38:27+00:00

Processed `dense_rhs_plus2_evidence_ultrahifi_start70_300k_vec8_s15` job `1674`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `02:00:12`
- Final eval: `884.75537109375` ± `28.268857955932617`
- Best eval: `884.75537109375` at step `300000`
- Horizon path: `3->4->5->5->6`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-01T23:38:29+00:00

Processed `dense_rhs_plus2_evidence_ultrahifi_start20_300k_vec8_s15` job `1676`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:56:26`
- Final eval: `849.2491455078125` ± `15.82135009765625`
- Best eval: `849.2491455078125` at step `300000`
- Horizon path: `3->3->4->5->5->4`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-01T23:53:59+00:00

Processed `dense_rhs_plus2_evidence_ultrahifi_returnlocal_300k_vec8_s15` job `1675`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `02:07:45`
- Final eval: `729.8106079101562` ± `17.057329177856445`
- Best eval: `740.24560546875` at step `250000`
- Horizon path: `3->4->5->6->6`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-02T00:10:00+00:00

Implemented RFC `shadow_horizon_training_stats` as a bounded Dense-RHS evidence patch.

- Equation: deployment utility is now optionally multiplied by `u_h^w`, where `u_h = norm_inv(L_probe(h) / h)` and `L_probe(h)` is the dense replay-prefix dynamics+reward probe loss.
- Config flags: `dense_rhs.learner_proxy_enabled`, `dense_rhs.learner_proxy_weight`, and `dense_rhs.learner_proxy_mode`.
- Expected benefit: discourage horizon switches that look good in short candidate env rollouts but have worse learner-facing model/reward prefix evidence.
- Expected failure mode: if the proxy is too strong, it may collapse back to fixed `h=3`; the dispatcher therefore launches bounded weights `0.25`, `0.5`, and `1.0`.
- Cleanup condition: remove or demote the proxy if all shadow-probe smokes underperform Sparse-HiFi without improving horizon stability.

## 2026-05-02T00:40:00+00:00

Recorded manual GPU-freeing intervention.

- Job `1688` was manually cancelled to free a GPU for other work.
- The campaign cap is reduced from `3` to `2` active steward GPUs so future ticks do not immediately refill the freed slot.
- The active shadow-probe jobs `1686` and `1687` continue to provide bounded evidence for weights `0.25` and `0.5`.
- Raise `constraints.max_active_gpus` explicitly if the steward should use three GPUs again.

## 2026-05-02T00:50:00+00:00

Added final reporting and clean/chaos comparison deliverables to the steward.

- The campaign must generate a detailed `dense_rhs_decision_report.tex` once final gates complete.
- The report must list every trained model, run artifacts, summary plots, score/runtime/horizon results, and the metric or plot evidence behind architecture decisions.
- `package-results` now writes model ranking and clean-vs-chaos comparison tables in addition to the score table and winning-algorithm report.
- After the clean +2% threshold is passed and chaos jobs are run, every scheduled environment must compare clean and chaos results for both no-RHS baseline and the winning Dense-RHS model family.

## 2026-05-02T08:50:00+00:00

Closed the failed shadow-probe branch and queued two clean +2 anchor retries.

- `shadow_horizon_training_stats` is marked `shadow_probe_failed`: the 120k smokes reached only `684.22`, `578.20`, and the promoted 300k run collapsed to `68.87` by 100k.
- The next queued profiles return to the known Sparse-HiFi family, where the best run was close to baseline at `922.63`.
- `dense_rhs_plus2_anchor_start70_margin10_300k_vec8_s15` tests the stable start-70 profile with a stricter incumbent margin `0.10`.
- `dense_rhs_plus2_anchor_start90_margin06_300k_vec8_s15` tests a longer fixed-horizon warmup to protect early learning, then a small local adaptive search.
- The steward cap remains `2` to preserve the manually freed GPU slot until explicitly changed.

## 2026-05-02T00:11:33+00:00

RFC dispatcher queued shadow-probe smokes: dense_rhs_plus2_shadowprobe_w0p25_120k_vec8_s15_start20k, dense_rhs_plus2_shadowprobe_w0p5_120k_vec8_s15_start20k, dense_rhs_plus2_shadowprobe_w1p0_120k_vec8_s15_start20k.

## 2026-05-02T00:11:38+00:00

Launched `dense_rhs_plus2_shadowprobe_w0p25_120k_vec8_s15_start20k` as job `1686`.

## 2026-05-02T00:12:05+00:00

Launched `dense_rhs_plus2_shadowprobe_w0p5_120k_vec8_s15_start20k` as job `1687`.

## 2026-05-02T00:12:26+00:00

Launched `dense_rhs_plus2_shadowprobe_w1p0_120k_vec8_s15_start20k` as job `1688`.

## 2026-05-02T00:47:06+00:00

Processed `dense_rhs_plus2_shadowprobe_w1p0_120k_vec8_s15_start20k` job `1688`: `failed`.

- SLURM state: `CANCELLED by 1002`
- Runtime: `00:30:43`
- Final eval: `274.5364074707031` ± `156.55780029296875`
- Best eval: `274.5364074707031` at step `40000`
- Horizon path: `3`
- Reason: SLURM state `CANCELLED by 1002`.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-02T01:02:01+00:00

Processed `dense_rhs_plus2_shadowprobe_w0p25_120k_vec8_s15_start20k` job `1686`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `00:48:44`
- Final eval: `684.2154541015625` ± `26.09765625`
- Best eval: `684.2154541015625` at step `120000`
- Horizon path: `3->3`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-02T01:02:03+00:00

Processed `dense_rhs_plus2_shadowprobe_w0p5_120k_vec8_s15_start20k` job `1687`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `00:47:00`
- Final eval: `578.2000732421875` ± `16.3250789642334`
- Best eval: `578.2000732421875` at step `120000`
- Horizon path: `2->2`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-02T01:02:05+00:00

Promoted best shadow-probe smoke `dense_rhs_plus2_shadowprobe_w0p25_120k_vec8_s15_start20k` (best_eval=684.22, job=1686) to full profile `dense_rhs_plus2_shadowprobe_w0p25_300k_vec8_s15_start20k`.

## 2026-05-02T01:02:09+00:00

Launched `dense_rhs_plus2_shadowprobe_w0p25_300k_vec8_s15_start20k` as job `1691`.

## 2026-05-02T01:56:12+00:00

Processed `dense_rhs_plus2_shadowprobe_w0p25_300k_vec8_s15_start20k` job `1691`: `failed`.

- SLURM state: `CANCELLED by 1002`
- Runtime: `00:53:46`
- Final eval: `68.86866760253906` ± `175.0941619873047`
- Best eval: `68.86866760253906` at step `100000`
- Horizon path: `3->4->4`
- Reason: SLURM state `CANCELLED by 1002`.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-02T08:49:45+00:00

Launched `dense_rhs_plus2_anchor_start70_margin10_300k_vec8_s15` as job `1692`.

## 2026-05-02T08:50:09+00:00

Launched `dense_rhs_plus2_anchor_start90_margin06_300k_vec8_s15` as job `1693`.

## 2026-05-02T10:44:29+00:00

Processed `dense_rhs_plus2_anchor_start70_margin10_300k_vec8_s15` job `1692`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:47:51`
- Final eval: `898.3875122070312` ± `21.15914535522461`
- Best eval: `898.3875122070312` at step `300000`
- Horizon path: `4->3->4->5->5`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-02T10:44:31+00:00

Processed `dense_rhs_plus2_anchor_start90_margin06_300k_vec8_s15` job `1693`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:47:01`
- Final eval: `906.2188720703125` ± `31.46767234802246`
- Best eval: `906.2188720703125` at step `300000`
- Horizon path: `3->4->4->5->4`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-02T10:56:44+00:00

RFC `horizon_trust_region_hcap4`: queue three clean +2 Dense-RHS variants after jobs 1692/1693 completed below target.

- Evidence: 1692 reached `898.39` with path `4->3->4->5->5`; 1693 reached `906.22` with path `3->4->4->5->4`. Both improved until 200k and then plateaued or regressed after h=5 appeared.
- Rule: keep Dense-RHS adaptive but restrict the candidate horizon set to `H={2,3,4}` for this diagnostic family, with `hmax=4` and bucket `[4]`. This is a horizon trust region, not a fixed-horizon control: the learner and planner still follow the selected horizon online.
- Expected benefit: preserve the near-winning Sparse-HiFi behavior while removing the empirically harmful h=5 transition and reducing query compute.
- Failure mode: if all hcap4 variants remain below the no-RHS baseline, the missing ingredient is not late h=5 drift; mark this trust-region family for removal from final algorithm candidates.
- Queued: `dense_rhs_plus2_hcap4_start70_margin04_300k_vec8_s15`, `dense_rhs_plus2_hcap4_start120_returnlite_300k_vec8_s15`, and `dense_rhs_plus2_hcap4_ei_start70_300k_vec8_s15`.

## 2026-05-02T13:17:59+00:00

Launched `dense_rhs_plus2_hcap4_start70_margin04_300k_vec8_s15` as job `1701`.

## 2026-05-02T13:18:38+00:00

Launched `dense_rhs_plus2_hcap4_start120_returnlite_300k_vec8_s15` as job `1702`.

## 2026-05-02T13:19:44+00:00

Launched `dense_rhs_plus2_hcap4_ei_start70_300k_vec8_s15` as job `1703`.

## 2026-05-02T13:52:42+00:00

RFC `deployment_utility_posterior`: add an explicit posterior over expected eval-reward gain by deployed horizon.

- Evidence: active job `1703` is the weakest current hcap4 run at the first eval (`46.98 ± 23.22` at 50k), while `1701` and `1702` are less bad (`164.06` and `216.53`). This makes `1703` the slot to replace with a qualitatively different evidence model rather than another transition-threshold tweak.
- Rule: after each clean eval, attribute the eval-return delta since the previous eval to the horizon deployed during that interval. Maintain per-horizon Gaussian-style sufficient statistics over these deltas and compute a UCB utility `mean_gain(h) + beta * uncertainty(h)`. At query time, combine this utility with the normalized Dense-RHS deployment score and allow it to override the query-selected horizon only among the evaluated candidates.
- Config flags: `DENSE_RHS_DEPLOYMENT_UTILITY_ENABLED`, `DENSE_RHS_DEPLOYMENT_UTILITY_WEIGHT`, `DENSE_RHS_DEPLOYMENT_UTILITY_EXPLORATION`, `DENSE_RHS_DEPLOYMENT_UTILITY_PRIOR_MEAN`, `DENSE_RHS_DEPLOYMENT_UTILITY_PRIOR_STD`, `DENSE_RHS_DEPLOYMENT_UTILITY_DENSE_SCORE_WEIGHT`, and `DENSE_RHS_DEPLOYMENT_UTILITY_MIN_OBSERVATIONS`.
- Expected benefit: improve horizon evidence quality by measuring downstream learner/eval gain rather than relying only on one-shot short candidate rollouts.
- Failure mode: delayed attribution is noisy because each eval interval includes continued learning and stochasticity; if the run switches erratically or stays below the hcap4 baselines, keep the logged metrics for analysis but do not promote this rule.
- Queued: `dense_rhs_plus2_hcap4_uplift_start70_300k_vec8_s15`.

## 2026-05-02T13:54:18+00:00

Processed `dense_rhs_plus2_hcap4_ei_start70_300k_vec8_s15` job `1703`: `failed`.

- SLURM state: `CANCELLED by 1002`
- Runtime: `00:34:28`
- Final eval: `46.981651306152344` ± `23.216461181640625`
- Best eval: `46.981651306152344` at step `50000`
- Horizon path: `2`
- Reason: SLURM state `CANCELLED by 1002`.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-02T13:54:24+00:00

Launched `dense_rhs_plus2_hcap4_uplift_start70_300k_vec8_s15` as job `1708`.

## 2026-05-02T14:49:55+00:00

Processed `dense_rhs_plus2_hcap4_start70_margin04_300k_vec8_s15` job `1701`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:25:44`
- Final eval: `896.6925048828125` ± `20.500463485717773`
- Best eval: `896.6925048828125` at step `300000`
- Horizon path: `3->3->2->2->3`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-02T14:49:57+00:00

Processed `dense_rhs_plus2_hcap4_start120_returnlite_300k_vec8_s15` job `1702`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:25:30`
- Final eval: `898.8212890625` ± `24.93979263305664`
- Best eval: `908.4658203125` at step `200000`
- Horizon path: `3->3->4->3`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-02T14:50:02+00:00

Hcap4 trust-region family completed below target (best=908.4658203125, best_label=dense_rhs_plus2_hcap4_start120_returnlite_300k_vec8_s15); queued H={3,4} late-exploitation profiles: dense_rhs_plus2_h34_start120_margin02_300k_vec8_s15, dense_rhs_plus2_h34_start160_returndom_300k_vec8_s15, dense_rhs_plus2_h34_ei_start120_300k_vec8_s15.

## 2026-05-02T14:50:06+00:00

Launched `dense_rhs_plus2_h34_start120_margin02_300k_vec8_s15` as job `1709`.

## 2026-05-02T14:50:54+00:00

Launched `dense_rhs_plus2_h34_start160_returndom_300k_vec8_s15` as job `1710`.

## 2026-05-02T15:45:54+00:00

Processed `dense_rhs_plus2_hcap4_uplift_start70_300k_vec8_s15` job `1708`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:24:33`
- Final eval: `863.4307861328125` ± `28.056758880615234`
- Best eval: `863.4307861328125` at step `300000`
- Horizon path: `2->2->2->2->2`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-02T15:46:02+00:00

Launched `dense_rhs_plus2_h34_ei_start120_300k_vec8_s15` as job `1711`.

## 2026-05-02T16:18:57+00:00

Processed `dense_rhs_plus2_h34_start120_margin02_300k_vec8_s15` job `1709`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:24:34`
- Final eval: `901.8836059570312` ± `34.625823974609375`
- Best eval: `919.34619140625` at step `250000`
- Horizon path: `3->3->3->3`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-02T17:01:27+00:00

Processed `dense_rhs_plus2_h34_start160_returndom_300k_vec8_s15` job `1710`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:24:12`
- Final eval: `904.8642578125` ± `23.15458106994629`
- Best eval: `904.8642578125` at step `300000`
- Horizon path: `4->4->4`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-02T17:34:12+00:00

RFC `full_horizon_evidence_search`: restore the full adaptive horizon set after diagnostic trust-region runs.

- Evidence: H={2,3,4} and H={3,4} runs were useful diagnostics but did not beat the clean +2 target. They also violate the intended final algorithm principle: the search should discover the useful horizon online rather than receive a hand-capped set.
- Rule: keep `H={2,...,30}` and use bucketed compilation only as an implementation detail. Queue one broad all-horizon evaluator that evaluates all 29 horizons at each query with lower per-horizon eval budget, plus one full-H Bayesian expected-improvement evaluator with global posterior candidate selection.
- Expected benefit: preserve the adaptive RHS spirit while directly testing whether better horizon evidence and softer transition decisions can avoid bad downshifts/upshifts without removing horizons.
- Failure mode: if full-H still underperforms, the next RFC should improve evidence attribution or query pairing, not narrow the horizon set again.
- Queued: `dense_rhs_plus2_fullh_all29_start70_300k_vec8_s15` and `dense_rhs_plus2_fullh_ei_b9_start70_300k_vec8_s15`.

## 2026-05-02T17:35:14+00:00

Launched `dense_rhs_plus2_fullh_all29_start70_300k_vec8_s15` as job `1712`.

## 2026-05-02T17:36:23+00:00

Launched `dense_rhs_plus2_fullh_ei_b9_start70_300k_vec8_s15` as job `1713`.

## 2026-05-02T17:42:30+00:00

Manual chaos-baseline override after removing the restricted `1711` diagnostic from the active set.

- `1711` was already terminal when cancellation was requested; it completed at `899.64` final clean eval and remains below the clean +2 target.
- Launched no-RHS current-MJX chaos baseline `no_rhs_chaos_parity_300k_vec8_s15_current` as job `1715`.
- Definition: chaos is applied during training only with `env.mjx_dmc.enable_domain_randomization=true`, `env.mjx_dmc.enable_observation_noise=true`, and `env.mjx_dmc.base_action_delay=1`; clean evaluation remains enabled every `50k` for `20` episodes.
- Legacy-match check: this matches the old experimental convention of randomized/noisy training with clean evaluation, but the current MJX port implements actuator-strength randomization, observation noise, wind/push perturbations, jitter/slip-style perturbation, and action delay. It does not currently randomize MuJoCo mass, damping, or friction parameters, so this is a current-JAX chaos baseline rather than full legacy-chaos parity.

## 2026-05-02T19:18:56+00:00

Processed `dense_rhs_plus2_fullh_all29_start70_300k_vec8_s15` job `1712`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:39:11`
- Final eval: `837.6513671875` ± `28.396848678588867`
- Best eval: `837.6513671875` at step `300000`
- Horizon path: `4->3->5->5->7`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-02T19:19:02+00:00

H={3,4} late-exploitation family completed below target (best=919.34619140625, best_label=dense_rhs_plus2_h34_start120_margin02_300k_vec8_s15). Steward needs a new algorithm RFC; do not silently lower the clean +2 goal.

## 2026-05-02T19:29:38+00:00

Processed `dense_rhs_plus2_fullh_ei_b9_start70_300k_vec8_s15` job `1713`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:43:33`
- Final eval: `879.5133056640625` ± `28.175579071044922`
- Best eval: `879.5133056640625` at step `300000`
- Horizon path: `5->7->7->7->7`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-02T19:29:40+00:00

Processed `no_rhs_chaos_parity_300k_vec8_s15_current` job `1715`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:30:18`
- Final eval: `505.26690673828125` ± `35.0055046081543`
- Best eval: `590.1742553710938` at step `200000`
- Horizon path: ``
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-02T19:29:43+00:00

H={3,4} late-exploitation family completed below target (best=919.34619140625, best_label=dense_rhs_plus2_h34_start120_margin02_300k_vec8_s15). Steward needs a new algorithm RFC; do not silently lower the clean +2 goal.

## 2026-05-02T19:40:28+00:00

H={3,4} late-exploitation family completed below target (best=919.34619140625, best_label=dense_rhs_plus2_h34_start120_margin02_300k_vec8_s15). Steward needs a new algorithm RFC; do not silently lower the clean +2 goal.

## 2026-05-02T19:51:00+00:00

H={3,4} late-exploitation family completed below target (best=919.34619140625, best_label=dense_rhs_plus2_h34_start120_margin02_300k_vec8_s15). Steward needs a new algorithm RFC; do not silently lower the clean +2 goal.

## 2026-05-02T19:52:30+00:00

RFC `full_horizon_utility_evidence`: keep full adaptive horizon search over `H={2,...,30}` and improve evidence attribution instead of applying a hard horizon cap.

- Evidence: full-H all-candidate and full-H EI runs both underperformed (`837.65` and `879.51`) and selected larger horizons (`5/7`) too aggressively. H-capped diagnostics got closer but violate the final algorithm principle.
- Rule: queue three full-H profiles that keep `H={2,...,30}` and bucketed compilation only as an implementation detail: deployment-utility posterior, learner-proxy scoring, and a combined utility-plus-learner-proxy profile.
- Expected benefit: horizon changes are guided by downstream eval-gain evidence and dense learner compatibility, not only by noisy short query returns.
- Failure mode: if these still drift to harmful large horizons, the next patch should improve query pairing/calibration or add a principled horizon-complexity prior, not manually remove horizon values.

## 2026-05-02T19:53:42+00:00

Launched `dense_rhs_plus2_fullh_du_start70_300k_vec8_s15` as job `1716`.

## 2026-05-02T19:54:11+00:00

Launched `dense_rhs_plus2_fullh_learnerproxy_start70_300k_vec8_s15` as job `1717`.

## 2026-05-02T19:54:46+00:00

Launched `dense_rhs_plus2_fullh_du_learnerproxy_start70_300k_vec8_s15` as job `1718`.

## 2026-05-03T09:56:30+00:00

Processed `dense_rhs_plus2_fullh_du_start70_300k_vec8_s15` job `1716`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:27:59`
- Final eval: `826.55810546875` ± `19.581214904785156`
- Best eval: `826.55810546875` at step `300000`
- Horizon path: `2->2->2->2->2`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-03T09:56:31+00:00

Processed `dense_rhs_plus2_fullh_learnerproxy_start70_300k_vec8_s15` job `1717`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:49:03`
- Final eval: `912.6260986328125` ± `29.712343215942383`
- Best eval: `912.6260986328125` at step `300000`
- Horizon path: `3->5->7->7->9`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-03T09:56:33+00:00

Processed `dense_rhs_plus2_fullh_du_learnerproxy_start70_300k_vec8_s15` job `1718`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:27:42`
- Final eval: `834.3712768554688` ± `20.812170028686523`
- Best eval: `834.3712768554688` at step `300000`
- Horizon path: `3->4->4->4->4`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-03T09:56:36+00:00

H={3,4} late-exploitation family completed below target (best=919.34619140625, best_label=dense_rhs_plus2_h34_start120_margin02_300k_vec8_s15). Steward needs a new algorithm RFC; do not silently lower the clean +2 goal.

## 2026-05-03T10:07:03+00:00

H={3,4} late-exploitation family completed below target (best=919.34619140625, best_label=dense_rhs_plus2_h34_start120_margin02_300k_vec8_s15). Steward needs a new algorithm RFC; do not silently lower the clean +2 goal.

## 2026-05-03T10:37:58+00:00

H={3,4} late-exploitation family completed below target (best=919.34619140625, best_label=dense_rhs_plus2_h34_start120_margin02_300k_vec8_s15). Steward needs a new algorithm RFC; do not silently lower the clean +2 goal.

## 2026-05-03T10:53:27+00:00

H={3,4} late-exploitation family completed below target (best=919.34619140625, best_label=dense_rhs_plus2_h34_start120_margin02_300k_vec8_s15). Steward needs a new algorithm RFC; do not silently lower the clean +2 goal.

## 2026-05-03T11:03:58+00:00

H={3,4} late-exploitation family completed below target (best=919.34619140625, best_label=dense_rhs_plus2_h34_start120_margin02_300k_vec8_s15). Steward needs a new algorithm RFC; do not silently lower the clean +2 goal.

## 2026-05-03T11:14:27+00:00

H={3,4} late-exploitation family completed below target (best=919.34619140625, best_label=dense_rhs_plus2_h34_start120_margin02_300k_vec8_s15). Steward needs a new algorithm RFC; do not silently lower the clean +2 goal.

## 2026-05-03T11:24:57+00:00

H={3,4} late-exploitation family completed below target (best=919.34619140625, best_label=dense_rhs_plus2_h34_start120_margin02_300k_vec8_s15). Steward needs a new algorithm RFC; do not silently lower the clean +2 goal.

## 2026-05-03T11:45:59+00:00

H={3,4} late-exploitation family completed below target (best=919.34619140625, best_label=dense_rhs_plus2_h34_start120_margin02_300k_vec8_s15). Steward needs a new algorithm RFC; do not silently lower the clean +2 goal.

## 2026-05-04T12:40:00+00:00

User closed the quadruped clean +2 architecture-search target. The campaign no longer searches for a Dense-RHS variant that beats the clean no-RHS baseline by 2%; instead, the selected Dense-RHS winner architecture is `dense_rhs_plus2_sparse_hifi_start70_300k_vec8_s15` from job `1663` (`922.63 +/- 21.34`, horizon path `3->3->3->3->3`).

The new steward objective is table-driven cross-environment evaluation:

- First fill MJX port gates and clean no-RHS paper-parity baselines for the scheduled environments.
- Then run the selected Dense-RHS winner architecture in clean/no-chaos mode.
- Then run no-RHS chaos baselines for missing environments.
- Finally run the selected Dense-RHS winner architecture in chaos mode.

New launches are blocked until `runs/results/final_results_table_template.tex` is reviewed and approved. The goal file now enforces `constraints.max_active_gpus=2`, `success.clean_plus2_gate_closed=true`, and `constraints.final_table_approved=false`.

## 2026-05-04T12:48:12+00:00

Launched `dense_rhs_winner_chaos_300k_vec8_s15` as job `1810`.

## 2026-05-04T14:44:40+00:00

Processed `dense_rhs_winner_chaos_300k_vec8_s15` job `1810`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:48:24`
- Final eval: `614.44970703125` ± `44.561885833740234`
- Best eval: `614.44970703125` at step `300000`
- Horizon path: `3->2->2->2->3`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-08T14:46:06+00:00

Table dispatcher queued clean_no_rhs profiles: fish_swim_clean_no_rhs_300k_vec8_s15, finger_turn_hard_clean_no_rhs_300k_vec8_s15, cheetah_run_clean_no_rhs_300k_vec8_s15, cartpole_swingup_clean_no_rhs_300k_vec8_s15, acrobot_swingup_clean_no_rhs_300k_vec8_s15, walker_run_clean_no_rhs_300k_vec8_s15, hopper_hop_clean_no_rhs_300k_vec8_s15.

## 2026-05-08T14:46:18+00:00

Launched `fish_swim_clean_no_rhs_300k_vec8_s15` as job `1959`.

## 2026-05-08T14:47:20+00:00

Launched `finger_turn_hard_clean_no_rhs_300k_vec8_s15` as job `1960`.

## 2026-05-08T15:40:33+00:00

Processed `fish_swim_clean_no_rhs_300k_vec8_s15` job `1959`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `00:53:38`
- Final eval: `505.16326904296875` ± `284.06915283203125`
- Best eval: `596.1466064453125` at step `250000`
- Horizon path: ``
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-08T15:40:58+00:00

Launched `cheetah_run_clean_no_rhs_300k_vec8_s15` as job `1962`.

## 2026-05-08T15:44:32+00:00

Processed `finger_turn_hard_clean_no_rhs_300k_vec8_s15` job `1960`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `00:56:55`
- Final eval: `980.5499877929688` ± `13.987404823303223`
- Best eval: `980.5499877929688` at step `300000`
- Horizon path: ``
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-08T15:44:52+00:00

Launched `cartpole_swingup_clean_no_rhs_300k_vec8_s15` as job `1963`.

## 2026-05-08T16:04:30+00:00

Launched `acrobot_swingup_clean_no_rhs_300k_vec8_s15` as job `1964`.

## 2026-05-08T16:37:34+00:00

Processed `cartpole_swingup_clean_no_rhs_300k_vec8_s15` job `1963`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `00:52:26`
- Final eval: `866.7747192382812` ± `0.26454320549964905`
- Best eval: `867.5153198242188` at step `200000`
- Horizon path: ``
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-08T16:37:43+00:00

Launched `walker_run_clean_no_rhs_300k_vec8_s15` as job `1966`.

## 2026-05-08T16:38:19+00:00

Processed `cheetah_run_clean_no_rhs_300k_vec8_s15` job `1962`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `00:56:55`
- Final eval: `845.8548583984375` ± `116.47073364257812`
- Best eval: `871.3776245117188` at step `150000`
- Horizon path: ``
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-08T16:57:01+00:00

Processed `acrobot_swingup_clean_no_rhs_300k_vec8_s15` job `1964`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `00:50:35`
- Final eval: `539.5477294921875` ± `162.24920654296875`
- Best eval: `539.5477294921875` at step `300000`
- Horizon path: ``
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-08T16:57:20+00:00

Launched `hopper_hop_clean_no_rhs_300k_vec8_s15` as job `1967`.

## 2026-05-08T18:05:38+00:00

Processed `walker_run_clean_no_rhs_300k_vec8_s15` job `1966`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:26:04`
- Final eval: `839.2001953125` ± `10.541329383850098`
- Best eval: `839.2001953125` at step `300000`
- Horizon path: ``
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-08T18:05:39+00:00

Processed `hopper_hop_clean_no_rhs_300k_vec8_s15` job `1967`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:06:43`
- Final eval: `287.27569580078125` ± `14.467547416687012`
- Best eval: `287.27569580078125` at step `300000`
- Horizon path: ``
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-08T18:05:47+00:00

Table dispatcher queued clean_dense_rhs_winner profiles: fish_swim_clean_dense_rhs_sparse_hifi_300k_vec8_s15, finger_turn_hard_clean_dense_rhs_sparse_hifi_300k_vec8_s15, cheetah_run_clean_dense_rhs_sparse_hifi_300k_vec8_s15, cartpole_swingup_clean_dense_rhs_sparse_hifi_300k_vec8_s15, acrobot_swingup_clean_dense_rhs_sparse_hifi_300k_vec8_s15, walker_run_clean_dense_rhs_sparse_hifi_300k_vec8_s15, hopper_hop_clean_dense_rhs_sparse_hifi_300k_vec8_s15.

## 2026-05-08T18:05:59+00:00

Launched `fish_swim_clean_dense_rhs_sparse_hifi_300k_vec8_s15` as job `1968`.

## 2026-05-08T18:06:00+00:00

Launched `finger_turn_hard_clean_dense_rhs_sparse_hifi_300k_vec8_s15` as job `1969`.

## 2026-05-08T18:11:54+00:00

Launched `cheetah_run_clean_dense_rhs_sparse_hifi_300k_vec8_s15` as job `1970`.

## 2026-05-08T19:10:05+00:00

Processed `fish_swim_clean_dense_rhs_sparse_hifi_300k_vec8_s15` job `1968`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:03:36`
- Final eval: `596.9766845703125` ± `232.79652404785156`
- Best eval: `708.2169189453125` at step `250000`
- Horizon path: `3->3->3->3->3`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-08T19:10:25+00:00

Launched `cartpole_swingup_clean_dense_rhs_sparse_hifi_300k_vec8_s15` as job `1971`.

## 2026-05-08T19:18:07+00:00

Processed `finger_turn_hard_clean_dense_rhs_sparse_hifi_300k_vec8_s15` job `1969`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:11:50`
- Final eval: `927.25` ± `102.99022674560547`
- Best eval: `963.8499755859375` at step `200000`
- Horizon path: `3->3->4->4->5`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-08T19:18:28+00:00

Launched `acrobot_swingup_clean_dense_rhs_sparse_hifi_300k_vec8_s15` as job `1972`.

## 2026-05-08T19:32:30+00:00

Processed `cheetah_run_clean_dense_rhs_sparse_hifi_300k_vec8_s15` job `1970`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:19:14`
- Final eval: `933.12890625` ± `37.509788513183594`
- Best eval: `933.12890625` at step `300000`
- Horizon path: `4->5->6->6->6`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-08T19:32:50+00:00

Launched `walker_run_clean_dense_rhs_sparse_hifi_300k_vec8_s15` as job `1973`.

## 2026-05-08T20:18:50+00:00

Processed `acrobot_swingup_clean_dense_rhs_sparse_hifi_300k_vec8_s15` job `1972`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `00:59:31`
- Final eval: `362.320556640625` ± `108.22805786132812`
- Best eval: `575.4118041992188` at step `200000`
- Horizon path: `3->3->3->3->3`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-08T20:19:10+00:00

Launched `hopper_hop_clean_dense_rhs_sparse_hifi_300k_vec8_s15` as job `1974`.

## 2026-05-08T20:21:27+00:00

Processed `cartpole_swingup_clean_dense_rhs_sparse_hifi_300k_vec8_s15` job `1971`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:10:34`
- Final eval: `882.39404296875` ± `0.22589746117591858`
- Best eval: `882.39404296875` at step `300000`
- Horizon path: `4->4->5->4->3`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-09T08:50:01+00:00

Processed `walker_run_clean_dense_rhs_sparse_hifi_300k_vec8_s15` job `1973`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:38:25`
- Final eval: `835.9410400390625` ± `9.586824417114258`
- Best eval: `835.9410400390625` at step `300000`
- Horizon path: `4->3->3->4->4`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-09T08:50:03+00:00

Processed `hopper_hop_clean_dense_rhs_sparse_hifi_300k_vec8_s15` job `1974`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:16:29`
- Final eval: `350.6131896972656` ± `12.311149597167969`
- Best eval: `351.2184143066406` at step `250000`
- Horizon path: `2->2->3->2->3`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-09T08:50:17+00:00

Table dispatcher queued chaos_no_rhs profiles: fish_swim_chaos_no_rhs_300k_vec8_s15, finger_turn_hard_chaos_no_rhs_300k_vec8_s15, cheetah_run_chaos_no_rhs_300k_vec8_s15, cartpole_swingup_chaos_no_rhs_300k_vec8_s15, acrobot_swingup_chaos_no_rhs_300k_vec8_s15, walker_run_chaos_no_rhs_300k_vec8_s15, hopper_hop_chaos_no_rhs_300k_vec8_s15.

## 2026-05-09T08:50:31+00:00

Launched `fish_swim_chaos_no_rhs_300k_vec8_s15` as job `1975`.

## 2026-05-09T08:50:33+00:00

Launched `finger_turn_hard_chaos_no_rhs_300k_vec8_s15` as job `1976`.

## 2026-05-09T08:50:34+00:00

Launched `cheetah_run_chaos_no_rhs_300k_vec8_s15` as job `1977`.

## 2026-05-09T10:03:15+00:00

Processed `fish_swim_chaos_no_rhs_300k_vec8_s15` job `1975`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:10:46`
- Final eval: `73.97637939453125` ± `35.65225601196289`
- Best eval: `82.80696105957031` at step `50000`
- Horizon path: ``
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-09T10:03:38+00:00

Launched `cartpole_swingup_chaos_no_rhs_300k_vec8_s15` as job `1978`.

## 2026-05-09T10:04:31+00:00

Processed `finger_turn_hard_chaos_no_rhs_300k_vec8_s15` job `1976`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:13:09`
- Final eval: `713.4000244140625` ± `403.3861083984375`
- Best eval: `713.4000244140625` at step `300000`
- Horizon path: ``
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-09T10:04:33+00:00

Processed `cheetah_run_chaos_no_rhs_300k_vec8_s15` job `1977`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:13:47`
- Final eval: `457.7344665527344` ± `76.45452117919922`
- Best eval: `457.7344665527344` at step `300000`
- Horizon path: ``
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-09T10:04:58+00:00

Launched `acrobot_swingup_chaos_no_rhs_300k_vec8_s15` as job `1979`.

## 2026-05-09T10:04:59+00:00

Launched `walker_run_chaos_no_rhs_300k_vec8_s15` as job `1980`.

## 2026-05-09T11:14:08+00:00

Processed `cartpole_swingup_chaos_no_rhs_300k_vec8_s15` job `1978`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:10:06`
- Final eval: `855.3677978515625` ± `2.8434972763061523`
- Best eval: `855.3677978515625` at step `300000`
- Horizon path: ``
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-09T11:14:09+00:00

Processed `acrobot_swingup_chaos_no_rhs_300k_vec8_s15` job `1979`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:08:01`
- Final eval: `292.6933288574219` ± `96.26347351074219`
- Best eval: `329.11444091796875` at step `250000`
- Horizon path: ``
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-09T11:14:33+00:00

Launched `hopper_hop_chaos_no_rhs_300k_vec8_s15` as job `1981`.

## 2026-05-09T17:51:44+00:00

Processed `walker_run_chaos_no_rhs_300k_vec8_s15` job `1980`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:43:12`
- Final eval: `591.030029296875` ± `30.23134994506836`
- Best eval: `591.030029296875` at step `300000`
- Horizon path: ``
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-09T17:51:48+00:00

Processed `hopper_hop_chaos_no_rhs_300k_vec8_s15` job `1981`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:19:11`
- Final eval: `198.45770263671875` ± `7.539370059967041`
- Best eval: `198.45770263671875` at step `300000`
- Horizon path: ``
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-09T17:52:03+00:00

Table dispatcher queued chaos_dense_rhs_winner profiles: fish_swim_chaos_dense_rhs_sparse_hifi_300k_vec8_s15, finger_turn_hard_chaos_dense_rhs_sparse_hifi_300k_vec8_s15, cheetah_run_chaos_dense_rhs_sparse_hifi_300k_vec8_s15, cartpole_swingup_chaos_dense_rhs_sparse_hifi_300k_vec8_s15, acrobot_swingup_chaos_dense_rhs_sparse_hifi_300k_vec8_s15, walker_run_chaos_dense_rhs_sparse_hifi_300k_vec8_s15, hopper_hop_chaos_dense_rhs_sparse_hifi_300k_vec8_s15.

## 2026-05-09T17:52:20+00:00

Launched `fish_swim_chaos_dense_rhs_sparse_hifi_300k_vec8_s15` as job `1982`.

## 2026-05-09T17:52:23+00:00

Launched `finger_turn_hard_chaos_dense_rhs_sparse_hifi_300k_vec8_s15` as job `1983`.

## 2026-05-09T17:52:27+00:00

Launched `cheetah_run_chaos_dense_rhs_sparse_hifi_300k_vec8_s15` as job `1984`.

## 2026-05-09T19:11:57+00:00

Processed `fish_swim_chaos_dense_rhs_sparse_hifi_300k_vec8_s15` job `1982`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:19:01`
- Final eval: `76.80537414550781` ± `35.97894287109375`
- Best eval: `90.21401977539062` at step `50000`
- Horizon path: `3->3->3->3->3`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-09T19:12:22+00:00

Launched `cartpole_swingup_chaos_dense_rhs_sparse_hifi_300k_vec8_s15` as job `1985`.

## 2026-05-09T19:18:54+00:00

Processed `finger_turn_hard_chaos_dense_rhs_sparse_hifi_300k_vec8_s15` job `1983`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:24:19`
- Final eval: `688.5999755859375` ± `405.9638671875`
- Best eval: `688.5999755859375` at step `300000`
- Horizon path: `3->3->3->3->3`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-09T19:19:16+00:00

Launched `acrobot_swingup_chaos_dense_rhs_sparse_hifi_300k_vec8_s15` as job `1986`.

## 2026-05-09T19:24:53+00:00

Processed `cheetah_run_chaos_dense_rhs_sparse_hifi_300k_vec8_s15` job `1984`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:31:02`
- Final eval: `491.46044921875` ± `76.12613677978516`
- Best eval: `491.46044921875` at step `300000`
- Horizon path: `3->4->3->4->5`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-09T19:25:15+00:00

Launched `walker_run_chaos_dense_rhs_sparse_hifi_300k_vec8_s15` as job `1987`.

## 2026-05-09T20:33:45+00:00

Processed `cartpole_swingup_chaos_dense_rhs_sparse_hifi_300k_vec8_s15` job `1985`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:19:03`
- Final eval: `876.3756103515625` ± `1.726609706878662`
- Best eval: `876.3756103515625` at step `300000`
- Horizon path: `3->3->3->2->3`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-09T20:34:08+00:00

Launched `hopper_hop_chaos_dense_rhs_sparse_hifi_300k_vec8_s15` as job `1988`.

## 2026-05-09T20:36:00+00:00

Processed `acrobot_swingup_chaos_dense_rhs_sparse_hifi_300k_vec8_s15` job `1986`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:16:17`
- Final eval: `353.9346923828125` ± `109.49887084960938`
- Best eval: `379.4981994628906` at step `250000`
- Horizon path: `4->4->4->4->4`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-09T21:21:07+00:00

Processed `walker_run_chaos_dense_rhs_sparse_hifi_300k_vec8_s15` job `1987`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:54:57`
- Final eval: `579.1027221679688` ± `34.14745330810547`
- Best eval: `627.7696533203125` at step `250000`
- Horizon path: `3->3->3->4->4`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.

## 2026-05-09T22:06:08+00:00

Processed `hopper_hop_chaos_dense_rhs_sparse_hifi_300k_vec8_s15` job `1988`: `completed`.

- SLURM state: `COMPLETED`
- Runtime: `01:31:31`
- Final eval: `173.11436462402344` ± `9.265361785888672`
- Best eval: `173.11436462402344` at step `300000`
- Horizon path: `2->2->2->3->3`
- Reason: Completed; no automatic classifier for this method.
- Follow-up: No automatic follow-up rule fired.
