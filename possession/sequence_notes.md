# Temporal CNN experiment (external-only)

New files: `sequence_tcn.py`, `sequence_experiment.py`, `sequence_evaluate.py`,
`sequence_robustness.py`, `sequence_run.py`, and `tests/test_sequence_tcn.py`.
No frozen v3 files were changed and no challenge labels were read.

## Design

TCN with 48 hidden channels and six noncausal residual dilated convolutional layers
(dilations 1,2,4,8,16,32). Each observed feature is normalized using only training
matches; missingness is an explicit extra channel. GroupNorm uses sequence-level
statistics, therefore inference is offline and half-local, not streaming.
Training crops 256 samples (51.2s at5Hz), batch16, 500 AdamW steps, seed2027.
Half of crop centers are sampled near external-reference transitions. Loss is
class-weighted cross-entropy plus bounded log-probability smoothness (weight.15).
No hyperparameters were chosen with the challenge public ground truth.

The existing training caches were reconstructed into exact half sequences:
2 degradation seeds ×2 team orientations ×2 halves, per external match. A strict
length check protects this reconstruction. Models are trained leaving one entire
match out and use symmetric direct/mirrored inference.

## Six external comparisons (three games ×two dropout seeds)

All numbers use weak event-derived references; neither variant is perfect truth.
The corrected reference fixes a self-match bug in legacy one-touch detection.

| Reference | HGB v3 spellF1 | TCN spellF1 | 50/50 probability ensemble |
| --- | ---: | ---: | ---: |
| Legacy reference | .62502 | .64790 | .65185 |
| Corrected weak reference | .55668 | .56902 | .57432 |

First dropout seed alone favored TCN. The ensemble was selected over all six
external development cases and both reference interpretations. These are not
six independent games, and selection on this comparison makes it development
validation rather than an untouched final test. Sportec is a separate domain test.

Outputs:
- `data/work/sequence_tcn/results.json`: first dropout seed, original references.
- `reference_audit_results.json`: same predictions, both references.
- `robustness.json`: independently degraded second dropout seed, both references.
- `fold_1.pt`, `fold_2.pt`, `fold_3.pt`: held-out-game checkpoints.

## Final candidate and reproduction

Trained on all3 external games, original weak labels, 500 steps:
`data/work/sequence_tcn_final/sequence_model.pt`
SHA256 `0fb25d7270798fa98334f6b84fe0d151859e2bbea437c57de6f898d81e50c581`.

`evaluation/results/sequence_tcn.csv` passes the official format validator.
SHA256 `403359779fbfdb9be2b75406b8de48918d1ea8b90c7ce08bf7aa354800464727`.
It is byte-identical on saved-model reproduction AND a new MPS training run
from scratch in this environment (PyTorch2.15.0, current frozen sources/data).
This is an empirical check, not a cross-platform bit-exact guarantee.

`evaluation/results/sequence_ensemble.csv` combines that TCN and the existing
HGB model (`data/work/temporal_v3/model.joblib`) 50/50 in probability space before
the original v3 decoder. It also passes validation, with 233 possessions:
H47,395 /A27,595 /D83,867 /X40,462.
SHA256 `8e6c49a3798910a2e369b80a2a973f574b89936bd305eacc401927d5377938d6`.

## Commands

```bash
OMP_NUM_THREADS=8 .venv/bin/python -m possession.sequence_experiment
OMP_NUM_THREADS=8 .venv/bin/python -m possession.sequence_evaluate
OMP_NUM_THREADS=8 .venv/bin/python -m possession.sequence_robustness
OMP_NUM_THREADS=8 .venv/bin/python -m possession.sequence_run --steps 500
OMP_NUM_THREADS=8 .venv/bin/python -m possession.sequence_run \
  --model data/work/sequence_tcn_final/sequence_model.pt \
  --hgb-model data/work/temporal_v3/model.joblib \
  --output evaluation/results/sequence_ensemble.csv \
  --work-dir data/work/sequence_ensemble
```

The runner requires the existing external feature caches when training. It
validates reconstructed lengths, but the cache identity does not hash source
files; stale caches must not be reused after source or reference changes.
Checkpoint loader deliberately supports only locally generated trusted files
and uses torch.load(weights_only=False), as checkpoints contain NumPy scalers.

API for new domains:
`load_sequence_model(path,device)` returns `(model,normalizer)`;
`match_probabilities(model,normalizer,Match,device)` returns `(p_HAD,frames)`.
The caller converts source data into the same metre/25Hz Match representation.
