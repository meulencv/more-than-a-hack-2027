# Mixed-domain temporal experiment

Additional model preserves all legacy temporal checkpoints.
Training: external Metrica games1/2/3 plus only Sportec `J03WMX`.
Sportec `J03WN1` is never part of training. This is external development validation,
not an untouched final test because it has already been inspected in other comparisons.

Native training augmentation uses `blind_corruption` seeds101/102 and both team
orientations; it never consults state or possession labels when adding corruption.
Native labels are read only as training targets. Unknown native labels (-1) are
ignored in cross-entropy. The model is the same48-channel6-layerTCN, fixed500steps.

| Native holdout J03WN1 | HGB Metrica-only | Original TCN | Mixed TCN | Mixed 50/50 HGB ensemble |
| --- | ---: | ---: | ---: | ---: |
| Clean spellF1 | .49121 | .37209 | .34060 | .43651 |
| Blind stress spellF1 | .27109 | .26667 | .44161 | .37566 |

The new model is much stronger under the specified blind tracking corruption,
but clearly worse on clean tracking. It is NOT a robust replacement for the
existing candidate. No new challenge CSV was generated from this mixed model
on the basis of these mixed results. A possible future direction is a quality-
conditioned mixture of experts, with any gate trained/calibrated on other data,
not retrospectively chosen from challenge public performance.

Artifacts ignored byGit:
`data/work/sequence_native_mix/mixed_model.pt`, `results.json`, and
`heldout_clean.npz` /`heldout_blind_stress.npz`.
Runner: `OMP_NUM_THREADS=8 .venv/bin/python -m possession.sequence_native_mix`.
Inference API remains `load_sequence_model` and `match_probabilities` in
`possession.sequence_run`; checkpoints are compatible with that API.
