# FoG detection pipeline (ankle IMU, ESP32 / TFLM)

Python 3.11, TF 2.15.1, tfmot 0.8.0 (conda env `fog`). Run every command from this directory:

```
PY=C:\Users\aarus\anaconda3\envs\fog\python.exe
```

## Files
| file | role |
|---|---|
| `config.py` | paths, windowing, split, MCU budget, training and augmentation hyperparameters |
| `data.py` | load, 60->50 Hz resample, 21 Hz LPF, windowing, labels, subject-grouped splits, leakage asserts |
| `augment.py` | IMU augmentation: rotation, scale, noise, time-warp, shift, channel dropout |
| `models.py` | `ds_tcn`, `inception`, `hybrid` (BatchNorm-free candidates) and param/MAC accounting |
| `train.py` | subject-grouped CV, ablation, final multi-seed training |
| `evaluate.py` | window and episode metrics, subject-level bootstrap CIs, FSM post-processing tuning, plots |
| `quantize_export.py` | QAT fine-tune, INT8 TFLite, C header, normalization arrays, op list, golden vectors and operating point |
| `verify_export.py` | run the exported model against all ten golden vectors (does not access test data) |

## Commands
```
%PY% data.py                                   # dataset report + leakage asserts
%PY% data.py --activity-report                 # decide WALK_ACTIVITIES for FA/hour
%PY% models.py                                 # params / MACs / INT8 KB per candidate
%PY% train.py --models ds_tcn inception hybrid --cv      # 5-fold subject-grouped CV (seed 0)
%PY% train.py --models ds_tcn --ablate                   # augmentation ablation via CV
%PY% train.py --models ds_tcn --final                    # 5 seeds, test subjects never touched
%PY% evaluate.py ds_tcn                                  # OOF metrics, thresholds, FSM tuning, plots
%PY% quantize_export.py --model ds_tcn                   # QAT / export (test not touched)
%PY% quantize_export.py --model ds_tcn --test            # single final test evaluation (do not rerun)
%PY% verify_export.py                                  # check model against golden vectors
```

## Design decisions
- **Label rule:** window is FoG if at least 50% of its samples are FoG. Boundary windows (0 < fraction < 0.5) stay negative in evaluation.
- **Window overlap:** 100 samples / 2.0 s at 50 Hz and stride 25 samples / 0.5 s gives 75% overlap (not 50%).
- **Split:** test subjects 18, 20, 21, 22 are held out whole. The 18 dev subjects form 5 subject-grouped folds balanced by FoG count. Asserts check subject, normalization and block disjointness.
- **Normalization:** per-channel z-score fit on training subjects only; 8 channels (6 raw + |acc| + |gyro|).
- **Imbalance:** balanced oversampling (capped at 4x), focal loss with label smoothing. `pos_weight` is not applied together with oversampling.
- **Quantization:** QAT via tfmot with custom Conv1D/DepthwiseConv1D configs, then full-INT8 conversion.

## Summary table
`outputs/final_summary.csv` is a compact machine-readable record. `cv_seed0_oof_pr_auc` is from the earlier exploratory CV with scale augmentation enabled; it is not the current-config CV estimate. The five fixed-validation scores use the current `--final` runs, all evaluated on subject 2; their mean/SD measures seed variation on that one subject, not cross-subject uncertainty. The held-out test columns are intentionally empty because the current seed-3 export has not been evaluated on the test split.

## Current export and evaluation status
- Current deployable checkpoint: `ds_tcn`, seed 3, selected by PR-AUC on a fixed development validation fold (subject 2); all five stability seeds used that same fold. Validation PR-AUC across seeds: **0.673 ± 0.044** (mean ± sample SD), measuring seed variability on this one subject only, not cross-subject generalization. Exact scores are in `outputs/train_results.json`.
- Current QAT validation comparison uses the same selected QAT weights: float PR-AUC 0.8114, INT8 PR-AUC 0.8016. This is a small validation quantization gap, not a test result.
- Current INT8 operating point, selected on the same validation subject: probability threshold 0.36328125, EMA alpha 0.7, two consecutive windows to turn on, threshold off 0.36328125 and two consecutive windows to turn off. The tuning objective targeted episode sensitivity 0.7 and obtained 0.769 episode sensitivity / 25.25 FA per walking hour on that validation subject. See `outputs/export/operating_point.h` and `ds_tcn_export.json`.
- **No held-out test evaluation exists for the current seed-3 export and its operating point. Do not use old test metrics below as estimates for this artifact.** The test set was already accessed once for the earlier seed-0 model, so this project did not re-evaluate it after changing seed selection, QAT weights or threshold.

## Historical test result (superseded; seed 0, old operating point)
- Test subjects 18, 20, 21, 22 were evaluated once using seed 0 and threshold 0.4823 / FSM (EMA 0, k=3). The old INT8 window metrics were PR-AUC 0.408, ROC-AUC 0.856, F1 0.459, precision 0.369, recall 0.608, specificity 0.862, MCC 0.383. Episode sensitivity was 0.579 (19 episodes), median delay 1.0 s, FA/hr 104.4. Subject-bootstrap 95% CIs are in `outputs/eval_test_ds_tcn.json`. These figures do not describe the current seed-3 export.
- The old float-vs-INT8 test PR-AUCs (0.206 vs 0.408) are **not a valid quantization gap**: they compare pre-QAT float weights with QAT-fine-tuned INT8 weights. Do not interpret this as a quantization effect.

## Deployment notes
- Only 3 dev subjects (1, 2, 19) hold most FoG windows, and 7 subjects have none; uncertainty is high. Activity-code meanings are undocumented, so validate `WALK_ACTIVITIES` before trusting FA/hr.
- Export: `outputs/export/ds_tcn_int8.tflite` (28.2 KB; 7,585 params; 271,232 estimated MACs), `model_data.h`, `norm.h`, `quantization.h`, `operating_point.h`, `golden_vectors.h/.npz`, and `tflm_ops.txt`.
- Input tensor is `[1,100,8]` INT8; current model IO scales/zero points are in `quantization.h`. Compute |acc| and |gyro| in raw sensor units, append before z-score, then quantize with those affine parameters. Verify with the golden vectors.
- Generated op list: ADD, AVERAGE_POOL_2D, BATCH_TO_SPACE_ND, CONV_2D, DEPTHWISE_CONV_2D, EXPAND_DIMS, FULLY_CONNECTED, LOGISTIC, MEAN, RESHAPE, SPACE_TO_BATCH_ND. Confirm op support against the firmware's exact TFLM version and measure tensor-arena use and latency on hardware before deployment.
