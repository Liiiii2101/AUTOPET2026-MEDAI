# Ensembling

Model ensembling is used at two levels to improve robustness across folds, tracers, and interaction steps:

1. **Cross-fold ensembling within a tracer/stage** — average softmax probabilities across the 10 trained folds
   of a given Stage‑1 or Stage‑2 model, using nnU-Net's stock ensembling utility (unmodified from upstream):

   ```bash
   nnUNetv2_predict -i INPUT_FOLDER -o FOLD0_OUT -d DATASET_ID -c 3d_fullres -tr MyCustomCurriculumTrainerSegPreSkel -f 0 --save_probabilities
   # ... repeat for each fold ...
   nnUNetv2_ensemble -i FOLD0_OUT FOLD1_OUT ... FOLD9_OUT -o ENSEMBLE_OUT
   ```

   (`nnUNetv2_ensemble` is provided by the [nnUNet-2.2](../nnUNet-2.2) install and wraps
   [`nnunetv2/ensembling/ensemble.py`](../nnUNet-2.2/nnunetv2/ensembling/ensemble.py).)

2. **Checkpoint/model ensembling across interaction steps** — Stage‑2 checkpoints saved at different points in
   the curriculum schedule (decoder-warmup vs. full-fine-tune) are combined the same way, to average out
   instability introduced by the scribble-simulation curriculum.

FDG and PSMA are always ensembled **separately** (never mixed) since each tracer branch is trained
independently end-to-end — see [`../stage1_lesion_segmentation`](../stage1_lesion_segmentation) and
[`../stage2_interactive_refinement`](../stage2_interactive_refinement).
