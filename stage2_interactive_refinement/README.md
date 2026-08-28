# Stage 2 — interactive scribble refinement

The second-stage network takes the Stage‑1 checkpoint (`../stage1_lesion_segmentation`) and continues training
so it can incorporate cumulative foreground/background scribbles and correct the initial prediction. As in
Stage 1, FDG and PSMA are fine-tuned independently.

- **Trainer**: `MyCustomCurriculumTrainerSegPreSkel` (`nnUNet-2.2/nnunetv2/training/nnUNetTrainer/curriculum/MyCustomCurriculumTrainerSegPreSkel.py`)
- **Backbone**: STU-Net-Small, encoder+decoder initialized from the finished **Stage‑1 checkpoint** (loaded via
  `STUNetTrainerSegPre`, not the raw MAE checkpoint)
- **Scribble simulation**: [`nnUNetDataLoader3DInteractive`](../nnUNet-2.2/nnunetv2/training/dataloading/data_loader_3d_interactive.py)
  skeletonizes the ground-truth lesion mask and dilates it into a short "tube" each iteration, standing in for a
  sparse user-drawn correction stroke during training.
- **Schedule**: same curriculum warmup phases as Stage 1, applied on top of the Stage‑1 weights.

## Running

```bash
./train_fdg.sh DATASET_ID_FDG FOLD /path/to/stage1/fold_F/checkpoint_final.pth
./train_psma.sh DATASET_ID_PSMA FOLD /path/to/stage1/fold_F/checkpoint_final.pth
```

Use the **Stage‑1 checkpoint for the matching fold and tracer** as `PRETRAINED_WEIGHTS` — Stage 2 is a
continuation of that specific fold's training, not a fresh run.

After all folds/tracers are trained, see [`../ensembling`](../ensembling) to combine checkpoints for inference.
