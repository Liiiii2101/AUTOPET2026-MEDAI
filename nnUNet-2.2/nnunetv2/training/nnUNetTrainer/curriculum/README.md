# `curriculum/` — curriculum-scheduled trainers (Stage 1 & Stage 2)

Implements the three-phase curriculum schedule described in the paper (Section "Training and test parameters"):
1. **Decoder warmup** (epochs 0–50): pretrained encoder frozen, only the randomly-initialized decoder trains, `lr = 1e-5`.
2. **Full-network warmup** (epochs 50–100): encoder unfrozen, whole network trains jointly.
3. **Full fine-tuning**: standard nnU-Net schedule, `lr = 1e-4`, DSC + CE loss.

| File | Pipeline stage | Backbone / dataloader |
|---|---|---|
| `MyCustomCurriculumTrainer.py` | **Stage 1** — initial lesion segmentation from PET+CT (+ MAE-pretrained encoder init) | `stunet.STUNetTrainer.STUNetTrainer_small_pretrain_location` / `..._DualEncoder_gatefuse`; plain nnU-Net dataloader |
| `MyCustomCurriculumTrainer_noSmooth.py` | Stage 1 variant, modified loss smoothing | same as above |
| `MyCustomCurriculumTrainerSegPre.py` | **Stage 2** setup — same curriculum schedule, but the backbone (`stunet.STUNetTrainerSegPre`) resumes from a finished Stage‑1 checkpoint instead of the MAE checkpoint | `stunet.STUNetTrainerSegPre`; plain nnU-Net dataloader |
| `MyCustomCurriculumTrainerSegPreSkel.py` | **Stage 2** — interactive refinement: extends `MyCustomCurriculumTrainerSegPre` with the scribble-simulation dataloader | `stunet.STUNetTrainerSegPre`; [`nnUNetDataLoader3DInteractive`](../../dataloading/data_loader_3d_interactive.py) (skeletonizes the lesion mask and dilates it into a "tube" to simulate a sparse foreground/background scribble each iteration) |

See [`stage1_lesion_segmentation`](../../../../../stage1_lesion_segmentation) and
[`stage2_interactive_refinement`](../../../../../stage2_interactive_refinement) at the repo root for runnable
training commands built on top of these trainers.
