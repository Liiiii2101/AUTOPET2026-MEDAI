# nnUNet Modifications

We have made specific changes to the nnUNet v2.2 codebase for the MEDAI AutoPET pipeline (see the
[top-level README](../README.md) for the full pipeline description). All other original nnU-Net code is
unchanged and only copied over.

## Changes

- **STU-Net backbones** — `nnunetv2/training/nnUNetTrainer/stunet/` (was a single flat `STUNetTrainer.py`;
  reorganized into a subpackage, see its [README](nnunetv2/training/nnUNetTrainer/stunet/README.md)):
  STU-Net trainer/architecture variants (`STUNetTrainer`, `_small`/`_base`/`_large`/`_huge`, dual-encoder
  gated-fusion variants, focal-loss and long-schedule ablations), plus `STUNetTrainerSegPre.py` which loads a
  finished Stage‑1 checkpoint instead of the raw MAE pretraining checkpoint.
- **Curriculum trainers** — `nnunetv2/training/nnUNetTrainer/curriculum/` (see its
  [README](nnunetv2/training/nnUNetTrainer/curriculum/README.md)): `MyCustomCurriculumTrainer*` classes
  implementing the paper's 3-phase schedule (decoder warmup → full-network warmup → full fine-tuning) for
  both Stage 1 (initial segmentation) and Stage 2 (interactive refinement).
- **Interactive/scribble dataloader** — `nnunetv2/training/dataloading/data_loader_3d_interactive.py`:
  `nnUNetDataLoader3DInteractive` simulates sparse user scribbles during Stage‑2 training by skeletonizing and
  dilating the ground-truth lesion mask.
- **Fine-tuning entry point** — `nnunetv2/run/run_finetuning_stunet.py`: wraps `run_training.py`'s entry point
  with STU-Net-compatible pretrained-weight loading (tolerant of a mismatched segmentation head, so an encoder
  pretrained on a different number of classes can still be loaded).

See the repo root ([`stage1_lesion_segmentation`](../stage1_lesion_segmentation),
[`stage2_interactive_refinement`](../stage2_interactive_refinement), [`ensembling`](../ensembling)) for how
these pieces are invoked end to end.
