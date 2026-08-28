# Stage 1 — initial lesion segmentation

The first-stage network consumes PET + CT and produces the initial lesion mask that Stage 2
(`../stage2_interactive_refinement`) later refines with scribbles. FDG and PSMA are trained as two
independent models on the same architecture and schedule (see the paper's Section 3, "Methods").

- **Trainer**: `MyCustomCurriculumTrainer` (`nnUNet-2.2/nnunetv2/training/nnUNetTrainer/curriculum/MyCustomCurriculumTrainer.py`)
- **Backbone**: STU-Net-Small, encoder initialized from the [MAE pretraining](../pretraining) checkpoint
- **Schedule**: decoder warmup (50 ep) → full-network warmup (50 ep) → full fine-tuning — see
  [`nnUNet-2.2/.../nnUNetTrainer/curriculum/README.md`](../nnUNet-2.2/nnunetv2/training/nnUNetTrainer/curriculum/README.md)
  for the exact phase breakdown

## Prerequisites

Standard nnU-Net dataset preparation and preprocessing for each tracer dataset:

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

## Running

```bash
./train_fdg.sh DATASET_ID_FDG FOLD /path/to/mae_pretrain_checkpoint.pth
./train_psma.sh DATASET_ID_PSMA FOLD /path/to/mae_pretrain_checkpoint.pth
```

Both scripts are thin wrappers around `nnUNet-2.2/nnunetv2/run/run_finetuning_stunet.py` with
`-tr MyCustomCurriculumTrainer` preset; run each of the 10 folds per tracer separately, then ensemble
(see [`../ensembling`](../ensembling)).

Output checkpoints land under nnU-Net's standard results folder:
`nnUNet_results/DatasetXXX_.../MyCustomCurriculumTrainer__nnUNetPlans__3d_fullres/fold_F/checkpoint_final.pth`
— this is the checkpoint Stage 2 resumes from.
