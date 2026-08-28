# MEDAI — Tracer-Aware Interactive Segmentation for AutoPET V

**MEDAI**'s submission to the [AutoPET V challenge](https://autopet-v.grand-challenge.org/) (interactive
lesion segmentation in whole-body PET/CT). The pipeline pairs a self-supervised masked-autoencoder-pretrained 3D
[STU-Net](https://github.com/uni-medical/STU-Net) backbone with tracer-specific (FDG/PSMA) branches, an
anatomical organ prior, and a second interactive stage that refines the initial prediction from sparse
corrective scribbles — built inside a modified [nnU-Net v2.2](https://github.com/MIC-DKFZ/nnUNet) framework.


## Pipeline overview

![Pipeline overview: MAE pretraining on PET/CT, then per-tracer 1st-stage segmentation (CT + organ context) producing an initial prediction, scribble correction, and a 2nd-stage model producing the final prediction.](docs/assets/pipeline_overview.png)

```
Pretrain (self-supervised)  →  Tracer routing  →  Stage 1 (initial segmentation)  →  Stage 2 (interactive refinement)  →  Ensemble  →  Predict
 pretraining/                   FDG / PSMA         PET + CT + organ context           + cumulative scribbles              (per tracer)
 asynchronous MAE on             (see note below)   MyCustomCurriculumTrainer          MyCustomCurriculumTrainerSegPreSkel
 CT/PET patches                                     stage1_lesion_segmentation/        stage2_interactive_refinement/
```

1. **Pretraining** (`pretraining/`): a 3D masked-autoencoder (`STUNet_MAE`) learns general PET/CT
   representations before any lesion labels are used, masking CT and PET **independently** (asynchronous
   masking) so the encoder must exploit cross-modal, not just spatial, redundancy.
2. **Tracer routing** (`tracer_classification/`): each study is routed to an FDG- or PSMA-specific branch,
   since the two tracers differ substantially in uptake pattern and lesion appearance. Routing uses the
   pretrained tracer classifier published by [Kalisch et al.](https://github.com/hakal104/autoPETIII/) for
   their AutoPET III submission — see that folder for details and citation.
3. **Stage 1 — initial segmentation** (`stage1_lesion_segmentation/`): PET + CT + an anatomical organ prior
   (`organ_segmentation/`) are consumed to produce an initial lesion mask, using an MAE-pretrained STU-Net
   fine-tuned with a 3-phase curriculum schedule.
4. **Stage 2 — interactive refinement** (`stage2_interactive_refinement/`): the Stage‑1 prediction is combined
   with cumulative foreground/background scribbles (simulated during training via skeletonization of the
   ground-truth mask) and refined by a second network initialized from the Stage‑1 checkpoint.
5. **Ensembling**: fold- and checkpoint-level ensembling (via nnU-Net's `nnUNetv2_ensemble`) improves
   robustness across interaction steps and cohorts. FDG and PSMA are always kept separate.


FDG and PSMA share the same architecture and training pipeline end to end but are trained fully independently,
to account for tracer-specific appearance and error modes.

<!-- ## Data

Trained and evaluated on the official AutoPET V training data only (no external imaging data):

| Tracer | Studies | Patients | Source | Cohort |
|---|---|---|---|---|
| FDG | 1,014 | 900 | University Hospital Tübingen | Malignant melanoma, lymphoma, lung cancer, and negative controls |
| PSMA | 597 | 378 | LMU University Hospital Munich | Prostate cancer, with and without PSMA-avid lesions |

Each study is a co-registered whole-body CT + PET (SUV) volume pair with a binary manual lesion mask, provided
in nnU-Net's NIfTI format (CT/PET as separate channels). Preprocessing uses nnU-Net's default planning, with
tracer-specific target spacing/patch size chosen automatically by the planner:

| Tracer | Target spacing (mm) | Patch size |
|---|---|---|
| FDG | 3.0 × 2.03 × 2.03 | 128 × 128 × 128 |
| PSMA | 4.07 × 3.27 × 4.07 | 112 × 192 × 112 |

## Repository layout

```
.
├── pretraining/                     Stage 0 — self-supervised MAE pretraining (standalone, not part of nnU-Net)
│   ├── dataloader.py                    NPZDataset: loads preprocessed CT/PET patches
│   ├── model_asynchronous.py            STUNet_MAE model definition
│   └── train_asynchronous.py            Training script (single-GPU or DDP)
│
├── tracer_classification/           Stage — FDG/PSMA routing via Kalisch et al.'s pretrained classifier (see its README)
├── organ_segmentation/              Auxiliary anatomical prior (not included in this snapshot, see its README)
│
├── stage1_lesion_segmentation/      Stage 1 — initial lesion segmentation (wrapper scripts + docs)
├── stage2_interactive_refinement/   Stage 2 — scribble-guided refinement (wrapper scripts + docs)
└── nnUNet-2.2/                      Forked nnU-Net v2.2 with STU-Net + curriculum + interactive additions
    └── nnunetv2/
        ├── run/run_finetuning_stunet.py                    Fine-tuning entry point (loads pretrained weights)
        ├── training/dataloading/data_loader_3d_interactive.py   Skeleton-based scribble simulation
        └── training/nnUNetTrainer/
            ├── stunet/          STU-Net backbones (Stage 1 & 2 network variants)
            └── curriculum/      Curriculum-scheduled trainers (Stage 1 & 2 training loops)
```

See [`nnUNet-2.2/readme.md`](nnUNet-2.2/readme.md) for exactly what was changed relative to upstream nnU-Net,
and [`pretraining/README.md`](pretraining/README.md) for pretraining usage.

## Setup

Requires Python >= 3.9 and a CUDA-capable GPU (pretraining and fine-tuning are both 3D and memory-hungry).

```bash
# 1. Install the (modified) nnU-Net framework in editable mode
pip install -e nnUNet-2.2

# 2. Install additional dependencies used by the pretraining scripts
pip install -r requirements.txt
```

## Usage

### 1. Pretrain the MAE encoder

```bash
python pretraining/train_asynchronous.py \
    --data_dir /path/to/nnUNet_preprocessed/DatasetXXX/.../fold_all \
    --save_dir /path/to/pretrain_output \
    --epochs 500 --lr 1e-4 --batch 4 --mask_ratio 0.5
```

See [`pretraining/README.md`](pretraining/README.md) for all arguments.

### 2. Stage 1 — initial segmentation

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
stage1_lesion_segmentation/train_fdg.sh DATASET_ID FOLD /path/to/pretrain_output/checkpoint.pth
```

See [`stage1_lesion_segmentation/README.md`](stage1_lesion_segmentation/README.md).

### 3. Stage 2 — interactive refinement

```bash
stage2_interactive_refinement/train_fdg.sh DATASET_ID FOLD /path/to/stage1/fold_F/checkpoint_final.pth
```

See [`stage2_interactive_refinement/README.md`](stage2_interactive_refinement/README.md).

### 4. Ensemble and predict

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c 3d_fullres -tr MyCustomCurriculumTrainerSegPreSkel --save_probabilities
nnUNetv2_ensemble -i FOLD0_OUT FOLD1_OUT ... -o ENSEMBLE_OUT
```

FDG and PSMA are ensembled separately (never mixed), since each tracer branch is trained independently
end to end.

## Results

10-fold cross-validation Dice on the official AutoPET V training split.

**Stage 1** (initial segmentation):

| Fold | FDG Dice | PSMA Dice |
|---|---|---|
| 0 | 0.5534 | 0.5990 |
| 1 | 0.5750 | 0.5343 |
| 2 | 0.6551 | 0.5949 |
| 3 | 0.5124 | 0.5712 |
| 4 | 0.6464 | 0.5579 |
| 5 | 0.6365 | 0.6074 |
| 6 | 0.6221 | 0.5966 |
| 7 | 0.5642 | 0.6234 |
| 8 | 0.5093 | 0.6025 |
| 9 | 0.6145 | 0.5459 |
| **Mean ± Std.** | **0.5889 ± 0.0537** | **0.5833 ± 0.0293** |

**Stage 2** (after interactive refinement) — evaluation is still in progress; folds not yet run are blank:

| Fold | FDG Dice | PSMA Dice |
|---|---|---|
| 0 | 0.5604 | 0.6717 |
| 1 | 0.6278 | 0.6134 |
| 2 | 0.6621 | 0.6383 |
| 3 | 0.5565 | 0.5937 |
| 4 | — | 0.6153 |
| 5 | 0.6393 | 0.6853 |
| 6 | 0.6766 | — |
| 7 | 0.6270 | 0.6340 |
| 8 | 0.5630 | 0.6357 |
| 9 | 0.6446 | 0.6562 |

Final test-set performance will be added after the challenge evaluation.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE). This project is based on
[nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and [STU-Net](https://github.com/uni-medical/STU-Net), also
Apache 2.0 licensed. Tracer routing uses the MIT-licensed classifier from
[hakal104/autoPETIII](https://github.com/hakal104/autoPETIII/) (see [`tracer_classification/README.md`](tracer_classification/README.md)). -->
