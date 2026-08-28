# `stunet/` — STU-Net backbones & network definitions

Trainer subclasses that swap nnU-Net's default architecture for **STU-Net** (scalable, transferable U-Net),
plus the STU-Net `nn.Module` definitions themselves (each trainer file also defines the network classes it
needs — this mirrors the upstream nnU-Net convention of colocating a trainer with its architecture).

| File | Role |
|---|---|
| `STUNetTrainer.py` | Base STU-Net trainer + size variants (`_small`, `_base`, `_large`, `_huge`) and encoder-loading variants used for **Stage 1** (initial lesion segmentation). `STUNetTrainer_small_pretrain_location` and `STUNetTrainer_small_pretrain_STUNet_DualEncoder_gatefuse` load encoder weights from the [MAE pretraining checkpoint](../../../../../pretraining) (see [`stage1_lesion_segmentation`](../../../../../stage1_lesion_segmentation)). |
| `STUNetTrainerSegPre.py` | Same class hierarchy as `STUNetTrainer.py`, but its encoder-loading variants instead load a **finished Stage‑1 segmentation checkpoint** (not the raw MAE checkpoint) as the starting point — "SegPre" = *initialized from a previous‑stage Segmentation checkpoint*. Used for **Stage 2** (interactive refinement), see [`stage2_interactive_refinement`](../../../../../stage2_interactive_refinement). |
| `STUNetTrainer_base.py`, `STUNetTrainer_large.py`, `STUNetTrainer_small.py` | Standalone architecture-size variants without pretrained-encoder loading. |
| `STUNetTrainer_small_1500ep.py`, `STUNetTrainer_small_FL.py`, `STUNetTrainer_small_noSmooth.py` | `STUNetTrainer_small` variants with a longer schedule, a focal-loss term, or a modified compound loss, respectively — used in ablations. |

`STUNetTrainer.py` and `STUNetTrainerSegPre.py` intentionally duplicate most of their class hierarchy; the only
functional difference is *which checkpoint gets loaded into the encoder* (hardcoded per training run — swap the
path before launching a fold/tracer). Both are kept because they are wired to different downstream curriculum
trainers (see [`../curriculum`](../curriculum)) rather than merged, to avoid changing training behavior during
this reorganization.

> The hardcoded checkpoint paths in these files (`/projects/lcy_data/...`) point at the team's original HPC
> cluster storage and **must be edited to your own paths** before training.
