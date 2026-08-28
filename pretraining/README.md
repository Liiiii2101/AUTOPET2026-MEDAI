# Pretraining (STU-Net Masked Autoencoder)

Self-supervised pretraining of a 3D STU-Net encoder via masked-patch reconstruction on unlabeled, preprocessed
CT/PET patches. Intended to produce encoder weights that are later loaded into the supervised STU-Net
fine-tuning trainer in [`nnUNet-2.2`](../nnUNet-2.2) (see the top-level [README](../README.md) for the full
pipeline).

## Files

| File | Purpose |
|---|---|
| `dataloader.py` | `NPZDataset` — loads 2-channel (CT + PET) `.npz` patches from a directory. Expects each `.npz` to contain a `patch` array. |
| `model_asynchronous.py` | `STUNet_MAE` — STU-Net-based encoder/decoder with random block masking and reconstruction loss. |
| `train_asynchronous.py` | Training loop. Supports single-GPU and multi-GPU (`torch.distributed` / DDP) training, mixed precision, and periodic reconstruction visualizations. |

## Data expectations

`--data_dir` must point at a folder of `.npz` files, each holding a `patch` array of shape
`(2, D, H, W)` (CT and PET channels stacked). These are typically produced by running nnU-Net's own
preprocessing over your dataset and pointing this script at the resulting patch folder, e.g.
`nnUNet_preprocessed/DatasetXXX_.../<plans>/fold_all`.

## Running

```bash
python train_asynchronous.py \
    --data_dir /path/to/preprocessed/patches \
    --save_dir /path/to/output \
    --epochs 500 \
    --lr 1e-4 \
    --batch 4 \
    --num_workers 4 \
    --mask_ratio 0.8
```

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | *(none — must be set)* | Directory of preprocessed `.npz` patches. |
| `--save_dir` | *(none — must be set)* | Output directory for checkpoints and reconstruction visualizations. |
| `--epochs` | 500 | Number of training epochs. |
| `--lr` | 1e-4 | Learning rate. |
| `--batch` | 4 | Batch size (per process, if using DDP). |
| `--num_workers` | 4 | DataLoader worker processes. |
| `--mask_ratio` | 0.9 | Fraction of each patch masked during reconstruction. A reasonable starting point differs by tracer/tumor type — the original experiments used ~0.5 for PSMA/FDG and ~0.8 for head & neck. |

> The `--data_dir`/`--save_dir` defaults baked into `train_asynchronous.py` are cluster paths from the
> original experiments — always pass your own via the CLI flags above.

### Multi-GPU (DDP)

Launch with `torchrun` to enable distributed training (the script detects `torch.distributed` environment
variables automatically):

```bash
torchrun --nproc_per_node=NUM_GPUS train_asynchronous.py --data_dir ... --save_dir ...
```

## Output

Checkpoints and (every few epochs) reconstruction visualizations of CT/PET slices — original vs. masked vs.
reconstructed — are written under `--save_dir` (visualizations in a `vis/` subfolder). The saved checkpoint's
encoder weights are what get passed to `run_finetuning_stunet.py` for supervised fine-tuning.
