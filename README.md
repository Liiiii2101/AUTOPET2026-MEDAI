# AUTOPET2026-MEDAI

Our pipeline for the [AutoPET 2026](https://autopet-iii.grand-challenge.org/) challenge (whole-body PET/CT lesion
segmentation), built on a masked-autoencoder pretraining stage followed by supervised fine-tuning of an
[STU-Net](https://github.com/uni-medical/STU-Net) architecture inside a modified [nnU-Net v2.2](https://github.com/MIC-DKFZ/nnUNet) framework.

## Pipeline overview

```
1. Pretrain (self-supervised)        2. Fine-tune (supervised)             3. Predict
   pretrain/train_asynchronous.py  →    nnUNet-2.2 STUNetTrainer         →    nnUNetv2_predict
   (masked-autoencoder on           (loads pretrained encoder weights      (sliding-window inference
    CT/PET patches, STUNet_MAE)      via run_finetuning_stunet.py)          on new cases)
```

1. **Pretraining** (`pretrain/`): a 3D masked-autoencoder (`STUNet_MAE`) is trained on unlabeled, preprocessed
   CT/PET patches (`.npz` files) to learn general PET/CT representations before any lesion labels are used.
2. **Fine-tuning** (`nnUNet-2.2/`): the pretrained encoder weights are loaded into an STU-Net trainer
   (`STUNetTrainer*`) and fine-tuned for lesion segmentation using the standard nnU-Net v2.2 training loop.
3. **Inference**: trained models are applied with nnU-Net's standard prediction entry points
   (`nnUNetv2_predict`).

## Repository layout

```
.
├── pretrain/          Self-supervised MAE pretraining code (standalone, not part of nnU-Net)
│   ├── dataloader.py            NPZDataset: loads preprocessed CT/PET patches
│   ├── model_asynchronous.py    STUNet_MAE model definition
│   └── train_asynchronous.py    Training script (single-GPU or DDP)
│
└── nnUNet-2.2/        Forked copy of nnU-Net v2.2 with STU-Net additions — see nnUNet-2.2/readme.md
    └── nnunetv2/
        ├── run/run_finetuning_stunet.py           Entry point for STU-Net fine-tuning with pretrained weights
        └── training/nnUNetTrainer/STUNetTrainer*.py  STU-Net trainer variants (base/small/large/...)
```

See [`nnUNet-2.2/readme.md`](nnUNet-2.2/readme.md) for exactly what was changed relative to upstream nnU-Net,
and [`pretrain/README.md`](pretrain/README.md) for pretraining usage.

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
python pretrain/train_asynchronous.py \
    --data_dir /path/to/nnUNet_preprocessed/DatasetXXX/.../fold_all \
    --save_dir /path/to/pretrain_output \
    --epochs 500 --lr 1e-4 --batch 4 --mask_ratio 0.8
```

`--data_dir` should point at a directory of preprocessed `.npz` patches (e.g. produced by nnU-Net's
preprocessing pipeline). See [`pretrain/README.md`](pretrain/README.md) for all arguments.

### 2. Fine-tune with nnU-Net + STU-Net

Standard nnU-Net dataset preparation/preprocessing (`nnUNetv2_plan_and_preprocess`) applies as usual, then
fine-tune with the STU-Net trainer, pointing it at the pretraining checkpoint:

```bash
python nnUNet-2.2/nnunetv2/run/run_finetuning_stunet.py \
    DATASET_ID 3d_fullres FOLD -tr STUNetTrainer -pretrained_weights /path/to/pretrain_output/checkpoint.pth
```

### 3. Predict

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c 3d_fullres -tr STUNetTrainer
```

## License

Apache License 2.0 — see [`LICENSE`](LICENSE). This project is based on
[nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and [STU-Net](https://github.com/uni-medical/STU-Net), also
Apache 2.0 licensed.
