# Submission — Grand Challenge inference container

This folder contains the exact code that built and ran our AutoPET-V final-test-set
submission (extracted directly from the succeeded submission image, byte-for-byte),
plus everything needed to rebuild it.

## Build

```bash
cd AUTOPET2026-MEDAI          # repo root
bash submission/download_weights.sh
docker build --platform=linux/amd64 -f submission/Dockerfile -t autopet_interactive_submit .
```

`download_weights.sh` fetches the trained weights from Google Drive (~5.5GB) into
`submission/weights/`, in the layout the Dockerfile expects. Weights are not
committed to git (see `.gitignore`) - they're too large for a repo.

## What's in the image

- `process_work.py` — the container entrypoint. Classifies each case as FDG or
  PSMA, runs the tracer-specific initial-prediction model (organ-conditioned
  nnU-Net, ensembled across folds), and on interactive iterations runs a
  second scribble-refinement model. Applies an SUV threshold and hard
  scribble-enforcement as post-processing. Caches the initial prediction and
  organ map under `/output/state/` so repeat iterations on the same case (the
  final test set persists `/output` between iterations) don't redo that work.
- `classify_pet.py` — FDG/PSMA tracer classifier (MIP + ResNet18), reusing the
  pretrained classifier published by Kalisch et al. for their AutoPET III
  submission (see `../tracer_classification/README.md`).
- `nnUNet-2.2/` (repo top level) — our fork of nnU-Net v2.2 with the STU-Net
  backbone, curriculum trainers, and two inference-time memory optimizations
  for very large whole-body volumes:
  - `float32` (not the nnU-Net default `float64`) on image loads.
  - Skips computing full softmax before argmax when only the hard
    segmentation is needed (mathematically exact, not an approximation —
    argmax(logits) == argmax(softmax(logits)) per voxel).
  - Converts the prediction array to plain numpy before it crosses the
    process boundary to the export worker, avoiding a `/dev/shm`-backed
    tensor IPC path that can overflow small shared-memory limits on large,
    many-class volumes.

## Models and folds actually used

| Model | Dataset | Trainer | Folds | Channels |
|---|---|---|---|---|
| `organ` | 101 (Organ10) | STUNetTrainer_small | 11 | CT |
| `fdg_initial` | 100 (autopet) | MyCustomCurriculumTrainer | 0,1,2,3,4,5,6,7,8,9,11 | CT, PET, organ |
| `psma_initial` | 247 (psma, organ-channel) | MyCustomCurriculumTrainer | 0,1,2,3,4,5,6,7,8,9,11 | CT, PET, organ |
| `psma_initial_small` | 245 (psma, no organ channel) | MyCustomCurriculumTrainer | 0 | CT, PET |
| `fdg_interactive` | 100 (autopet) | MyCustomCurriculumTrainerSegPre | 0,1,3,4,5,6,7,8,9,11 | CT, PET, interaction |
| `psma_interactive` | 247 (psma, organ-channel) | MyCustomCurriculumTrainerSegPre | 0,1,2,3,4,5,6,7,8,9,11 | CT, PET, interaction |

`psma_initial_small` is a fallback used only for PSMA cases above a CT-voxel-count
threshold, to avoid an out-of-memory failure mode confirmed on very large volumes
running the organ-channel model. See `process_work.py` for the exact threshold and
routing logic.

## Local testing

Grand Challenge runs the container with fixed resource flags (30GB memory, 8 CPUs,
single NVIDIA A10G / 24GB VRAM, 20 minutes/case, no network, `--cap-drop=ALL`).
For local sanity testing with the same flags:

```bash
docker run --rm \
    --memory=30g --memory-swap=30g \
    --network=none --cap-drop=ALL --security-opt=no-new-privileges \
    --shm-size=2g --gpus=all \
    -v "$(pwd)/input/:/input/" -v "$(pwd)/output/:/output/" \
    autopet_interactive_submit
```

`/input/images/ct/`, `/input/images/pet/` should each contain one `.mha` file for
the case; `/input/lesion-clicks.json` (optional) carries interactive scribbles in
the `{"points": [{"name": "tumor"|"background", "point": [x,y,z]}]}` format.
