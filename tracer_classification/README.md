# Tracer classification (FDG vs. PSMA routing)

Each incoming study is routed to the FDG- or PSMA-specific branch (see
[`../stage1_lesion_segmentation`](../stage1_lesion_segmentation) /
[`../stage2_interactive_refinement`](../stage2_interactive_refinement)) before segmentation runs.

**This repository does not train its own tracer classifier.** Routing uses the pretrained tracer classifier
published by Kalisch et al. for their AutoPET III submission:

- Repo: [hakal104/autoPETIII](https://github.com/hakal104/autoPETIII/) (MIT License) — see `classify_pet.py`
  and pretrained weights (`tracer_classifier.pt`, linked from that repo's README).
- Paper: H. Kalisch, F. Hörst, K. Herrmann, J. Kleesiek, C. Seibold, *"AutoPET III challenge: Incorporating
  anatomical knowledge into nnUNet for lesion segmentation in PET/CT,"* [arXiv:2409.12155](https://arxiv.org/abs/2409.12155).

## Usage

1. Fetch `classify_pet.py` and `tracer_classifier.pt` from the repo above (see its README for the weights link).
2. Run it on each incoming PET/CT study to get an FDG/PSMA label.
3. Feed FDG-labeled studies into the FDG branch and PSMA-labeled studies into the PSMA branch of
   [`../stage1_lesion_segmentation`](../stage1_lesion_segmentation).

During our own training/validation, tracer identity was instead taken directly from the AutoPET dataset
metadata (known at training time), so FDG and PSMA models were trained and validated fully independently
without needing the classifier. The classifier above is what is used to route un-labeled cases at inference.
