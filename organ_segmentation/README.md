# Auxiliary organ segmentation

Per the paper, an auxiliary organ segmentation model provides explicit anatomical context to the lesion
segmentation networks (see [`../stage1_lesion_segmentation`](../stage1_lesion_segmentation)), helping
distinguish normal physiological tracer uptake from malignant lesions.

**Status: not included in this repository snapshot.** No standalone organ-segmentation training code or
pretrained organ model is present in this repo yet. If/when it is added, it belongs here, with its predictions
consumed by Stage 1 as an extra input channel alongside PET and CT.
