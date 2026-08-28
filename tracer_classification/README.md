# Tracer classification (FDG vs. PSMA routing)

Per the paper, a dedicated classifier inspects each incoming study and routes it to the FDG- or PSMA-specific
branch (see [`../stage1_lesion_segmentation`](../stage1_lesion_segmentation) /
[`../stage2_interactive_refinement`](../stage2_interactive_refinement)) before segmentation runs.

**Status: not included in this repository snapshot.** During development, FDG and PSMA cases were routed using
the dataset's tracer metadata directly rather than a learned classifier, since tracer identity is known at
training time. The two branches were trained, validated, and ensembled completely independently
(never mixed) on that basis. Code for a standalone tracer-classification model — needed for e.g. a submission
container that cannot rely on metadata — is planned but not yet added here.
