# =============================================================================
# Grand Challenge submission image (x86-64). Runs as a non-root user.
#
# Build from the REPO ROOT (not this directory) so the build context can see
# both nnUNet-2.2/ (top level) and this submission/ folder:
#
#   cd AUTOPET2026-MEDAI
#   bash submission/download_weights.sh          # fetches trained weights first
#   docker build --platform=linux/amd64 -f submission/Dockerfile \
#                -t autopet_interactive_submit .
#
# This is byte-for-byte the code that produced our final-test-set submission
# (extracted directly from that image), plus the trained weights fetched by
# download_weights.sh. See submission/README.md for details.
# =============================================================================
FROM --platform=linux/amd64 pytorch/pytorch:2.12.0-cuda12.6-cudnn9-runtime
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# --- non-root user + writable dirs (GC rejects images that run as root) ------
RUN groupadd -r algorithm && \
    useradd -m --no-log-init -r -g algorithm algorithm && \
    mkdir -p /opt/algorithm /input /output /output/images/tumor-lesion-segmentation && \
    chown -R algorithm:algorithm /opt/algorithm /input /output

USER algorithm
WORKDIR /opt/algorithm
ENV PATH="/home/algorithm/.local/bin:${PATH}"

# --- python deps (installed into the user site) ------------------------------
RUN python -m pip install -U pip && \
    python -m pip install SimpleITK nibabel monai timm fvcore batchgeneratorsv2 && \
    python -m pip install "numpy<2"

COPY --chown=algorithm:algorithm nnUNet-2.2/ /opt/algorithm/nnUNet-2.2/
RUN python -m pip install --user /opt/algorithm/nnUNet-2.2/nnunetv2/dynamic-network-architectures && \
    python -m pip install --user /opt/algorithm/nnUNet-2.2

# --- algorithm code ----------------------------------------------------------
COPY --chown=algorithm:algorithm submission/process_work.py /opt/algorithm/
COPY --chown=algorithm:algorithm submission/classify_pet.py /opt/algorithm/

# --- weights -------------------------------------------------------------
# Populated by submission/download_weights.sh before building - see
# submission/README.md. Only the fold(s) actually used by MODELS in
# process_work.py are included.
COPY --chown=algorithm:algorithm submission/weights/nnUNet_results /opt/algorithm/nnUNet_results
COPY --chown=algorithm:algorithm submission/weights/classifier/tracer_classifier.pt /opt/algorithm/classifier/tracer_classifier.pt

# --- nnU-Net env -------------------------------------------------------------
RUN mkdir -p /opt/algorithm/nnUNet_raw /opt/algorithm/nnUNet_preprocessed
ENV nnUNet_raw="/opt/algorithm/nnUNet_raw"
ENV nnUNet_preprocessed="/opt/algorithm/nnUNet_preprocessed"
ENV nnUNet_results="/opt/algorithm/nnUNet_results"

ENTRYPOINT ["python", "-m", "process_work"]
