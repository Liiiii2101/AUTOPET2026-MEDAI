import hashlib
import json
import os
import shutil
import subprocess

import numpy as np
import nibabel as nib
import SimpleITK
import torch
from scipy.ndimage import gaussian_filter


# =============================================================================
#  CONFIGURATION  ---  EDIT THESE TO MATCH YOUR TRAINED MODELS
# =============================================================================
TRAINER = "nnUNetTrainer"      # default trainer if a model omits "trainer"
PLANS = "nnUNetPlans"          # default plans   if a model omits "plans"

# Channel sources:
#   "ct"          -> CT volume                       (/input/images/ct)
#   "pet"         -> PET/SUV volume                  (/input/images/pet)
#   "organ"       -> organ-prediction map            (from the "organ" model)
#   "interaction" -> combined initial-pred+scribbles (built per interactive step)
MODELS = {
    # Organ segmentation. CT-only. Used by BOTH tracers' initial models now.
    "organ":            {"dataset_id": 101, "config": "3d_fullres", "fold": "11",  # matches proven submit11 checkpoint (verified byte-identical)
                         "trainer": "STUNetTrainer_small",
                         "channels": ["ct"]},

    # ----- INITIAL prediction (no scribbles yet) -----
    # Both initial models now ensemble the full 10-fold CV (0-9) + the fold_11
    # full-data retrain (fold_12 exists on disk too but is deliberately left
    # out - same as everywhere else in this file - it's an extra/holdout
    # fold, not part of the intended ensemble).
    "fdg_initial":      {"dataset_id": 100, "config": "3d_fullres", "fold": "0 1 2 3 4 5 6 7 8 9 11",
                         "trainer": "MyCustomCurriculumTrainer",
                         "channels": ["ct", "pet", "organ"]},
    "psma_initial":     {"dataset_id": 247, "config": "3d_fullres", "fold": "0 1 2 3 4 5 6 7 8 9 11",
                         "trainer": "MyCustomCurriculumTrainer",
                         "channels": ["ct", "pet", "organ"]},   # NOW includes organ channel
    # Fallback for oversized PSMA volumes (see PSMA_BIG_VOXEL_THRESHOLD below):
    # the proven, no-organ-channel Dataset245 model. GC confirmed a real OOM
    # kill on a 963-slice PSMA case running the organ-channel path above; the
    # same case (and everything smaller we've tried) is fine on this one, so
    # route oversized PSMA cases here instead of paying the extra organ-model
    # pass that path needs.
    "psma_initial_small": {"dataset_id": 245, "config": "3d_fullres", "fold": "0",
                         "trainer": "MyCustomCurriculumTrainer",
                         "channels": ["ct", "pet"]},

    # ----- INTERACTIVE prediction -----
    # interaction channel = tracer-specific scheme, see INTERACTION below.
    # fdg: full 0-9+11 ensemble except fold_2, which is deliberately excluded
    # - that checkpoint on disk is literally named "fold_2_wrong_5f" (a
    # known-bad training run).
    "fdg_interactive":  {"dataset_id": 100, "config": "3d_fullres", "fold": "0 1 3 4 5 6 7 8 9 11",
                         "trainer": "MyCustomCurriculumTrainerSegPre", "plans": "nnUNetPlans192",
                         "channels": ["ct", "pet", "interaction"]},
    # psma: full 0-9 CV + the fold_11 full-data retrain, same as the initial
    # models above (fold_12 exists on disk too but stays excluded - extra/
    # holdout fold, not part of the intended ensemble).
    "psma_interactive": {"dataset_id": 247, "config": "3d_fullres", "fold": "0 1 2 3 4 5 6 7 8 9 11",
                         "trainer": "MyCustomCurriculumTrainerSegPre",
                         "channels": ["ct", "pet", "interaction"]},
}

CLASSIFIER_CKPT = "/opt/algorithm/classifier/tracer_classifier.pt"

# Label values burned into the interaction channel. Both tracers now use the
# same scheme: initial prediction as base (label 1) + fg scribbles (2) + bg
# scribbles (3). (Previously PSMA was scribbles-only, no initial-pred base -
# the new re-uploaded psma_interactive model was retrained to use it too.)
INTERACTION = {
    "fdg":  {"use_initial": True, "pred_label": 1, "fg_label": 2, "bg_label": 3},
    "psma": {"use_initial": True, "pred_label": 1, "fg_label": 2, "bg_label": 3},
}

# SUV thresholds for post-processing (PET below this is zeroed).
SUV_THRESHOLD = {"fdg": 1.5, "psma": 1.0}

# Above this many CT voxels, PSMA cases route to "psma_initial_small" (no
# organ channel) instead of "psma_initial" (organ channel) - see MODELS
# above. Only PSMA is size-gated: FDG's organ+3-channel initial path is
# unchanged from the proven config and has been confirmed fine even on a
# much bigger (105M-voxel) case, so this isn't a general "big volume" guard,
# just a targeted workaround for the one confirmed-new failure mode.
# Calibration (from all 600 PSMA cases in the training set,
# check_head_low444729.out): sizes cluster tightly, median 17.16M voxels,
# p99.5 20.3M, with one other outlier at 24.3M. The GC-confirmed OOM case
# (200x200x963 = 38.52M voxels) is the single LARGEST of all 600 cases by a
# wide margin - a genuine, singular outlier, not just "on the large side".
# 28M sits in the clear gap between the normal cluster (<=24.3M) and that
# one case (38.52M), so this should only ever trigger for genuinely
# exceptional cases, not the bulk of normal-sized PSMA scans.
PSMA_BIG_VOXEL_THRESHOLD = 28_000_000

# Persistent, mounted location that survives between iterations (NOT the image
# filesystem, which is wiped by `docker run --rm`). Confirmed by organisers
# (2026-08-07): "/output" persists across iterations on the FINAL test set
# (NOT the preliminary set, which is single-iteration / no persistence - see
# [[autopet-persistence-and-metrics]]). Used to cache the initial prediction
# and organ map so interactive steps 2-6 don't redo that work from scratch.
STATE_BASE = "/output/state"
# =============================================================================


class AutopetInteractive:

    def __init__(self):
        self.input_path = "/input/"
        self.output_path = "/output/images/tumor-lesion-segmentation/"
        self.gc_json_path = os.path.join(self.input_path, "lesion-clicks.json")
        self.work_base = "/opt/algorithm/work"   # ephemeral scratch (per run)
        self.case_id = "case"
        self._case_key_cache = {}   # ct_nii path -> md5 key, memoized within this run

    # ------------------------------------------------------------------ #
    #  config validation (fail fast on typos)                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def validate_models():
        valid_channels = {"ct", "pet", "organ", "interaction"}
        for key, cfg in MODELS.items():
            for required in ("dataset_id", "config", "fold", "channels"):
                if required not in cfg:
                    raise KeyError(
                        f"MODELS['{key}'] is missing '{required}' "
                        f"(check for typos / missing commas). Got keys: {list(cfg)}")
            bad = set(cfg["channels"]) - valid_channels
            if bad:
                raise ValueError(f"MODELS['{key}'] has unknown channels {bad}")
            tr = cfg.get("trainer", TRAINER)
            pl = cfg.get("plans", PLANS)
            print(f"[cfg] {key:16s} -d {cfg['dataset_id']} -c {cfg['config']} "
                  f"-f {cfg['fold']} -tr {tr} -p {pl}  channels={cfg['channels']}")

    # ------------------------------------------------------------------ #
    #  io helpers                                                        #
    # ------------------------------------------------------------------ #
    def convert_mha_to_nii(self, mha_input_path, nii_out_path):
        SimpleITK.WriteImage(SimpleITK.ReadImage(mha_input_path), nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):
        SimpleITK.WriteImage(SimpleITK.ReadImage(nii_input_path), mha_out_path, True)

    @staticmethod
    def _clean_dir(path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    def check_gpu(self):
        print("GPU available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("Device:", torch.cuda.get_device_name(0))

    # ------------------------------------------------------------------ #
    #  persistent per-case state                                         #
    # ------------------------------------------------------------------ #
    def _case_key(self, ct_nii):
        if ct_nii in self._case_key_cache:
            return self._case_key_cache[ct_nii]
        h = hashlib.md5()
        with open(ct_nii, "rb") as f:
            h.update(f.read())
        key = h.hexdigest()[:16]
        self._case_key_cache[ct_nii] = key
        return key

    def _state_dir(self, ct_nii):
        d = os.path.join(STATE_BASE, self._case_key(ct_nii))
        os.makedirs(d, exist_ok=True)
        return d

    def _cached_file(self, ct_nii, name):
        """Return the path to a persisted artifact from an earlier iteration
        of THIS case, if present, else None. Opportunistic and fail-open: on
        the preliminary test set /output has no cross-iteration persistence
        (and is only ever queried once per case anyway), so any failure here
        just means "recompute", never a hard error."""
        try:
            path = os.path.join(self._state_dir(ct_nii), name)
            if os.path.exists(path):
                print(f"[STATE] CACHE HIT for {name} -> reusing result from an earlier "
                      f"iteration, skipping recompute")
                return path
            print(f"[STATE] CACHE MISS for {name} -> no earlier-iteration result found, "
                  f"falling back to recomputing it now")
        except Exception as e:
            print(f"[STATE] cache lookup for {name} failed ({e}); falling back to recomputing")
        return None

    def _save_cache(self, ct_nii, src_path, name):
        """Persist an artifact for reuse by later iterations of this case.
        Best-effort - a failure to persist must never fail the current
        prediction, so this only ever logs and continues."""
        try:
            dst = os.path.join(self._state_dir(ct_nii), name)
            shutil.copyfile(src_path, dst)
            print(f"[STATE] cached {name} -> {dst}")
        except Exception as e:
            print(f"[STATE] failed to cache {name} ({e}); continuing without persistence")

    # ------------------------------------------------------------------ #
    #  inputs                                                            #
    # ------------------------------------------------------------------ #
    def load_inputs(self):
        os.makedirs(self.work_base, exist_ok=True)
        src_dir = os.path.join(self.work_base, "src")
        self._clean_dir(src_dir)

        ct_mha = os.listdir(os.path.join(self.input_path, "images/ct/"))[0]
        pet_mha = os.listdir(os.path.join(self.input_path, "images/pet/"))[0]
        uuid = os.path.splitext(ct_mha)[0]

        ct_nii = os.path.join(src_dir, "ct.nii.gz")
        pet_nii = os.path.join(src_dir, "pet.nii.gz")
        self.convert_mha_to_nii(os.path.join(self.input_path, "images/ct/", ct_mha), ct_nii)
        self.convert_mha_to_nii(os.path.join(self.input_path, "images/pet/", pet_mha), pet_nii)

        scribbles = {"tumor": [], "background": []}
        if os.path.exists(self.gc_json_path):
            with open(self.gc_json_path, "r") as f:
                scribbles = self.gc_to_swfastedit_format(json.load(f))

        is_initial = (len(scribbles["tumor"]) + len(scribbles["background"])) == 0
        print(f"uuid={uuid} is_initial={is_initial} "
              f"#fg={len(scribbles['tumor'])} #bg={len(scribbles['background'])}")
        return uuid, ct_nii, pet_nii, scribbles, is_initial

    @staticmethod
    def gc_to_swfastedit_format(gc_dict):
        swfast = {"tumor": [], "background": []}
        for p in gc_dict.get("points", []):
            if p["name"] == "tumor":
                swfast["tumor"].append(p["point"])
            elif p["name"] == "background":
                swfast["background"].append(p["point"])
        return swfast

    # ------------------------------------------------------------------ #
    #  tracer classifier  (FDG vs PSMA)                                  #
    # ------------------------------------------------------------------ #
    def classify_tracer(self, pet_nii):
        """Return "fdg" or "psma" from the PET volume only, using the MIP
        ResNet18 classifier in classify_pet.py."""
        if not os.path.exists(CLASSIFIER_CKPT):
            print(f"[WARN] no classifier at {CLASSIFIER_CKPT}; defaulting to 'psma'.")
            return "psma"
        from classify_pet import classify_pet   # lazy import (needs torchvision)
        tracer = classify_pet(pet_nii, CLASSIFIER_CKPT)
        if tracer not in ("fdg", "psma"):
            raise ValueError(f"classify_pet returned unexpected tracer: {tracer!r}")
        return tracer

    # ------------------------------------------------------------------ #
    #  channel builders                                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _burn_points(vol, points, label, shape):
        """Write `label` at each valid 3D point. Skips malformed points
        (not length-3) and out-of-bounds coordinates instead of crashing."""
        for p in points or []:
            if not hasattr(p, "__len__") or len(p) != 3:
                print(f"[WARN] skipping malformed scribble point: {p!r}")
                continue
            x, y, z = int(p[0]), int(p[1]), int(p[2])
            if 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]:
                vol[x, y, z] = label

    def build_interaction_channel(self, tracer, initial_pred_nii, scribbles,
                                  ref_pet_nii, out_path):
        """Build the (single) interaction channel using the tracer-specific
        scheme in INTERACTION. FDG uses the initial prediction as a base; PSMA
        uses scribbles only."""
        scheme = INTERACTION[tracer]
        ref = nib.load(ref_pet_nii)
        shape = ref.shape

        if scheme["use_initial"]:
            if initial_pred_nii is None or not os.path.exists(initial_pred_nii):
                raise ValueError(f"{tracer} interaction needs an initial prediction")
            # dtype=float32: get_fdata() defaults to float64, doubling RAM for
            # no reason - this is a 0/1 label mask, exact in float32 either way.
            pred = nib.load(initial_pred_nii).get_fdata(dtype=np.float32)
            if pred.shape != shape:
                raise ValueError(f"initial pred shape {pred.shape} != ref {shape}")
            vol = (pred > 0).astype(np.float32) * scheme["pred_label"]
        else:
            vol = np.zeros(shape, dtype=np.float32)

        self._burn_points(vol, scribbles.get("tumor", []), scheme["fg_label"], shape)
        self._burn_points(vol, scribbles.get("background", []), scheme["bg_label"], shape)

        nib.save(nib.Nifti1Image(vol, ref.affine, ref.header), out_path)

    def run_organ_model(self, ct_nii):
        cached = self._cached_file(ct_nii, "organ.nii.gz")
        if cached:
            return cached
        out = os.path.join(self.work_base, "organ_result")
        result = self._predict("organ", {"ct": ct_nii}, out)
        self._save_cache(ct_nii, result, "organ.nii.gz")
        return result

    # ------------------------------------------------------------------ #
    #  generic nnUNet prediction                                         #
    # ------------------------------------------------------------------ #
    def _stage_inputs(self, model_key, sources, stage_dir):
        self._clean_dir(stage_dir)
        for idx, ch in enumerate(MODELS[model_key]["channels"]):
            if ch not in sources:
                raise KeyError(f"Model '{model_key}' needs channel '{ch}'; "
                               f"have {list(sources)}")
            shutil.copyfile(sources[ch],
                            os.path.join(stage_dir, f"{self.case_id}_{idx:04d}.nii.gz"))

    def _model_dir(self, cfg):
        """Resolve the trained-model folder for a MODELS entry."""
        from nnunetv2.utilities.dataset_name_id_conversion import (
            convert_id_to_dataset_name)
        from nnunetv2.paths import nnUNet_results
        ds_name = convert_id_to_dataset_name(cfg["dataset_id"])
        tr = cfg.get("trainer", TRAINER)
        pl = cfg.get("plans", PLANS)
        return os.path.join(nnUNet_results, ds_name,
                            f"{tr}__{pl}__{cfg['config']}")

    @staticmethod
    def _parse_folds(fold_str):
        """"0 1 all" -> (0, 1, 'all'). nnU-Net keeps the 'all' fold name as a
        string (checkpoints live under a literal 'fold_all/' dir); every other
        token is a numbered fold and must be cast to int."""
        return tuple(tok if tok == "all" else int(tok) for tok in str(fold_str).split())

    def _build_predictor(self, model_key):
        """Build a fresh nnUNetPredictor for this model. We intentionally do NOT
        cache it: on GC each iteration is a separate container invocation and each
        model runs once per call, so caching gives no reuse but keeps weights
        resident (peak-memory risk on the 16 GB / FDG 3-model path). The speed
        win comes from running in-process (no per-model torch/nnU-Net re-import),
        not from caching."""
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        cfg = MODELS[model_key]
        folds = self._parse_folds(cfg["fold"])
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,          # equivalent to --disable_tta
            perform_everything_on_gpu=True,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(
            self._model_dir(cfg),
            use_folds=folds,
            checkpoint_name="checkpoint_final.pth",
        )
        return predictor

    def _predict(self, model_key, sources, out_dir):
        cfg = MODELS[model_key]
        stage_dir = os.path.join(self.work_base, f"{model_key}_in")
        self._stage_inputs(model_key, sources, stage_dir)
        self._clean_dir(out_dir)
        print(f"[{model_key}] d={cfg['dataset_id']} c={cfg['config']} "
              f"f={cfg['fold']} tr={cfg.get('trainer', TRAINER)} "
              f"p={cfg.get('plans', PLANS)} channels={cfg['channels']}")

        predictor = self._build_predictor(model_key)
        try:
            predictor.predict_from_files(
                stage_dir,
                out_dir,
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
                folder_with_segs_from_prev_stage=None,
                num_parts=1,
                part_id=0,
            )
        finally:
            # Free the model before the next one loads, to cap peak memory.
            # Note: we do NOT move the network to CPU first (that would briefly
            # raise host RAM, which is the tighter limit here); del + gc + GPU
            # cache clear is enough.
            del predictor
            import gc as _gc
            _gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        result_nii = os.path.join(out_dir, f"{self.case_id}.nii.gz")
        if not os.path.exists(result_nii):
            raise FileNotFoundError(f"[{model_key}] missing output {result_nii}")
        return result_nii

    # ------------------------------------------------------------------ #
    #  routing                                                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ct_voxel_count(ct_nii):
        """Cheap size check - just the header shape, no need to load pixel
        data. Used to route oversized PSMA cases away from the organ-channel
        model (see PSMA_BIG_VOXEL_THRESHOLD)."""
        shape = nib.load(ct_nii).shape
        n = 1
        for d in shape:
            n *= d
        return n

    def _run_initial(self, tracer, ct_nii, pet_nii):
        """Run the initial model for this tracer and return the post-processed
        initial mask. On the final test set (persistent /output, see
        STATE_BASE) this is only actually computed once per case: the result
        is cached, and every later interactive iteration - which also needs
        this as the base of its interaction channel, since INTERACTION now
        has use_initial=True for both tracers - reuses the cached file
        instead of rerunning the organ + initial model passes. On the
        preliminary set (no persistence) the cache simply never hits and this
        recomputes every time, same as before.
        Driven by MODELS[...]['channels'], not hard-coded per tracer, so this
        stays correct regardless of which tracer's initial model needs the
        organ map (originally FDG-only; psma_initial now needs it too since
        it moved to the organ-channel Dataset247 model) - except for
        oversized PSMA volumes, which route to psma_initial_small instead
        (GC-confirmed OOM kill on the organ-channel path for a 963-slice
        PSMA case; see PSMA_BIG_VOXEL_THRESHOLD)."""
        cached = self._cached_file(ct_nii, "initial_pred_pp.nii.gz")
        if cached:
            return cached

        if tracer == "fdg":
            model_key = "fdg_initial"
        else:
            n_voxels = self._ct_voxel_count(ct_nii)
            if n_voxels > PSMA_BIG_VOXEL_THRESHOLD:
                model_key = "psma_initial_small"
                print(f"[STAGE] -> PSMA case has {n_voxels:,} CT voxels (> "
                      f"{PSMA_BIG_VOXEL_THRESHOLD:,}); routing to "
                      f"psma_initial_small (no organ channel) to avoid the "
                      f"GC-confirmed OOM on the organ-channel path")
            else:
                model_key = "psma_initial"
        sources = {"ct": ct_nii, "pet": pet_nii}
        if "organ" in MODELS[model_key]["channels"]:
            sources["organ"] = self.run_organ_model(ct_nii)
        seg = self._predict(model_key, sources,
                            os.path.join(self.work_base, f"{model_key}_result"))
        result = self.postprocess(seg, pet_nii, tracer, interaction_nii=None)
        self._save_cache(ct_nii, result, "initial_pred_pp.nii.gz")
        return result

    def predict(self, ct_nii, pet_nii, scribbles, is_initial):
        tracer = self.classify_tracer(pet_nii)
        print("Predicted tracer:", tracer)

        if is_initial:
            return self._run_initial(tracer, ct_nii, pet_nii)

        # Interactive step. The interaction channel is tracer-specific:
        #   FDG  -> regenerate the initial prediction in-call and use it as the
        #           channel base (+ scribbles).
        #   PSMA -> scribbles only, so no initial model is run here (also faster).
        scheme = INTERACTION[tracer]
        initial_pred = None
        if scheme["use_initial"]:
            initial_pred = os.path.join(self.work_base, "initial_pred.nii.gz")
            shutil.copyfile(self._run_initial(tracer, ct_nii, pet_nii), initial_pred)

        interaction = os.path.join(self.work_base, "interaction.nii.gz")
        self.build_interaction_channel(tracer, initial_pred, scribbles, pet_nii, interaction)

        sources = {"ct": ct_nii, "pet": pet_nii, "interaction": interaction}
        model_key = "fdg_interactive" if tracer == "fdg" else "psma_interactive"
        seg = self._predict(model_key, sources,
                            os.path.join(self.work_base, f"{model_key}_result"))
        # SUV threshold + enforce scribbles using the tracer-specific interaction map.
        return self.postprocess(seg, pet_nii, tracer, interaction_nii=interaction)

    # ------------------------------------------------------------------ #
    #  post-processing                                                   #
    # ------------------------------------------------------------------ #
    def postprocess(self, seg_nii, pet_nii, tracer, interaction_nii=None):
        """
        Apply an SUV threshold (zero out predictions where PET is too low) and,
        when an interaction map is present, force foreground scribbles to lesion
        and background scribbles to 0. Returns the path to the cleaned mask.
        """
        # dtype=float32 on every get_fdata() below: defaults to float64 (2x RAM
        # for no gain - the PET niis are stored as float32 on disk to begin
        # with, so float64 promotion here never added real precision, just
        # wasted memory; the label/mask arrays are small exact integers,
        # lossless in float32 regardless). Matters on big whole-body cases.
        seg_img = nib.load(seg_nii)
        seg_arr = seg_img.get_fdata(dtype=np.float32)
        pet_arr = nib.load(pet_nii).get_fdata(dtype=np.float32)
        if pet_arr.shape != seg_arr.shape:
            raise ValueError(f"PET shape {pet_arr.shape} != seg shape {seg_arr.shape}")

        thr = SUV_THRESHOLD.get(tracer, 1.0)
        seg_arr[pet_arr < thr] = 0

        if interaction_nii is not None and os.path.exists(interaction_nii):
            scheme = INTERACTION[tracer]
            prior = nib.load(interaction_nii).get_fdata(dtype=np.float32)
            seg_arr[prior == scheme["fg_label"]] = 1   # foreground scribble -> lesion
            seg_arr[prior == scheme["bg_label"]] = 0   # background scribble -> not lesion

        seg_arr = np.where(seg_arr == 1, 1, 0).astype(np.int16)

        out_path = seg_nii.replace(".nii.gz", "_pp.nii.gz")
        nib.save(nib.Nifti1Image(seg_arr, seg_img.affine, seg_img.header), out_path)
        return out_path

    # ------------------------------------------------------------------ #
    #  outputs / entry                                                   #
    # ------------------------------------------------------------------ #
    def write_outputs(self, uuid, seg_nii):
        os.makedirs(self.output_path, exist_ok=True)
        out_mha = os.path.join(self.output_path, uuid + ".mha")
        self.convert_nii_to_mha(seg_nii, out_mha)
        print("Output written to:", out_mha)

    def _write_empty_mask(self, uuid, ref_nii):
        """Write an all-zero mask matching the reference geometry, as a safe
        fallback so a single failing case does not fail the whole submission."""
        os.makedirs(self.output_path, exist_ok=True)
        ref = nib.load(ref_nii)
        empty = np.zeros(ref.shape, dtype=np.int16)
        tmp = os.path.join(self.work_base, "empty.nii.gz")
        nib.save(nib.Nifti1Image(empty, ref.affine, ref.header), tmp)
        out_mha = os.path.join(self.output_path, uuid + ".mha")
        self.convert_nii_to_mha(tmp, out_mha)
        print(f"[WARN] wrote empty fallback mask to {out_mha}")

    def process(self):
        self.check_gpu()
        self.validate_models()
        uuid, ct_nii, pet_nii, scribbles, is_initial = self.load_inputs()
        try:
            seg_nii = self.predict(ct_nii, pet_nii, scribbles, is_initial)
            self.write_outputs(uuid, seg_nii)
        except Exception as e:
            import traceback
            print(f"[ERROR] prediction failed for {uuid}: {e}")
            traceback.print_exc()
            # Always emit a valid output so the case (and submission) does not fail.
            self._write_empty_mask(uuid, pet_nii)
        print("Done")


if __name__ == "__main__":
    print("START")
    AutopetInteractive().process()