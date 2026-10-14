"""
Multi-dataset 3D trajectory prediction dataset for Molmo2 SFT.

Supports EgoDex, DROID, and MolmoSpaces with clip-based sampling,
H>1 history frames, and per-dataset data loading.

Text format (H=3, F=8, P=16 example):
  Input: Predict the future 3D point coordinates of 16 points over 8 timestamps,
         given action: {caption}, and 3 history frames:
         <tracks coords="0.0 1 0 0 0 ...;1.0 1 DX DY DZ ...;2.0 1 DX DY DZ ...">object trajectories</tracks>

  Output: <tracks coords="3.0 1 DX DY DZ ...;4.0 1 DX DY DZ ...;...;10.0 ...">object trajectories</tracks>

Encoding:
- Raw 3D, anchor-relative delta: anchor = first point at frame t (last history frame)
- Quantize: int(round(delta * 1000))
- Invisible/padded points omitted from text
- Valid mask: (P, F) bool — True if visible AND within clip bounds

Clip-based sampling (training):
1. Sample object (prob ∝ point count)
2. Sample clip (prob ∝ extended frame count)
3. Sample timestep t in valid range
4. History = [t-H+1..t], Future = [t+1..t+F] — incremental frames
5. Sample P points visible at frame t
"""

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from molmo_motion.data.dataset import Dataset
from molmo_motion.data.video_loader import VideoFrames
from molmo_motion.tokenizer import POINT_FEATURE_TOKEN, TWO_D_FEAT_START_TOKEN, TWO_D_FEAT_END_TOKEN

log = logging.getLogger(__name__)

# ── Data roots ──────────────────────────────────────────────────────────

# Legacy split metadata (only for datasets NOT in molmo-motion-1m: davis,
# hotworld — not part of the released recipe).
METADATA_ROOT = os.environ.get(
    "MOLMO_MOTION_METADATA_ROOT", "data/motion_filtering_splits")

# Root of the MolmoMotion-1M corpus (egodex, ytvis, hepic, xperience, droid,
# stereo4d, molmospaces). Each dataset lives under
# `<MOLMO_MOTION_1M_ROOT>/<dataset>/` with sub-directories `annotations/`,
# `tracks/`, `videos/`. Download from HF `allenai/molmo-motion-1m`; see the
# top-level README. Set via `export MOLMO_MOTION_1M_ROOT=...`.
MOLMO_MOTION_1M_ROOT = os.environ.get(
    "MOLMO_MOTION_1M_ROOT", "data/molmo-motion-1m")

# Root of the PointMotionBench eval suite (hot3d, worldtrack, davis).
# Download from HF `allenai/PointMotionBench`; set via
# `export POINTMOTIONBENCH_ROOT=...`.
POINTMOTIONBENCH_ROOT = os.environ.get(
    "POINTMOTIONBENCH_ROOT", "data/PointMotionBench")

DATASET_CONFIG = {
    # ── molmo-motion-1m datasets ────────────────────────────────────────
    "egodex": {
        "data_root_env": "MOLMO_MOTION_1M_ROOT",
        "data_root_default": f"{MOLMO_MOTION_1M_ROOT}/egodex",
        "split_file": "egodex/annotations/egodex_split.json",
        "split_is_absolute": True,  # path is under MOLMO_MOTION_1M_ROOT
    },
    "ytvis": {
        "data_root_env": "MOLMO_MOTION_1M_ROOT",
        "data_root_default": f"{MOLMO_MOTION_1M_ROOT}/ytvis",
        "split_file": "ytvis/annotations/ytvis_split.json",
        "split_is_absolute": True,
    },
    # NOTE: internal dataset token is "hepic" but the released corpus stores
    # HD-EPIC under "hdepic/" with "hdepic_*" annotation filenames.
    "hepic": {
        "data_root_env": "MOLMO_MOTION_1M_ROOT",
        "data_root_default": f"{MOLMO_MOTION_1M_ROOT}/hdepic",
        "split_file": "hdepic/annotations/hdepic_split.json",
        "split_is_absolute": True,
    },
    "xperience": {
        "data_root_env": "MOLMO_MOTION_1M_ROOT",
        "data_root_default": f"{MOLMO_MOTION_1M_ROOT}/xperience",
        "split_file": "xperience/annotations/xperience_split.json",
        "split_is_absolute": True,
    },
    "droid": {
        "data_root_env": "MOLMO_MOTION_1M_ROOT",
        "data_root_default": f"{MOLMO_MOTION_1M_ROOT}/droid",
        "split_file": "droid/annotations/droid_split.json",
        "split_is_absolute": True,
    },
    "stereo4d": {
        "data_root_env": "MOLMO_MOTION_1M_ROOT",
        "data_root_default": f"{MOLMO_MOTION_1M_ROOT}/stereo4d",
        "split_file": "stereo4d/annotations/stereo4d_split.json",
        "split_is_absolute": True,
    },
    "molmospaces": {
        "data_root_env": "MOLMO_MOTION_1M_ROOT",
        "data_root_default": f"{MOLMO_MOTION_1M_ROOT}/molmospaces",
        "split_file": "molmospaces/annotations/molmospaces_split.json",
        "split_is_absolute": True,
    },
    # Hand-only sister datasets (share videos with the main dataset; tracks
    # come from the dedicated hand split files under the same root).
    "egodex_hand": {
        "data_root_env": "MOLMO_MOTION_1M_ROOT",
        "data_root_default": f"{MOLMO_MOTION_1M_ROOT}/egodex",
        "split_file": "egodex/annotations/egodex_hand_split.json",
        "split_is_absolute": True,
    },
    "xperience_hand": {
        "data_root_env": "MOLMO_MOTION_1M_ROOT",
        "data_root_default": f"{MOLMO_MOTION_1M_ROOT}/xperience",
        "split_file": "xperience/annotations/xperience_hand_split.json",
        "split_is_absolute": True,
    },
    # ── Legacy datasets (NOT in molmo-motion-1m) ────────────────────────
    "davis": {
        "data_root_env": "DAVIS_DATA_ROOT",
        "data_root_default": "data/davis",
        "split_file": "davis_split.json",
    },
    # HotWorld = WorldTrack_GT + HOT3D filtered. Entries store absolute paths
    # to per-source files, so the data_root_default is essentially unused.
    "hotworld": {
        "data_root_env": "HOTWORLD_DATA_ROOT",
        "data_root_default": "data/hot3d-benchmark",
        "split_file": "hotworld_split.json",
    },
    # ── PointMotionBench eval-only benchmarks ────────────────────────────
    # Per-sub-clip motion-filtered bundles with their own on-disk layouts.
    # Entries are enumerated at init time (walk the directory once), so no
    # `split_file` exists; the loader's `_build_<bench>_entries` produces
    # the list of dicts.
    "hot3d_bench": {
        "data_root_env": "HOT3D_BENCH_ROOT",
        "data_root_default": f"{POINTMOTIONBENCH_ROOT}/hot3d",
        "split_file": None,
    },
    "worldtrack_bench": {
        "data_root_env": "WT_BENCH_ROOT",
        "data_root_default": f"{POINTMOTIONBENCH_ROOT}/worldtrack",
        "split_file": None,
    },
    "davis_bench": {
        "data_root_env": "DAVIS_BENCH_ROOT",
        "data_root_default": f"{POINTMOTIONBENCH_ROOT}/davis",
        "split_file": None,
    },
}

# Hand datasets: objects are always left_hand/right_hand, clip = full video
HAND_OBJECTS = ["left_hand", "right_hand"]

LABEL_TEXT_HISTORY = "3d object history"
LABEL_TEXT_FUTURE = "3d object trajectories"
LABEL_TEXT_ENDPOINT = "3d object endpoints"
LABEL_TEXT_2D_COORD = "2d object point"
# v1-era (traj3d_droid_v1) label: a single "object points" string was used for
# both prompt and answer <tracks> blocks. Preserved for the _v1match flag.
LABEL_TEXT_V1 = "object points"
# Under molmo-motion-1m, DROID MP4s are pre-resampled to 15 fps (matching
# the track rate). The old multiplier from when video was 60 fps and tracks
# were 15 fps is no longer needed; we keep the constant for any consumer
# that imports it but set it to 1.
DROID_FPS_MULTIPLIER = 1
MOLMOSPACES_VIS_THRESHOLD = 0.5
# Gripper moves too slowly in the source clips; decimate by this factor so each
# strided track frame = every Nth raw frame (and the video is seeked 3× further
# per step). Applied everywhere molmospaces is loaded (entries, 3D, 2D, video).
# Overridable per-job via env var MOLMOSPACES_STRIDE (e.g. set to 2 for F=30
# finetune so molmospaces clips have enough strided frames to fit H+F=33).
MOLMOSPACES_TIME_STRIDE = int(os.environ.get("MOLMOSPACES_STRIDE", "4"))


class Trajectory3DDataset(Dataset):
    """Multi-dataset 3D trajectory prediction for Molmo2 SFT."""

    # Class-level fallbacks for newer flags so that an already-pickled
    # dataset instance (e.g. one created before these attrs existed and
    # then unpickled into a worker running newer code) doesn't crash with
    # AttributeError when newer methods inspect `self.<flag>`.
    use_camera_frame = False
    use_depth_token = False
    depth_target_size = 378
    eval_first_h_frames = False

    def __init__(self, split="train", num_points=16, num_future_frames=8,
                 history_size=3, datasets=("egodex", "droid", "molmospaces"),
                 use_2d_point_features=False, max_eval_per_dataset=None,
                 dataset_weighting="sqrt", pred_end_point_first=False,
                 use_2d_coordinate=False, predict_history_3d=False,
                 v1_match_format=False, mixed_history=True,
                 downsample_molmospaces=False, downsample_seed=42,
                 use_camera_frame=False, use_depth_token=False,
                 depth_target_size=378,
                 eval_first_h_frames=False,
                 merge_train_test=False):
        self.split = split
        self.num_points = num_points
        self.num_future_frames = num_future_frames
        # `history_size` is the default / nominal value used when
        # `mixed_history=False`. When True we ignore it and sample H per
        # example from {1, 3} with 50/50 probability; the value here only
        # affects the eval-mode deterministic configs (we fix H=3 there).
        self.history_size = history_size
        # If True, sample H ∈ {1, 3} uniformly per training example. This
        # is the regime described in the paper (one shared model that has
        # seen both single-frame and 3-frame histories).
        self.mixed_history = mixed_history
        # If True, randomly sub-sample MolmoSpaces entries to half the
        # DROID entry count. Both `droid` and `molmospaces` must be in the
        # `datasets` list for this to take effect; the down-sample uses
        # `downsample_seed` for reproducibility.
        self.downsample_molmospaces = downsample_molmospaces
        self.downsample_seed = downsample_seed
        # "train_test" = load train entries but use deterministic test-style eval
        self.is_eval = split in ("validation", "test", "train_test")
        self.use_2d_point_features = use_2d_point_features
        # "sqrt": per-dataset prob ∝ sqrt(size); "naive": ∝ size (just concatenate);
        # "uniform": each dataset equally likely. Only affects training, not eval.
        self.dataset_weighting = dataset_weighting
        # Endpoint-first answer mode: prepend a single-frame <tracks>... endpoint block
        # (last future frame, labelled "object endpoints") before the full trajectory.
        self.pred_end_point_first = pred_end_point_first
        # If True, prepend a "2d point coordinates: <points coords="(H-1).0 1 x y 2 x y .../>"
        # field to the prompt, using normalized [0,1] 2D coords × 1000 rounded to ints.
        self.use_2d_coordinate = use_2d_coordinate
        # If True, drop the "history 3d point coordinates" field from the prompt and
        # emit it as part of the model answer (as a <tracks>...</tracks> block with
        # label "3d object history"). The answer is then supervised on history too.
        self.predict_history_3d = predict_history_3d
        # If True, emit the *exact* v1-era prompt + answer format used by
        # traj3d_droid_v1 (label "object points", hardcoded F in prompt text,
        # bare caption, "and {H} history frames:\n<tracks>…" connector, no
        # quotes, no Oxford comma, no trailing period). Overrides and ignores
        # the other prompt-augmenting flags (_pointfeat, _2dcoord, _endfirst,
        # _predhist) so the prompt is byte-identical to v1.
        self.v1_match_format = v1_match_format
        # If True, transform every 3D point in `traj_clean` into the
        # camera-of-frame-`t` coordinate frame before the existing
        # delta-from-anchor step. This makes the model's coordinate axes
        # track the camera at the query time t (paper t_0 convention).
        # Supported per-dataset (egodex/ytvis/hepic/xperience/molmospaces use
        # per-frame extrinsics; droid is no-op since data is already in camera
        # frame; stereo4d skipped — no extrinsics in the released NPZs).
        self.use_camera_frame = use_camera_frame
        # If True, load a (H, W) metric-depth map at the query frame t and
        # include it in the example metadata as `depth_t`. The model side
        # (Molmo2TrajectoryDepth) then runs it through the LingBot-Depth
        # encoder and injects the projected features into the SigLIP2 image
        # feature map at the query-frame patches.
        # Stochastic schedule per example: 30% monocular (depth zeroed out),
        # 30% patch-masked (per-pixel Bernoulli p=0.5), 40% full depth.
        # stereo4d is deterministically pinned to the monocular bucket
        # because no depth is shipped for stereo4d in molmo-motion-1m.
        self.use_depth_token = use_depth_token
        self.depth_target_size = depth_target_size
        # Eval-only override: emit exactly ONE config per entry with the
        # query frame fixed at t = H − 1 (history = clip's first H frames)
        # and future = [H, …, H + num_future_frames − 1] (clipped to clip
        # length). Activated by the `_t0first` token in the dataset name.
        self.eval_first_h_frames = eval_first_h_frames
        # `merge_train_test` is accepted for backwards compatibility with
        # the source repo's call sites; the release `Trajectory3DDataset`
        # always uses every annotated clip in `split == "train"` mode
        # regardless of this flag, so it has no effect here.
        self.merge_train_test = merge_train_test
        # Per-instance LRU cache of opened zip/h5/hdf5 file handles. Datasets
        # that store depth in archives (egodex/ytvis/hepic zips,
        # xperience hdf5, droid h5) benefit a lot from keeping handles open
        # across consecutive __getitem__ calls inside the same worker.
        self._depth_handle_cache = {}

        # Resolve per-dataset data roots
        self.data_roots = {}
        for ds in datasets:
            cfg = DATASET_CONFIG[ds]
            if cfg["data_root_env"] == "MOLMO_MOTION_1M_ROOT":
                # data_root_default already folds in the env-resolved corpus
                # root with the per-dataset subdirectory appended; consulting
                # the env var again would drop the subdirectory.
                self.data_roots[ds] = Path(cfg["data_root_default"])
            else:
                self.data_roots[ds] = Path(
                    os.environ.get(cfg["data_root_env"], cfg["data_root_default"]))

        # Load train/test splits.
        # Release behavior: train mode always uses ALL annotated data
        # (train + test concatenated) — the held-out split metadata still
        # exists in the JSONs but is not exposed as a separate mode in the
        # public API. Evaluation in the release path is done against
        # PointMotionBench, not a held-out slice of the human corpus.
        self.entries = []
        for ds in datasets:
            cfg = DATASET_CONFIG[ds]
            # Benchmark datasets (hot3d_bench/worldtrack_bench/davis_bench)
            # have no split JSON — enumerate sub-clips from disk.
            if cfg.get("split_file") is None:
                ds_entries = self._build_bench_entries(ds)
            else:
                # Some configs point under MOLMO_MOTION_1M_ROOT, others under
                # the legacy METADATA_ROOT (davis, hotworld).
                if cfg.get("split_is_absolute"):
                    split_path = os.path.join(MOLMO_MOTION_1M_ROOT, cfg["split_file"])
                else:
                    split_path = os.path.join(METADATA_ROOT, cfg["split_file"])
                with open(split_path) as f:
                    split_data = json.load(f)
                if self.is_eval:
                    ds_entries = split_data["test"]
                else:
                    # Train mode: use every annotated clip in the split JSON.
                    ds_entries = list(split_data["train"]) + list(split_data["test"])

            for entry in ds_entries:
                entry["_dataset"] = ds
            self.entries.extend(ds_entries)

        self.entries.sort(key=lambda e: (e["_dataset"], e.get("file", "")))

        # Filter entries against the per-NPZ track-keys cache. Some upstream
        # split JSONs reference obj keys (e.g. `obj1`) that the corresponding
        # `_3d.npz` doesn't actually expose (upstream re-clustered tracks but
        # didn't update the split). Build the cache with
        # `python scripts/build_track_keys_cache.py` if it's missing.
        self.entries = self._filter_entries_by_npz_keys(self.entries, datasets)

        # Optional: down-sample MolmoSpaces to half of DROID's entry count.
        # Has no effect at eval time (we want the deterministic full eval).
        if (self.downsample_molmospaces and not self.is_eval
                and "molmospaces" in datasets and "droid" in datasets):
            droid_n = sum(1 for e in self.entries if e["_dataset"] == "droid")
            target = droid_n // 2
            molmo_idxs = [i for i, e in enumerate(self.entries) if e["_dataset"] == "molmospaces"]
            if len(molmo_idxs) > target > 0:
                _rng = np.random.RandomState(self.downsample_seed)
                keep = set(_rng.choice(molmo_idxs, size=target, replace=False).tolist())
                before = len(self.entries)
                self.entries = [e for i, e in enumerate(self.entries)
                                if e["_dataset"] != "molmospaces" or i in keep]
                log.info(
                    f"[Traj3D] downsample_molmospaces: kept {target}/{len(molmo_idxs)} "
                    f"molmospaces entries (half of DROID's {droid_n}); "
                    f"total entries {before} → {len(self.entries)}"
                )

        # Molmospaces: compress num_frames + clip ranges to strided units so the
        # rest of the pipeline (clip sampling, eval expansion, video seeking)
        # treats frame index k as original raw frame k * STRIDE.
        s = MOLMOSPACES_TIME_STRIDE
        if s > 1:
            for entry in self.entries:
                if entry["_dataset"] != "molmospaces":
                    continue
                entry["num_frames"] = (entry["num_frames"] + s - 1) // s
                new_clips = {}
                for obj, ranges in entry["clips_by_object"].items():
                    kept = [[st // s, en // s] for st, en in ranges if en // s >= st // s]
                    if kept:
                        new_clips[obj] = kept
                entry["clips_by_object"] = new_clips

        # NOTE: short motion clips are NOT hard-dropped here. The training
        # sampler / eval-config builder pick the longest extended clip and
        # `_build_example` pads short futures by repeating the last value;
        # `valid_mask` keeps loss masked on the padded positions. We previously
        # hard-dropped clips with ext_len < H+F, but that disproportionately
        # culled molmospaces under long-horizon (F=30/F=32) configs, so we
        # reverted to the soft pad+mask policy.

        if self.is_eval:
            # Expand entries into all (entry, obj, clip, t) eval configs
            # Per spec: all objects, all clips, 3 t values per clip
            all_configs = self._build_eval_configs()
            if max_eval_per_dataset is not None:
                all_configs = self._stratified_subsample(all_configs, max_eval_per_dataset)
            self.eval_configs = all_configs
        else:
            self.eval_configs = None

        # Pre-group entries by dataset for weighted training-time sampling.
        self._ds_to_indices = {}
        for idx, e in enumerate(self.entries):
            self._ds_to_indices.setdefault(e["_dataset"], []).append(idx)
        self._ds_names_sorted = sorted(self._ds_to_indices.keys())

        if self.is_eval or dataset_weighting == "naive":
            self._ds_probs = None  # fall through to item % len(entries)
        elif isinstance(dataset_weighting, dict):
            # Manual per-dataset weights. Every dataset in the mix MUST appear
            # in the weight dict; extras are tolerated (warned about) so the
            # same env-var spec can cover multiple training configs.
            missing = [d for d in self._ds_names_sorted if d not in dataset_weighting]
            if missing:
                raise ValueError(
                    f"manual dataset_weighting is missing weights for: {missing} "
                    f"(provided keys: {sorted(dataset_weighting.keys())})")
            extra = [d for d in dataset_weighting if d not in self._ds_names_sorted]
            if extra:
                print(f"[Traj3D] WARN: manual weights specify unused datasets: {extra}")
            w = np.array([float(dataset_weighting[d]) for d in self._ds_names_sorted],
                         dtype=np.float64)
            if (w < 0).any() or w.sum() <= 0:
                raise ValueError(f"manual weights must be non-negative and sum > 0: {w}")
            self._ds_probs = w / w.sum()
        else:
            counts = np.array([len(self._ds_to_indices[d]) for d in self._ds_names_sorted],
                              dtype=np.float64)
            if dataset_weighting == "sqrt":
                w = np.sqrt(counts)
            elif dataset_weighting == "uniform":
                w = np.ones_like(counts)
            else:
                raise NotImplementedError(f"dataset_weighting={dataset_weighting}")
            self._ds_probs = w / w.sum()

        # Print stats
        ds_counts = {}
        for e in self.entries:
            ds_counts[e["_dataset"]] = ds_counts.get(e["_dataset"], 0) + 1
        stats = ", ".join(f"{k}={v}" for k, v in sorted(ds_counts.items()))
        n_eval = f" -> {len(self.eval_configs)} eval configs" if self.eval_configs else ""
        if self.is_eval:
            mix = ""
        elif isinstance(dataset_weighting, dict):
            mix = " mix=manual"
        else:
            mix = f" mix={dataset_weighting}"
        if self._ds_probs is not None:
            mix_probs = ", ".join(
                f"{d}={p:.3f}" for d, p in zip(self._ds_names_sorted, self._ds_probs))
            mix += f" ({mix_probs})"
        print(f"[Traj3D] {split}: {len(self.entries)} entries ({stats}), "
              f"H={history_size}, F={num_future_frames}, P={num_points}{n_eval}{mix}")

    # ── NPZ key filtering (against upstream JSON / NPZ key mismatch) ──

    # Datasets whose `points_3d` is an object-array dict {obj: ndarray}.
    # Per-file dict keys are cached at
    # $HOME/.cache/molmo_motion_1m_track_keys/{ds}.json (build with
    # `scripts/build_track_keys_cache.py`). Other datasets either ship a
    # flat single-object array (egodex flat case, droid, xperience,
    # xperience_hand) or use canonical fixed keys (hand sets) for which
    # the filter is a no-op.
    _DICT_FORMAT_DATASETS = ("egodex", "hepic", "ytvis", "stereo4d", "molmospaces")

    def _filter_entries_by_npz_keys(self, entries, datasets):
        """Drop entries / `clips_by_object` keys not present in the NPZ.

        Loads per-dataset `track_keys` caches once. For each entry whose
        dataset is dict-format, intersects `clips_by_object` keys with the
        NPZ's actual keys; drops entries whose intersection is empty.
        Entries whose dataset is flat ("__flat__" sentinel) or missing
        from the cache file are passed through unchanged (with a one-time
        warning per dataset).
        """
        cache_dir = Path(os.environ.get(
            "MOLMO_MOTION_1M_TRACK_KEYS_CACHE",
            str(Path.home() / ".cache" / "molmo_motion_1m_track_keys")))

        # Per-dataset key map: {file_id: list[str] or None}
        key_maps = {}
        for ds in datasets:
            if ds not in self._DICT_FORMAT_DATASETS:
                continue
            cp = cache_dir / f"{ds}.json"
            if not cp.exists():
                raise FileNotFoundError(
                    f"track-keys cache missing for {ds}: {cp}. "
                    f"Build it with: python scripts/build_track_keys_cache.py "
                    f"--datasets {ds}"
                )
            with open(cp) as f:
                key_maps[ds] = json.load(f)

        kept = []
        n_drop_entry = {ds: 0 for ds in datasets}
        n_drop_obj   = {ds: 0 for ds in datasets}
        n_missing_fid = {ds: 0 for ds in datasets}
        for e in entries:
            ds = e["_dataset"]
            if ds not in key_maps:
                kept.append(e)
                continue
            fid = e["file"]
            available = key_maps[ds].get(fid)
            if available is None:
                # Missing/corrupted NPZ — drop entry entirely.
                n_drop_entry[ds] += 1
                n_missing_fid[ds] += 1
                continue
            if available == ["__flat__"]:
                # Loader ignores obj_name for flat NPZs — keep entry as-is.
                kept.append(e)
                continue
            avail_set = set(available)
            new_clips = {o: r for o, r in e["clips_by_object"].items() if o in avail_set}
            dropped_objs = len(e["clips_by_object"]) - len(new_clips)
            if dropped_objs:
                n_drop_obj[ds] += dropped_objs
            if not new_clips:
                n_drop_entry[ds] += 1
                continue
            if dropped_objs:
                # Shallow copy so we don't mutate the upstream split JSON dict.
                e = {**e, "clips_by_object": new_clips,
                     "num_clips_total": sum(len(r) for r in new_clips.values())}
            kept.append(e)

        # Log only datasets that had any drops or corrupt files.
        for ds in datasets:
            if n_drop_entry[ds] or n_drop_obj[ds]:
                msg = (
                    f"[Traj3D] {ds}: filter dropped {n_drop_entry[ds]} entries"
                    f" and pruned {n_drop_obj[ds]} obj keys missing in NPZs"
                )
                if n_missing_fid[ds]:
                    msg += f" ({n_missing_fid[ds]} corrupted/missing NPZs)"
                log.info(msg)
        return kept

    # ── Hand track existence check ────────────────────────────────────

    def _hand_tracks_exist(self, ds, entry):
        if ds == "egodex_hand":
            stem = entry["file"]
            path = self.data_roots[ds] / "tracks" / "hand" / f"{stem}_3d.npz"
            return path.exists()
        elif ds == "xperience_hand":
            stem = entry["file"]
            base = self.data_roots[ds] / "tracks"
            return ((base / f"{stem}_left_hand.npz").exists()
                    or (base / f"{stem}_right_hand.npz").exists())
        return True

    # ── Per-dataset 3D + visibility loading ─────────────────────────────

    def _load_3d_and_vis(self, entry, obj_name):
        """Load (N_obj, T, 3) points and (N_obj, T) bool visibility."""
        ds = entry["_dataset"]
        if ds == "egodex":
            return self._load_egodex_3d(entry, obj_name)
        elif ds == "droid":
            return self._load_droid_3d(entry, obj_name)
        elif ds == "molmospaces":
            return self._load_molmospaces_3d(entry, obj_name)
        elif ds == "hepic":
            return self._load_hepic_3d(entry, obj_name)
        elif ds == "xperience":
            return self._load_xperience_3d(entry, obj_name)
        elif ds == "egodex_hand":
            return self._load_egodex_hand_3d(entry, obj_name)
        elif ds == "xperience_hand":
            return self._load_xperience_hand_3d(entry, obj_name)
        elif ds == "davis":
            return self._load_davis_3d(entry, obj_name)
        elif ds == "ytvis":
            return self._load_ytvis_3d(entry, obj_name)
        elif ds == "stereo4d":
            return self._load_stereo4d_3d(entry, obj_name)
        elif ds == "hotworld":
            return self._load_hotworld_3d(entry, obj_name)
        elif ds == "hot3d_bench":
            return self._load_hot3d_bench_3d(entry, obj_name)
        elif ds == "worldtrack_bench":
            return self._load_worldtrack_bench_3d(entry, obj_name)
        elif ds == "davis_bench":
            return self._load_davis_bench_3d(entry, obj_name)
        raise ValueError(f"Unknown dataset: {ds}")

    # ─── molmo-motion-1m unified loaders ──────────────────────────────
    #
    # The release ships per-dataset NPZs with two flavors:
    #   FLAT  — `points_3d` is a single ndarray; we look it up directly.
    #   DICT  — `points_3d` is a 0-d object array wrapping a dict
    #           {obj_name: (N, T, 3)}; we look up by `obj_name`.
    # 2D files are analogous: `tracks` (T, N, 2) + `visibility` (T, N) + `dim`,
    # also either flat or dict per dataset. Loaders below convert to a
    # uniform `(pts_3d (N, T, 3), vis (N, T))` shape per chosen object.

    @staticmethod
    def _unpack_3d_from_dict(npz_data, vis_raw, obj_name):
        """Pull (pts_3d (N,T,3), vis (N,T) bool) for one object out of dict-format NPZ data."""
        pts_dict = npz_data.item() if hasattr(npz_data, "item") and npz_data.dtype == object else npz_data
        if obj_name not in pts_dict:
            raise KeyError(f"obj_name='{obj_name}' not in pts dict keys: {list(pts_dict.keys())}")
        pts_3d = pts_dict[obj_name].astype(np.float32)  # (N, T, 3)
        if vis_raw is not None and getattr(vis_raw, "dtype", None) == object:
            vis_dict = vis_raw.item()
            vis = vis_dict.get(obj_name, None)
            if vis is None:
                vis = np.isfinite(pts_3d).all(axis=-1)
            else:
                if vis.ndim == 3:
                    vis = vis.squeeze(-1)
                vis = vis.astype(bool)
        else:
            vis = np.isfinite(pts_3d).all(axis=-1)
        return pts_3d, vis

    def _load_egodex_3d(self, entry, obj_name):
        stem = entry["file"]
        path = self.data_roots["egodex"] / "tracks" / "object" / f"{stem}_3d.npz"
        with np.load(str(path), allow_pickle=True) as f:
            pts_raw = f["points_3d"]
            vis_raw = f.get("visibility", None) if hasattr(f, "get") else None
            # numpy NpzFile doesn't have .get; use 'in' instead
            if vis_raw is None and "visibility" in f.files:
                vis_raw = f["visibility"]
        if pts_raw.dtype == object:
            return self._unpack_3d_from_dict(pts_raw, vis_raw, obj_name)
        # FLAT: (N, T, 3) — only one object per egodex object-mode clip
        pts_3d = pts_raw.astype(np.float32)
        if vis_raw is not None:
            vis = vis_raw
            if vis.ndim == 3:
                vis = vis.squeeze(-1)
            vis = vis.astype(bool)
        else:
            vis = np.isfinite(pts_3d).all(axis=-1)
        return pts_3d, vis

    def _load_egodex_hand_3d(self, entry, obj_name):
        # `obj_name` is 'left_hand' or 'right_hand'.
        stem = entry["file"]
        path = self.data_roots["egodex_hand"] / "tracks" / "hand" / f"{stem}_3d.npz"
        with np.load(str(path), allow_pickle=True) as f:
            pts_raw = f["points_3d"]
            vis_raw = f["visibility"] if "visibility" in f.files else None
        return self._unpack_3d_from_dict(pts_raw, vis_raw, obj_name)

    def _load_droid_3d(self, entry, obj_name):
        # In molmo-motion-1m, `file` already includes the `__{cam}` suffix.
        stem = entry["file"]
        path = self.data_roots["droid"] / "tracks" / f"{stem}_3d.npz"
        with np.load(str(path), allow_pickle=True) as f:
            pts_3d = f["points_3d"].astype(np.float32)  # (T, N, 3)
            valid_3d = f["valid_3d"]                     # (T, N) bool
        return np.transpose(pts_3d, (1, 0, 2)), valid_3d.T.astype(bool)

    def _load_molmospaces_3d(self, entry, obj_name):
        # obj_name is 'body_{int}'.
        stem = entry["file"]
        path = self.data_roots["molmospaces"] / "tracks" / f"{stem}_3d.npz"
        s = MOLMOSPACES_TIME_STRIDE
        with np.load(str(path), allow_pickle=True) as f:
            pts_dict = f["points_3d"].item()
            vis_dict = f["visibility"].item()
        if obj_name not in pts_dict:
            raise KeyError(f"obj_name='{obj_name}' not in {list(pts_dict.keys())}")
        pts_3d = pts_dict[obj_name].astype(np.float32)   # (N, T, 3)
        vis    = vis_dict[obj_name].astype(bool)          # (N, T)
        # Apply 4× temporal stride to compress slow-moving sim clips.
        pts_3d = pts_3d[:, ::s, :]
        vis    = vis[:, ::s]
        return pts_3d, vis

    def _load_hepic_3d(self, entry, obj_name):
        # Multi-object dict with keys 'left_hand', 'right_hand', 'object'.
        stem = entry["file"]
        path = self.data_roots["hepic"] / "tracks" / f"{stem}_3d.npz"
        with np.load(str(path), allow_pickle=True) as f:
            pts_raw = f["points_3d"]
            vis_raw = f["visibility"] if "visibility" in f.files else None
        return self._unpack_3d_from_dict(pts_raw, vis_raw, obj_name)

    def _load_xperience_3d(self, entry, obj_name):
        # Per-object NPZ: `{stem}_{object,left_hand,right_hand}.npz`.
        stem = entry["file"]
        track_key = obj_name if obj_name in ("object", "left_hand", "right_hand") else "object"
        path = self.data_roots["xperience"] / "tracks" / f"{stem}_{track_key}.npz"
        with np.load(str(path), allow_pickle=True) as f:
            pts_3d = f["points_3d"].astype(np.float32)   # (N, T, 3)
            vis    = f["visibility"].astype(bool)         # (N, T)
        return pts_3d, vis

    def _load_xperience_hand_3d(self, entry, obj_name):
        # Hand-only path; obj_name ∈ {left_hand, right_hand}.
        stem = entry["file"]
        path = self.data_roots["xperience_hand"] / "tracks" / f"{stem}_{obj_name}.npz"
        with np.load(str(path), allow_pickle=True) as f:
            pts_3d = f["points_3d"].astype(np.float32)
            vis    = f["visibility"].astype(bool)
        return pts_3d, vis

    def _load_ytvis_3d(self, entry, obj_name):
        # Dict NPZ keyed by object cluster name (e.g. 'parrot_0').
        stem = entry["file"]
        path = self.data_roots["ytvis"] / "tracks" / f"{stem}_3d.npz"
        with np.load(str(path), allow_pickle=True) as f:
            pts_raw = f["points_3d"]
            vis_raw = f["visibility"] if "visibility" in f.files else None
        return self._unpack_3d_from_dict(pts_raw, vis_raw, obj_name)

    def _load_stereo4d_3d(self, entry, obj_name):
        # Dict NPZ keyed by 'obj0', 'obj1', ...
        stem = entry["file"]
        path = self.data_roots["stereo4d"] / "tracks" / f"{stem}_3d.npz"
        with np.load(str(path), allow_pickle=True) as f:
            pts_raw = f["points_3d"]
            vis_raw = f["visibility"] if "visibility" in f.files else None
        return self._unpack_3d_from_dict(pts_raw, vis_raw, obj_name)

    def _load_hotworld_3d(self, entry, obj_name):
        """HotWorld: dispatch on `source_dataset`.

        WorldTrack: NPZ contains `tracks_world` (T, N, 3) world-frame tracks +
        `tracks_XYZ` camera-frame + `extrinsics_w2c` + `fx_fy_cx_cy` +
        `images_jpeg_bytes` + per-point `object_ids`. We use `tracks_world` for
        the model input (camera moves; world-frame is stable). Cache JPEG
        bytes + camera-frame XYZ + intrinsics for use in 2D / video helpers.

        HOT3D: separate `tracks_3d_npz` (`points_3d` shape (2000, T, 3) in
        camera frame, `visibility` (2000, T, 1)). Object name encodes block
        index in `h3_object_block_index`; each object is a fixed 100-pt block.
        """
        src = entry["source_dataset"]
        if src == "worldtrack":
            with np.load(entry["tracks_npz"], allow_pickle=True) as f:
                pts_world = f["tracks_world"].astype(np.float32)   # (T, N, 3)
                xyz_cam   = f["tracks_XYZ"].astype(np.float32)      # (T, N, 3)
                vis       = f["visibility"].astype(bool)            # (T, N)
                obj_ids   = f["object_ids"]                          # (N,)
                jpegs     = f["images_jpeg_bytes"]                   # (T,) bytes
                fx_fy_cx_cy = f["fx_fy_cx_cy"].astype(np.float64)
            oid = int(obj_name.replace("obj", ""))
            mask = obj_ids == oid
            if not mask.any():
                raise KeyError(f"WT entry {entry['file']} has no points with object_id={oid}")
            pts_world_obj = np.transpose(pts_world[:, mask, :], (1, 0, 2))  # (N', T, 3)
            xyz_cam_obj   = np.transpose(xyz_cam[:, mask, :], (1, 0, 2))    # (N', T, 3)
            vis_obj       = vis[:, mask].T                                  # (N', T)
            entry["_cached_jpegs"] = jpegs
            entry["_cached_wt_xyz_cam"] = xyz_cam_obj
            entry["_cached_wt_intr"] = fx_fy_cx_cy
            return pts_world_obj, vis_obj
        elif src == "hot3d":
            d = np.load(entry["tracks_3d_npz"])
            pts = d["points_3d"].astype(np.float32)  # (2000, T, 3) camera-frame
            vis = d["visibility"].astype(bool)        # (2000, T, 1) or (2000, T)
            if vis.ndim == 3:
                vis = vis.squeeze(-1)
            block = int(entry["h3_object_block_index"])
            s, e = block * 100, (block + 1) * 100
            return pts[s:e], vis[s:e]
        raise ValueError(f"Unknown hotworld source_dataset: {src}")

    def _load_hotworld_2d(self, entry, obj_name, chosen_indices, t):
        """HotWorld 2D — WT projects camera-frame XYZ via intrinsics; HOT3D
        reads the precomputed 2D tracks NPZ."""
        src = entry["source_dataset"]
        if src == "worldtrack":
            xyz = entry.pop("_cached_wt_xyz_cam", None)
            intr = entry.get("_cached_wt_intr", None)
            jpegs = entry.get("_cached_jpegs", None)
            if xyz is None or intr is None or jpegs is None:
                # Re-run 3D loader to refill the caches
                self._load_hotworld_3d(entry, obj_name)
                xyz = entry.pop("_cached_wt_xyz_cam")
                intr = entry["_cached_wt_intr"]
                jpegs = entry["_cached_jpegs"]
            # Decode the first JPEG to get image dim
            import io
            from PIL import Image
            img0 = np.array(Image.open(io.BytesIO(bytes(jpegs[0]))))
            img_h, img_w = img0.shape[:2]
            fx, fy, cx, cy = intr
            X = xyz[chosen_indices, t, 0]
            Y = xyz[chosen_indices, t, 1]
            Z = xyz[chosen_indices, t, 2]
            Z_safe = np.where(np.abs(Z) < 1e-6, 1e-6, Z)
            u = fx * X / Z_safe + cx
            v = fy * Y / Z_safe + cy
            coords = np.stack([u / img_w, v / img_h], axis=-1).astype(np.float32)  # (P, 2)
            coords = np.nan_to_num(coords, nan=0.5)
            return np.clip(coords, 0.0, 1.0)
        elif src == "hot3d":
            d = np.load(entry["tracks_2d_npz"])
            tracks = np.transpose(d["tracks"].astype(np.float32), (1, 0, 2))  # (N, T, 2)
            dim = d["dim"]
            block = int(entry["h3_object_block_index"])
            s, e = block * 100, (block + 1) * 100
            tracks_obj = tracks[s:e]
            coords = tracks_obj[chosen_indices, t, :].copy()
            img_h, img_w = int(dim[0]), int(dim[1])
            coords = np.nan_to_num(coords, nan=0.5 * img_w)
            coords[:, 0] /= img_w
            coords[:, 1] /= img_h
            return np.clip(coords, 0.0, 1.0).astype(np.float32)
        return None

    def _load_davis_3d(self, entry, obj_name):
        vid_id = entry["file"]
        path = self.data_roots["davis"] / "vipe" / "davis_final_tracks" / f"{vid_id}_3d.npz"
        with np.load(str(path), allow_pickle=True) as f:
            pts_dict = f["points_3d"][()]
        pts_3d = pts_dict[obj_name].astype(np.float32)  # (N, T, 3)
        vis = np.isfinite(pts_3d).all(axis=-1)           # (N, T) from NaN
        return pts_3d, vis

    # ── PointMotionBench benchmark loaders (eval-only) ────────────────────

    def _build_bench_entries(self, ds):
        """Walk the benchmark directory once and return a list of entry dicts
        with `file`, `num_frames`, `fps`, `caption`, `clips_by_object`. These
        benchmarks have no train/test split — every clip is a test entry."""
        if ds == "hot3d_bench":
            return self._build_hot3d_bench_entries()
        if ds == "worldtrack_bench":
            return self._build_worldtrack_bench_entries()
        if ds == "davis_bench":
            return self._build_davis_bench_entries()
        return []

    # ── hot3d_bench ───────────────────────────────────────────────────────
    def _build_hot3d_bench_entries(self):
        root = self.data_roots["hot3d_bench"]
        captions_path = root / "hot3d_annotations.json"
        if not captions_path.exists():
            raise FileNotFoundError(
                f"HOT3D annotations not found at {captions_path}. Download the "
                f"PointMotionBench dataset (ships `hot3d/hot3d_annotations.json`) "
                f"from allenai/PointMotionBench; see pointmotionbench/README.md."
            )
        captions = json.load(open(captions_path))
        entries = []
        for npz_path in sorted((root / "tracks").glob("*_3d.npz")):
            stem = npz_path.stem[:-3]   # strip "_3d"
            # Peek shape without loading full data
            with np.load(npz_path, allow_pickle=True) as f:
                T = f["points_3d"].shape[1]
            cap = captions.get(stem, {}).get("caption", "")
            entries.append({
                "file": stem,
                "num_frames": int(T),
                "fps": 30,
                "caption": cap,
                "clips_by_object": {"obj0": [[0, int(T) - 1]]},
                "num_clips_total": 1,
            })
        return entries

    def _load_hot3d_bench_3d(self, entry, obj_name):
        path = self.data_roots["hot3d_bench"] / "tracks" / f"{entry['file']}_3d.npz"
        with np.load(str(path), allow_pickle=True) as f:
            pts = f["points_3d"].astype(np.float32)       # (2000, T, 3)
            vis = f["visibility"].astype(bool).squeeze(-1)  # (2000, T, 1) → (2000, T)
        return pts, vis

    def _load_hot3d_bench_2d(self, entry, obj_name, chosen_indices, t):
        path = self.data_roots["hot3d_bench"] / "tracks" / f"{entry['file']}_2d.npz"
        with np.load(str(path), allow_pickle=True) as f:
            tracks = f["tracks"]              # (T, 2000, 2)
            dim = f["dim"].astype(int)        # [H, W]
        # tracks: (T, N, 2) — same convention _sample_2d expects.
        return self._sample_2d(tracks, chosen_indices, t, int(dim[0]), int(dim[1]))

    # ── worldtrack_bench ──────────────────────────────────────────────────
    def _build_worldtrack_bench_entries(self):
        root = self.data_roots["worldtrack_bench"]
        entries = []
        for split_name in ("adt_mini", "ds_mini", "po_mini", "pstudio_mini"):
            split_root = root / split_name
            if not split_root.exists():
                continue
            for clip_dir in sorted(split_root.iterdir()):
                if not clip_dir.is_dir():
                    continue
                npz_path = clip_dir / f"{clip_dir.name}.npz"
                if not npz_path.exists():
                    continue
                caption = ""
                cap_path = clip_dir / "caption.json"
                if cap_path.exists():
                    try:
                        caption = json.load(open(cap_path)).get("caption", "")
                    except Exception:
                        pass
                with np.load(npz_path, allow_pickle=True) as f:
                    T = int(f["tracks_XYZ"].shape[0])
                entries.append({
                    "file": f"{split_name}/{clip_dir.name}",
                    "num_frames": T,
                    "fps": 30,
                    "caption": caption,
                    "clips_by_object": {"obj0": [[0, T - 1]]},
                    "num_clips_total": 1,
                })
        return entries

    def _load_worldtrack_bench_3d(self, entry, obj_name):
        path = self.data_roots["worldtrack_bench"] / entry["file"]
        npz_path = path / f"{path.name}.npz"
        with np.load(str(npz_path), allow_pickle=True) as f:
            tracks_xyz = f["tracks_XYZ"].astype(np.float32)   # (T, N, 3) world frame
            vis = f["visibility"].astype(bool)                 # (T, N)
        # Convert (T, N, 3) → (N, T, 3) to match our convention.
        pts = np.transpose(tracks_xyz, (1, 0, 2))
        vis = vis.T
        return pts, vis

    def _load_worldtrack_bench_2d(self, entry, obj_name, chosen_indices, t):
        path = self.data_roots["worldtrack_bench"] / entry["file"]
        npz_path = path / f"{path.name}.npz"
        with np.load(str(npz_path), allow_pickle=True) as f:
            tracks_xyz = f["tracks_XYZ"]     # (T, N, 3) camera frame
            fx, fy, cx, cy = f["fx_fy_cx_cy"].astype(np.float64)
            jpeg = f["images_jpeg_bytes"][0]
        # Decode just the first JPEG to find image size.
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(jpeg))
        W, H = img.size
        # Project per chosen point at frame t.
        ti = min(int(t), tracks_xyz.shape[0] - 1)
        xyz = tracks_xyz[ti, chosen_indices, :]                # (P, 3)
        z = np.clip(xyz[:, 2], 1e-6, None)
        u = xyz[:, 0] / z * fx + cx
        v = xyz[:, 1] / z * fy + cy
        coords = np.stack([u / W, v / H], axis=-1).astype(np.float32)
        return np.clip(coords, 0.0, 1.0)

    # ── davis_bench ───────────────────────────────────────────────────────
    def _build_davis_bench_entries(self):
        root = self.data_roots["davis_bench"]
        captions_path = root / "davis_captions.json"
        if not captions_path.exists():
            raise FileNotFoundError(
                f"DAVIS captions not found at {captions_path}. Download the "
                f"PointMotionBench dataset (ships `davis/davis_captions.json`) "
                f"from allenai/PointMotionBench."
            )
        captions = json.load(open(captions_path))
        entries = []
        for npz_path in sorted((root / "tracks").glob("*_3d.npz")):
            seq = npz_path.stem[:-3]
            # davis_bench 3D is dict {obj_name: (N, T, 3)}; pick the FIRST obj
            # and use it for the entry (full benchmark could iterate all obj
            # but for first-pass eval one-per-clip is sufficient).
            with np.load(npz_path, allow_pickle=True) as f:
                d = f["points_3d"].item()
            obj0 = sorted(d.keys())[0]
            T = int(d[obj0].shape[1])
            entries.append({
                "file": seq,
                "num_frames": T,
                "fps": 30,
                "caption": captions[seq]["description"],
                "clips_by_object": {obj0: [[0, T - 1]]},
                "num_clips_total": 1,
            })
        return entries

    def _load_davis_bench_3d(self, entry, obj_name):
        path = self.data_roots["davis_bench"] / "tracks" / f"{entry['file']}_3d.npz"
        with np.load(str(path), allow_pickle=True) as f:
            d = f["points_3d"].item()
        pts = d[obj_name].astype(np.float32)              # (N, T, 3)
        vis = np.isfinite(pts).all(axis=-1)                # (N, T)
        return pts, vis

    def _load_davis_bench_2d(self, entry, obj_name, chosen_indices, t):
        path = self.data_roots["davis_bench"] / "tracks" / f"{entry['file']}_2d.npz"
        with np.load(str(path), allow_pickle=True) as f:
            tracks_raw = f["tracks"]
            dim = f["dim"].astype(int)
        if tracks_raw.dtype == object:
            d = tracks_raw.item()
            tracks = d[obj_name]            # (T, N, 2)
        else:
            tracks = tracks_raw
        return self._sample_2d(tracks, chosen_indices, t, int(dim[0]), int(dim[1]))

    # ── Per-dataset camera-pose loading for camera-frame-at-t0 ──────────
    #
    # Returns a (4, 4) world-to-camera matrix at the given track-space frame
    # `t`, used by `_build_example` to transform points into the query-camera
    # frame before the delta-from-anchor step. Returns None for datasets where
    # the released NPZs don't ship per-frame extrinsics (stereo4d) — caller
    # falls back to the world-frame delta.
    #
    # Convention: all w2c here are world→camera with the world origin equal
    # to camera-frame-0 (egodex/ytvis/hepic frame 0 is identity; xperience /
    # molmospaces ship c2w that we invert; droid is no-op since the released
    # `points_3d` is already in the camera frame).
    _CAMERA_FRAME_SUPPORTED = (
        "egodex", "ytvis", "hepic", "xperience", "molmospaces", "droid",
        "stereo4d",
    )

    @staticmethod
    def _w2c_from_pose_dict(pose, t):
        """Helper: look up w2c[t] from a {data:(T,4,4), inds:(T,)} NPZ-like
        dict, using `inds` to map the requested track frame `t` to the
        corresponding row of `data`. Falls back to direct indexing when
        `inds` is missing or just a 0..T-1 range."""
        data = pose["data"]
        if "inds" in pose:
            inds = pose["inds"]
            # Vast majority of files have `inds == arange(T)`; only do the
            # search when the frame numbering is non-trivial.
            if len(inds) and (inds[0] != 0 or inds[-1] != len(inds) - 1):
                hits = np.where(inds == t)[0]
                if hits.size:
                    return data[hits[0]].astype(np.float32)
        idx = min(max(0, int(t)), data.shape[0] - 1)
        return data[idx].astype(np.float32)

    def _load_w2c_for_frame(self, entry, t):
        """Return a (4, 4) world-to-camera matrix for `entry` at track-space
        frame `t`, or None if the dataset has no extrinsics on disk."""
        ds = entry["_dataset"]
        stem = entry["file"]
        if ds in ("egodex", "ytvis", "hepic"):
            # vipe pipeline: README labels `data` as w2c but the matrices are
            # actually c2w (verified by reprojection on ytvis — direct project
            # of `data` lands ~12 px off, inverted projection matches the
            # recorded 2D to sub-pixel). So invert here.
            path = self.data_roots[ds] / "camera" / "pose" / f"{stem}.npz"
            if not path.exists():
                return None
            with np.load(str(path), allow_pickle=True) as f:
                c2w = self._w2c_from_pose_dict(f, t)
            R = c2w[:3, :3]; tvec = c2w[:3, 3]
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = R.T
            w2c[:3, 3]  = -R.T @ tvec
            return w2c
        if ds == "xperience":
            path = self.data_roots[ds] / "camera" / "poses" / f"{stem}.npz"
            if not path.exists():
                return None
            with np.load(str(path), allow_pickle=True) as f:
                c2w_all = f["c2w"]
                idx = min(max(0, int(t)), c2w_all.shape[0] - 1)
                c2w = c2w_all[idx].astype(np.float32)
            # invert c2w to get w2c
            R = c2w[:3, :3]; tvec = c2w[:3, 3]
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = R.T
            w2c[:3, 3]  = -R.T @ tvec
            return w2c
        if ds == "molmospaces":
            path = self.data_roots[ds] / "camera" / f"{stem}.npz"
            if not path.exists():
                return None
            with np.load(str(path), allow_pickle=True) as f:
                c2w_all = f["cam_poses"]  # (T_raw, 4, 4) at raw fps
            # The track loader strides molmospaces by MOLMOSPACES_TIME_STRIDE;
            # `t` is in stride-frame space, so map back to raw-frame index.
            t_raw = int(t) * MOLMOSPACES_TIME_STRIDE
            idx = min(max(0, t_raw), c2w_all.shape[0] - 1)
            c2w = c2w_all[idx].astype(np.float32)
            R = c2w[:3, :3]; tvec = c2w[:3, 3]
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = R.T
            w2c[:3, 3]  = -R.T @ tvec
            return w2c
        if ds == "droid":
            # `points_3d` is already in the camera frame, so the world frame
            # IS the camera frame: w2c[t] = identity for all t.
            return np.eye(4, dtype=np.float32)
        if ds == "stereo4d":
            # camera/{slug}.npz ships `cam_pos` (T, 3) + `cam_fwd` (T, 3).
            # World convention is Y-up (verified by reprojection at ~1.3 px
            # median error with constant intrinsics fx=fy=443.405, cx=cy=256).
            # Construct w2c via look-at: camera-forward = cam_fwd, world-up =
            # +Y, camera-right = up × fwd, camera-down = fwd × right.
            path = self.data_roots[ds] / "camera" / f"{stem}.npz"
            if not path.exists():
                return None
            with np.load(str(path), allow_pickle=True) as f:
                pos = f["cam_pos"]; fwd = f["cam_fwd"]
            idx = min(max(0, int(t)), pos.shape[0] - 1)
            p = pos[idx].astype(np.float32); fd = fwd[idx].astype(np.float32)
            z = fd / (np.linalg.norm(fd) + 1e-9)
            up_world = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            x = np.cross(up_world, z)
            nx = np.linalg.norm(x)
            if nx < 1e-6:
                # cam_fwd is parallel to world Y — degenerate; fall back to no
                # transform rather than emit a garbage R.
                return None
            x = x / nx
            y = np.cross(z, x)
            R = np.stack([x, y, z], axis=0)        # rows are cam axes in world
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = R
            w2c[:3, 3]  = -R @ p
            return w2c
        # davis, hotworld, *_hand: no per-frame extrinsics shipped.
        return None

    # ── Per-dataset depth loading for the depth-token branch ─────────────
    #
    # Returns a (H_target, W_target) float32 metric-depth map at the query
    # frame `t`, normalized to the depth encoder's input size via bilinear
    # resampling. Invalid depth (0, NaN, +inf) is preserved as 0 — the encoder
    # treats 0 as the "no signal" marker per `remap_depth_in: log`.
    #
    # Stereo4d always returns None (no depth in molmo-motion-1m), so the
    # caller's 30/30/40 dropout schedule clamps stereo4d into the monocular
    # bucket automatically.

    _DEPTH_SUPPORTED = (
        "egodex", "ytvis", "hepic", "xperience", "droid", "molmospaces",
    )

    def _get_handle(self, key, opener):
        """LRU-style: open a handle on first miss and cache for reuse within
        the same dataloader worker."""
        h = self._depth_handle_cache.get(key)
        if h is None:
            h = opener()
            self._depth_handle_cache[key] = h
        return h

    @staticmethod
    def _resize_depth(depth, size):
        """Bilinear-resize a (H, W) float32 depth map to (size, size). Done in
        torch for parity with the model side (and to avoid any cv2 dependency
        at training time)."""
        import torch
        # Copy out of any mmap'd / non-writable buffer first so torch doesn't
        # warn (the npz mmap path returns read-only arrays).
        depth_np = np.ascontiguousarray(np.asarray(depth, dtype=np.float32))
        t = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)
        t = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear",
                                             align_corners=False)
        return t.squeeze(0).squeeze(0).numpy()

    def _load_depth_at_t(self, entry, t):
        """Return (size, size) float32 metric depth at the query frame, or
        None if the dataset has no shipped depth (stereo4d)."""
        ds = entry["_dataset"]
        stem = entry["file"]
        size = self.depth_target_size

        if ds in ("egodex", "ytvis", "hepic"):
            # EXR-in-zip: one .exr per frame, named like `{idx:05d}.exr`.
            import zipfile, io, tempfile, OpenEXR, Imath
            zp = self.data_roots[ds] / "depth" / f"{stem}.zip"
            if not zp.exists():
                return None
            z = self._get_handle(("zip", str(zp)), lambda: zipfile.ZipFile(str(zp)))
            target_name = f"{int(t):05d}.exr"
            if target_name not in z.namelist():
                return None
            # OpenEXR needs a real path. Write to a tmp once per call.
            with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as tf:
                tf.write(z.read(target_name)); tmp = tf.name
            try:
                exr = OpenEXR.InputFile(tmp)
                hdr = exr.header(); dw = hdr["dataWindow"]
                W = dw.max.x - dw.min.x + 1; H = dw.max.y - dw.min.y + 1
                ch = "Z" if "Z" in hdr["channels"] else next(iter(hdr["channels"]))
                buf = exr.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT))
                depth = np.frombuffer(buf, dtype=np.float32).reshape(H, W)
            finally:
                os.unlink(tmp)
            return self._resize_depth(depth, size)

        if ds == "xperience":
            # depth/{uuid}__{ep}.hdf5 at key "depth/depth": (N, 256, 256) fp32.
            # frame_indices/{slug}.npy gives the (T,) → episode-frame mapping.
            import h5py
            # slug = entry["file"] is "<uuid>__ep<int>__clip_<...>"
            stem_parts = stem.split("__")
            if len(stem_parts) < 2:
                return None
            ep_key = "__".join(stem_parts[:2])    # "<uuid>__ep<int>"
            hdf5_p = self.data_roots[ds] / "depth" / f"{ep_key}.hdf5"
            idx_p  = self.data_roots[ds] / "camera" / "frame_indices" / f"{stem}.npy"
            if not hdf5_p.exists() or not idx_p.exists():
                return None
            indices = np.load(str(idx_p))
            if int(t) >= len(indices):
                return None
            ep_frame = int(indices[int(t)])
            # `locking=False` is required on weka/NFS — default locking
            # raises OSError(errno=37, 'No locks available').
            h5 = self._get_handle(("h5", str(hdf5_p)),
                                  lambda: h5py.File(str(hdf5_p), "r", locking=False))
            depth = h5["depth/depth"][ep_frame].astype(np.float32)
            return self._resize_depth(depth, size)

        if ds == "droid":
            # depth/{ep}.h5 at "{serial}+ext/depth": (T, 360, 640) uint16.
            # Slug format `{episode}__{serial}`.
            import h5py
            if "__" not in stem:
                return None
            ep, serial = stem.rsplit("__", 1)
            h5p = self.data_roots[ds] / "depth" / f"{ep}.h5"
            if not h5p.exists():
                return None
            h5 = self._get_handle(("h5", str(h5p)),
                                  lambda: h5py.File(str(h5p), "r", locking=False))
            key = f"{serial}+ext/depth"
            if key not in h5:
                return None
            d = h5[key]
            if int(t) >= d.shape[0]:
                return None
            depth = d[int(t)].astype(np.float32)
            # droid depth is uint16 mm → convert to meters.
            depth = depth / 1000.0
            return self._resize_depth(depth, size)

        if ds == "molmospaces":
            # camera/{slug}.npz `depth_frames` (T_raw, H, W). t is in
            # stride-frame space, so map back to raw.
            npz_p = self.data_roots[ds] / "camera" / f"{stem}.npz"
            if not npz_p.exists():
                return None
            mm = self._get_handle(
                ("npz", str(npz_p)),
                lambda: np.load(str(npz_p), allow_pickle=True, mmap_mode="r"))
            if "depth_frames" not in mm:
                return None
            t_raw = int(t) * MOLMOSPACES_TIME_STRIDE
            arr = mm["depth_frames"]
            if t_raw >= arr.shape[0]:
                return None
            depth = arr[t_raw].astype(np.float32)
            return self._resize_depth(depth, size)

        # stereo4d + other: no shipped depth.
        return None

    @staticmethod
    def _apply_depth_dropout(depth, dataset_name, rng):
        """30 / 30 / 40 dropout schedule per example. Stereo4d clamps to
        monocular regardless of the random draw. Returns (depth, bucket)
        where bucket ∈ {'monocular', 'masked', 'full'}.

        - monocular: depth replaced with all-zeros (no signal)
        - masked: per-pixel Bernoulli with p_keep=0.5
        - full: depth passed through unchanged
        """
        if dataset_name == "stereo4d" or depth is None:
            H, W = (None, None) if depth is None else depth.shape[-2:]
            return None, "monocular"
        r = rng.random()
        if r < 0.30:
            return np.zeros_like(depth), "monocular"
        elif r < 0.60:
            keep_mask = (rng.random(depth.shape) < 0.5).astype(depth.dtype)
            return depth * keep_mask, "masked"
        else:
            return depth, "full"

    # ── Per-dataset 2D track loading ──────────────────────────────────────

    def _load_2d_coords(self, entry, obj_name, chosen_indices, t):
        """Load normalized [0,1] 2D pixel coords at frame t for chosen points.

        Returns:
            coords_2d: (P, 2) float32 in [0, 1], or None if unavailable
        """
        ds = entry["_dataset"]
        try:
            if ds == "egodex":
                return self._load_egodex_2d(entry, obj_name, chosen_indices, t)
            elif ds == "droid":
                return self._load_droid_2d(entry, obj_name, chosen_indices, t)
            elif ds == "molmospaces":
                return self._load_molmospaces_2d(entry, obj_name, chosen_indices, t)
            elif ds == "hepic":
                return self._load_hepic_2d(entry, obj_name, chosen_indices, t)
            elif ds == "xperience":
                return self._load_xperience_2d(entry, obj_name, chosen_indices, t)
            elif ds == "egodex_hand":
                return self._load_egodex_hand_2d(entry, obj_name, chosen_indices, t)
            elif ds == "xperience_hand":
                return self._load_xperience_hand_2d(entry, obj_name, chosen_indices, t)
            elif ds == "davis":
                return self._load_davis_2d(entry, obj_name, chosen_indices, t)
            elif ds == "ytvis":
                return self._load_ytvis_2d(entry, obj_name, chosen_indices, t)
            elif ds == "stereo4d":
                return self._load_stereo4d_2d(entry, obj_name, chosen_indices, t)
            elif ds == "hotworld":
                return self._load_hotworld_2d(entry, obj_name, chosen_indices, t)
            elif ds == "hot3d_bench":
                return self._load_hot3d_bench_2d(entry, obj_name, chosen_indices, t)
            elif ds == "worldtrack_bench":
                return self._load_worldtrack_bench_2d(entry, obj_name, chosen_indices, t)
            elif ds == "davis_bench":
                return self._load_davis_bench_2d(entry, obj_name, chosen_indices, t)
        except (IndexError, KeyError, FileNotFoundError):
            # 2D/3D point count mismatch or missing file — skip 2D for this example
            return None
        return None

    # ─── molmo-motion-1m unified 2D loaders ────────────────────────────
    @staticmethod
    def _sample_2d(tracks_NT2, chosen_indices, t, img_h, img_w):
        """Common tail: sample (P, 2) tracks at frame t, normalize to [0,1], clip."""
        coords = tracks_NT2[chosen_indices, t, :].copy()
        coords = np.nan_to_num(coords, nan=0.5 * img_w)
        coords[:, 0] /= img_w
        coords[:, 1] /= img_h
        return np.clip(coords, 0.0, 1.0).astype(np.float32)

    def _load_egodex_2d(self, entry, obj_name, chosen_indices, t):
        stem = entry["file"]
        path = self.data_roots["egodex"] / "tracks" / "object" / f"{stem}_2d.npz"
        if not path.exists():
            return None
        with np.load(str(path), allow_pickle=True) as f:
            tracks_raw = f["tracks"]
            dim = f["dim"]
        if tracks_raw.dtype == object:
            tdict = tracks_raw.item()
            if obj_name not in tdict:
                return None
            tracks_2d = np.transpose(tdict[obj_name].astype(np.float32), (1, 0, 2))  # (T,N,2)→(N,T,2)
        else:
            # Flat (T, N, 2) → (N, T, 2)
            tracks_2d = np.transpose(tracks_raw.astype(np.float32), (1, 0, 2))
        return self._sample_2d(tracks_2d, chosen_indices, t, int(dim[0]), int(dim[1]))

    def _load_droid_2d(self, entry, obj_name, chosen_indices, t):
        stem = entry["file"]
        path = self.data_roots["droid"] / "tracks" / f"{stem}_2d.npz"
        with np.load(str(path), allow_pickle=True) as f:
            tracks_2d = np.transpose(f["tracks_2d"].astype(np.float32), (1, 0, 2))  # (T,N,2)→(N,T,2)
            ds_dim = f["ds_dim"]
        return self._sample_2d(tracks_2d, chosen_indices, t, int(ds_dim[0]), int(ds_dim[1]))

    def _load_molmospaces_2d(self, entry, obj_name, chosen_indices, t):
        stem = entry["file"]
        path = self.data_roots["molmospaces"] / "tracks" / f"{stem}_2d.npz"
        s = MOLMOSPACES_TIME_STRIDE
        with np.load(str(path), allow_pickle=True) as f:
            tracks_raw = f["tracks"]
            dim = f["dim"]
        tdict = tracks_raw.item()
        if obj_name not in tdict:
            return None
        # Per-object (T, N, 2) → strided (T', N, 2) → transposed (N, T', 2)
        tr = tdict[obj_name].astype(np.float32)[::s, :, :]
        tracks_2d = np.transpose(tr, (1, 0, 2))
        return self._sample_2d(tracks_2d, chosen_indices, t, int(dim[0]), int(dim[1]))

    def _load_xperience_2d(self, entry, obj_name, chosen_indices, t):
        # 2D coords live INSIDE the 3D NPZ as tracks_2d (object) or pixel_coords (hand).
        stem = entry["file"]
        track_key = obj_name if obj_name in ("object", "left_hand", "right_hand") else "object"
        path = self.data_roots["xperience"] / "tracks" / f"{stem}_{track_key}.npz"
        if not path.exists():
            return None
        with np.load(str(path), allow_pickle=True) as f:
            if "tracks_2d" in f.files:
                tracks_2d = f["tracks_2d"].astype(np.float32)        # (N, T, 2)
            elif "pixel_coords" in f.files:
                tracks_2d = f["pixel_coords"].astype(np.float32)     # (N, T, 2)
            else:
                return None
        img_h, img_w = 512, 512
        return self._sample_2d(tracks_2d, chosen_indices, t, img_h, img_w)

    def _load_egodex_hand_2d(self, entry, obj_name, chosen_indices, t):
        stem = entry["file"]
        path = self.data_roots["egodex_hand"] / "tracks" / "hand" / f"{stem}_2d.npz"
        if not path.exists():
            return None
        with np.load(str(path), allow_pickle=True) as f:
            tracks_raw = f["tracks"]
            dim = f["dim"]
        if tracks_raw.dtype != object:
            return None
        tdict = tracks_raw.item()
        if obj_name not in tdict:
            return None
        tracks_2d = np.transpose(tdict[obj_name].astype(np.float32), (1, 0, 2))
        return self._sample_2d(tracks_2d, chosen_indices, t, int(dim[0]), int(dim[1]))

    def _load_ytvis_2d(self, entry, obj_name, chosen_indices, t):
        stem = entry["file"]
        path = self.data_roots["ytvis"] / "tracks" / f"{stem}_2d.npz"
        if not path.exists():
            return None
        with np.load(str(path), allow_pickle=True) as f:
            tracks_raw = f["tracks"]
            dim = f["dim"]
        tdict = tracks_raw.item() if tracks_raw.dtype == object else None
        if tdict is None or obj_name not in tdict:
            return None
        tracks_2d = np.transpose(tdict[obj_name].astype(np.float32), (1, 0, 2))
        return self._sample_2d(tracks_2d, chosen_indices, t, int(dim[0]), int(dim[1]))

    def _load_stereo4d_2d(self, entry, obj_name, chosen_indices, t):
        stem = entry["file"]
        path = self.data_roots["stereo4d"] / "tracks" / f"{stem}_2d.npz"
        if not path.exists():
            return None
        with np.load(str(path), allow_pickle=True) as f:
            tracks_raw = f["tracks"]
            dim = f["dim"]
        tdict = tracks_raw.item() if tracks_raw.dtype == object else None
        if tdict is None or obj_name not in tdict:
            return None
        tracks_2d = np.transpose(tdict[obj_name].astype(np.float32), (1, 0, 2))
        return self._sample_2d(tracks_2d, chosen_indices, t, int(dim[0]), int(dim[1]))

    def _load_hepic_2d(self, entry, obj_name, chosen_indices, t):
        # AllTracker pixel-space tracks. The 2D npz is a flat (T, N_total, 2)
        # tensor where N_total is the concatenation of per-object blocks in
        # the dict-iteration order of the matching _3d.npz. There's no
        # explicit per-object slicing key — re-derive the offset from 3D
        # shapes. (Per hepic README: 2D is NOT a reprojection of 3D; they
        # share seeds but are produced by independent pipelines.)
        stem = entry["file"]
        p2 = self.data_roots["hepic"] / "tracks" / f"{stem}_2d.npz"
        p3 = self.data_roots["hepic"] / "tracks" / f"{stem}_3d.npz"
        if not p2.exists():
            return None
        with np.load(str(p3), allow_pickle=True) as f3:
            pts_dict = f3["points_3d"].item()
        if obj_name not in pts_dict:
            return None
        offset = 0
        for k, v in pts_dict.items():
            if k == obj_name:
                break
            offset += v.shape[0]
        n_obj = pts_dict[obj_name].shape[0]
        with np.load(str(p2), allow_pickle=True) as f2:
            tracks_raw = f2["tracks"].astype(np.float32)  # (T, N_total, 2)
            dim = f2["dim"]
        # Slice this object's block then transpose (T,N,2) → (N,T,2).
        tracks_block = np.transpose(tracks_raw[:, offset:offset + n_obj, :], (1, 0, 2))
        return self._sample_2d(tracks_block, chosen_indices, t, int(dim[0]), int(dim[1]))

    def _load_davis_2d(self, entry, obj_name, chosen_indices, t):
        # Use cached data from _load_davis_3d
        if "_cached_2d" in entry:
            tracks_2d = entry.pop("_cached_2d")  # (N, T, 2)
            img_h, img_w = entry.pop("_cached_2d_dim")
        else:
            vid_id = entry["file"]
            path_2d = self.data_roots["davis"] / "vipe" / "davis_final_tracks" / f"{vid_id}_2d.npz"
            if not path_2d.exists():
                return None
            with np.load(str(path_2d), allow_pickle=True) as f:
                tracks_dict = f["tracks"][()]
            if obj_name not in tracks_dict:
                return None
            tracks_2d = np.transpose(tracks_dict[obj_name].astype(np.float32), (1, 0, 2))
            img_h, img_w = 480, 854
        coords = tracks_2d[chosen_indices, t, :].copy()
        coords = np.nan_to_num(coords, nan=0.5 * img_w)
        coords[:, 0] /= img_w
        coords[:, 1] /= img_h
        return np.clip(coords, 0.0, 1.0).astype(np.float32)

    def _load_xperience_hand_2d(self, entry, obj_name, chosen_indices, t):
        stem = entry["file"]
        path = self.data_roots["xperience_hand"] / "tracks" / f"{stem}_{obj_name}.npz"
        if not path.exists():
            return None
        with np.load(str(path), allow_pickle=True) as f:
            pixel_coords = f["pixel_coords"].astype(np.float32)  # (N, T, 2)
        img_h, img_w = 512, 512
        return self._sample_2d(pixel_coords, chosen_indices, t, img_h, img_w)

    # ── Per-dataset object helpers ──────────────────────────────────────

    def _get_object_names(self, entry):
        return sorted(entry["clips_by_object"].keys())

    def _get_point_count(self, entry, obj_name):
        """Get point count for an object. Uses cheap methods to avoid loading large NPZs.

        For most datasets this is either a constant, derivable from cached metadata,
        or the entry only has 1 object so this is never called in a weighted loop.
        """
        ds = entry["_dataset"]
        if ds == "egodex":
            # molmo-motion-1m egodex has one object cluster per clip; the legacy
            # multi-cluster `_egodex_filter_meta` path is no longer used.
            return 100
        elif ds == "droid":
            return 100  # DROID: single object, ~60-95 points. Exact count not needed.
        elif ds == "molmospaces":
            return 100  # Approximate; exact count only matters for multi-object weighting
        elif ds == "hepic":
            return 100  # Fixed 100 points per object block
        elif ds == "xperience":
            return 100  # Single object, no weighting needed
        elif ds == "egodex_hand":
            return 100  # Approximate; hands typically ~50-100 points
        elif ds == "xperience_hand":
            return 21  # Xperience hands: typically 21 points per hand
        elif ds == "davis":
            return 100  # DAVIS: typically ~50-100 points per object
        elif ds == "ytvis":
            return 100  # YT-VIS: typically ~50-100 points per object (dict format)
        elif ds == "stereo4d":
            return 100  # Stereo4D: ~100-300 points per object cluster
        elif ds == "hotworld":
            # WT: 100 (typical per object after object_id mask).
            # HOT3D: 100 (fixed block size).
            return 100
        elif ds == "hot3d_bench":
            return 2000   # filtered HOT3D-Clips ships exactly 2000 surface points per sub-clip
        elif ds == "worldtrack_bench":
            return 100    # per-object after object_id mask (similar to hotworld)
        elif ds == "davis_bench":
            return 100    # AllTracker samples ~100 query points per object via k-means
        raise ValueError(f"Unknown dataset: {ds}")

    # ── Per-dataset video helpers ───────────────────────────────────────

    def _get_video_path(self, entry):
        ds = entry["_dataset"]
        # The seven molmo-motion-1m datasets all use the uniform layout
        # `<root>/videos/<file>.mp4` (DROID videos already include the
        # `__{cam}` suffix in `file`). The two legacy datasets (davis,
        # hotworld) keep their previous bespoke paths below.
        if ds in ("egodex", "ytvis", "hepic", "xperience", "droid", "stereo4d",
                  "molmospaces", "egodex_hand", "xperience_hand"):
            return str(self.data_roots[ds] / "videos" / f"{entry['file']}.mp4")
        elif ds == "davis":
            vid_id = entry["file"]
            return str(self.data_roots["davis"] / "vipe" / "vipe_results" / "rgb" / f"{vid_id}.mp4")
        elif ds == "hotworld":
            src = entry["source_dataset"]
            if src == "worldtrack":
                # WT has no mp4 — RGB frames are JPEGs inline in the same NPZ
                # that holds the tracks. _load_hotworld_3d caches them on entry.
                jpegs = entry.pop("_cached_jpegs", None)
                if jpegs is None:
                    with np.load(entry["tracks_npz"], allow_pickle=True) as f:
                        jpegs = f["images_jpeg_bytes"]
                # Return the bytes array; _read_video_frames will detect this.
                return jpegs
            return entry["rgb_mp4"]
        elif ds == "hot3d_bench":
            return str(self.data_roots["hot3d_bench"] / "rgbs" / f"{entry['file']}.mp4")
        elif ds == "worldtrack_bench":
            clip_dir = self.data_roots["worldtrack_bench"] / entry["file"]
            npz_path = clip_dir / f"{clip_dir.name}.npz"
            with np.load(str(npz_path), allow_pickle=True) as f:
                return f["images_jpeg_bytes"]
        elif ds == "davis_bench":
            return str(self.data_roots["davis_bench"] / "videos" / "input_480p" / f"{entry['file']}.mp4")
        raise ValueError(f"Unknown dataset: {ds}")

    def _map_frame_to_video(self, entry, frame_indices):
        """Map track-space frame indices to video-space frame indices.

        Under molmo-motion-1m, DROID's MP4s are re-encoded at 15 fps to match
        the track rate, so the old 60→15 stride no longer applies. MolmoSpaces
        retains its 4× temporal stride because the loader internally
        sub-samples the track arrays (see `_load_molmospaces_3d`).
        """
        ds = entry["_dataset"]
        if ds == "molmospaces":
            return [idx * MOLMOSPACES_TIME_STRIDE for idx in frame_indices]
        # All others (egodex, hepic, xperience, droid, hands, davis, ytvis,
        # stereo4d, hotworld): direct mapping.
        return list(frame_indices)

    def _read_video_frames(self, mp4_path, frame_indices):
        """Read specific frames from video as (N, H, W, 3) uint8 RGB.

        Optimized: if frames are consecutive, seek once to first frame
        and read sequentially (2× faster than per-frame seeking).

        Special-case: when `mp4_path` is a numpy array of JPEG bytes (the
        format used by WorldTrack inside HotWorld — RGB frames are stored
        inline in the tracks NPZ, no mp4 file), decode each requested frame
        directly with PIL and skip the OpenCV path.
        """
        if isinstance(mp4_path, np.ndarray):
            import io
            from PIL import Image
            frames = []
            for fi in frame_indices:
                fi_clamped = max(0, min(int(fi), len(mp4_path) - 1))
                img = np.array(Image.open(io.BytesIO(bytes(mp4_path[fi_clamped]))))
                if img.ndim == 2:
                    img = np.stack([img, img, img], axis=-1)
                if img.shape[-1] == 4:
                    img = img[..., :3]
                frames.append(img)
            return np.stack(frames, axis=0).astype(np.uint8)
        if not os.path.exists(mp4_path):
            raise FileNotFoundError(
                f"Video not found: {mp4_path}. The molmo-motion-1m / PointMotionBench "
                f"downloads do not include raw videos for every dataset — run the "
                f"per-dataset reconstruction script first (e.g. "
                f"`<dataset>/reconstruct_videos.py` for molmo-motion-1m, or the "
                f"`pointmotionbench/hot3d/` rgbs rebuild for HOT3D eval). See the "
                f"per-dataset README."
            )
        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"Could not open video (corrupt file or unsupported codec): {mp4_path}"
            )
        frames = []

        # Check if indices are consecutive → use fast sequential read
        sorted_indices = sorted(frame_indices)
        is_consecutive = all(
            sorted_indices[i + 1] - sorted_indices[i] == 1
            for i in range(len(sorted_indices) - 1)
        ) if len(sorted_indices) > 1 else True

        if is_consecutive and len(sorted_indices) > 0:
            # Seek once to first frame, then read consecutively
            cap.set(cv2.CAP_PROP_POS_FRAMES, sorted_indices[0])
            for _ in sorted_indices:
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                elif frames:
                    frames.append(frames[-1].copy())
                else:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 854
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                    frames.append(np.zeros((h, w, 3), dtype=np.uint8))
        else:
            # Non-consecutive: seek per frame
            for fi in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                elif frames:
                    frames.append(frames[-1].copy())
                else:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 854
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                    frames.append(np.zeros((h, w, 3), dtype=np.uint8))

        cap.release()
        return np.stack(frames, axis=0)  # (N, H, W, 3)

    # ── Caption ─────────────────────────────────────────────────────────

    @staticmethod
    def _get_caption(entry):
        caption = entry.get("caption", "")
        if caption and caption.strip():
            return caption.strip()
        # EgoDex fallback: extract task name from stem
        ds = entry.get("_dataset", "")
        if ds == "egodex":
            stem = entry.get("file", "")
            # stem like "part1_add_remove_lid_0" → "add remove lid"
            parts = stem.split("_")
            if len(parts) >= 3:
                return " ".join(parts[1:-1])
        return ""

    # ── Eval config expansion ──────────────────────────────────────────

    def _build_eval_configs(self):
        """Expand all entries into deterministic eval configs.

        Per entry: pick ONE (object, clip) — the object with the most points
        (ties broken alphabetically) and its longest extended clip — then emit
        up to three timestamps for that clip: t_min (earliest valid), t_mid
        (midpoint of [t_min, t_max]), t_max (latest valid). If the clip is
        too short to fit H + F, falls back to a single clamped t with
        need_padding=True.

        When `self.eval_first_h_frames` is True (set by the `_t0first` token),
        the eval emits exactly ONE config per entry with the query frame
        pinned at t = H − 1 — i.e. history is the clip's first H frames and
        future is [H, ..., H + F − 1] (clipped to clip length). This is the
        protocol used for the PointMotionBench benchmark evals.
        """
        H = self.history_size
        F = self.num_future_frames
        if self.eval_first_h_frames:
            return self._build_eval_configs_first_h(H, F)
        configs = []
        for entry_idx, entry in enumerate(self.entries):
            num_frames = entry["num_frames"]
            obj_names = sorted(entry["clips_by_object"].keys())
            if not obj_names:
                continue

            # Object with the most points (alphabetical tie-break).
            try:
                pc = [(self._get_point_count(entry, n), n) for n in obj_names]
            except Exception:
                continue
            max_count = max(c for c, _ in pc)
            best_obj = next(n for c, n in pc if c == max_count)

            # Longest extended clip for that object.
            clips = entry["clips_by_object"][best_obj]
            ext_clips = [(max(0, s - 1), min(num_frames - 1, e + 2)) for s, e in clips]
            clip_lens = [ee - es + 1 for es, ee in ext_clips]
            best_clip_idx = clip_lens.index(max(clip_lens))
            ext_start, ext_end = ext_clips[best_clip_idx]
            anchor_s = clips[best_clip_idx][0]

            # Up to 3 timestamps: earliest, middle, latest.
            t_min = ext_start + H - 1
            t_max = ext_end - F
            if t_max < t_min:
                t_clamp = max(ext_start, min(ext_start + H - 1, ext_end))
                t_list = [t_clamp]
                need_padding_list = [True]
            else:
                t_mid = (t_min + t_max) // 2
                # Dedup preserving order min → mid → max.
                seen = set(); t_list = []
                for t in (t_min, t_mid, t_max):
                    if t not in seen:
                        t_list.append(t); seen.add(t)
                need_padding_list = [False] * len(t_list)

            for t, need_padding in zip(t_list, need_padding_list):
                hist_start = t - H + 1
                hist_frames = list(range(max(ext_start, hist_start), t + 1))
                while len(hist_frames) < H:
                    hist_frames.insert(0, hist_frames[0])
                future_end = min(ext_end, t + F)
                future_frames = list(range(t + 1, future_end + 1))

                configs.append({
                    "entry_idx": entry_idx,
                    "obj_name": best_obj,
                    "clip_idx": best_clip_idx,
                    "anchor_s": anchor_s,
                    "t": t,
                    "hist_frames": hist_frames,
                    "future_frames": future_frames,
                    "need_padding": need_padding,
                })
        return configs

    def _build_eval_configs_first_h(self, H, F):
        """First-H-frames eval mode: one config per entry, t = H − 1,
        future = [H, …, H + F − 1] clipped to clip length.

        For each entry: pick the object with the most points (alphabetical
        tie-break) and the first clip range. Build a single config with
        deterministic frame indices anchored at the clip start.
        """
        configs = []
        for entry_idx, entry in enumerate(self.entries):
            num_frames = entry["num_frames"]
            obj_names = sorted(entry["clips_by_object"].keys())
            if not obj_names:
                continue
            try:
                pc = [(self._get_point_count(entry, n), n) for n in obj_names]
            except Exception:
                continue
            best_obj = max(pc)[1]
            clips = entry["clips_by_object"][best_obj]
            if not clips:
                continue
            anchor_s = clips[0][0]

            # Need at least H frames of history. If the clip is too short
            # even for H, skip it (no useful eval for this entry).
            if num_frames < H:
                continue
            # Query frame t = H − 1 within the clip; the clip uses
            # absolute frame indices into the track tensor, so for benchmark
            # datasets that start at 0 this is exactly frame H-1.
            t = H - 1
            hist_frames = list(range(0, H))
            # Future ends at min(num_frames - 1, t + F).
            future_end = min(num_frames - 1, t + F)
            future_frames = list(range(t + 1, future_end + 1))
            # If the clip is too short to even reach one future frame, skip.
            if not future_frames:
                continue

            configs.append({
                "entry_idx": entry_idx,
                "obj_name": best_obj,
                "clip_idx": 0,
                "anchor_s": anchor_s,
                "t": t,
                "hist_frames": hist_frames,
                "future_frames": future_frames,
                "need_padding": len(future_frames) < F,
            })
        return configs

    def _stratified_subsample(self, configs, max_per_dataset):
        """Deterministic per-video subsampling.

        max_per_dataset is a cap on the number of VIDEOS per dataset. All
        (obj, clip) configs for each selected video are kept. Videos are
        picked in sorted (entry_idx) order so every eval job sees the same
        videos.
        """
        # Group configs by dataset, then by entry_idx (video)
        ds_by_video = {}
        for cfg in configs:
            entry = self.entries[cfg["entry_idx"]]
            ds = entry["_dataset"]
            ds_by_video.setdefault(ds, {}).setdefault(cfg["entry_idx"], []).append(cfg)

        result = []
        for ds in sorted(ds_by_video.keys()):
            by_video = ds_by_video[ds]
            video_indices = sorted(by_video.keys())[:max_per_dataset]

            selected_configs = []
            for vidx in video_indices:
                selected_configs.extend(by_video[vidx])

            result.extend(selected_configs)
            log.info(f"[Traj3D] {ds}: {len(selected_configs)} configs from "
                     f"{len(video_indices)} videos (target ≤{max_per_dataset} videos)")

        return result

    # ── Text formatting ─────────────────────────────────────────────────

    @staticmethod
    def _quantize(val):
        return int(round(val * 1000))

    def _format_tracks_text(self, timestamps, points_delta, visibility, label):
        """Format delta coords into <tracks> text.

        Args:
            timestamps: list of float timestamps
            points_delta: (P, F, 3) delta values
            visibility: (P, F) bool
            label: trailing label text (e.g. "object history" / "object trajectories")
        """
        P = points_delta.shape[0]
        frame_strings = []
        for fi, ts in enumerate(timestamps):
            parts = [f"{ts:.1f}"]
            for pi in range(P):
                if not visibility[pi, fi]:
                    continue
                obj_id = pi + 1
                x = Trajectory3DDataset._quantize(points_delta[pi, fi, 0])
                y = Trajectory3DDataset._quantize(points_delta[pi, fi, 1])
                z = Trajectory3DDataset._quantize(points_delta[pi, fi, 2])
                parts.append(f"{obj_id} {x} {y} {z}")
            if len(parts) > 1:
                frame_strings.append(" ".join(parts))
        coords_str = ";".join(frame_strings)
        return f'<tracks coords="{coords_str}">{label}</tracks>'

    # ── Example building ────────────────────────────────────────────────

    def _build_example(self, entry, obj_name, pts_3d, visibility, chosen_indices,
                       hist_frames, future_frames, t, need_padding, anchor_s=None):
        """Build Molmo2-compatible example dict.

        Args:
            entry: metadata dict with _dataset field
            obj_name: chosen object name
            pts_3d: (N_obj, T_total, 3) all points for this object
            visibility: (N_obj, T_total) bool
            chosen_indices: (P,) indices into N_obj
            hist_frames: list of H frame indices (track space)
            future_frames: list of <=F frame indices (track space)
            t: current timestep (last history frame)
            need_padding: whether future frames need padding
            anchor_s: deprecated (still accepted for backwards compatibility,
                but ignored). The 2D point-feature anchor is now the last
                history frame `t`; the anchor is no longer the segment start.
        """
        # Per-example H is determined by the length of `hist_frames` itself
        # (the sampler decides whether to use H=1 or H=3 per example).
        H = len(hist_frames)
        P = self.num_points

        all_frames = hist_frames + future_frames
        n_actual_future = len(future_frames)
        # Effective F: at least num_future_frames (for padding short clips),
        # but expand when _get_train samples a longer future horizon.
        F = max(self.num_future_frames, n_actual_future)

        # Extract trajectory for chosen points at all frames
        traj = pts_3d[chosen_indices][:, all_frames, :]  # (P, H+n_actual, 3)

        # Clean NaN → 0 (non-interpolated tracks like DAVIS, hand datasets may have NaN)
        traj_clean = np.nan_to_num(traj, nan=0.0)

        # Optional: world → camera-frame-at-t (paper t_0 convention).
        # Applied uniformly across all history+future frames so every point's
        # coordinates are expressed in the camera frame of the QUERY time t.
        # Unsupported datasets (stereo4d, davis, hotworld, *_hand) fall back
        # to the world-frame delta (same as the historical behavior).
        if self.use_camera_frame and entry["_dataset"] in self._CAMERA_FRAME_SUPPORTED:
            w2c = self._load_w2c_for_frame(entry, t)
            if w2c is not None:
                R = w2c[:3, :3]; tvec = w2c[:3, 3]
                # traj_clean shape (P, T_total, 3) → X_cam = X_world @ R.T + t
                traj_clean = traj_clean @ R.T + tvec

        # Anchor: first point at frame t (last history frame, index H-1)
        anchor_idx = min(H - 1, traj_clean.shape[1] - 1)
        anchor = traj_clean[0, anchor_idx, :].copy()  # (3,)

        # Delta encoding
        traj_delta = traj_clean - anchor[np.newaxis, np.newaxis, :]

        # Pad future if clip too short
        if traj_delta.shape[1] < H + F:
            pad_count = (H + F) - traj_delta.shape[1]
            last_val = traj_delta[:, -1:, :]
            traj_delta = np.concatenate(
                [traj_delta, np.repeat(last_val, pad_count, axis=1)], axis=1)

        # Visibility for chosen points
        vis_chosen = visibility[chosen_indices]  # (P, T_total)

        # History visibility
        hist_vis = np.ones((P, H), dtype=bool)
        for hi, fi in enumerate(hist_frames):
            hist_vis[:, hi] = vis_chosen[:, fi]

        # Future valid mask: visible AND within actual clip bounds (not padded)
        valid_mask = np.zeros((P, F), dtype=bool)
        for fi_idx in range(n_actual_future):
            frame_idx = future_frames[fi_idx]
            valid_mask[:, fi_idx] = vis_chosen[:, frame_idx]
        # Padded future frames remain False

        # Split delta into history and future
        hist_delta = traj_delta[:, :H, :]    # (P, H, 3)
        future_delta = traj_delta[:, H:, :]  # (P, F, 3)

        # Zero out invisible positions in both deltas so text formatting never
        # picks up junk values (raw tracks may carry NaN there; nan_to_num
        # above turned those into 0 relative to the raw anchor, which is not
        # the same as a 0-delta). Sparse emission then skips them.
        hist_delta[~hist_vis] = 0.0
        future_delta[~valid_mask] = 0.0

        # ── Text format ─────────────────────────────────────────────
        hist_timestamps = [float(i) for i in range(H)]
        future_timestamps = [float(H + i) for i in range(F)]

        # v1-era sparse emission: only visible entries (raw tracks carry NaN at
        # invisible positions, so deltas there are junk — skip them in text).
        # `valid_mask` already excludes padded-future positions; `hist_vis`
        # reflects real per-frame visibility within the (always full-length) H
        # history window.
        # When _v1match is on, use the single-label "object points" for BOTH
        # input and output <tracks> blocks (matches traj3d_droid_v1 exactly).
        lbl_hist = LABEL_TEXT_V1 if self.v1_match_format else LABEL_TEXT_HISTORY
        lbl_fut = LABEL_TEXT_V1 if self.v1_match_format else LABEL_TEXT_FUTURE
        input_tracks = self._format_tracks_text(hist_timestamps, hist_delta, hist_vis, lbl_hist)
        output_tracks = self._format_tracks_text(future_timestamps, future_delta, valid_mask, lbl_fut)

        # Endpoint-first mode: emit a single-frame <tracks>...<LABEL_TEXT_ENDPOINT> block
        # covering only the last valid future frame, prepended to the full trajectory
        # answer. Supervised by the normal answer CE loss (appears first in the answer,
        # so gets gradient before the full trajectory tokens).
        endpoint_tracks = ""
        if (not self.v1_match_format) and self.pred_end_point_first and n_actual_future > 0:
            last_idx = n_actual_future - 1
            endpoint_tracks = self._format_tracks_text(
                [future_timestamps[last_idx]],
                future_delta[:, last_idx:last_idx + 1, :],
                valid_mask[:, last_idx:last_idx + 1],
                LABEL_TEXT_ENDPOINT)

        caption = self._get_caption(entry)

        # ── 2D point features (optional) ────────────────────────────
        # Two independent flags:
        # 2D anchor frame for both point-feature injection and coord text
        # is now the LAST HISTORY FRAME `t` (= paper's t_0). Both
        # `use_2d_point_features` and `use_2d_coordinate` consume the same
        # per-point coords loaded at frame t.
        coords_2d_t = None
        points_inline = ""
        coords_2d_inline = ""
        if self.use_2d_point_features or self.use_2d_coordinate:
            coords_2d_t = self._load_2d_coords(entry, obj_name, chosen_indices, int(t))

        if self.use_2d_point_features and coords_2d_t is not None:
            # One per-point feature per example, sampled from frame t's ViT
            # patches (no extra anchor frame in the video stack). Format:
            # <points anchor 1 <2d_feat_start><|pf|><2d_feat_end> ... P .../>
            wrapped_pf = f"{TWO_D_FEAT_START_TOKEN}{POINT_FEATURE_TOKEN}{TWO_D_FEAT_END_TOKEN}"
            pts = " ".join(f"{pi + 1} {wrapped_pf}" for pi in range(P))
            points_inline = f"<points anchor {pts}/>"

        if self.use_2d_coordinate and coords_2d_t is not None:
            # One-frame <points coords="..."> with integer [0, 1000] x/y per point
            # at the last history timestamp (H-1).0.
            ts = f"{float(H - 1):.1f}"
            xy = (np.clip(coords_2d_t, 0.0, 1.0) * 1000.0).round().astype(int)
            parts = [ts]
            for pi in range(P):
                parts.append(f"{pi + 1} {int(xy[pi, 0])} {int(xy[pi, 1])}")
            coords_str = " ".join(parts)
            coords_2d_inline = f'<points coords="{coords_str}">{LABEL_TEXT_2D_COORD}</points>'

        if self.v1_match_format:
            # Byte-exact v1 prompt + answer. Ignores all other format flags
            # (_pointfeat / _2dcoord / _endfirst / _predhist) so the result
            # matches traj3d_droid_v1 verbatim: bare caption, hardcoded nominal
            # F timestamps, "and {H} history frames:\n<tracks>" connector,
            # single "object points" label. Answer is the single future block.
            F_nominal = self.num_future_frames
            question = (
                f"Predict the future 3D trajectories of {P} points over {F_nominal} timestamps, "
                f"given action: {caption}, and {H} history frames:\n"
                f"{input_tracks}")
            # output_tracks stays as the single future block (no endpoint / history prepend)
        else:
            # ── Build prompt: "given action", optional 2D fields, and (unless
            #    predict_history_3d) the history-3D field, joined with Oxford-comma "and".
            fields = [f'given action: "{caption}"']
            if coords_2d_inline:
                fields.append(f'2d point coordinates: "{coords_2d_inline}"')
            if points_inline:
                fields.append(f'2d history point features: "{points_inline}"')
            if not self.predict_history_3d:
                fields.append(f'history 3d point coordinates: "{input_tracks}"')
            if len(fields) >= 2:
                fields[-1] = f"and {fields[-1]}"
            question = (f"Predict the future 3D point coordinates of {P} points over "
                        f"{n_actual_future} timestamps, " + ", ".join(fields) + ".")
            if self.pred_end_point_first:
                question += " predict the endpoint 3d coordinate first."

            # ── Build answer: optional endpoint → optional history → full trajectory ──
            answer_blocks = []
            if endpoint_tracks:
                answer_blocks.append(endpoint_tracks)
            if self.predict_history_3d:
                answer_blocks.append(input_tracks)
            answer_blocks.append(output_tracks)
            output_tracks = " ".join(answer_blocks)

        # ── Video frames ────────────────────────────────────────────
        # Feed the H history frames `[t-H+1, …, t]` directly. We no longer
        # prepend a separate anchor frame: the 2D-feature anchor *is* the
        # last history frame `t`, which is already the last frame here.
        video_path = self._get_video_path(entry)
        video_frame_indices = self._map_frame_to_video(entry, list(hist_frames))
        frames_rgb = self._read_video_frames(video_path, video_frame_indices)

        n_video_frames = len(hist_frames)
        timestamps_arr = np.arange(n_video_frames, dtype=np.float64) * 1.0
        video_frames = VideoFrames(
            frames=frames_rgb,
            timestamps=timestamps_arr,
            target_fps=1.0,
        )

        # ── Metadata ────────────────────────────────────────────────
        ds = entry["_dataset"]
        file_id = entry.get("file", "")
        if ds == "droid":
            file_id = f"{entry['file']}_{entry['cam']}"

        example_id = f"traj3d_{ds}_{file_id}_{obj_name}_t{t}"

        metadata = {
            "example_id": example_id,
            "task": "3d_trajectory",
            "expression": caption,
            "video": file_id,
            "task_name": ds,
            "obj_label": obj_name,
            "dataset_name": ds,
            "t": int(t),
            "hist_frames": [int(x) for x in hist_frames],
            "future_frames": [int(x) for x in future_frames],
            # GT for metric computation (raw 3D space)
            "gt_answer": output_tracks,
            "gt_anchor": anchor.tolist(),
            # IMPORTANT: gt_future_raw MUST be in the SAME coordinate frame as
            # gt_anchor. `traj_clean` has the optional `use_camera_frame` transform
            # already applied, so it's the right tensor to slice from. Reading
            # from raw `pts_3d` here was a bug that made v3/v4 (camera-frame)
            # eval L2 incomparable to v2 — the metric was computing
            # ||pred_in_cam - gt_in_world|| instead of ||pred - gt|| in one frame.
            "gt_future_raw": (traj_clean[:, H:H + n_actual_future, :].astype(np.float32)
                              if n_actual_future > 0 else np.zeros((P, 0, 3), dtype=np.float32)).tolist(),
            "gt_future_vis": valid_mask.tolist(),
            "pass_idx": None,
            "point_indices": chosen_indices.tolist(),
        }

        # Pad gt_future_raw if needed (for padded frames, use last actual value)
        if n_actual_future < F:
            gt_raw = np.array(metadata["gt_future_raw"], dtype=np.float32)  # (P, n_actual, 3)
            if gt_raw.shape[1] > 0:
                pad_count = F - gt_raw.shape[1]
                last_val = gt_raw[:, -1:, :]
                gt_raw = np.concatenate([gt_raw, np.repeat(last_val, pad_count, axis=1)], axis=1)
            else:
                gt_raw = np.zeros((P, F, 3), dtype=np.float32)
            metadata["gt_future_raw"] = gt_raw.tolist()

        # Pass coords_2d through metadata for the model hook to extract.
        # These are the coords at frame t (= the last history frame); the
        # model side reads the corresponding ViT patches from `raw_vit[:, -1]`.
        if self.use_2d_point_features and coords_2d_t is not None:
            metadata["coords_2d"] = coords_2d_t  # (P, 2) float32, normalized [0,1]
            metadata["coords_2d_frame"] = int(t)

        # Depth token branch: load depth at frame t, apply 30/30/40 dropout,
        # ship through metadata as (H, W) float32 (or None if monocular bucket).
        if self.use_depth_token and entry["_dataset"] in self._DEPTH_SUPPORTED + ("stereo4d",):
            raw_depth = self._load_depth_at_t(entry, int(t))
            # Use a per-example RNG seeded by entry idx + t for reproducible
            # dropout across runs (cheap and self-contained).
            d_rng = np.random.RandomState(abs(hash((entry["_dataset"], entry.get("file",""), int(t)))) & 0xffffffff)
            dropped, bucket = self._apply_depth_dropout(raw_depth, entry["_dataset"], d_rng)
            metadata["depth_t"] = dropped     # (H, W) float32 metric meters OR None
            metadata["depth_bucket"] = bucket

        return {
            "video": video_frames,  # VideoFrames object, preprocessor accepts this directly
            "message_list": [{"style": "video_qa", "question": question,
                              "answer": output_tracks}],
            "metadata": metadata,
        }

    # ── Dataset interface ───────────────────────────────────────────────

    @classmethod
    def download(cls, n_procs=1):
        pass

    def __len__(self):
        if self.is_eval:
            return len(self.eval_configs)
        return len(self.entries)

    def get(self, item, rng):
        if self.is_eval:
            return self._get_eval(item)
        return self._get_train(item, rng)

    # ── Training sampling ───────────────────────────────────────────────

    def _get_train(self, item, rng):
        if self._ds_probs is not None:
            ds = self._ds_names_sorted[rng.choice(len(self._ds_probs), p=self._ds_probs)]
            indices = self._ds_to_indices[ds]
            entry = self.entries[indices[rng.randint(0, len(indices))]]
        else:
            entry = self.entries[item % len(self.entries)]
        # Per-example history length. When `mixed_history` is on, draw H from
        # {1, 3} with 50/50 probability (paper §4 setting). Otherwise use
        # the constructor-fixed `self.history_size`.
        if self.mixed_history:
            H = 1 if rng.rand() < 0.5 else 3
        else:
            H = self.history_size
        F = self.num_future_frames
        P = self.num_points
        num_frames = entry["num_frames"]

        # Step 1: Sample object (weighted by point count)
        obj_names = self._get_object_names(entry)
        if len(obj_names) == 0:
            log.warning(f"No objects for {entry.get('file', '?')}, skipping")
            return self._get_train(rng.randint(0, len(self.entries)), rng)

        try:
            weights = np.array([self._get_point_count(entry, n) for n in obj_names],
                               dtype=np.float64)
        except Exception as e:
            log.warning(f"Failed to get point counts for {entry.get('file', '?')}: {e}")
            return self._get_train(rng.randint(0, len(self.entries)), rng)

        if weights.sum() == 0:
            log.warning(f"All point counts zero for {entry.get('file', '?')}, skipping")
            return self._get_train(rng.randint(0, len(self.entries)), rng)

        weights = weights / weights.sum()
        chosen_obj = obj_names[rng.choice(len(obj_names), p=weights)]

        # Step 2: Sample clip (weighted by extended frame count)
        clips = entry["clips_by_object"][chosen_obj]
        ext_clips = [(max(0, s - 1), min(num_frames - 1, e + 2)) for s, e in clips]
        clip_weights = np.array([ee - es + 1 for es, ee in ext_clips], dtype=np.float64)
        clip_weights = clip_weights / clip_weights.sum()
        clip_idx = rng.choice(len(clips), p=clip_weights)
        ext_start, ext_end = ext_clips[clip_idx]

        # Step 3: Load 3D + visibility first (needed to find valid t)
        try:
            pts_3d, visibility = self._load_3d_and_vis(entry, chosen_obj)
        except Exception as e:
            log.warning(f"Failed to load 3D for {entry.get('file', '?')}: {e}")
            return self._get_train(rng.randint(0, len(self.entries)), rng)

        # Step 4: Sample timestep t, retry within clip if no visible points
        t_min = ext_start + H - 1
        t_max = ext_end - F
        need_padding = t_max < t_min
        if need_padding:
            t = min(ext_start + H - 1, ext_end)
            t = max(ext_start, t)
        else:
            t = rng.randint(t_min, t_max + 1)

        # Clamp t to valid range for this track. Visibility is filtered at
        # frame t (= last history frame = 2D-feature anchor in the new
        # paper-aligned setup).
        t_clamped = min(t, pts_3d.shape[1] - 1)
        visible_idx = np.where(visibility[:, t_clamped])[0]

        # If no visible points at t, try other t values within the clip.
        if len(visible_idx) == 0 and not need_padding:
            all_t = list(range(t_min, t_max + 1))
            rng.shuffle(all_t)
            for alt_t in all_t:
                alt_t_clamped = min(alt_t, pts_3d.shape[1] - 1)
                alt_idx = np.where(visibility[:, alt_t_clamped])[0]
                if len(alt_idx) > 0:
                    t = alt_t
                    t_clamped = alt_t_clamped
                    visible_idx = alt_idx
                    break

        if len(visible_idx) == 0:
            # Truly no visible points at the filter frame — skip to another entry
            return self._get_train(rng.randint(0, len(self.entries)), rng)

        # Step 5: Build frame indices (incremental, not uniform)
        hist_start = t - H + 1
        hist_frames = list(range(max(ext_start, hist_start), t + 1))
        while len(hist_frames) < H:
            hist_frames.insert(0, hist_frames[0])

        future_end = min(ext_end, t + F)
        future_frames = list(range(t + 1, future_end + 1))

        if len(visible_idx) >= P:
            chosen = rng.choice(visible_idx, P, replace=False)
        else:
            extra = rng.choice(visible_idx, P - len(visible_idx), replace=True)
            chosen = np.concatenate([visible_idx, extra])
            rng.shuffle(chosen)

        return self._build_example(entry, chosen_obj, pts_3d, visibility,
                                   chosen, hist_frames, future_frames, t, need_padding)

    # ── Eval (deterministic, all objects × clips × 3 t-values) ────────

    def _get_eval(self, item):
        """Deterministic eval: uses pre-expanded configs from _build_eval_configs."""
        P = self.num_points
        cfg = self.eval_configs[item % len(self.eval_configs)]
        entry = self.entries[cfg["entry_idx"]]
        obj_name = cfg["obj_name"]
        t = cfg["t"]
        hist_frames = cfg["hist_frames"]
        future_frames = cfg["future_frames"]
        need_padding = cfg["need_padding"]

        pts_3d, visibility = self._load_3d_and_vis(entry, obj_name)
        # Visibility filter at frame t (= the 2D-feature anchor).
        filter_idx = min(t, pts_3d.shape[1] - 1)
        visible_idx = np.where(visibility[:, filter_idx])[0]

        if len(visible_idx) == 0:
            visible_idx = np.arange(pts_3d.shape[0])

        # Deterministic uniform spacing (same as modeling/ eval_sampler)
        if len(visible_idx) >= P:
            step = len(visible_idx) / P
            chosen = visible_idx[[int(i * step) for i in range(P)]]
        else:
            reps = P // len(visible_idx) + 1
            chosen = np.tile(visible_idx, reps)[:P]

        return self._build_example(entry, obj_name, pts_3d, visibility,
                                   chosen, hist_frames, future_frames, t, need_padding)
