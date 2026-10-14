"""
Full-enumeration shuffled, resumable ROLLOUT eval for a single checkpoint.

Differs from auto_eval_full.py:
  - Each eval example is a `n_rollouts`-step rolling prediction with `stride`
    per rollout (model predicts F per forward; we keep the first `stride` and
    feed the last H of those kept back as the next rollout's 3D history;
    VIDEO frames also advance each rollout to match the predicted time).
  - Per clip, 3 configs with start-concentrated t-values: {s, s+3, s+6}
    (where s,e = original clip endpoints). Configs where t + n_rollouts*stride
    overruns the clip are skipped.
  - No `max_eval_per_dataset` cap.
  - Deterministic shuffle across (entry, t) configs for fast category coverage.
  - Per-example row written to predictions.jsonl; resume by skipping example_ids
    already present.

Usage:
    torchrun --nproc-per-node=1 scripts/auto_eval_full_rollout.py \\
        --checkpoint_dir .../step18000 \\
        --output_dir eval_results/foo_rollout \\
        --dataset_name trajectory_3d \\
        --dataset_suffix _droid_p8_h3_f8 \\
        --split test \\
        --num_points 8 --history 3 --future 8 \\
        --stride 6 --n_rollouts 3 --prompt_style new
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
log = logging.getLogger(__name__)

INPUT_TOKEN_BUDGET = 1500

LABEL_V1 = "object points"
LABEL_HISTORY_NEW = "3d object history"


def quantize(val):
    return int(round(val * 1000))


def format_tracks(points_delta, visibility, label):
    """Emit `<tracks coords="...">label</tracks>` for H history frames.
    points_delta: (P, H, 3)  visibility: (P, H) bool  label: str.
    """
    P_, H_, _ = points_delta.shape
    frame_strings = []
    for hi in range(H_):
        parts = [f"{float(hi):.1f}"]
        for pi in range(P_):
            if not visibility[pi, hi]:
                continue
            x = quantize(points_delta[pi, hi, 0])
            y = quantize(points_delta[pi, hi, 1])
            z = quantize(points_delta[pi, hi, 2])
            parts.append(f"{pi + 1} {x} {y} {z}")
        if len(parts) > 1:
            frame_strings.append(" ".join(parts))
    return f'<tracks coords="{";".join(frame_strings)}">{label}</tracks>'


def build_question(style, num_points, H, F, caption, hist_delta, hist_vis):
    if style == "v1":
        tracks = format_tracks(hist_delta, hist_vis, LABEL_V1)
        return (
            f"Predict the future 3D trajectories of {num_points} points over {F} timestamps, "
            f"given action: {caption}, and {H} history frames:\n{tracks}"
        )
    elif style == "new":
        tracks = format_tracks(hist_delta, hist_vis, LABEL_HISTORY_NEW)
        return (
            f"Predict the future 3D point coordinates of {num_points} points over {F} timestamps, "
            f"given action: \"{caption}\", and history 3d point coordinates: \"{tracks}\"."
        )
    raise ValueError(style)


def read_checkpoint_max_seq_len(checkpoint_path):
    cfg_path = os.path.join(checkpoint_path, "config.yaml")
    if not os.path.exists(cfg_path):
        return None
    import yaml
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("model", {}).get("llm", {}).get("max_sequence_length")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dataset_name", default="trajectory_3d")
    p.add_argument("--dataset_suffix", default="",
                   help="e.g. '_droid_p8_h3_f8'")
    p.add_argument("--split", choices=["test", "traintest"], required=True)
    p.add_argument("--num_points", type=int, default=8)
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--future", type=int, default=8)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--n_rollouts", type=int, default=3)
    p.add_argument("--prompt_style", choices=["v1", "new"], default="new")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_entries", type=int, default=None,
                   help="Debug: limit to first N entries.")
    p.add_argument("--object_name", type=str, default=None,
                   help="If set, force this object name for every entry "
                        "(skip entries that don't have it). Useful for hand "
                        "datasets where the default 'most-points' picker "
                        "always selects 'left_hand'; pass 'right_hand' to "
                        "evaluate the other side.")
    p.add_argument("--all_points", action="store_true",
                   help="Decode ALL points visible at the anchor frame, "
                        "not just `num_points` sampled. Splits candidates "
                        "into N // num_points contiguous batches; each batch "
                        "is one rollout chain. Per-(entry,t,batch) records "
                        "share the same example_id prefix and resume cleanly.")
    p.add_argument("--max_points_per_clip", type=int, default=None,
                   help="When combined with --all_points: if the visible "
                        "point pool for a clip exceeds this value, randomly "
                        "sub-sample (per-clip deterministic seed) to this "
                        "size BEFORE chunking. Caps the per-clip forward-pass "
                        "count to ceil(max_points_per_clip / num_points). "
                        "Pass --max_points_per_clip 24 with --num_points 8 "
                        "for exactly 3 forward passes per clip — the recipe "
                        "the released paper numbers use.")
    p.add_argument("--fixed_t0", type=int, default=None,
                   help="If set, ignore the (s, s+3, s+6) sampling and use "
                        "this single absolute frame index as t0 for every "
                        "entry. One config per entry. Use 2 for H=3 to make "
                        "history = [0,1,2].")
    p.add_argument("--config_jsonl_dir", type=str, default=None,
                   help="If set, build configs from all *.jsonl files in this "
                        "directory; each line must have keys "
                        "{video, obj, clip_idx, t0}. Each line → one config. "
                        "Overrides default per-entry sampling.")
    p.add_argument("--shard_idx", type=int, default=0,
                   help="Index of the shard this job processes (0-based).")
    p.add_argument("--num_shards", type=int, default=1,
                   help="Total number of shards. Configs are evenly split "
                        "deterministically (after seed-shuffle) across shards.")
    p.add_argument("--load_all_splits", action="store_true",
                   help="Also load the entries from the OTHER split "
                        "(test ↔ traintest) and merge into one entries list. "
                        "Useful for evaluating across the full dataset rather "
                        "than just one split.")
    p.add_argument("--source_dataset_filter", type=str, default=None,
                   help="If set, keep only entries whose `source_dataset` "
                        "field equals this value (e.g. 'worldtrack' to "
                        "evaluate only the WorldTrack subset of HotWorld).")
    args = p.parse_args()

    P = args.num_points
    H = args.history
    F = args.future
    stride = args.stride
    n_rollouts = args.n_rollouts
    assert stride >= H and stride <= F, f"need H<=stride<=F, got H={H} stride={stride} F={F}"
    total_future = stride * n_rollouts

    from molmo_motion.util import prepare_torchrun_environment, select_checkpoint
    prepare_torchrun_environment()

    checkpoint = select_checkpoint(args.checkpoint_dir)
    log.info(f"Checkpoint: {checkpoint}")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.max_new_tokens is None:
        max_seq = read_checkpoint_max_seq_len(checkpoint) or 4096
        args.max_new_tokens = max_seq - INPUT_TOKEN_BUDGET
        log.info(f"Auto max_new_tokens = {max_seq} - {INPUT_TOKEN_BUDGET} = {args.max_new_tokens}")

    # Build dataset — we use its entries list + helper methods but compute
    # configs ourselves (not via ds.eval_configs).
    dataset_name_for_test = f"{args.dataset_name}_{args.split}{args.dataset_suffix}"
    os.environ.pop("TRAJ3D_MAX_EVAL_PER_DATASET", None)
    from molmo_motion.data.get_dataset import get_dataset_by_name
    ds = get_dataset_by_name(dataset_name_for_test, split="test")
    entries = list(ds.entries)
    if args.load_all_splits:
        other_split = "test" if args.split == "traintest" else "traintest"
        other_dataset_name = f"{args.dataset_name}_{other_split}{args.dataset_suffix}"
        ds_other = get_dataset_by_name(other_dataset_name, split="test")
        added = list(ds_other.entries)
        entries.extend(added)
        log.info(f"--load_all_splits: appended {len(added)} entries from '{other_split}' split")
    if args.source_dataset_filter is not None:
        before = len(entries)
        entries = [e for e in entries if e.get("source_dataset") == args.source_dataset_filter]
        log.info(f"--source_dataset_filter='{args.source_dataset_filter}': "
                 f"kept {len(entries)}/{before} entries")
    if args.max_entries is not None:
        entries = entries[:args.max_entries]
    log.info(f"Final entry count: {len(entries)}")

    # ---- helpers ----
    def _best_obj_clip(entry_):
        """Return (best_obj, best_clip_idx) using the 'most-points + longest-
        ext-clip' convention. Returns (None, None) if entry has no objects."""
        obj_names = sorted(entry_["clips_by_object"].keys())
        if not obj_names:
            return None, None
        if args.object_name is not None:
            if args.object_name not in obj_names:
                return None, None
            best_obj_ = args.object_name
        else:
            try:
                pc = [(ds._get_point_count(entry_, n), n) for n in obj_names]
            except Exception:
                return None, None
            max_count = max(c for c, _ in pc)
            best_obj_ = next(n for c, n in pc if c == max_count)
        clips_ = entry_["clips_by_object"][best_obj_]
        ext_clips = [(max(0, ss - 3), min(entry_["num_frames"] - 1, ee + 2))
                     for ss, ee in clips_]
        clip_lens = [b_ - a_ + 1 for a_, b_ in ext_clips]
        best_clip_idx_ = clip_lens.index(max(clip_lens))
        return best_obj_, best_clip_idx_

    def _adaptive_n_rollouts(t, num_frames):
        """Largest r ≥ 1 such that t + r*stride ≤ num_frames-1; clamped by args.n_rollouts.
        A clip shorter than one stride still gets 1 rollout (scored on its available frames)."""
        max_r = (num_frames - 1 - t) // stride
        if max_r == 0 and num_frames - 1 - t >= 1:
            max_r = 1
        return min(args.n_rollouts, int(max_r))

    # Build a video-id → entry_idx mapping (used by --config_jsonl_dir)
    file_to_entry = {}
    for idx, entry in enumerate(entries):
        ds_name = entry["_dataset"]
        if ds_name == "droid":
            fid = f"{entry['file']}_{entry['cam']}"
        else:
            fid = entry["file"]
        file_to_entry[fid] = idx

    # ---- build configs depending on mode ----
    configs = []

    if args.config_jsonl_dir is not None:
        # Mode 1: per-line JSONL configs.
        n_jsonl_files = 0
        n_lines_total = 0
        n_lines_skipped = 0
        for jpath in sorted(Path(args.config_jsonl_dir).glob("*.jsonl")):
            n_jsonl_files += 1
            with open(jpath) as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    n_lines_total += 1
                    rec_in = json.loads(ln)
                    vid = rec_in["video"]
                    if vid not in file_to_entry:
                        n_lines_skipped += 1
                        continue
                    e_idx = file_to_entry[vid]
                    e = entries[e_idx]
                    obj_name_ = rec_in["obj"]
                    if obj_name_ not in e["clips_by_object"]:
                        n_lines_skipped += 1
                        continue
                    cidx = int(rec_in.get("clip_idx", 0))
                    if cidx >= len(e["clips_by_object"][obj_name_]):
                        n_lines_skipped += 1
                        continue
                    t_val = int(rec_in["t0"])
                    if t_val - (H - 1) < 0:
                        n_lines_skipped += 1
                        continue
                    n_r = _adaptive_n_rollouts(t_val, e["num_frames"])
                    if n_r < 1:
                        n_lines_skipped += 1
                        continue
                    configs.append({
                        "entry_idx": e_idx,
                        "obj": obj_name_,
                        "clip_idx": cidx,
                        "t": t_val,
                        "n_rollouts": int(n_r),
                        "category": rec_in.get("category", ""),
                    })
        log.info(f"Built {len(configs)} configs from {n_jsonl_files} JSONL files "
                 f"({n_lines_total} lines, {n_lines_skipped} skipped due to "
                 f"missing entries / bad indices / left-edge / no-fit)")

    elif args.fixed_t0 is not None:
        # Mode 2: one config per entry, with t = fixed_t0 (absolute).
        t_val = int(args.fixed_t0)
        for entry_idx, entry in enumerate(entries):
            num_frames = entry["num_frames"]
            best_obj, best_clip_idx = _best_obj_clip(entry)
            if best_obj is None:
                continue
            if t_val - (H - 1) < 0:
                continue
            n_r = _adaptive_n_rollouts(t_val, num_frames)
            if n_r < 1:
                continue
            configs.append({
                "entry_idx": entry_idx,
                "obj": best_obj,
                "clip_idx": best_clip_idx,
                "t": t_val,
                "n_rollouts": int(n_r),
            })
        log.info(f"Built {len(configs)} configs (one per entry, fixed t0={t_val})")

    else:
        # Mode 3 (default): existing 3 start-concentrated t-values.
        for entry_idx, entry in enumerate(entries):
            num_frames = entry["num_frames"]
            best_obj, best_clip_idx = _best_obj_clip(entry)
            if best_obj is None:
                continue
            s, e = entry["clips_by_object"][best_obj][best_clip_idx]
            for t in (s, s + 3, s + 6):
                if t - (H - 1) < 0:
                    continue
                n_r = _adaptive_n_rollouts(t, num_frames)
                if n_r < 1:
                    continue
                configs.append({
                    "entry_idx": entry_idx,
                    "obj": best_obj,
                    "clip_idx": best_clip_idx,
                    "t": int(t),
                    "n_rollouts": int(n_r),
                })
        log.info(f"Built {len(configs)} configs ({len(configs)//3 if len(configs)%3==0 else '~'} "
                 f"entries × 3 start-concentrated t)")

    # Deterministic shuffle
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(configs))

    # Apply shard split (same shuffle seed across shards → disjoint partitions).
    if args.num_shards > 1:
        original_n = len(order)
        order = order[args.shard_idx :: args.num_shards]
        log.info(f"Shard {args.shard_idx + 1}/{args.num_shards}: "
                 f"{len(order)}/{original_n} configs in this shard")

    # Load existing predictions for resume
    pred_path = os.path.join(args.output_dir, "predictions.jsonl")
    done_ids = set()
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done_ids.add(rec.get("example_id"))
                except Exception:
                    continue
        log.info(f"Resume: {len(done_ids)} already saved.")

    # Load model
    from molmo_motion.models.model_config import BaseModelConfig
    from molmo_motion.train.checkpointer import load_model_state
    from molmo_motion.torch_util import seed_all
    seed_all(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    model_cfg = BaseModelConfig.load(os.path.join(checkpoint, "config.yaml"), key="model")
    with torch.device("meta"):
        model = model_cfg.build_model()
    model.to_empty(device=device)
    load_model_state(checkpoint, model)
    model.eval()
    model.to(torch.bfloat16)
    torch.cuda.empty_cache()
    log.info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    preprocessor = model_cfg.build_preprocessor(for_inference=True, is_training=False)
    tok = preprocessor.tokenizer

    from molmo_motion.data.video_loader import VideoFrames
    from molmo_motion.eval.egodex_3d_evaluator import parse_tracks_text, tracks_to_array

    # Beaker progress
    beaker_client, beaker_workload, beaker_orig_desc = None, None, ""
    if os.environ.get("BEAKER_EXPERIMENT_ID"):
        try:
            from beaker import Beaker
            beaker_client = Beaker.from_env()
            beaker_workload = beaker_client.workload.get(os.environ["BEAKER_EXPERIMENT_ID"])
            beaker_orig_desc = beaker_workload.experiment.description or ""
        except Exception as e:
            log.warning(f"Beaker init failed: {e}")

    pred_f = open(pred_path, "a", buffering=1)
    t_start = time.time()
    n_skipped = 0
    n_processed = 0

    try:
        for idx_i, cfg_idx in enumerate(order):
            cfg = configs[int(cfg_idx)]
            entry_idx, obj, clip_idx, t0 = cfg["entry_idx"], cfg["obj"], cfg["clip_idx"], cfg["t"]
            # Per-config rollout count (short clips get fewer rollouts so they
            # aren't wholesale dropped). Falls back to args.n_rollouts for
            # backwards compat with old configs that don't carry the field.
            n_rollouts = int(cfg.get("n_rollouts", args.n_rollouts))
            total_future = stride * n_rollouts
            entry = entries[entry_idx]
            ds_name = entry["_dataset"]
            file_id = entry.get("file", "")
            if ds_name == "droid":
                file_id = f"{entry['file']}_{entry['cam']}"
            ex_id_prefix = f"rollout_{ds_name}_{file_id}_{obj}_clip{clip_idx}_t{t0}"

            # When `--all_points` is OFF this is a single-batch config and the
            # legacy example_id (no `_b{batch}` suffix) is used. When ON we'll
            # iterate `n_batches` batches and each batch gets its own example_id
            # with `_b{batch}` appended for resume.
            if not args.all_points and ex_id_prefix in done_ids:
                n_skipped += 1
                continue

            # Load 3D + video paths
            try:
                pts_3d, visibility = ds._load_3d_and_vis(entry, obj)
                video_path = ds._get_video_path(entry)
            except Exception as e:
                log.warning(f"Load failed {ex_id_prefix}: {e}")
                continue

            N_pts = pts_3d.shape[0]
            T = entry["num_frames"]

            # Candidate point set: visible at all H rollout-0 history frames.
            hist0 = [t0 - H + 1 + i for i in range(H)]  # [t0-2, t0-1, t0]
            vis_all_hist = np.ones(N_pts, dtype=bool)
            for fi in hist0:
                vis_all_hist &= visibility[:, min(fi, visibility.shape[1] - 1)]
            cand_full = np.where(vis_all_hist)[0]
            if len(cand_full) < P and not args.all_points:
                cand_full = np.where(visibility[:, min(t0, visibility.shape[1] - 1)])[0]
            if len(cand_full) == 0:
                log.warning(f"No visible points for {ex_id_prefix}; skipping")
                continue

            # Decide batches.
            if args.all_points:
                # Sort and split into N // P contiguous batches; drop remainder
                # (per user spec). If fewer than P candidates, skip.
                cand_full = np.sort(cand_full)
                # Optional cap: random sub-sample to at most
                # `max_points_per_clip` before chunking. Per-clip
                # deterministic seed → identical subset on resume / reruns.
                if (args.max_points_per_clip is not None
                        and len(cand_full) > args.max_points_per_clip):
                    clip_rng = np.random.RandomState(
                        (hash(ex_id_prefix) & 0xFFFFFFFF) ^ args.seed
                    )
                    cand_full = np.sort(clip_rng.choice(
                        cand_full, args.max_points_per_clip, replace=False
                    ))
                n_batches = len(cand_full) // P
                if n_batches == 0:
                    log.warning(f"Only {len(cand_full)} visible points (<P={P}) "
                                f"for {ex_id_prefix}; skipping in --all_points mode")
                    continue
            else:
                n_batches = 1

            caption = ds._get_caption(entry)

            for batch_idx in range(n_batches):
                if args.all_points:
                    chosen = cand_full[batch_idx * P : (batch_idx + 1) * P]
                    ex_id = f"{ex_id_prefix}_b{batch_idx}"
                else:
                    cand = cand_full
                    if len(cand) >= P:
                        step_s = len(cand) / P
                        chosen = cand[[int(i * step_s) for i in range(P)]]
                    else:
                        reps = P // len(cand) + 1
                        chosen = np.tile(cand, reps)[:P]
                    ex_id = ex_id_prefix

                if ex_id in done_ids:
                    n_skipped += 1
                    continue

                # Ground-truth future (t0+1 .. t0+total_future) for metrics later
                future_abs = np.arange(t0 + 1, t0 + 1 + total_future)
                future_abs = np.minimum(future_abs, pts_3d.shape[1] - 1)
                gt_future_raw = pts_3d[chosen][:, future_abs, :]   # (P, total_future, 3)
                gt_future_vis = visibility[chosen][:, future_abs]  # (P, total_future)
                # Frames past the clip end were clamped above; don't score them.
                n_future_valid = int(min(total_future, pts_3d.shape[1] - 1 - t0))
                gt_future_vis[:, n_future_valid:] = False

                # Initial 3D history (raw absolute) for THIS batch
                current_hist_raw = pts_3d[chosen][:, hist0, :].astype(np.float32).copy()
                current_hist_raw = np.nan_to_num(current_hist_raw, nan=0.0)

                rollouts_records = []
                combined_kept_raw = []  # list of (P, stride, 3) per rollout

                for r_idx in range(n_rollouts):
                    # Absolute video-space frames for this rollout's history
                    hist_track_abs = [t0 - H + 1 + i + r_idx * stride for i in range(H)]
                    # Clamp in case of off-by-one near end
                    hist_track_abs = [min(max(0, x), T - 1) for x in hist_track_abs]
                    video_frame_indices = ds._map_frame_to_video(entry, hist_track_abs)
                    try:
                        frames_rgb = ds._read_video_frames(video_path, video_frame_indices)
                    except Exception as e:
                        log.warning(f"Video read failed {ex_id} r{r_idx}: {e}")
                        frames_rgb = None
                    if frames_rgb is None or frames_rgb.shape[0] != H:
                        log.warning(f"Bad video frames {ex_id} r{r_idx}; skipping config")
                        break

                    # Anchor: first chosen point at last history frame (training convention)
                    anchor = current_hist_raw[0, H - 1, :].astype(np.float32).copy()
                    hist_delta = current_hist_raw - anchor[None, None, :]
                    hist_vis_for_prompt = np.ones((P, H), dtype=bool)

                    question = build_question(args.prompt_style, P, H, F, caption,
                                              hist_delta, hist_vis_for_prompt)

                    video_frames = VideoFrames(
                        frames=frames_rgb,
                        timestamps=np.arange(H, dtype=np.float64) * 1.0,
                        target_fps=1.0,
                    )
                    example = {
                        "video": video_frames,
                        "message_list": [{"style": "video_qa", "question": question, "answer": ""}],
                        "metadata": {"example_id": f"{ex_id}_r{r_idx}"},
                    }
                    processed = preprocessor(example, np.random)
                    batch = {}
                    for k, v in processed.items():
                        if isinstance(v, np.ndarray):
                            batch[k] = torch.from_numpy(v).unsqueeze(0).to(device)
                    if "input_tokens" in batch:
                        batch["input_ids"] = batch.pop("input_tokens")
                    batch.pop("target_tokens", None)
                    batch.pop("loss_masks", None)
                    if "images" in batch:
                        n_images = batch["images"].shape[1]
                        batch["image_masks"] = torch.ones(1, n_images, dtype=torch.bool, device=device)

                    try:
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            out = model.generate(batch, max_steps=args.max_new_tokens, beam_size=1)
                        tok_ids = out.token_ids[0, 0]
                        pred_text = tok.decode(tok_ids[tok_ids >= 0].tolist())
                    except Exception as e:
                        log.warning(f"Gen failed {ex_id} r{r_idx}: {e}")
                        pred_text = ""

                    parsed = parse_tracks_text(pred_text)
                    if parsed is None:
                        pred_delta = np.zeros((P, F, 3), dtype=np.float32)
                        pred_vis_full = np.zeros((P, F), dtype=bool)
                    else:
                        pred_delta, pred_vis_full = tracks_to_array(parsed, P, F,
                                                                    start_timestamp=float(H))
                    pred_raw_full = pred_delta + anchor[None, None, :]  # (P, F, 3)
                    kept = pred_raw_full[:, :stride, :]                 # (P, stride, 3)
                    combined_kept_raw.append(kept)

                    rollouts_records.append({
                        "rollout_idx": r_idx,
                        "history_track_abs": hist_track_abs,
                        "video_frame_indices": list(video_frame_indices),
                        "anchor": anchor.tolist(),
                        "pred_text": pred_text,
                        "pred_raw_full": pred_raw_full.tolist(),
                        "pred_raw_kept": kept.tolist(),
                        "parsed_ok": parsed is not None,
                    })

                    # Advance: next rollout uses last H frames of kept predictions
                    current_hist_raw = kept[:, -H:, :].astype(np.float32).copy()

                # If we broke early (video read failure), combined_kept_raw may be short
                if len(combined_kept_raw) != n_rollouts:
                    log.warning(f"Partial rollout for {ex_id}; skipping save")
                    continue

                combined_raw = np.concatenate(combined_kept_raw, axis=1)  # (P, total_future, 3)

                # Metrics: L2 on visible GT positions
                valid = gt_future_vis  # (P, total_future)
                diff = (combined_raw - gt_future_raw) ** 2
                l2_per = np.sqrt(diff.sum(axis=2))  # (P, total_future)
                n_vis = int(valid.sum())
                l2 = float(l2_per[valid].mean()) if n_vis > 0 else float("nan")
                mae_mask = np.repeat(valid[:, :, None], 3, axis=2)
                mae = float(np.abs(combined_raw - gt_future_raw)[mae_mask].mean()) \
                    if mae_mask.any() else float("nan")

                rec = {
                    "example_id": ex_id,
                    "order_i": int(idx_i),
                    "entry_idx": int(entry_idx),
                    "dataset": ds_name,
                    "video": file_id,
                    "obj": obj,
                    "clip_idx": int(clip_idx),
                    "t0": int(t0),
                    "batch_idx": int(batch_idx),
                    "n_batches": int(n_batches),
                    "caption": caption,
                    "point_indices": [int(x) for x in chosen],
                    "stride": int(stride),
                    "n_rollouts": int(n_rollouts),
                    "l2": l2,
                    "mae": mae,
                    "n_visible": n_vis,
                    "n_future_valid": n_future_valid,
                    "parsed_ok": [r["parsed_ok"] for r in rollouts_records],
                    "rollouts": rollouts_records,
                    "gt_future_raw": gt_future_raw.tolist(),
                    "gt_future_vis": gt_future_vis.tolist(),
                    "pred_raw_combined": combined_raw.tolist(),
                }
                pred_f.write(json.dumps(rec) + "\n")
                done_ids.add(ex_id)
                n_processed += 1

                if n_processed == 1 or n_processed % 5 == 0:
                    el = time.time() - t_start
                    rate = n_processed / max(el, 1e-6)
                    pct = 100 * len(done_ids) / max(len(configs) * max(1, n_batches if args.all_points else 1), 1)
                    eta_secs = (len(configs) - idx_i) / max(rate, 1e-6) * (n_batches if args.all_points else 1)
                    eta = f"{int(eta_secs // 3600)}h{int((eta_secs % 3600)//60):02d}m"
                    log.info(f"[cfg {idx_i+1}/{len(configs)} batch {batch_idx+1}/{n_batches}]  "
                             f"new={n_processed} skipped={n_skipped}  rate={rate:.2f}/s  "
                             f"eta={eta}  l2={l2:.4f}")
                    if beaker_client is not None:
                        try:
                            beaker_client.workload.update(
                                beaker_workload,
                                description=f"[cfg {idx_i+1}/{len(configs)}; eta={eta}] {beaker_orig_desc}")
                        except Exception:
                            pass
    finally:
        pred_f.close()

    # Final summary
    all_l2, all_mae = [], []
    total = 0
    with open(pred_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                total += 1
                if not np.isnan(r.get("l2", float("nan"))):
                    all_l2.append(r["l2"])
                if not np.isnan(r.get("mae", float("nan"))):
                    all_mae.append(r["mae"])
            except Exception:
                continue

    summary = {
        "checkpoint": checkpoint,
        "dataset_name": dataset_name_for_test,
        "split": args.split,
        "n_configs_total": len(configs),
        "n_done": total,
        "mean_l2": float(np.mean(all_l2)) if all_l2 else float("nan"),
        "mean_mae": float(np.mean(all_mae)) if all_mae else float("nan"),
        "elapsed_sec": time.time() - t_start,
        "stride": stride,
        "n_rollouts_max": int(args.n_rollouts),
        "n_rollouts_per_config": [int(c.get("n_rollouts", args.n_rollouts)) for c in configs],
        "P": P, "H": H, "F": F, "prompt_style": args.prompt_style,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if total >= len(configs):
        with open(os.path.join(args.output_dir, "DONE"), "w") as f:
            f.write(json.dumps(summary, indent=2))
        log.info(f"All {len(configs)} complete — DONE written.")
    else:
        log.info(f"Partial ({total}/{len(configs)}) — no DONE. Re-run to continue.")


if __name__ == "__main__":
    main()
