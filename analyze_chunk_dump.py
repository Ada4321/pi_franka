"""Diagnose whether zigzag motion comes from WITHIN chunks or BETWEEN chunks.

Reads the .npz produced by `inference_pi05.py --dump-chunks PATH` and
computes two metrics:

  1) WITHIN-CHUNK roughness
       For each chunk individually, sum the squared discrete second
       differences along the time axis, per joint. Smooth motion has small
       roughness; a sample-to-sample zigzag has large roughness.

  2) BETWEEN-CHUNK disagreement
       For each absolute timestep t covered by >=2 chunks, gather all
       predictions targeting t (chunk_q[t - q] for every chunk queried at
       step q with t - q < CHUNK_LEN) and take the max-min spread per joint.
       Low spread means consecutive chunks agree on what to do at time t;
       high spread means they don't, which is the signature of closed-loop
       covariate shift.

Each metric is split into an "early" phase (assumed in-distribution
approach motion) and a "late" phase (where the user reported swing). The
ratio late/early tells you where the zigzag concentrates.

Usage:
  python analyze_chunk_dump.py recordings/run01_chunks.npz
  python analyze_chunk_dump.py recordings/run01_chunks.npz --late-frac 0.4
"""

from __future__ import annotations

import argparse
import numpy as np

JOINT_NAMES = ["j0", "j1", "j2", "j3", "j4", "j5", "j6", "gripper"]


def within_chunk_roughness(chunks: np.ndarray) -> np.ndarray:
    """Sum of squared 2nd-differences along time, per chunk per joint.

    chunks: (N, T, 8) -> returns (N, 8) in units of (rad / step^2)^2 (or m^2
    for the gripper column).
    """
    # 2nd diff: x[k+1] - 2*x[k] + x[k-1], shape (N, T-2, 8)
    second_diff = chunks[:, 2:, :] - 2.0 * chunks[:, 1:-1, :] + chunks[:, :-2, :]
    return (second_diff ** 2).sum(axis=1)


def between_chunk_spread(
    chunks: np.ndarray, query_steps: np.ndarray, chunk_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """For each absolute timestep covered by >=2 chunks, return per-joint
    max-min spread across the overlapping predictions.

    Returns
    -------
    spreads : (M, 8)  spread per absolute step (only steps with overlap)
    steps   : (M,)    the absolute step index for each row of `spreads`
    """
    n_chunks = chunks.shape[0]
    qmin, qmax = int(query_steps.min()), int(query_steps.max())
    rows: list[np.ndarray] = []
    steps: list[int] = []
    for t in range(qmin, qmax + chunk_len):
        preds = []
        for ci in range(n_chunks):
            q = int(query_steps[ci])
            offset = t - q
            if 0 <= offset < chunk_len:
                preds.append(chunks[ci, offset, :])
        if len(preds) < 2:
            continue
        arr = np.stack(preds, axis=0)  # (Kt, 8)
        rows.append(arr.max(axis=0) - arr.min(axis=0))
        steps.append(t)
    if not rows:
        return np.zeros((0, chunks.shape[2])), np.zeros((0,), dtype=np.int64)
    return np.stack(rows, axis=0), np.asarray(steps, dtype=np.int64)


def _fmt_row(name: str, vals: list[float], widths: list[int]) -> str:
    return "  " + name.ljust(widths[0]) + "".join(
        f"{v:>{w}.6f}" if v is not None else " " * w
        for v, w in zip(vals, widths[1:])
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help=".npz produced by inference_pi05.py --dump-chunks")
    ap.add_argument("--late-frac", type=float, default=0.5,
                    help="fraction of chunks (in chunk-index order) that count as 'late' "
                         "for the early/late comparison. 0.5 = second half is late.")
    args = ap.parse_args()

    d = np.load(args.path)
    chunks = np.asarray(d["chunks_raw"], dtype=np.float64)       # (N, T, 8)
    query_steps = np.asarray(d["query_steps"], dtype=np.int64)   # (N,)
    chunk_steps = int(d.get("chunk_steps", 1)) if "chunk_steps" in d.files else None
    control_hz = float(d.get("control_hz", 0.0)) if "control_hz" in d.files else None
    diffusion_K = int(d.get("diffusion_samples", 1)) if "diffusion_samples" in d.files else None

    N, T, D = chunks.shape
    print(f"Loaded {N} chunks of shape {T}x{D} from {args.path}")
    if control_hz is not None:
        print(f"  control_hz={control_hz}  chunk_steps={chunk_steps}  diffusion_K={diffusion_K}")
    print(f"  query steps: t={int(query_steps.min())}..{int(query_steps.max())}")
    print()

    if N < 2:
        print("Need at least 2 chunks to diagnose. Run a longer rollout.")
        return

    # ---- 1) Within-chunk roughness, per chunk per joint --------------------
    within = within_chunk_roughness(chunks)  # (N, 8)

    # Early/late split by chunk index (proxy for time, since chunks are
    # appended in order).
    split = max(1, int(round(N * (1.0 - args.late_frac))))
    within_early = within[:split].mean(axis=0)
    within_late = within[split:].mean(axis=0)
    print(f"WITHIN-CHUNK ROUGHNESS  (sum of squared 2nd-difference per chunk per joint)")
    print(f"  split: first {split} chunks = early, last {N - split} = late")
    print(f"  {'joint':<8} {'overall':>12} {'early':>12} {'late':>12} {'late/early':>12}")
    for j in range(D):
        ratio = within_late[j] / max(within_early[j], 1e-12)
        print(f"  {JOINT_NAMES[j]:<8} {within.mean(axis=0)[j]:>12.6f} "
              f"{within_early[j]:>12.6f} {within_late[j]:>12.6f} {ratio:>12.2f}x")
    print()

    # ---- 2) Between-chunk spread, per absolute step ------------------------
    between, between_steps = between_chunk_spread(chunks, query_steps, chunk_len=T)
    if between.shape[0] == 0:
        print("BETWEEN-CHUNK SPREAD: no overlapping comparisons available.")
        print(f"  (looks like --chunk-steps >= chunk_len={T}; re-run with smaller chunk_steps)")
        print()
        between_early_mean = between_late_mean = np.zeros(D)
    else:
        # Split between-steps by their position relative to the early/late split
        split_step = int(query_steps[split - 1]) if split > 0 else int(query_steps[0])
        early_mask = between_steps <= split_step
        late_mask = ~early_mask
        between_early_mean = (
            between[early_mask].mean(axis=0) if early_mask.any() else np.zeros(D)
        )
        between_late_mean = (
            between[late_mask].mean(axis=0) if late_mask.any() else np.zeros(D)
        )
        print(f"BETWEEN-CHUNK SPREAD  (max-min of overlapping predictions per abs step)")
        print(f"  {between.shape[0]} comparable steps "
              f"(early: {int(early_mask.sum())}, late: {int(late_mask.sum())})")
        print(f"  {'joint':<8} {'mean':>12} {'early':>12} {'late':>12} {'late/early':>12} {'overall p95':>14} {'overall max':>14}")
        for j in range(D):
            ratio = between_late_mean[j] / max(between_early_mean[j], 1e-12)
            print(f"  {JOINT_NAMES[j]:<8} {between[:, j].mean():>12.5f} "
                  f"{between_early_mean[j]:>12.5f} {between_late_mean[j]:>12.5f} "
                  f"{ratio:>12.2f}x "
                  f"{np.percentile(between[:, j], 95):>14.5f} "
                  f"{between[:, j].max():>14.5f}")
        print()

    # ---- Verdict using the early phase as the in-distribution baseline -----
    # The user reports approach (early) is smooth, grasp (late) swings. So
    # compare late-phase metrics to the early-phase baseline; large ratios
    # tell us which axis of the problem dominates.
    JOINT_COLS = slice(0, 7)  # exclude gripper for verdict
    within_ratio = float(
        (within_late[JOINT_COLS].sum() + 1e-12)
        / (within_early[JOINT_COLS].sum() + 1e-12)
    )
    between_ratio = float(
        (between_late_mean[JOINT_COLS].sum() + 1e-12)
        / (between_early_mean[JOINT_COLS].sum() + 1e-12)
    )

    print("=" * 64)
    print("DIAGNOSIS")
    print(f"  late/early within-chunk roughness  (joints summed): {within_ratio:>7.2f}x")
    print(f"  late/early between-chunk spread    (joints summed): {between_ratio:>7.2f}x")
    print()
    HI = 3.0  # "meaningfully worse than early phase"
    w_hi = within_ratio > HI
    b_hi = between_ratio > HI
    if w_hi and not b_hi:
        print("  --> the chunks THEMSELVES get jagged in the late phase. The policy is")
        print("      outputting zigzag-shaped chunks during grasp. Most likely cause:")
        print("      observation distribution shift (wrist camera view OOD vs training).")
        print("      Client filters can only mask this; the real fix is upstream.")
    elif b_hi and not w_hi:
        print("  --> chunks stay internally smooth, but consecutive chunks DISAGREE more")
        print("      in the late phase. Classic closed-loop covariate shift. Try in")
        print("      order: (1) larger --chunk-steps during grasp (open-loop the contact),")
        print("      (2) Real-Time Chunking on the server, (3) heavier temporal ensembling.")
    elif w_hi and b_hi:
        print("  --> BOTH within-chunk roughness AND between-chunk disagreement spike")
        print("      late. Most likely a single root cause (observation OOD) feeding both")
        print("      symptoms. Address inputs first (camera viewpoint, wrist USB3,")
        print("      preprocessing match) before more client-side filtering.")
    else:
        print("  --> neither metric is meaningfully worse late. The visible swing is")
        print("      probably downstream of the chunks — impedance controller behavior,")
        print("      streaming/timing, gripper perturbation, or measurement-space slew")
        print("      lag. Check `--verbose` per-step prints for SLEW-CLIPPED / DIV-CAPPED.")


if __name__ == "__main__":
    main()
