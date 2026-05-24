"""Verify server-side RTC by inspecting a chunk dump.

If RTC is working, every chunk after the first should have its first K
positions exactly match the previous chunk's in-flight tail
(prev_chunk[offset : offset + K] where offset = current_query_step -
previous_query_step). The match is exact to float32 round-trip
precision because the server clamps those positions to the values the
client sent in `prefix_actions`.

The script auto-detects the effective anchor length K per chunk pair and
reports it -- it does NOT require you to remember which --rtc-prefix-len
you used. If K is 0 across the board, RTC is not active. If K matches
your --chunk-steps (or whatever you set --rtc-prefix-len to), RTC is
correctly wired.

Usage:
  python verify_rtc.py recordings/with_rtc.npz
  python verify_rtc.py recordings/with_rtc.npz --expected-k 5
  python verify_rtc.py recordings/no_rtc.npz   # baseline: should report K=0
"""

from __future__ import annotations

import argparse
import numpy as np

JOINT_NAMES = ["j0", "j1", "j2", "j3", "j4", "j5", "j6", "gripper"]


def detect_anchor_length(
    prev_chunk: np.ndarray, curr_chunk: np.ndarray, offset: int, tol: float
) -> int:
    """Largest K such that prev_chunk[offset:offset+K] == curr_chunk[:K]
    within `tol` per-element absolute tolerance."""
    T_prev, T_curr = prev_chunk.shape[0], curr_chunk.shape[0]
    max_K = min(T_curr, T_prev - offset)
    K = 0
    for k in range(max_K):
        if float(np.abs(prev_chunk[offset + k] - curr_chunk[k]).max()) > tol:
            break
        K = k + 1
    return K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help=".npz produced by inference_pi05.py --dump-chunks")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="per-element absolute tolerance for considering two action "
                         "values equal (default 1e-4 covers float32 round-trip plus "
                         "minor sampling-time numerical noise)")
    ap.add_argument("--expected-k", type=int, default=None,
                    help="expected anchor length K (typically the value of "
                         "--rtc-prefix-len or --chunk-steps used at rollout time). "
                         "When set, the script reports pass/fail vs this value.")
    args = ap.parse_args()

    d = np.load(args.path)
    chunks = np.asarray(d["chunks_raw"], dtype=np.float64)
    query_steps = np.asarray(d["query_steps"], dtype=np.int64)
    chunk_steps = int(d["chunk_steps"]) if "chunk_steps" in d.files else None

    N, T, D = chunks.shape
    print(f"Loaded {N} chunks of shape {T}x{D} from {args.path}")
    if chunk_steps is not None:
        print(f"  rollout used chunk_steps={chunk_steps}; tol={args.tol}")
    print()

    if N < 2:
        print("Need at least 2 chunks to verify RTC. Run a longer rollout.")
        return

    detected_K: list[int] = []
    pairs: list[tuple[int, int, int, int, int, float]] = []
    for i in range(1, N):
        q_prev = int(query_steps[i - 1])
        q_curr = int(query_steps[i])
        offset = q_curr - q_prev
        if offset <= 0 or offset >= T:
            continue
        K = detect_anchor_length(chunks[i - 1], chunks[i], offset, args.tol)
        if K < T and offset + K < T:
            boundary_diff = float(np.abs(chunks[i - 1][offset + K] - chunks[i][K]).max())
        else:
            boundary_diff = float("nan")
        detected_K.append(K)
        pairs.append((i, q_prev, q_curr, offset, K, boundary_diff))

    if not detected_K:
        print("No comparable chunk pairs (offsets all >= chunk length). "
              "Re-run with smaller --chunk-steps.")
        return

    K_arr = np.asarray(detected_K, dtype=np.int64)
    print(f"Examined {len(K_arr)} consecutive chunk pairs")
    print(f"  detected anchor length K -- mean={K_arr.mean():.2f}  "
          f"median={int(np.median(K_arr))}  min={int(K_arr.min())}  max={int(K_arr.max())}")
    # Histogram with explicit zeros so the user can see the distribution at a glance.
    hist = np.bincount(K_arr, minlength=T + 1)
    print(f"  K histogram (index = K value, value = count):")
    print(f"  {hist.tolist()}")
    print()

    if args.expected_k is not None:
        match = K_arr == args.expected_k
        n_ok = int(match.sum())
        print(f"Verifying expected K={args.expected_k}:")
        print(f"  {n_ok}/{len(K_arr)} pairs match the expected anchor length")
        if match.all():
            print(f"  PASS: every chunk pair anchors exactly K={args.expected_k} positions.")
        else:
            print(f"  FAIL: {len(K_arr) - n_ok} pairs deviate.")
            bad_idx = np.where(~match)[0]
            print(f"  first {min(5, len(bad_idx))} deviating pairs:")
            for idx in bad_idx[:5]:
                i, q_prev, q_curr, offset, K, bd = pairs[idx]
                print(f"    pair {i}: q_prev={q_prev}  q_curr={q_curr}  offset={offset}  "
                      f"detected K={K}  boundary_diff={bd:.3e}")
        print()

    # Diagnosis when no expected K was provided.
    if args.expected_k is None:
        print("DIAGNOSIS:")
        if int(K_arr.max()) == 0:
            print("  K=0 for every chunk pair -> RTC is NOT active.")
            print("  Either --rtc wasn't set, the server ignores prefix_actions,")
            print("  or the prefix never got sent (first-query-only edge case).")
        elif int(K_arr.min()) > 0 and float(K_arr.std()) < 0.5:
            print(f"  K is consistently {int(K_arr.min())} across all pairs ->")
            print(f"  RTC is correctly anchoring {int(K_arr.min())} positions per chunk.")
        else:
            print(f"  K varies between {int(K_arr.min())} and {int(K_arr.max())} ->")
            print(f"  partial / inconsistent anchoring. May indicate the server is")
            print(f"  applying RTC only some of the time, or the tolerance ({args.tol})")
            print(f"  is too tight for this server's numerical noise.")
        print()

    # Per-joint diagnostic at position 0 (should be exact passthrough under RTC).
    pos0 = np.stack([
        np.abs(chunks[i - 1][offset] - chunks[i][0])
        for (i, _, _, offset, _, _) in pairs
    ])  # (n_pairs, 8)
    print("Per-joint diff at anchor position 0 (max across all pairs):")
    for j in range(D):
        print(f"  {JOINT_NAMES[j]:<8} max={pos0[:, j].max():.3e}  mean={pos0[:, j].mean():.3e}")


if __name__ == "__main__":
    main()
