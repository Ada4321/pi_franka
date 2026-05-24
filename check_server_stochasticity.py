"""Probe whether the policy server uses stochastic (noise-sampled) inference.

Calls `infer()` N times against the openpi server with a single FIXED dummy
observation and reports the max abs spread of the returned action chunks
across the N samples.

If the spread on the joint columns is ~0, the server is deterministic and
diffusion / sampling ensembling will have no effect — you'd need to enable
noise sampling on the server side first. If the spread is meaningful (>~1e-3
rad), the server is stochastic and `--diffusion-samples K` in inference_pi05.py
is worth trying.

Standalone — no robot, no cameras needed. Just connects to the policy server.

Usage:
  python check_server_stochasticity.py --remote-host 0.0.0.0 --remote-port 8000
"""

from __future__ import annotations

import argparse
import numpy as np

from ws_client import WebsocketClientPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote-host", default="0.0.0.0")
    ap.add_argument("--remote-port", type=int, default=8000)
    ap.add_argument("--prompt", default="pick up the object")
    ap.add_argument("--n-samples", type=int, default=5,
                    help="number of repeated infer() calls with the IDENTICAL observation")
    args = ap.parse_args()

    # Fixed dummy observation. Values are arbitrary but reused on every call,
    # so any difference in returned actions must come from stochastic sampling
    # inside the policy (i.e. the flow-matching ODE's noise input).
    rng = np.random.default_rng(0)
    ext = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    # A plausible joint pose + gripper width (matches RESET_JOINTS).
    state = np.array(
        [0.3463, -0.0387, -0.3453, -2.3377, -0.0176, 2.3012, 0.7983, 0.02],
        dtype=np.float32,
    )
    request = {
        "observation/image": ext,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": args.prompt,
    }

    print(f"Connecting to {args.remote_host}:{args.remote_port}...")
    client = WebsocketClientPolicy(args.remote_host, args.remote_port)
    try:
        print(f"Calling infer() {args.n_samples}x with the SAME observation...")
        samples = []
        for i in range(args.n_samples):
            resp = client.infer(request)
            actions = np.asarray(resp["actions"], dtype=np.float64)
            samples.append(actions)
            print(f"  [{i}] shape={actions.shape}  "
                  f"j0_step0={actions[0, 0]:+.5f}  j6_step0={actions[0, 6]:+.5f}  "
                  f"g_step0={actions[0, 7]:.4f}")
        samples = np.stack(samples, axis=0)  # (N, 10, 8)

        spread = samples.max(axis=0) - samples.min(axis=0)  # (10, 8)
        joint_spread = spread[:, :7]
        gripper_spread = spread[:, 7]

        print()
        print(f"Max-min spread across the {args.n_samples} samples:")
        print(f"  joints (per-step max abs spread): "
              f"{joint_spread.max(axis=1).round(5).tolist()}")
        print(f"  joints  overall max:  {float(joint_spread.max()):.5f} rad")
        print(f"  gripper overall max:  {float(gripper_spread.max()):.5f} m")

        thresh = 1e-3
        max_joint = float(joint_spread.max())
        print()
        if max_joint < thresh:
            print(f"VERDICT: server appears DETERMINISTIC "
                  f"(max joint spread {max_joint:.2e} < {thresh:.0e}).")
            print("         Diffusion / sampling ensembling will NOT help client-side.")
            print("         To enable: turn on noise sampling on the server (flow-matching")
            print("         seed per call, or temperature > 0).")
        else:
            print(f"VERDICT: server appears STOCHASTIC "
                  f"(max joint spread {max_joint:.4f} rad).")
            print(f"         Diffusion ensembling is worth trying. Suggested: "
                  f"--diffusion-samples 3")
            # Variance hint: averaging K samples should reduce std by sqrt(K).
            std_single = samples.std(axis=0).mean()
            print(f"         Mean per-element std across samples: {std_single:.5f} rad")
            print(f"         Expected std after K=3 averaging:    {std_single / np.sqrt(3):.5f} rad")
            print(f"         Expected std after K=5 averaging:    {std_single / np.sqrt(5):.5f} rad")
    finally:
        client.close()


if __name__ == "__main__":
    main()
