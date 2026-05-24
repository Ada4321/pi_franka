"""Real-Franka inference for openpi pi0.5 trained with
`pi05_real_world_canonical_rlds_rel` config.

Server-side contract this client mirrors (verified from
`/home/hez2/code/openpi/src/openpi/training/config.py` RLDSIsaacLabDataConfig
and `openpi.policies.droid_policy.DroidInputs`):

  request_data = {
      "observation/image":        (224, 224, 3) uint8 RGB,  # exterior view
      "observation/wrist_image":  (224, 224, 3) uint8 RGB,  # wrist view
      "observation/state":        (8,)  float32  [j0..j6, gripper_width_m],
      "prompt":                   str,
  }
  response["actions"] -> (10, 8) float32
      [:, :7]  absolute target joint angles (rad)   # AbsoluteActions already
                                                    # added current state on
                                                    # the server side.
      [:, 7]   absolute target gripper width (m)    # ~0.0 .. ~0.04 in shards;
                                                    # we sign-binarize vs the
                                                    # current gripper width
                                                    # (a la libero/main_pro.py).

Hardware:
  - 2 x Intel RealSense (exterior + wrist) selected by serial.
  - Franka via frankapy; dynamic joint streaming a la
    `frankapy/examples/run_dynamic_joints.py`.

Image preprocessing matches the sim2real intrinsics alignment:
  D435 640x480  -> center crop to 480x480  -> bilinear resize to 224x224.
This makes the per-pixel fx/fy of the policy input identical to what the sim
rendered at 256x256 (with horizontal_aperture chosen accordingly in
conf/env/sim2real.yaml).

Run from a python env that has rospy + frankapy + pyrealsense2 + msgpack/
websockets/numpy/pillow (openpi_client is only used for msgpack_numpy here;
the websocket client is a local Py3.7-compatible shim in ws_client.py).
"""

from __future__ import annotations  # keep `list[X] | None` etc. valid on Py3.7

import argparse
import contextlib
import os
import signal
import time

import numpy as np
import pyrealsense2 as rs
import rospy

from PIL import Image
# Local Py3.7-compatible shim; openpi_client.websocket_client_policy requires
# websockets>=12 which needs Python>=3.8.
from ws_client import WebsocketClientPolicy

from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage
from frankapy.proto_utils import make_sensor_group_msg, sensor_proto2ros_msg
from franka_interface_msgs.msg import SensorDataGroup


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------


class RealsenseStream:
    """Single-device RealSense color stream. Returns BGR->RGB uint8 frames."""

    def __init__(self, serial: str, width: int = 640, height: int = 480, fps: int = 30):
        self.serial = serial
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(cfg)
        # Drain a few frames so auto-exposure settles before we start.
        for _ in range(5):
            self.pipeline.wait_for_frames()

    def get_rgb(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError(f"RealSense {self.serial}: no color frame")
        bgr = np.asanyarray(color.get_data())  # (H, W, 3) uint8 BGR
        rgb = bgr[..., ::-1]  # RGB
        return np.ascontiguousarray(rgb)

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Joint streaming (matches frankapy/examples/run_dynamic_joints.py)
# ---------------------------------------------------------------------------


class JointStreamer:
    """Wraps fa.goto_joints(..., dynamic=True) + JointPositionSensorMessage pub.

    `k_gains` / `d_gains` go directly to frankapy's
    `JointImpedanceFeedbackController` (active when `dynamic=True`).
    Default frankapy gains are very soft on J5-J7; scale up for tighter
    tracking of the policy's joint setpoints.
    """

    def __init__(
        self,
        fa: FrankaArm,
        duration_budget: float = 60.0,
        k_gains: list[float] | None = None,
        d_gains: list[float] | None = None,
    ):
        self.fa = fa
        self.pub = rospy.Publisher(
            FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000
        )
        self.duration_budget = duration_budget
        self.k_gains = k_gains
        self.d_gains = d_gains
        self._init_time = None
        self._msg_id = 0
        self._active = False

    def start(self):
        """Open the dynamic joint skill at the current joint pose."""
        current = self.fa.get_joints()
        # Use a long buffer_time so we have headroom to stream; the skill
        # remains active until we send ShouldTerminate.
        self.fa.goto_joints(
            current,
            duration=self.duration_budget,
            dynamic=True,
            buffer_time=self.duration_budget + 5.0,
            k_gains=self.k_gains,
            d_gains=self.d_gains,
            ignore_virtual_walls=False,
        )
        self._init_time = rospy.Time.now().to_time()
        self._msg_id = 0
        self._active = True

    def send(self, joints: np.ndarray):
        assert self._active, "JointStreamer.start() must be called first"
        self._msg_id += 1
        msg = JointPositionSensorMessage(
            id=self._msg_id,
            timestamp=rospy.Time.now().to_time() - self._init_time,
            joints=list(map(float, joints)),
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                msg, SensorDataMessageType.JOINT_POSITION
            )
        )
        self.pub.publish(ros_msg)

    def stop(self):
        if not self._active:
            return
        term = ShouldTerminateSensorMessage(
            timestamp=rospy.Time.now().to_time() - self._init_time,
            should_terminate=True,
        )
        ros_msg = make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term, SensorDataMessageType.SHOULD_TERMINATE
            )
        )
        self.pub.publish(ros_msg)
        self._active = False


# ---------------------------------------------------------------------------
# Gripper (threshold-binarized, non-blocking)
# ---------------------------------------------------------------------------


class GripperController:
    """Sign-based binarization, mirrors examples/libero/main_pro.py
    `_gripper_target_to_cmd`:

      target < current - deadband -> close (grasp)
      target > current + deadband -> open
      within deadband              -> hold last command

    Direction (vs. current width), not absolute position, decides the cmd —
    same idea as `AbsoluteActions` on the arm side. `block=False` so we don't
    pause the joint stream; we only re-fire on transitions.
    """

    def __init__(self, fa: FrankaArm, deadband_m: float = 0.001, hold_deadband_m: float = 0.005):
        self.fa = fa
        self.deadband_m = deadband_m
        # Asymmetric hysteresis: protect an active grasp from per-frame policy
        # noise by requiring a clearly positive delta before flipping to open.
        # close->open uses `hold_deadband_m`; everything else uses `deadband_m`
        # so the open->close transition stays snappy.
        self.hold_deadband_m = hold_deadband_m
        self._last_cmd: str | None = None  # "open" | "close" | None

    def reset_open(self):
        self.fa.open_gripper(block=True)
        self._last_cmd = "open"

    def step(self, target_m: float, current_m: float):
        delta = target_m - current_m
        if self._last_cmd == "close":
            if delta > self.hold_deadband_m:
                cmd = "open"
            elif delta < -self.deadband_m:
                cmd = "close"
            else:
                return
        else:
            if delta < -self.deadband_m:
                cmd = "close"
            elif delta > self.deadband_m:
                cmd = "open"
            else:
                return
        if cmd == self._last_cmd:
            return
        if cmd == "open":
            self.fa.open_gripper(block=False)
        else:
            self.fa.close_gripper(grasp=True, block=False)
        self._last_cmd = cmd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    interrupted = False
    original = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original)
        if interrupted:
            raise KeyboardInterrupt


def _center_crop_resize(img: np.ndarray, out_size: int = 224) -> np.ndarray:
    """640x480 -> center crop to min(H, W) square -> bilinear resize to
    (out_size, out_size).

    Matches the sim2real intrinsics alignment in
    `conf/env/sim2real.yaml` (head/wrist horizontal_aperture chosen so that
    sim 256x256 fx/fy equals real D435 fx/fy after exactly this preprocessing
    on a 640x480 capture).
    """
    h, w = img.shape[:2]
    s = min(h, w)
    top = (h - s) // 2
    left = (w - s) // 2
    sq = img[top:top + s, left:left + s]
    return np.array(Image.fromarray(sq).resize((out_size, out_size), Image.BILINEAR))


def _compute_gains(k_scale: float, damping_ratio: float) -> tuple[list[float], list[float]]:
    """Scale frankapy's DEFAULT_K_GAINS and recompute d = 2*sqrt(k)*damping_ratio.

    DEFAULT_K_GAINS = [600, 600, 600, 600, 250, 150, 50] is soft on J5-J7;
    k_scale=2.0 ~ snappier tracking, 3.0 ~ approaches Franka's built-in
    joint-impedance feel, >4.0 risks joint torque limits.
    """
    k = (np.array(FC.DEFAULT_K_GAINS, dtype=float) * float(k_scale))
    d = 2.0 * np.sqrt(k) * float(damping_ratio)
    return k.tolist(), d.tolist()


def _make_video_writer(path: str, fps: float, shape: tuple):
    """Lazy-import cv2 and open an mp4 writer sized to `shape` (H, W, ...)."""
    import cv2  # lazy: only required when --record-video is set
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = shape[:2]
    writer = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(
            f"cv2.VideoWriter failed to open {path} — check ffmpeg/codec install"
        )
    return writer


def _smooth_chunk(chunk: np.ndarray, window: int) -> np.ndarray:
    """Symmetric moving-average LPF along the time axis of a (T, 8) chunk,
    applied only to the 7 joint columns. Gripper column is passed through so
    sign-binarized open/close transitions aren't delayed.

    Uses reflect padding for zero-phase response (no lag). `window` is forced
    to an odd value; window<=1 is a no-op.
    """
    if window <= 1:
        return chunk
    if window % 2 == 0:
        window += 1
    half = window // 2
    padded = np.pad(chunk[:, :7], ((half, half), (0, 0)), mode="reflect")
    kernel = np.ones(window, dtype=chunk.dtype) / window
    out = chunk.copy()
    for d in range(7):
        out[:, d] = np.convolve(padded[:, d], kernel, mode="valid")
    return out


def build_state(fa: FrankaArm) -> np.ndarray:
    joints = np.asarray(fa.get_joints(), dtype=np.float32)  # (7,)
    # frankapy `get_gripper_width()` returns TOTAL distance between fingers
    # ([0, 0.08] m). Training `state[:, 7]` was taken from IsaacLab
    # `state/gripper_pos[0]` (PER-FINGER, [0, 0.04] m), so we divide by 2 to
    # match the policy's expected units.
    width = float(fa.get_gripper_width()) / 2.0
    return np.concatenate([joints, np.array([width], dtype=np.float32)])  # (8,)


# ---------------------------------------------------------------------------
# Main rollout
# ---------------------------------------------------------------------------


RESET_JOINTS = [0.3463, -0.0387, -0.3453, -2.3377, -0.0176, 2.3012, 0.7983]


def rollout(args, fa, exterior_cam, wrist_cam, client):
    print(f"\n=== New rollout. Resetting arm. ===")
    # Two-step reset: reset_joints() goes to FC.HOME_JOINTS and uses a code
    # path that reliably preempts any leftover dynamic skill from a prior
    # rollout. goto_joints() then moves to our task-specific pose.
    fa.reset_joints()
    fa.goto_joints(RESET_JOINTS, ignore_virtual_walls=False)
    actual = fa.get_joints().tolist()
    err = max(abs(a - b) for a, b in zip(actual, RESET_JOINTS))
    print(f"[reset] target={[round(x,3) for x in RESET_JOINTS]}")
    print(f"[reset] actual={[round(x,3) for x in actual]}  max_err={err:.4f}")
    if err > 0.05:
        raise RuntimeError(f"Reset failed (max_err={err:.4f} rad). Arm not at RESET_JOINTS.")
    gripper = GripperController(
        fa,
        deadband_m=args.gripper_deadband,
        hold_deadband_m=args.gripper_hold_deadband,
    )
    gripper.reset_open()

    instruction = args.instruction or input("Enter instruction: ").strip()
    if not instruction:
        print("(empty instruction; abort)")
        return

    k_gains, d_gains = _compute_gains(args.k_scale, args.damping_ratio)
    streamer = JointStreamer(
        fa,
        duration_budget=args.max_steps / args.control_hz + 5.0,
        k_gains=k_gains,
        d_gains=d_gains,
    )
    print(f"joint impedance gains: k={[round(x,1) for x in k_gains]}  d={[round(x,1) for x in d_gains]}")
    print(f"[pre-streamer]  joints={[round(x,3) for x in fa.get_joints().tolist()]}")
    streamer.start()
    print(f"[post-streamer] joints={[round(x,3) for x in fa.get_joints().tolist()]}")
    time.sleep(0.2)
    print(f"[+200ms]        joints={[round(x,3) for x in fa.get_joints().tolist()]}")

    dt = 1.0 / args.control_hz
    # Command-rate interpolation: send `substeps` lerp'd commands per policy step.
    # 1 = original behavior (one send per policy step).
    if args.stream_hz and args.stream_hz > args.control_hz:
        substeps = max(1, int(round(args.stream_hz / args.control_hz)))
        print(f"[stream] interpolating {substeps} cmds/policy-step "
              f"(effective {substeps * args.control_hz:.0f} Hz to impedance controller)")
    else:
        substeps = 1
    # Seed prev command at current pose so the first policy step lerps from
    # the actual arm pose, not from a stale value.
    prev_cmd_target = np.asarray(fa.get_joints(), dtype=np.float64)

    # ---- video recording setup (writers opened lazily on first frame) ------
    video_paths = None
    writers: dict | None = None
    if args.record_video:
        os.makedirs(args.video_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        video_paths = {
            "ext": os.path.join(args.video_dir, f"rec_{stamp}_exterior.mp4"),
            "wrist": os.path.join(args.video_dir, f"rec_{stamp}_wrist.mp4"),
        }

    # Raw chunk dump for offline within/between zigzag diagnosis. Captures
    # what the server returned *before* any client-side smoothing/ensembling
    # so analyze_chunk_dump.py can answer "is the zigzag inside each chunk
    # or in the disagreement between consecutive chunks?"
    chunk_dump: list = []

    # Chunk history for temporal action ensembling. Each entry is
    # (query_step, chunk[10,8]). At control step t we use offset = t - query_step
    # into each chunk where 0 <= offset < CHUNK_LEN, weighted-averaging if
    # --ensemble is on, else just taking the latest chunk's prediction.
    CHUNK_LEN = 10
    chunk_history: list[tuple[int, np.ndarray]] = []
    if args.ensemble:
        print(f"[ensemble] on (decay m={args.ensemble_decay}); "
              f"chunk_steps={args.chunk_steps} -> up to {min(CHUNK_LEN, max(1, CHUNK_LEN // max(1, args.chunk_steps)))} overlapping chunks per step")
    if args.chunk_smooth > 1:
        print(f"[chunk-smooth] zero-phase moving avg, window={args.chunk_smooth} "
              f"(joint columns only; gripper passed through)")
    try:
        for t in range(args.max_steps):
            t0 = time.time()

            # ---- observations --------------------------------------------------
            ext_rgb = exterior_cam.get_rgb()
            wrist_rgb = wrist_cam.get_rgb()
            ext_224 = _center_crop_resize(ext_rgb)
            wrist_224 = _center_crop_resize(wrist_rgb)
            state = build_state(fa)

            if args.record_video:
                import cv2  # cheap after first import
                if writers is None:
                    writers = {
                        "ext": _make_video_writer(video_paths["ext"], args.control_hz, ext_224.shape),
                        "wrist": _make_video_writer(video_paths["wrist"], args.control_hz, wrist_224.shape),
                    }
                    print(f"[record] {video_paths['ext']} ({ext_224.shape[1]}x{ext_224.shape[0]}) @ {args.control_hz} fps")
                    print(f"[record] {video_paths['wrist']} ({wrist_224.shape[1]}x{wrist_224.shape[0]}) @ {args.control_hz} fps")
                writers["ext"].write(cv2.cvtColor(ext_224, cv2.COLOR_RGB2BGR))
                writers["wrist"].write(cv2.cvtColor(wrist_224, cv2.COLOR_RGB2BGR))

            if args.debug_snapshot and t == 0:
                Image.fromarray(ext_rgb).save("/Desktop/debug_ext_raw.png")
                Image.fromarray(ext_224).save("/Desktop/debug_ext_224.png")
                Image.fromarray(wrist_rgb).save("/Desktop/debug_wrist_raw.png")
                Image.fromarray(wrist_224).save("/Desktop/debug_wrist_224.png")
                print("[debug] first-frame snapshots saved to /Desktop/debug_{ext,wrist}_{raw,224}.png")

            # ---- query server if it's time for a fresh chunk -------------------
            last_q = chunk_history[-1][0] if chunk_history else None
            need_query = last_q is None or (t - last_q) >= args.chunk_steps
            if need_query:
                request = {
                    "observation/image": ext_224.astype(np.uint8),
                    "observation/wrist_image": wrist_224.astype(np.uint8),
                    "observation/state": state.astype(np.float32),
                    "prompt": instruction,
                }
                # Server-side diffusion ensembling: ask the server to draw K
                # flow-matching samples in one batched call and average. Older
                # servers will ignore this field and behave as K=1.
                if args.diffusion_samples > 1:
                    request["num_samples"] = int(args.diffusion_samples)
                # RTC: anchor the new chunk's first K positions to the
                # in-flight tail of the previous chunk (what we WOULD have
                # executed had we not re-queried). Server-side inpainting in
                # the flow-matching sampler then generates a chunk that's
                # continuous with the prior plan.
                if args.rtc and chunk_history:
                    last_q_rtc, last_chunk = chunk_history[-1]
                    offset = t - last_q_rtc  # next index we'd have consumed
                    k_rtc = (
                        args.rtc_prefix_len
                        if args.rtc_prefix_len is not None
                        else args.chunk_steps
                    )
                    end = min(offset + k_rtc, CHUNK_LEN)
                    if end > offset:
                        prefix = np.asarray(last_chunk[offset:end], dtype=np.float32)
                        request["prefix_actions"] = prefix
                        request["prefix_len"] = int(end - offset)
                with prevent_keyboard_interrupt():
                    resp = client.infer(request)
                new_chunk = np.asarray(resp["actions"], dtype=np.float64)
                assert new_chunk.shape == (CHUNK_LEN, 8), f"unexpected actions shape {new_chunk.shape}"
                if args.dump_chunks:
                    chunk_dump.append((int(t), state.copy(), new_chunk.copy()))
                new_chunk = _smooth_chunk(new_chunk, args.chunk_smooth)
                chunk_history.append((t, new_chunk))
                if args.verbose:
                    print(
                        f"[t={t}] new chunk; "
                        f"j_target[0]={new_chunk[0,:7].round(3).tolist()} "
                        f"g_target[0]={new_chunk[0,7]:.3f}"
                    )

            # Drop chunks whose 10-sample horizon has passed.
            chunk_history = [(q, ch) for (q, ch) in chunk_history if (t - q) < CHUNK_LEN]

            # ---- pick action ---------------------------------------------------
            if args.ensemble and len(chunk_history) > 1:
                # Weighted-average all predictions targeting absolute step t.
                # Weights: w_age = exp(-m * age) where age=0 is most recent chunk.
                preds = []
                ws = []
                for k, (q, ch) in enumerate(chunk_history):
                    age = len(chunk_history) - 1 - k
                    preds.append(ch[t - q])
                    ws.append(float(np.exp(-args.ensemble_decay * age)))
                preds = np.stack(preds, axis=0)              # (N, 8)
                ws = np.asarray(ws, dtype=np.float64)
                ws /= ws.sum()
                action = (preds * ws[:, None]).sum(axis=0)   # (8,)
                if args.verbose:
                    print(f"  t={t} ensemble N={len(chunk_history)} weights={ws.round(3).tolist()}")
            else:
                q, ch = chunk_history[-1]
                action = ch[t - q]

            joint_target = np.asarray(action[:7], dtype=np.float64)
            gripper_target = float(action[7])

            # ---- command-space slew + divergence safety cap -------------------
            # 1) Slew limit in COMMAND space (relative to last commanded target,
            #    not to current joints). The setpoint marches forward by at most
            #    `joint_max_delta` each step toward the policy's intent, so the
            #    command frontier advances at a fixed slew rate regardless of
            #    how fast the impedance controller is tracking — endpoint
            #    accuracy is preserved.
            # 2) Safety cap in MEASUREMENT space. If the arm physically fails
            #    to track (collision, torque limit, payload), bound how far the
            #    command can drift from current joints. Much larger than
            #    joint_max_delta so it never interferes during normal motion.
            current_joints = np.asarray(state[:7], dtype=np.float64)
            slew_delta = joint_target - prev_cmd_target
            joint_target = np.clip(
                joint_target,
                prev_cmd_target - args.joint_max_delta,
                prev_cmd_target + args.joint_max_delta,
            )
            joint_target = np.clip(
                joint_target,
                current_joints - args.joint_max_divergence,
                current_joints + args.joint_max_divergence,
            )

            if args.verbose:
                applied_delta = joint_target - prev_cmd_target
                max_raw = float(np.abs(slew_delta).max())
                max_applied = float(np.abs(applied_delta).max())
                divergence = float(np.abs(joint_target - current_joints).max())
                clip_tag = " [SLEW-CLIPPED]" if max_applied < max_raw - 1e-9 else ""
                div_tag = " [DIV-CAPPED]" if divergence >= args.joint_max_divergence - 1e-9 else ""
                print(
                    f"  t={t} slew_want={max_raw:.4f} "
                    f"slew_cmd={max_applied:.4f}{clip_tag} "
                    f"div={divergence:.4f}{div_tag} "
                    f"g_target={gripper_target:.3f} g_cur={float(state[7]):.3f}"
                )

            # ---- execute -------------------------------------------------------
            # Spread `substeps` linearly-interpolated commands across the time
            # remaining in this policy period. When substeps==1 this reduces
            # to "send joint_target immediately, sleep until next period".
            t_exec = time.time()
            exec_remaining = max(1e-6, (t0 + dt) - t_exec)
            sub_dt_eff = exec_remaining / substeps
            for i in range(substeps):
                if i == substeps - 1:
                    cmd = joint_target
                else:
                    alpha = (i + 1) / substeps
                    cmd = prev_cmd_target + alpha * (joint_target - prev_cmd_target)
                sleep_until = t_exec + i * sub_dt_eff
                now = time.time()
                if now < sleep_until:
                    time.sleep(sleep_until - now)
                streamer.send(cmd)
                if i == 0:
                    # Fire the gripper once per policy step (not per substep).
                    gripper.step(target_m=gripper_target, current_m=float(state[7]))
            prev_cmd_target = joint_target

            # Land on the policy-step boundary.
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n(Ctrl+C — stopping rollout)")
    finally:
        streamer.stop()
        if writers is not None:
            for w in writers.values():
                w.release()
            print(f"[record] wrote {video_paths['ext']} and {video_paths['wrist']}")
        if args.dump_chunks and chunk_dump:
            qs = np.array([r[0] for r in chunk_dump], dtype=np.int64)
            states = np.stack([r[1] for r in chunk_dump], axis=0).astype(np.float32)
            raw_chunks = np.stack([r[2] for r in chunk_dump], axis=0).astype(np.float32)
            out_dir = os.path.dirname(args.dump_chunks) or "."
            os.makedirs(out_dir, exist_ok=True)
            np.savez(
                args.dump_chunks,
                query_steps=qs,
                query_states=states,
                chunks_raw=raw_chunks,
                chunk_steps=int(args.chunk_steps),
                control_hz=float(args.control_hz),
                diffusion_samples=int(args.diffusion_samples),
            )
            print(f"[dump] saved {len(chunk_dump)} raw chunks -> {args.dump_chunks}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote-host", default="0.0.0.0", help="policy server host")
    ap.add_argument("--remote-port", type=int, default=8000)
    ap.add_argument(
        "--exterior-serial",
        required=True,
        help="RealSense serial for the exterior view (must roughly match one of the 4 sim training viewpoints)",
    )
    ap.add_argument(
        "--wrist-serial",
        required=True,
        help="RealSense serial for the wrist-mounted camera",
    )
    ap.add_argument("--instruction", default=None,
                    help="task prompt; if omitted, ask interactively per rollout")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--chunk-steps", type=int, default=10,
                    help="how many actions from each 10-step chunk to execute before re-querying. "
                         "Smaller = more overlap between consecutive chunks (and more inference load). "
                         "With --ensemble, smaller values give stronger smoothing (try 1-3).")
    ap.add_argument("--ensemble", action="store_true",
                    help="temporal action ensembling: keep recent chunks and weighted-average all "
                         "predictions targeting each absolute timestep. Smooths chunk-boundary jumps "
                         "and intra-chunk noise without adding lag. Pair with smaller --chunk-steps.")
    ap.add_argument("--ensemble-decay", type=float, default=0.5,
                    help="exponential decay m for ensemble weights: w_age = exp(-m * age). "
                         "0 = uniform average (heaviest smoothing); larger = trust latest chunk more. "
                         "Typical: 0.05-0.5.")
    ap.add_argument("--chunk-smooth", type=int, default=1,
                    help="zero-phase moving-average window applied to each chunk's joint "
                         "columns along the time axis. 1=off; try 3 (mild) or 5 (strong). "
                         "Targets intra-chunk high-freq jitter that ensembling can't fix.")
    ap.add_argument("--control-hz", type=float, default=10.0)
    ap.add_argument("--stream-hz", type=float, default=None,
                    help="if set, linearly interpolate between consecutive policy joint targets "
                         "and stream commands to the impedance controller at this rate (e.g. 50-100). "
                         "Smooths the 10 Hz step inputs without adding lag. Unset = no interpolation.")
    ap.add_argument("--gripper-deadband", type=float, default=0.001,
                    help="sign-based gripper binarization: |target - current| below this holds last cmd")
    ap.add_argument("--gripper-hold-deadband", type=float, default=0.005,
                    help="while currently grasping, require target - current > this (m) to release. "
                         "Asymmetric: protects the grasp from frame-to-frame policy noise without "
                         "slowing open->close (which still uses --gripper-deadband)")
    ap.add_argument("--joint-max-delta", type=float, default=0.01,
                    help="COMMAND-SPACE slew: per-step cap on |commanded - PREVIOUS COMMAND| "
                         "joint angle (rad). Setpoint marches forward by at most this much each "
                         "step regardless of impedance tracking speed. At 10 Hz: 0.05 -> "
                         "~0.5 rad/s ceiling. Preserves both smoothness AND endpoint accuracy.")
    ap.add_argument("--joint-max-divergence", type=float, default=0.3,
                    help="safety bound on |commanded - CURRENT JOINTS| (rad). Caps how far the "
                         "command can drift ahead if the arm physically fails to track. Should "
                         "be much larger than --joint-max-delta so it doesn't interfere during "
                         "normal slew operation; only kicks in when something goes wrong.")
    ap.add_argument("--dump-chunks", default=None, metavar="PATH",
                    help="if set, save every RAW server chunk (before _smooth_chunk and before "
                         "ensembling) along with query step + state into PATH (.npz). Feed to "
                         "analyze_chunk_dump.py to diagnose whether jitter is within-chunk or "
                         "between-chunk.")
    ap.add_argument("--diffusion-samples", type=int, default=1,
                    help="diffusion sampling ensemble. Sent as `num_samples` in the request "
                         "payload — the SERVER draws K flow-matching samples in a single batched "
                         "call and returns the averaged chunk. 1=off; try 3-10. Requires server "
                         "support for the `num_samples` field; older servers will ignore it and "
                         "behave as K=1. Run check_server_stochasticity.py first.")
    ap.add_argument("--rtc", action="store_true",
                    help="Real-Time Chunking: send the next K in-flight actions as "
                         "`prefix_actions` in each request. The server constrains the new chunk's "
                         "first K positions to match (inpainting-style flow-matching), "
                         "eliminating the between-chunk seam. Requires server-side RTC support; "
                         "no-op against servers that don't read the field.")
    ap.add_argument("--rtc-prefix-len", type=int, default=None,
                    help="RTC anchor length K. Default = --chunk-steps (anchor exactly the "
                         "samples we'd have executed had we not re-queried). Override only if "
                         "you want to experiment with a different in-flight horizon.")
    ap.add_argument("--k-scale", type=float, default=1.0,
                    help="multiplier on frankapy DEFAULT_K_GAINS (1.0=soft default, 2.0=recommended, 3+=stiff)")
    ap.add_argument("--damping-ratio", type=float, default=1.0,
                    help="d_gains = 2*sqrt(k_gains)*ratio (1.0=critical, <1=underdamped/snappier, >1=overdamped)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--debug-snapshot", action="store_true",
                    help="dump first-frame raw + 22 d4x224 images of both cameras to /tmp")
    ap.add_argument("--record-video", action="store_true",
                    help="record both raw camera streams to mp4 per rollout")
    ap.add_argument("--video-dir", default="./recordings",
                    help="output directory for recorded videos (created if missing)")
    args = ap.parse_args()

    print("Connecting to Franka...")
    fa = FrankaArm()
    print("Opening RealSense cameras...")
    exterior_cam = RealsenseStream(args.exterior_serial)
    wrist_cam = RealsenseStream(args.wrist_serial)
    print(f"Connecting to policy server {args.remote_host}:{args.remote_port}...")
    client = WebsocketClientPolicy(args.remote_host, args.remote_port)
    if args.diffusion_samples > 1:
        print(f"[diffusion-ensemble] server-side, num_samples={args.diffusion_samples}")
    if args.rtc:
        k_rtc = args.rtc_prefix_len if args.rtc_prefix_len is not None else args.chunk_steps
        print(f"[rtc] on, anchoring first {k_rtc} positions of each new chunk")

    try:
        while True:
            rollout(args, fa, exterior_cam, wrist_cam, client)
            if input("Do one more rollout? (y/n) ").lower().strip() != "y":
                break
    finally:
        client.close()
        exterior_cam.stop()
        wrist_cam.stop()


if __name__ == "__main__":
    main()
