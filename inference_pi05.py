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

Run from a python env that has rospy + frankapy + pyrealsense2 + openpi-client.
If openpi_client is missing, install from this machine:
  pip install -e /home/hez2/code/openpi/packages/openpi-client
"""

from __future__ import annotations  # keep `list[X] | None` etc. valid on Py3.7

import argparse
import contextlib
import signal
import time

import numpy as np
import pyrealsense2 as rs
import rospy

from openpi_client import image_tools
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

    def __init__(self, fa: FrankaArm, deadband_m: float = 0.001):
        self.fa = fa
        self.deadband_m = deadband_m
        self._last_cmd: str | None = None  # "open" | "close" | None

    def reset_open(self):
        self.fa.open_gripper(block=True)
        self._last_cmd = "open"

    def step(self, target_m: float, current_m: float):
        delta = target_m - current_m
        if delta < -self.deadband_m:
            cmd = "close"
        elif delta > self.deadband_m:
            cmd = "open"
        else:
            return  # within deadband: hold last cmd
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


def _compute_gains(k_scale: float, damping_ratio: float) -> tuple[list[float], list[float]]:
    """Scale frankapy's DEFAULT_K_GAINS and recompute d = 2*sqrt(k)*damping_ratio.

    DEFAULT_K_GAINS = [600, 600, 600, 600, 250, 150, 50] is soft on J5-J7;
    k_scale=2.0 ~ snappier tracking, 3.0 ~ approaches Franka's built-in
    joint-impedance feel, >4.0 risks joint torque limits.
    """
    k = (np.array(FC.DEFAULT_K_GAINS, dtype=float) * float(k_scale))
    d = 2.0 * np.sqrt(k) * float(damping_ratio)
    return k.tolist(), d.tolist()


def build_state(fa: FrankaArm) -> np.ndarray:
    joints = np.asarray(fa.get_joints(), dtype=np.float32)  # (7,)
    width = float(fa.get_gripper_width())                   # scalar, meters
    return np.concatenate([joints, np.array([width], dtype=np.float32)])  # (8,)


# ---------------------------------------------------------------------------
# Main rollout
# ---------------------------------------------------------------------------


def rollout(args, fa, exterior_cam, wrist_cam, client):
    print(f"\n=== New rollout. Resetting arm. ===")
    fa.reset_joints()
    gripper = GripperController(fa, deadband_m=args.gripper_deadband)
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
    streamer.start()

    dt = 1.0 / args.control_hz
    rate = rospy.Rate(args.control_hz)

    pred_chunk: np.ndarray | None = None
    chunk_idx = 0
    try:
        for t in range(args.max_steps):
            t0 = time.time()

            # ---- observations --------------------------------------------------
            ext_rgb = exterior_cam.get_rgb()
            wrist_rgb = wrist_cam.get_rgb()
            ext_224 = image_tools.resize_with_pad(ext_rgb, 224, 224)
            wrist_224 = image_tools.resize_with_pad(wrist_rgb, 224, 224)
            state = build_state(fa)

            # ---- query server if chunk exhausted -------------------------------
            if pred_chunk is None or chunk_idx >= args.chunk_steps:
                request = {
                    "observation/image": ext_224.astype(np.uint8),
                    "observation/wrist_image": wrist_224.astype(np.uint8),
                    "observation/state": state.astype(np.float32),
                    "prompt": instruction,
                }
                with prevent_keyboard_interrupt():
                    resp = client.infer(request)
                pred_chunk = np.asarray(resp["actions"])  # (10, 8)
                assert pred_chunk.shape == (10, 8), f"unexpected actions shape {pred_chunk.shape}"
                chunk_idx = 0
                if args.verbose:
                    print(
                        f"[t={t}] new chunk; "
                        f"j_target[0]={pred_chunk[0,:7].round(3).tolist()} "
                        f"g_target[0]={pred_chunk[0,7]:.3f}"
                    )

            action = pred_chunk[chunk_idx]
            chunk_idx += 1
            joint_target = np.asarray(action[:7], dtype=np.float64)
            gripper_target = float(action[7])

            # ---- execute -------------------------------------------------------
            streamer.send(joint_target)
            gripper.step(target_m=gripper_target, current_m=float(state[7]))

            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
            else:
                # Stay on schedule even if a single step ran long.
                rate.last_time = rospy.Time.now()

    except KeyboardInterrupt:
        print("\n(Ctrl+C — stopping rollout)")
    finally:
        streamer.stop()


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
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--chunk-steps", type=int, default=5,
                    help="how many actions from each 10-step chunk to execute before re-querying")
    ap.add_argument("--control-hz", type=float, default=10.0)
    ap.add_argument("--gripper-deadband", type=float, default=0.001,
                    help="sign-based gripper binarization: |target - current| below this holds last cmd")
    ap.add_argument("--k-scale", type=float, default=2.0,
                    help="multiplier on frankapy DEFAULT_K_GAINS (1.0=soft default, 2.0=recommended, 3+=stiff)")
    ap.add_argument("--damping-ratio", type=float, default=1.0,
                    help="d_gains = 2*sqrt(k_gains)*ratio (1.0=critical, <1=underdamped/snappier, >1=overdamped)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print("Connecting to Franka...")
    fa = FrankaArm()
    print("Opening RealSense cameras...")
    exterior_cam = RealsenseStream(args.exterior_serial)
    wrist_cam = RealsenseStream(args.wrist_serial)
    print(f"Connecting to policy server {args.remote_host}:{args.remote_port}...")
    client = WebsocketClientPolicy(args.remote_host, args.remote_port)

    try:
        while True:
            rollout(args, fa, exterior_cam, wrist_cam, client)
            if input("Do one more rollout? (y/n) ").lower().strip() != "y":
                break
    finally:
        exterior_cam.stop()
        wrist_cam.stop()


if __name__ == "__main__":
    main()
