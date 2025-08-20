import subprocess
import shlex
import re

def ign_set_pose(entity_name: str,
                 x: float, y: float, z: float,
                 qx: float, qy: float, qz: float, qw: float,
                 world: str = "/world/all_training",
                 timeout_ms: int = 2000) -> bool:
    """
    Calls `ign service -s {world}/set_pose` to teleport `entity_name` and returns True on success.
    Raises RuntimeError on subprocess failure, ValueError on parse failure.
    """
    # Construct the protobuf-style request payload
    req = f"""
name: "{entity_name}"
position {{
  x: {x}
  y: {y}
  z: {z}
}}
orientation {{
  x: {qx}
  y: {qy}
  z: {qz}
  w: {qw}
}}
""".strip()

    cmd = [
        "ign", "service", "-s", f"{world}/set_pose",
        "--reqtype", "ignition.msgs.Pose",
        "--reptype", "ignition.msgs.Boolean",
        "--timeout", str(timeout_ms),
        "--req", req
    ]

    # Run the command and capture its stdout/stderr
    result = subprocess.run(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"`{' '.join(shlex.quote(c) for c in cmd)}` failed:\n"
            f"{result.stderr}"
        )

    # The service prints something like:
    #   data: true
    # or
    #   data: false
    m = re.search(r"data:\s*(true|false)", result.stdout)
    if not m:
        raise ValueError(f"Could not parse ign response:\n{result.stdout!r}")
    return (m.group(1) == "true")



success = ign_set_pose(
    entity_name="jetauto",
    x=1.0, y=2.0, z=0.0,
    qx=0.0, qy=0.0, qz=0.0, qw=1.0
)
print("Teleport succeeded?" , success)
