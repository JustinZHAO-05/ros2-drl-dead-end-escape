import time, subprocess, re, shutil


# 预编译正则
_RE_BOOL = re.compile(r"\b(data|result)\s*:\s*(true|false)", re.IGNORECASE)

def _which_ign() -> str:
    """
    确保 'ign' 可执行存在；返回绝对路径，避免 PATH 抖动。
    """
    exe = shutil.which("ign")
    if not exe:
        raise FileNotFoundError("找不到 'ign' 可执行，请确认已安装并在 PATH 中。")
    return exe


def ign_set_pose(
    entity_name: str,
    x: float, y: float, z: float,
    qx: float, qy: float, qz: float, qw: float,
    world: str = "/world/all_training",
    timeout_ms: int = 2000,
    retries: int = 3,
    retry_delay: float = 0.1,
) -> bool:
    """
    Linux：调用 `ign service -s <world>/set_pose`
    含义：
      -s <svc>                  ：服务名
      --reqtype <proto>        ：请求类型
      --reptype <proto>        ：返回类型
      --timeout <ms>           ：毫秒超时
      --req "<text-format>"    ：protobuf 文本格式请求体
    """
    ign = _which_ign()
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
        ign, "service", "-s", f"{world}/set_pose",
        "--reqtype", "ignition.msgs.Pose",
        "--reptype", "ignition.msgs.Boolean",
        "--timeout", str(timeout_ms),
        "--req", req,
    ]

    backoff = retry_delay
    for attempt in range(1, retries + 1):
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            # print(f"[ign_set_pose] attempt {attempt} stderr:\n{res.stderr}")
            pass
        else:
            m = _RE_BOOL.search(res.stdout)
            if m and m.group(2).lower() == "true":
                return True
        if attempt < retries:
            time.sleep(backoff)
            backoff = min(backoff * 2, 1.0)  # 指数退避，最多 1s
    return False
