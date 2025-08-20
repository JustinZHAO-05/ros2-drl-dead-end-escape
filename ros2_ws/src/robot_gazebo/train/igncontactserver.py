import threading,  subprocess, json, time, shutil

class IgnContactsWatcher:
    def __init__(self, topic: str):
        self.topic = topic
        self.proc = None
        self.thread = None
        self.alive = threading.Event()
        self.has_contact = False
        self.last_ts = 0.0

    def start(self):
        ign = shutil.which("ign")
        if not ign:
            raise FileNotFoundError("ign not found")
        cmd = [ign, "topic", "-e", "-t", self.topic,
               "-m", "ignition.msgs.Contacts", "--json-output"]
        # 持续订阅（不加 -n 1）
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.alive.set()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        for line in self.proc.stdout:
            if not self.alive.is_set():
                break
            line = line.strip()
            if not line or line[0] != "{":
                continue
            try:
                msg = json.loads(line)
                contacts = msg.get("contact")
                self.has_contact = isinstance(contacts, list) and len(contacts) > 0
                self.last_ts = time.time()
            except json.JSONDecodeError:
                continue

    def stop(self):
        self.alive.clear()
        try:
            if self.proc:
                self.proc.terminate()
        except Exception:
            pass

    def collided_recently(self, within_sec: float = 0.5) -> bool:
        """在最近 within_sec 秒内是否侦测到过接触。"""
        return self.has_contact and (time.time() - self.last_ts) <= within_sec
