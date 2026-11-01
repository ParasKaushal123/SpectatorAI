"""A busy GPU, queued render or active transfer must never trip idle shutdown."""
import time


class IdleGuard:
    def __init__(self, seconds=300, now=None):
        self.seconds = max(1, float(seconds))
        self.last_busy = time.monotonic() if now is None else now
        self.activity_token = None

    def expired(self, health, jobs, now=None):
        now = time.monotonic() if now is None else now
        busy = health.get("active_transfers", 0) > 0 or any(
            j.get("status") in ("queued", "running")
            or j.get("render", {}).get("status") in ("queued", "running") for j in jobs
        )
        token = health.get("last_activity")
        if busy or (token is not None and token != self.activity_token):
            self.last_busy = now
        self.activity_token = token
        return now - self.last_busy >= self.seconds
