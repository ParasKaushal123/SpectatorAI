"""Actual uploads/downloads keep a session active; status polling does not."""
import threading
import time


class TransferActivity:
    def __init__(self):
        self.lock = threading.Lock()
        self.transfers = 0
        self.last_activity = time.monotonic()

    def start(self):
        with self.lock:
            self.transfers += 1
            self.last_activity = time.monotonic()

    def finish(self):
        with self.lock:
            self.transfers -= 1
            self.last_activity = time.monotonic()

    def snapshot(self):
        with self.lock:
            return {"active_transfers": self.transfers, "last_activity": self.last_activity}


activity = TransferActivity()


class TransferMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        tracked = scope["type"] == "http" and (
            (scope.get("method") == "POST" and path == "/jobs")
            or (scope.get("method") == "GET" and path.startswith("/jobs/") and "/files/" in path)
        )
        if tracked:
            activity.start()
        try:
            # Await the complete response body, including FileResponse streaming.
            await self.app(scope, receive, send)
        finally:
            if tracked:
                activity.finish()
