"""Run from repo root: .venv/Scripts/python.exe -m dashboard.tests.test_rooms"""
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from dashboard.backend import rooms
from tools.generate_room_fixture import generate


def test_rooms():
    fixture = generate()
    app = FastAPI()
    app.include_router(rooms.router)
    with tempfile.TemporaryDirectory(dir=fixture.parent) as tmp, patch.object(rooms, "ROOT", fixture.parent), patch.object(rooms.C, "RAW", Path(tmp)):
        client = TestClient(app)
        entries = client.get('/api/rooms').json()
        assert entries == [{"id": "synthetic-room", "has_video": False, "has_render": False, "has_robot": True}]
        base = '/api/rooms/synthetic-room'
        assert client.get(base + '/web/motion.json').json()['n'] == 90
        for asset in ['room.glb', 'hands_left.bin', 'hands_right.bin', 'robot.glb', 'robot_motion.bin']:
            r = client.get(base + '/web/' + asset, headers={'Range': 'bytes=0-31'})
            assert r.status_code == 206 and len(r.content) == 32
        assert client.get(base + '/web/viewer.html').status_code == 404
        assert client.get(base + '/web/robot.json').json()['n'] == 10
        assert client.get(base + '/web/robot_traj.json').status_code == 200
        assert client.get('/api/rooms/unknown/stats').status_code == 404
        r = client.get(base + '/video')
        assert r.status_code == 404 and 'raw_clips' in r.json()['detail']
        original = Path(tmp) / 'synthetic-room.mp4'
        original.write_bytes(b'synthetic-range-data')
        assert client.get('/api/rooms').json()[0]['has_video']
        r = client.get(base + '/video', headers={'Range': 'bytes=0-8'})
        assert r.status_code == 206 and r.content == b'synthetic'
    print('PASS: synthetic room catalogue, mesh assets, byte ranges and missing files')


if __name__ == '__main__':
    test_rooms()
