"""Generate a tiny synthetic scan + 90-frame mesh sequence. Contains no real footage."""
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "dashboard/frontend/src/__fixtures__/room-twin/synthetic-room"


def generate():
    web = DEST / "web"
    web.mkdir(parents=True, exist_ok=True)
    positions = [-.5, -.5, -.2, .5, -.5, -.2, .5, .5, -.2, -.5, .5, -.2,
                 -.5, -.5, -.3, .5, -.5, -.3, .5, .5, -.3, -.5, .5, -.3]
    colors = [.4, .7, .5] * 8
    indices = [0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1,
               1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0]
    binary = struct.pack('<48f36H', *(positions + colors + indices))
    gltf = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}], "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "indices": 2}]}],
            "buffers": [{"byteLength": len(binary)}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 96},
                            {"buffer": 0, "byteOffset": 96, "byteLength": 96},
                            {"buffer": 0, "byteOffset": 192, "byteLength": 72}],
            "accessors": [{"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3", "min": [-.5, -.5, -.3], "max": [.5, .5, -.2]},
                          {"bufferView": 1, "componentType": 5126, "count": 8, "type": "VEC3"},
                          {"bufferView": 2, "componentType": 5123, "count": 36, "type": "SCALAR"}]}
    js = json.dumps(gltf).encode(); js += b' ' * (-len(js) % 4)
    (web / 'room.glb').write_bytes(struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(binary)) +
                                   struct.pack('<II', len(js), 0x4E4F534A) + js + struct.pack('<II', len(binary), 0x004E4942) + binary)
    n = 90
    motion = {"fps": 30, "n": n, "quant": .00005, "focus": [0, 0, 0], "eye0": [0, -.9, .6],
              "head": [[0, -.3, .3]] * n, "quat": [[0, 0, 0, 1]] * n, "gaze": [[0, 1, 0]] * n,
              "faces": [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3], "hands": {}}
    for side, x in [('left', -.15), ('right', .15)]:
        anchors = [[x + .1 * math.sin(i / 14), .1 * math.cos(i / 14), 0] for i in range(n)]
        motion['hands'][side] = {"src": [1] * n, "joints": [[v for j in range(21) for v in [a[0] + j * .002, a[1], a[2]]] for a in anchors],
                                  "mesh": True, "has_mesh": [1] * n, "anchor": anchors, "verts": 4}
        tetra = [0, 0, 0, 1000, 0, 0, 0, 1500, 0, 0, 0, 700]
        (web / f'hands_{side}.bin').write_bytes(struct.pack('<' + 'h' * (12 * n), *(tetra * n)))
    (web / 'motion.json').write_text(json.dumps(motion))
    (DEST / 'room_stats.json').write_text(json.dumps({"clips_used": ["synthetic"], "web_triangles": 12, "registration": {}}))
    # A 10-frame, two-link robot in the same Z-up frame. No real robot outputs are copied.
    robot_gltf = json.loads(json.dumps(gltf))
    robot_gltf['nodes'] = [{"name": name, "mesh": 0, "scale": [.08, .08, .08]}
                           for name in ['left__link', 'right__link']]
    robot_gltf['scenes'] = [{"nodes": [0, 1]}]
    rjs = json.dumps(robot_gltf).encode(); rjs += b' ' * (-len(rjs) % 4)
    (web / 'robot.glb').write_bytes(struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(rjs) + 8 + len(binary)) +
                                  struct.pack('<II', len(rjs), 0x4E4F534A) + rjs + struct.pack('<II', len(binary), 0x004E4942) + binary)
    summary = {"tracked_frames": 9, "copyable_share": .6, "out_of_reach_share": .2,
               "too_fast_share": .2, "median_pos_error_cm": .4}
    robot = {"fps": 30, "n": 10, "links": ['left__link', 'right__link'],
             "grade": {"left": [0, 1, 1, 1, 2, 2, 3, 3, 1, 1], "right": [1, 1, 2, 2, 1, 1, 3, 3, 1, 1]},
             "summary": {"left": summary, "right": summary, "copyable_share": .6}}
    (web / 'robot.json').write_text(json.dumps(robot))
    poses = [v for f in range(10) for x in [-.2, .2] for v in [x + f * .015, .1, .05 + f * .008, 0, 0, 0, 1]]
    (web / 'robot_motion.bin').write_bytes(struct.pack('<' + 'f' * len(poses), *poses))
    (web / 'robot_traj.json').write_text(json.dumps({"n": 10, "joints": [[0] * 7] * 10}))
    print(DEST)
    return DEST


if __name__ == '__main__':
    folder = generate()
    if '--video' in sys.argv:
        import imageio_ffmpeg
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'lavfi', '-i',
                        'testsrc2=size=320x180:rate=30:duration=3', '-an', '-c:v', 'libx264', '-pix_fmt',
                        'yuv420p', '-movflags', '+faststart', str(folder / 'original.mp4')], check=True)
