"""Scene frame conventions and cache/decode equivalence without neural models."""
import sys
import tempfile
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from ps4v4.video import video_meta, iter_frames, read_frame, cache_video, clear_video_cache, _frames
from ps4v4.geometry import FisheyeCamera
from ps4v4.scene import scene_metadata


def main():
    with tempfile.TemporaryDirectory(dir=CODE, prefix="scene-test-") as td:
        video = str(Path(td) / "clip.mp4")
        writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
        for i in range(10):
            writer.write(np.full((48, 64, 3), i * 20, np.uint8))
        writer.release()
        meta = video_meta(video)
        decoded = [f.copy() for _, f in iter_frames(meta)]
        assert not cache_video(meta, max_bytes=1), "Size guard must fall back to decoding"
        assert cache_video(meta)
        assert all(np.array_equal(a, b) for a, (_, b) in zip(decoded, iter_frames(meta)))
        assert np.array_equal(decoded[5], read_frame(meta, 5))
        assert not read_frame(meta, 5).flags.writeable
        R = Rotation.from_euler("z", np.linspace(0, .5, 10)[:, None]).as_matrix()
        down_ref = np.array([0., 1., 0.])
        camera_down = R[7].T @ down_ref
        seq = {s: {"src": np.ones(10, int), "J_head": np.tile([[[.1, .2, .3]]], (10, 21, 1))}
               for s in ("left", "right")}
        cam = FisheyeCamera.from_fov(64, 48, 150)
        scene = scene_metadata(meta, cam, {"R": R}, seq, {"gravity_samples": [{"frame": 7, "down": camera_down}]})
        assert np.allclose(scene["gravity_ref"], down_ref)
        assert np.allclose(Rotation.from_quat(scene["head_quat_xyzw"]).as_matrix(), R)
        assert np.allclose(scene["hand_centre_ref"], [.1, .2, .3])
        seq["left"]["src"][:] = 0
        seq["right"]["src"][:] = 0
        fallback = scene_metadata(meta, cam, {"R": R}, seq, {})
        assert fallback["gravity_source"] == "camera_y" and np.isfinite(fallback["hand_centre_ref"]).all()
        clear_video_cache(video)
        assert not _frames
        print("SCENE/CACHE CHECKS PASSED: bounded cache, exact pixels, random access, release, gravity and quaternion frames")


if __name__ == "__main__":
    main()
