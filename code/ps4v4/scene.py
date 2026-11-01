"""Browser scene metadata in the same reference frame as kp3d_head."""
import numpy as np
from scipy.spatial.transform import Rotation


def scene_metadata(meta, camera, head, seq, camera_data):
    samples = camera_data.get("gravity_samples", [])
    down = [head["R"][min(s["frame"], meta.n_frames - 1)] @ np.array(s["down"]) for s in samples]
    gravity = np.median(down, axis=0) if down else np.array([0., 1., 0.])
    gravity /= max(np.linalg.norm(gravity), 1e-9)
    wrists = [d["J_head"][d["src"] == 1, 0] for d in (seq["left"], seq["right"])]
    wrists = np.concatenate(wrists)
    wrists = wrists[np.isfinite(wrists).all(axis=1)]
    centre = np.median(wrists, axis=0) if len(wrists) else np.array([0., .25, .45])
    return {"fps": meta.fps, "width": meta.width, "height": meta.height, "n_frames": meta.n_frames,
            "camera": camera.to_dict(), "frame": "camera at frame 0; x right, y down, z forward; metres",
            "head_quat_xyzw": Rotation.from_matrix(head["R"]).as_quat().tolist(),
            "gravity_ref": gravity.tolist(), "gravity_source": "geocalib" if down else "camera_y",
            "hand_centre_ref": centre.tolist()}
