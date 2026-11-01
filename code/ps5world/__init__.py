"""PS5 world view: the wearer's head (camera), gaze, body axis and both hands in one room-fixed 3D frame.

Built on top of PS4 v4 results (hands.jsonl, camera.json, head_rot.pkl) and a SLAM backend:
  * vipe      NVIDIA ViPE, wide-angle (unified/Mei) camera, hands and people masked, near-metric scale
  * rotation  fallback: PS4's own head rotation, head position held fixed (seated clips)
"""

VERSION = "1.3"   # 1.3: hand meshes (PS4 hands_mesh.npz) in render + world_hands_mesh.npz; 1.2: fix --max-seconds when the PS4 result is longer; 1.1: SLAM retries, mask guard, NaN checks
