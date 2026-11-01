"""Room twin: an accurate, continuous 3D copy of a workstation from seated head-cam clips, plus the worker's motion.

Seated clips have almost no parallax, so geometry comes from metric monocular depth (MoGe-2) on virtual pinhole
views of the fisheye frames, scaled to the SLAM map (ViPE) and fused with a TSDF into one coloured mesh. Several
clips of the same seat are registered (gravity-aligned ICP) into the room frame of the target clip, the frame of
its world_hands.jsonl, so the hand meshes sit on the reconstructed bench.

Outputs: room_mesh.ply (full), room.glb (web), room_video.mp4 (fly-through with the hands), web/ (viewer).
"""

VERSION = "1.0"
