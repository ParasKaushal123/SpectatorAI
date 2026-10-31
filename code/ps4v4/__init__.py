"""PS4 v4: tracklet-first 3D hand tracking for egocentric fisheye factory video."""

SIDES = ("left", "right")
STATES = ("OUT_OF_VIEW", "VISIBLE", "PARTIAL", "EXITING", "REACQUIRING")

# 21-joint layout (MediaPipe / OpenPose order) used everywhere
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
                    (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
BONES = [(c[i], c[i + 1]) for c in CHAINS for i in range(4)]
