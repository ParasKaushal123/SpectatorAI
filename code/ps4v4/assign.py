"""M3A: decide, per tracklet, LEFT / RIGHT / REJECT, using all of its evidence at once.

Hand score (log-odds "this is one of the wearer's hands", per frame, averaged over the tracklet):
  + detector score
  - far away compared with the wearer's typical hand depth   (another person's hand)
  - implausibly close                                         (junk filling the lens)
  - forearm running away from the camera / arm entering from the top   (another person)
  - hand-model keypoints do not fill the detector box         (junk)
  - static in the scene for the tracklet's whole life         (table, slab, chair parts)
  - very short tracklets

Side score (log-odds "right", per frame):
  forearm direction in the image (Betancourt et al. 2016), detector handedness, a small position prior
  (dropped inside crossings, where position is wrong by definition).

Then a labelling over all tracklets:
  * two tracklets that share a frame can never get the same side (one left, one right at most)
  * consecutive tracklets whose motion links up (before/after a crossing or a short loss) prefer
    the same side (Rangesh et al. 2016: resolve identity from motion once the hands separate)
  * solved greedily (strongest tracklets first) then refined by local moves with eviction.
Output: per-side tracks in the same format v3's tracker produced, plus a table of every tracklet.
"""
from __future__ import annotations

import logging

import numpy as np

from . import SIDES, STATES
from .probe import probe_features
from .tracklets import SRC_DET, SRC_FLOW, SRC_NONE, SRC_PRED, SRC_SAM, bsize, center

log = logging.getLogger(__name__)
S = {name: i for i, name in enumerate(STATES)}
LABELS = ("left", "right", "reject")


# ---- features ------------------------------------------------------------------------
def _static_spread(tr: dict) -> float:
    """How far the tracklet moves relative to the scene (head rotation removed), in box sizes."""
    path = np.cumsum(tr["resid"], 0)
    med = np.median(path, 0)
    spread = np.percentile(np.linalg.norm(path - med, axis=1), 90)
    return float(spread / np.median([bsize(b) for b in tr["boxes"]]))


def tracklet_features(tracklets: list[dict], probe: dict, W: int, H: int, fps: float,
                      away_z: float = 0.45) -> tuple[list[dict], float]:
    rows = []
    for tr in tracklets:
        det = tr["src"] == SRC_DET
        f = {"id": tr["id"], "start": int(tr["frames"][0]), "end": int(tr["frames"][-1]), "n": len(tr["frames"]),
             "n_det": int(det.sum()), "crossing": bool(tr["crossing"]),
             "score": float(np.mean(tr["score"][det])) if det.any() else 0.0,
             "p_right_det": float(np.nanmean(tr["p_right"][det])) if det.any() else 0.5,
             "spread": _static_spread(tr), "dur_s": len(tr["frames"]) / fps,
             "x_rel": float(np.mean([(center(b)[0] / W - 0.5) * 2 for b in tr["boxes"]])),
             "size_frac": float(np.median([bsize(b) for b in tr["boxes"]]) / np.hypot(W, H))}
        per = []
        p = probe.get(tr["id"])
        if p is not None:
            for k, i in enumerate(p["idx"]):
                if p["valid"][k] and np.isfinite(p["joints"][k]).all():
                    pf = probe_features(p["joints"][k], p["kp2d"][k], tr["boxes"][i], W, H)
                    pf["score"] = float(tr["score"][i])
                    pf["p_right"] = float(tr["p_right"][i])
                    pf["p_hand"] = float(p["p_hand"][k]) if "p_hand" in p else np.nan
                    pf["other"] = p["other"][k] if "other" in p else ""
                    per.append(pf)
        f["per_frame"] = per
        rows.append(f)
    _mark_on_forearm(tracklets, probe, rows, away_z)
    # the wearer's typical hand distance from long, confident tracklets (robust to a wrong focal). A low
    # percentile, not the median: another person's hands are farther and must not pull it up.
    zs = [pf["dist"] for f in rows if f["n_det"] >= 10 and f["score"] >= 0.6 for pf in f["per_frame"]]
    if len(zs) < 10:
        zs = [pf["dist"] for f in rows for pf in f["per_frame"]]
    z_ref = float(np.clip(np.percentile(zs, 35), 0.15, 1.5)) if zs else 0.45
    return rows, z_ref


def _arm_of(tr: dict, p: dict | None):
    """Per tracklet index: (wrist 2D, unit forearm direction 2D, hand size px) from the nearest probe frame,
    shifted with the box so it follows the hand between probe frames."""
    if p is None or not len(p["idx"]):
        return None
    ok = [k for k in range(len(p["idx"])) if p["valid"][k] and np.isfinite(p["kp2d"][k]).all()]
    if not ok:
        return None
    pidx = np.array([p["idx"][k] for k in ok])
    out = []
    for i in range(len(tr["frames"])):
        k = ok[int(np.argmin(np.abs(pidx - i)))]
        kp = p["kp2d"][k]
        w, kn = kp[0], kp[[5, 9, 13, 17]].mean(0)
        d = w - kn
        L = float(np.linalg.norm(kp[9] - kp[0])) + 1e-6
        if np.linalg.norm(d) < 3:
            out.append(None); continue
        shift = center(tr["boxes"][i]) - center(tr["boxes"][p["idx"][k]])
        out.append((w + shift, d / np.linalg.norm(d), L))
    return out


def _wearer_like(f: dict, away_z: float) -> bool:
    pf = f["per_frame"]
    if len(pf) < 3:
        return False
    toward = np.mean([p["toward_z"] < away_z for p in pf])
    ph = [p["p_hand"] for p in pf if np.isfinite(p.get("p_hand", np.nan))]
    return toward >= 0.7 and (not ph or np.median(ph) >= 0.5)


def _mark_on_forearm(tracklets: list[dict], probe: dict, rows: list[dict], away_z: float = 0.45) -> None:
    """Flag probe frames whose box sits on ANOTHER, stronger hand's forearm (beyond its wrist, along the
    arm). The detector often fires on a forearm or elbow; a hand never sits on its own arm."""
    arms = {tr["id"]: _arm_of(tr, probe.get(tr["id"])) for tr in tracklets}
    at = {}
    for tr, f in zip(tracklets, rows):
        if f["n_det"] < 10 or f["score"] < 0.5 or arms[tr["id"]] is None or not _wearer_like(f, away_z):
            continue
        for i, t in enumerate(tr["frames"]):
            if arms[tr["id"]][i] is not None:
                at.setdefault(int(t), []).append((tr["id"], f["score"], arms[tr["id"]][i]))
    for tr, f in zip(tracklets, rows):
        p = probe.get(tr["id"])
        if p is None:
            continue
        j = 0
        for k, i in enumerate(p["idx"]):
            if not (p["valid"][k] and np.isfinite(p["joints"][k]).all()):
                continue
            pf = f["per_frame"][j]; j += 1
            c = center(tr["boxes"][i])
            pf["on_arm"] = False
            for oid, osc, (w, u, L) in at.get(int(tr["frames"][i]), []):
                if oid == tr["id"] or osc < f["score"]:
                    continue
                v = c - w
                along = float(v @ u)
                perp = float(np.linalg.norm(v - along * u))
                if 0.8 * L < along < 4.5 * L and perp < 0.7 * L:
                    pf["on_arm"] = True
                    break


def score_tracklet(f: dict, z_ref: float, ac) -> None:
    """Fills f['hand'] (mean log-odds), f['side'] (mean log-odds right), f['reasons']."""
    reasons = []
    q, s = [], []
    for pf in f["per_frame"]:
        v = ac.w_score * (np.clip(pf["score"], 0, 1) - 0.5) * 4
        zr = pf["dist"] / z_ref
        # close work (hands near the lens, long reaches): a forearm that leaves through the BOTTOM of the image is
        # the wearer's own arm, so "far", "too close" and "arm running away" are not evidence of another person
        cw = bool(ac.get("close_work", False)) and np.isfinite(pf["exit_y"]) and \
            pf["exit_y"] >= ac.get("bottom_exit_frac", 0.85)
        far_r = ac.get("far_ratio_bottom", 3.0) if cw else ac.far_ratio
        near_r = ac.get("near_ratio_bottom", 0.12) if cw else ac.near_ratio
        if zr > far_r:
            v -= ac.w_far * min(1.0, (zr - far_r) / 0.5 + 0.5)
        if zr < near_r or pf["dist"] < ac.abs_min_m:
            v -= ac.w_near
        if not cw:
            v -= ac.w_away * np.clip((pf["toward_z"] - ac.away_z) / 0.4, 0, 1)
        if np.isfinite(pf["exit_y"]) and pf["exit_y"] < ac.top_exit_frac:
            v -= ac.w_top
        if pf["kp_iou"] < ac.kp_iou_min:
            v -= ac.w_kpbox
        if pf.get("on_arm"):
            v -= ac.w_onarm
        wearer_like = (pf["toward_z"] < ac.away_z - 0.25 and zr < 1.5 and
                       not (np.isfinite(pf["exit_y"]) and pf["exit_y"] < ac.top_exit_frac)) or (cw and zr < far_r)
        if wearer_like and np.isfinite(pf.get("p_hand", np.nan)):
            # a clear hand AND the wearer's geometry (arm runs back toward the camera, arm's-reach distance):
            # lets a partly hidden hand (only wrist and back of the hand visible) count. Never applied to a
            # hand that looks like another person's (far, arm running away, arm from the top).
            ph = np.clip(pf["p_hand"], 0.01, 0.99)
            v += min(ac.clip_boost_max, ac.w_clip_boost * max(0.0, float(np.log(ph / (1 - ph)))))
        q.append(v)
        # detector handedness counts less when the hand model could not fit the box (a hidden hand: its
        # left/right look is a guess); the forearm exit point still counts in full
        rel = 1.0 if pf["kp_iou"] >= ac.kp_iou_min else 0.3
        det = ac.w_det_side * (pf["p_right"] - 0.5) * 2 * rel if np.isfinite(pf["p_right"]) else 0.0
        s.append(ac.w_arm * pf["arm_exit"] + det)
    if q:
        hand = float(np.mean(q))
        zr_med = np.median([pf["dist"] for pf in f["per_frame"]]) / z_ref
        if zr_med > ac.far_ratio: reasons.append(f"far x{zr_med:.1f}")
        if zr_med < ac.near_ratio: reasons.append("too close")
        if np.mean([pf["toward_z"] > ac.away_z for pf in f["per_frame"]]) > 0.5: reasons.append("arm away")
        if np.mean([pf["kp_iou"] < ac.kp_iou_min for pf in f["per_frame"]]) > 0.5: reasons.append("kp/box")
        if np.mean([pf.get("on_arm", False) for pf in f["per_frame"]]) > 0.3: reasons.append("on a forearm")
        ph = [pf["p_hand"] for pf in f["per_frame"] if np.isfinite(pf["p_hand"])]
        f["p_hand"] = float(np.median(ph)) if ph else np.nan
        if ph:
            # semantic check, per tracklet (single crops are noisy: a hand gripping a plank can read as "foot"):
            # a veto only when clearly not a hand, never a boost (another person's hand IS a hand)
            lg = lambda x: float(np.log(x / (1 - x)))
            hand += ac.w_clip * min(0.0, lg(np.clip(f["p_hand"], 0.01, 0.99)) - lg(ac.clip_veto_below))
        if ph and f["p_hand"] < ac.clip_veto_below:
            names = [pf["other"] for pf in f["per_frame"] if pf["other"]]
            reasons.append("looks like " + max(set(names), key=names.count) if names else "not a hand")
    else:                                                    # nothing reconstructed: detector only, discounted
        hand = ac.w_score * (f["score"] - 0.5) * 4 - ac.no_probe_penalty
        reasons.append("no 3D")
    if f["dur_s"] >= ac.static_min_s and f["spread"] < ac.static_spread:
        hand -= ac.w_static
        reasons.append("static")
    if f["n_det"] < ac.min_det:
        hand -= 10.0
        reasons.append("short")
    side = float(np.mean(s)) if s else ac.w_det_side * (f["p_right_det"] - 0.5) * 2
    if not f["crossing"]:
        side += ac.w_pos * f["x_rel"]
    f["hand"], f["side"], f["reasons"] = hand, side, reasons


# ---- labelling ------------------------------------------------------------------------
def _links(tracklets: list[dict], fps: float, ac) -> dict:
    """(a, b) -> strength in [0, 1] when b can be the continuation of a (same hand)."""
    gap_max = int(round(ac.link_gap_s * fps))
    links = {}
    for ka, a in enumerate(tracklets):
        fa = a["frames"]
        ca = np.array([center(b) for b in a["boxes"][-5:]])
        v = (ca[-1] - ca[0]) / max(len(ca) - 1, 1)
        for kb, b in enumerate(tracklets):
            gap = b["frames"][0] - fa[-1]
            if kb == ka or gap < 1 or gap > gap_max:
                continue
            pred = ca[-1] + v * gap
            sz = bsize(a["boxes"][-1])
            err = np.linalg.norm(center(b["boxes"][0]) - pred) / sz
            ratio = bsize(b["boxes"][0]) / sz
            if 0.5 < ratio < 2.0 and err < 2.5:
                links[(ka, kb)] = float(np.exp(-0.5 * (err / 0.8) ** 2))
    return links


def _conflicts(tracklets: list[dict]) -> list[set]:
    n = len(tracklets)
    sets = [set(tr["frames"].tolist()) for tr in tracklets]
    rng = [(tr["frames"][0], tr["frames"][-1]) for tr in tracklets]
    out = [set() for _ in range(n)]
    order = sorted(range(n), key=lambda k: rng[k][0])
    for ii, a in enumerate(order):
        for b in order[ii + 1:]:
            if rng[b][0] > rng[a][1]:
                break
            if sets[a] & sets[b]:
                out[a].add(b); out[b].add(a)
    return out


def split_at_overlaps(tracklets: list[dict], feats: list[dict], ac) -> tuple[list[dict], list[dict], list[int]]:
    """Cut tracklets where they overlap another tracklet, so a short overlap (a duplicate box, two
    tracks briefly on one hand) costs only the overlapping frames, not the whole tracklet.
    Pieces keep their parent's hand/side scores; consecutive pieces of one parent are linked strongly."""
    conf = _conflicts(tracklets)
    pieces, pfeats, parent = [], [], []
    for k, (tr, f) in enumerate(zip(tracklets, feats)):
        fr = tr["frames"]
        cuts = set()
        for j in conf[k]:
            common = np.intersect1d(fr, tracklets[j]["frames"])
            for c in (common[0], common[-1] + 1):
                i = int(np.searchsorted(fr, c))
                if 0 < i < len(fr):
                    cuts.add(i)
        bounds = [0]
        for i in sorted(cuts):                           # drop cuts that leave tiny pieces
            if i - bounds[-1] >= ac.min_piece and len(fr) - i >= ac.min_piece:
                bounds.append(i)
        bounds.append(len(fr))
        for a, b in zip(bounds[:-1], bounds[1:]):
            pc = {key: (v[a:b] if isinstance(v, np.ndarray) else v) for key, v in tr.items()}
            pf = dict(f)
            pf["n_det"] = int((pc["src"] == SRC_DET).sum())
            pieces.append(pc); pfeats.append(pf); parent.append(k)
    return pieces, pfeats, parent


def solve_labels(feats: list[dict], tracklets: list[dict], fps: float, ac, parent: list[int] | None = None) -> list[str]:
    n = len(feats)
    weight = np.array([min(f["n_det"], ac.max_weight_frames) for f in feats], float)
    unary = np.zeros((n, 3))
    for k, f in enumerate(feats):
        h = (f["hand"] - ac.accept_bias) * weight[k]          # acceptance is decided by the hand score alone;
        sd = f["side"] * weight[k]                              # the side score only costs the less likely side
        unary[k, 0] = h - max(0.0, sd)                          # left
        unary[k, 1] = h - max(0.0, -sd)                         # right
        if f["hand"] < ac.hard_reject:
            unary[k, :2] = -1e9
    links = _links(tracklets, fps, ac)
    nbr = [[] for _ in range(n)]
    for (a, b), st in links.items():
        w = ac.w_link * st * min(weight[a], weight[b])
        nbr[a].append((b, w)); nbr[b].append((a, w))
    if parent is not None:                                  # pieces of one tracklet: strongly the same hand
        for k in range(n - 1):
            if parent[k] == parent[k + 1]:
                w = ac.w_parent_link * min(weight[k], weight[k + 1])
                nbr[k].append((k + 1, w)); nbr[k + 1].append((k, w))
    conf = _conflicts(tracklets)
    lab = np.full(n, 2)

    is_hand = [f["hand"] >= ac.accept_bias for f in feats]

    def local(k: int, l: int) -> float:
        u = unary[k, l]
        if l < 2 and is_hand[k]:                                # links pick a side; they never make a hand
            u += sum(w for j, w in nbr[k] if lab[j] == l)
        return u

    sib = [[] for _ in range(n)]                            # pieces of the same tracklet never take opposite sides
    if parent is not None:
        groups = {}
        for k, pa in enumerate(parent):
            groups.setdefault(pa, []).append(k)
        for g in groups.values():
            for k in g:
                sib[k] = [j for j in g if j != k]

    def allowed(k: int, l: int) -> bool:
        return l == 2 or (all(lab[j] != l for j in conf[k]) and all(lab[j] in (2, l) for j in sib[k]))

    def _options(j: int, taken: int, k: int) -> list[int]:
        return [m for m in range(3) if m != taken and (m == 2 or (all(lab[x] != m for x in conf[j] if x != k)
                                                                    and all(lab[x] in (2, m) for x in sib[j])))]

    order = np.argsort(-unary[:, :2].max(1))
    for _ in range(ac.sweeps):
        changed = False
        for k in order:
            best_l, best_u = lab[k], local(k, lab[k])
            for l in range(3):
                if allowed(k, l) and local(k, l) > best_u + 1e-9:
                    best_l, best_u = l, local(k, l)
            # eviction move: take a side from conflicting tracklets if the total gets better
            for l in range(2):
                if allowed(k, l):
                    continue
                losers = [j for j in conf[k] if lab[j] == l]
                gain = local(k, l) - local(k, lab[k])
                alt = {j: max(_options(j, l, k), key=lambda m: local(j, m)) for j in losers}
                for j in losers:
                    gain -= local(j, l) - local(j, alt[j])
                if gain > 1e-9 and local(k, l) > best_u:
                    for j in losers:
                        lab[j] = alt[j]
                    best_l, best_u = l, local(k, l)
            if best_l != lab[k]:
                lab[k] = best_l
                changed = True
        if not changed:
            break
    return [LABELS[l] for l in lab]


# ---- output ---------------------------------------------------------------------------
def build_side_tracks(meta, tracklets: list[dict], labels: list[str], exit_margin: float) -> dict:
    N = meta.n_frames
    out = {s: {"state": np.zeros(N, np.int8), "box": np.full((N, 4), np.nan), "score": np.zeros(N),
               "src": np.zeros(N, np.int8), "p_right": np.full(N, np.nan), "tid": np.full(N, -1, int)}
           for s in SIDES}
    W, H, diag = meta.width, meta.height, meta.diag
    for tr, lab in zip(tracklets, labels):
        if lab == "reject":
            continue
        o = out[lab]
        for k, t in enumerate(tr["frames"]):
            b, src = tr["boxes"][k], int(tr["src"][k])
            if src == SRC_DET:
                st = "VISIBLE"
            else:
                edge = min(b[0], b[1], W - b[2], H - b[3]) / diag
                st = "EXITING" if edge < exit_margin else "PARTIAL"
            o["state"][t] = S[st]; o["box"][t] = b; o["score"][t] = tr["score"][k]
            o["src"][t] = src if src in (SRC_DET, SRC_FLOW, SRC_SAM, SRC_PRED) else SRC_NONE
            o["p_right"][t] = tr["p_right"][k]; o["tid"][t] = tr["id"]
    return out


def run_assign(meta, tracklets: list[dict], probe: dict, cfg) -> dict:
    ac = cfg.assign
    feats, z_ref = tracklet_features(tracklets, probe, meta.width, meta.height, meta.fps, ac.away_z)
    for f in feats:
        score_tracklet(f, z_ref, ac)
    pieces, pfeats, parent = split_at_overlaps(tracklets, feats, ac) if feats else ([], [], [])
    plabels = solve_labels(pfeats, pieces, meta.fps, ac, parent) if pfeats else []
    tracks = build_side_tracks(meta, pieces, plabels, ac.exit_margin_frac)
    labels = []                                              # per original tracklet: e.g. "left" or "left+reject"
    for k in range(len(tracklets)):
        ls = [l for l, p in zip(plabels, parent) if p == k]
        labels.append("+".join(dict.fromkeys(ls)) if ls else "reject")
    table = []
    for f, lab in zip(feats, labels):
        if "reject" in lab and f["hand"] >= ac.accept_bias and "short" not in f["reasons"]:
            f["reasons"].append("side taken")
        if lab == "reject" and not f["reasons"]:
            f["reasons"].append("weak")
        table.append({k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items() if k != "per_frame"}
                     | {"label": lab, "reasons": ";".join(f["reasons"]), "n_probe": len(f["per_frame"])})
    rejected = {}
    for pc, lab, k in zip(pieces, plabels, parent):
        f = feats[k]
        if lab == "reject" and f["n_det"] >= 3:
            for i, t in enumerate(pc["frames"]):
                if pc["src"][i] == SRC_DET:
                    rejected.setdefault(int(t), []).append((pc["boxes"][i], f["reasons"][0] if f["reasons"] else ""))
    n_acc = sum(l != "reject" for l in labels)
    log.info("M3A assign: %d tracklets (%d pieces) -> %d accepted, z_ref=%.2f m", len(labels), len(pieces), n_acc,
             z_ref)
    return {"tracks": tracks, "table": table, "z_ref": z_ref, "rejected": rejected,
            "labels": dict(zip([tr["id"] for tr in tracklets], labels))}
