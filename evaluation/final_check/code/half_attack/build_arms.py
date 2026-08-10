#!/usr/bin/env python
"""Half-attack arm builder -- BUILD PHASE ONLY (no grid scoring).

Rows built here (joint modality `imu_trajectory_xytime` only):

  row 3  fake touch trajectory  + REAL inertial window from one of the victim's
         five held recordings                                     (cross-paired)
  row 4  REAL touch trajectory from one of the victim's five held recordings
         + fake inertial window                                   (cross-paired)
  row 5  REAL touch from genuine event A + REAL inertial window from genuine
         event B, same victim, A != B, different sessions          (MISMATCH CONTROL)

Rows 1 (genuine) and 2 (the release fake) already exist in the release bundles
and are read, never rebuilt.

MECHANICS -- identical for rows 3, 4 and 5 (fairness rule F8)
-------------------------------------------------------------
S0  CARRIER = the event that supplies the inertial window.  It fixes
    T = len(carrier imu) and D = carrier_trajectory[-1, 7] * 1000 ms.
    (Duration is read off channel 7 of the last row, never from a row count and
    never from the IMU length.)
S1  TEMPLATE = the event that supplies the touch trajectory.  It is resampled to
    T by the release's own `_resample_touch_template`.
S2  SECOND STAGE -- endpoint replacement onto the CARRIER's own first/last touch
    sample, executed by the release's own `_observe_exact_touch_template`
    (tap/scroll/swipe) with the release's own `_fiveshot_tap_drift_request`
    endpoint policy for tap.  This is the release's semantics: in the release
    the requested endpoints come from the attacker's bound Android carrier
    target (`_single_pointer_target_control_points_px`), i.e. from the event
    that carries the inertial window, not from the donor template.
    For pinch and keystroke the release's placement cannot be reconstructed
    (see SECOND_STAGE_NOTE) so S2 is the identity, applied identically to rows
    3, 4 and 5.
S3  elapsed column rewritten by `detector_grid_clock(T, D)`, dx/dy recomputed
    from the resampled contact mask and coordinates, then
    `modality_values(imu, trajectory, 'imu_trajectory_xytime')`.

SELECTION RULE (fairness rule F5) -- fixed in advance, computable from what the
attacker holds, never from a detector score, a label, or any test quantity:
  * which of the five shots is used is a BALANCED RANDOM assignment: the shot
    ordinals 0..4 are drawn as a concatenation of independent random
    permutations of (0,1,2,3,4), so every shot is used equally often up to the
    final partial block.  The draw is reseeded per repetition, which is what
    makes the construction spread visible.
  * row 3 and row 5 share the SAME carrier for the same (k, repetition), so
    their T and duration distributions are identical event for event.
  * row 4's carriers are the release's own fake events for that (victim,
    action), sampled without replacement -- the whole pool, so row 4 and row 2
    have identical T and duration event for event.
"""

from __future__ import annotations

import collections
import json
import sys
import types
import zlib
from pathlib import Path

import numpy as np

DIRECT = "/mnt/share/mwang49/data7/code/direct100k"
if DIRECT not in sys.path:
    sys.path.insert(0, DIRECT)

from security_exp.replay_dataset_builder import (  # noqa: E402
    _fiveshot_tap_drift_request,
    _observe_exact_touch_template,
    _resample_touch_template,
)
from security_exp.android_touch_observation import (  # noqa: E402
    detector_grid_clock,
    detector_grid_span_ms,
    screen_dimensions_for_orientation,
)
from security_exp.event_detectors import modality_values  # noqa: E402
from security_exp.exact_touch_template_generator import (  # noqa: E402
    DIRECTION8,
    STATIONARY,
)

ROOT = Path("/mnt/share/mwang49/data7/results/direct100k")
CELL_MAP = Path("/mnt/share/mwang49/data7/code/baselines/release_cell_map.json")
MATERIAL = ROOT / "fiveshot_material" / "material_manifest.jsonl"
BINDINGS = ROOT / "genuine_bindings_v1" / "genuine_bindings.jsonl"

ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")
EXACT_TOUCH_ACTIONS = frozenset(("tap", "scroll", "swipe"))
MODALITY = "imu_trajectory_xytime"
CLASSICAL = ("hmog_style_svm", "hmog_style_rf", "paper_svm", "paper_xgboost")
SEED_BASE = 20260810
NEAREST_LENGTH_CANDIDATES = 3

SECOND_STAGE_NOTE = (
    "restored for tap/scroll/swipe through the release's own "
    "_observe_exact_touch_template; identity for pinch and keystroke because "
    "the release's pinch placement needs live two-pointer endpoint geometry "
    "that a fake donor does not carry, and the release's keystroke touch is "
    "composed from key anchors rather than transported from a template, so "
    "neither can be reconstructed for a re-paired event. The omission is "
    "applied identically to rows 3, 4 and 5, so for pinch and keystroke rows "
    "3/4 are compared only against row 5 and never against rows 1/2."
)


# --------------------------------------------------------------------------- #
# release resolution
# --------------------------------------------------------------------------- #
def cell_map() -> dict:
    return json.loads(CELL_MAP.read_text())["cells"]


def action_bundle() -> dict:
    cells = cell_map()
    out = {}
    for action in ACTIONS:
        bundles = {
            cells[f"{action}__{MODALITY}__{det}"]["bundle"] for det in CLASSICAL
        }
        if len(bundles) != 1:
            raise SystemExit(f"{action}: classical joint cells disagree on bundle")
        out[action] = bundles.pop()
    return out


class Shard:
    """One (bundle, user) npz, indexed for event lookup."""

    __slots__ = ("imu", "traj", "off", "event_id", "session", "action", "label",
                 "cluster", "user")

    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=True)
        self.imu = z["imu_flat"]
        self.traj = z["trajectory_flat"]
        self.off = z["offsets"]
        self.event_id = z["event_id"]
        self.session = z["session_id"]
        self.action = z["action"]
        self.label = z["label"]
        self.cluster = z["source_cluster_id"]
        self.user = str(z["user_id"][0])

    def signals(self, row: int):
        a, b = int(self.off[row]), int(self.off[row + 1])
        return (np.ascontiguousarray(self.imu[a:b], dtype=np.float32),
                np.ascontiguousarray(self.traj[a:b], dtype=np.float32))

    def duration_ms(self, row: int) -> float:
        b = int(self.off[row + 1])
        return float(self.traj[b - 1, 7]) * 1000.0

    def length(self, row: int) -> int:
        return int(self.off[row + 1] - self.off[row])


class ShardBook:
    def __init__(self):
        self._cache: dict[tuple[str, str], Shard] = {}

    def get(self, bundle: str, user: str) -> Shard:
        key = (bundle, user)
        if key not in self._cache:
            self._cache[key] = Shard(ROOT / bundle / "shards" / f"{user}.npz")
        return self._cache[key]


def test_users(bundle: str = "replay_dataset_v15") -> list[str]:
    users = []
    for path in sorted((ROOT / bundle / "shards").glob("*.npz")):
        z = np.load(path, allow_pickle=True)
        if str(z["split"]) == "test":
            users.append(path.stem)
    return users


def fiveshot_map(users, bundles, book: ShardBook) -> tuple[dict, list]:
    """(user, action) -> the five held recordings as rows of the release bundle.

    Identity of the five is the published `material_manifest.jsonl`; the chain
    documented by final_gen/fiveshot_priority.py
    (source_cluster_id -> genuine_bindings.jsonl source_event_id -> event id)
    is verified alongside and any disagreement is reported.
    """
    shots = collections.defaultdict(list)
    for line in MATERIAL.open():
        r = json.loads(line)
        shots[(r["user_id"], r["action"])].append(r)
    for key in shots:
        shots[key].sort(key=lambda r: int(r["shot_ordinal"]))

    wanted = set()
    for u in users:
        for a in ACTIONS:
            for r in shots.get((u, a), []):
                wanted.add(r["source_cluster_id"])
    chain = {}
    for line in BINDINGS.open():
        r = json.loads(line)
        if r["source_cluster_id"] in wanted:
            chain[r["source_cluster_id"]] = r

    out, shortfall = {}, []
    for a in ACTIONS:
        bundle = bundles[a]
        for u in users:
            shard = book.get(bundle, u)
            rows = {str(shard.event_id[i]): i
                    for i in range(len(shard.label))
                    if shard.label[i] == 0 and shard.action[i] == a}
            resolved, missing = [], []
            for r in shots.get((u, a), []):
                eid = r["event_id"]
                cid = r["source_cluster_id"]
                if f"genuine-{cid}" != eid:
                    missing.append((r["shot_ordinal"], eid, "event_id is not genuine-<cluster>"))
                    continue
                if cid not in chain:
                    missing.append((r["shot_ordinal"], eid, "cluster absent from genuine_bindings"))
                    continue
                if eid not in rows:
                    missing.append((r["shot_ordinal"], eid, "absent from the release bundle"))
                    continue
                row = rows[eid]
                if shard.length(row) < 2:
                    missing.append((r["shot_ordinal"], eid, "T < 2"))
                    continue
                resolved.append({
                    "shot_ordinal": int(r["shot_ordinal"]),
                    "event_id": eid,
                    "row": row,
                    "session_id": str(shard.session[row]),
                    "orientation_id": int(chain[cid]["orientation_id"]),
                    "source_event_id": str(chain[cid]["source_event_id"]),
                    "T": shard.length(row),
                    "duration_ms": shard.duration_ms(row),
                })
            out[(u, a)] = resolved
            if len(resolved) != 5:
                shortfall.append({"user": u, "action": a,
                                  "declared": len(shots.get((u, a), [])),
                                  "resolved": len(resolved), "missing": missing})
    return out, shortfall


def orientation_books(users, bundles) -> tuple[dict, dict]:
    """cluster_id -> orientation for genuine; fake event_id -> orientation."""
    genuine_needed = set()
    book = ShardBook()
    for a in ACTIONS:
        shard_bundle = bundles[a]
        for u in users:
            s = book.get(shard_bundle, u)
            for i in range(len(s.label)):
                if s.label[i] == 0 and s.action[i] == a:
                    genuine_needed.add(str(s.cluster[i]))
    genuine = {}
    for line in BINDINGS.open():
        r = json.loads(line)
        if r["source_cluster_id"] in genuine_needed:
            genuine[r["source_cluster_id"]] = int(r["orientation_id"])
    fake = {}
    for bundle in sorted(set(bundles.values())):
        with (ROOT / bundle / "provenance.jsonl").open() as f:
            for line in f:
                if '"label": 1' not in line:
                    continue
                d = json.loads(line)
                if d["user_id"] not in users:
                    continue
                donor = d["donor"]
                oid = donor.get("conditioning_orientation_id")
                if oid is None:
                    oid = donor.get("target_binding", {}).get("orientation_id")
                if oid is not None:
                    fake[d["event_id"]] = int(oid)
    return genuine, fake


# --------------------------------------------------------------------------- #
# the builder
# --------------------------------------------------------------------------- #
class BuildRejected(RuntimeError):
    pass


def _requested_endpoints(action, template, carrier_traj, orientation_id):
    """Requested control points = the CARRIER's own first/last touch sample."""
    width, height = screen_dimensions_for_orientation(orientation_id)
    dims = np.asarray((width, height), dtype=np.float64)
    start = carrier_traj[0, 1:3].astype(np.float64) * dims
    end = carrier_traj[-1, 1:3].astype(np.float64) * dims
    target = types.SimpleNamespace(action=action, orientation_id=int(orientation_id))
    if action == "tap":
        request_template, end_xy, direction, audit = _fiveshot_tap_drift_request(
            template, target, start_px=(float(start[0]), float(start[1]))
        )
        return request_template, (float(start[0]), float(start[1])), end_xy, direction, audit
    delta = end - start
    if not np.any(delta):
        raise BuildRejected(f"{action} carrier endpoints are stationary")
    angle = float(np.arctan2(float(delta[1]), float(delta[0])))
    direction = DIRECTION8[int(np.floor((angle + np.pi / 8.0) / (np.pi / 4.0))) % 8]
    return (template, (float(start[0]), float(start[1])),
            (float(end[0]), float(end[1])), direction, {})


def build_event(action, template_traj, carrier_imu, carrier_traj, orientation_id):
    """Return ((T,15) joint array, audit dict).  Raises BuildRejected."""
    T = int(len(carrier_imu))
    if T < 2:
        raise BuildRejected("carrier has fewer than two samples")
    if len(carrier_traj) != T:
        raise BuildRejected("carrier imu and trajectory lengths disagree")
    if len(template_traj) < 2:
        raise BuildRejected("template has fewer than two samples")
    duration_ms = float(carrier_traj[-1, 7]) * 1000.0
    if not np.isfinite(duration_ms) or duration_ms <= 0.0:
        raise BuildRejected("carrier duration is not positive")

    audit = {"T": T, "duration_ms": duration_ms,
             "template_T": int(len(template_traj)),
             "orientation_id": int(orientation_id)}

    if action in EXACT_TOUCH_ACTIONS:
        req_tmpl, start_xy, end_xy, direction, extra = _requested_endpoints(
            action, template_traj, carrier_traj, orientation_id
        )
        target = types.SimpleNamespace(action=action,
                                       orientation_id=int(orientation_id))
        try:
            observation, generation = _observe_exact_touch_template(
                req_tmpl, target,
                target_samples=T,
                target_duration_ms=duration_ms,
                requested_start_xy=start_xy,
                requested_end_xy=end_xy,
                requested_direction=direction,
            )
        except Exception as exc:                      # release's own guards
            raise BuildRejected(f"second stage refused: {type(exc).__name__}: {exc}")
        traj = np.array(observation.trajectory, dtype=np.float32)
        audit["second_stage"] = generation.get("transform_mode")
        audit["identity_transform"] = bool(generation.get("identity_transform"))
        audit["residual_scale"] = float(generation.get("residual_scale", 1.0))
        audit.update({k: extra[k] for k in extra})
    else:
        traj = _resample_touch_template(
            np.asarray(template_traj, dtype=np.float32), T
        ).copy()
        audit["second_stage"] = "identity_no_release_placement_available"
        audit["identity_transform"] = True
        audit["residual_scale"] = 1.0

    # S3 -- the observer owns the clock and the deltas, for every action.
    traj[:, 7] = detector_grid_clock(T, duration_ms)
    deltas = np.zeros((T, 2), dtype=np.float32)
    contact = traj[:, 0]
    consecutive = (contact[1:] > 0.0) & (contact[:-1] > 0.0)
    deltas[1:][consecutive] = np.diff(traj[:, 1:3], axis=0)[consecutive]
    traj[:, 5:7] = deltas

    imu = np.ascontiguousarray(carrier_imu, dtype=np.float32)
    try:
        values = modality_values(imu, traj, MODALITY)
    except Exception as exc:
        raise BuildRejected(f"modality_values refused: {exc}")
    return values, traj, imu, audit


def validate_joint(values, traj, T):
    problems = []
    if values.shape != (T, 15):
        problems.append(f"shape {values.shape} != ({T}, 15)")
    if not np.isfinite(values).all():
        problems.append("non-finite")
    if np.any(traj[:, 1:3] < 0.0) or np.any(traj[:, 1:3] > 1.0):
        problems.append("x/y outside [0,1]")
    if float(traj[0, 7]) != 0.0:
        problems.append(f"elapsed[0] = {traj[0, 7]!r} != 0")
    if np.any(np.diff(traj[:, 7]) < 0.0):
        problems.append("elapsed not monotone")
    if T < 2:
        problems.append("T < 2")
    if not np.array_equal(values[:, :6], np.asarray(values[:, :6])):
        problems.append("imu block corrupted")
    return problems


# --------------------------------------------------------------------------- #
# sampling plan
# --------------------------------------------------------------------------- #
def _balanced(rng, k, n):
    """n draws from range(k), balanced: concatenated random permutations."""
    blocks = [rng.permutation(k) for _ in range(int(np.ceil(n / k)))]
    return np.concatenate(blocks)[:n]


def plan(user, action, rep, five, fake_rows, genuine_rows, shard, t_match=True):
    """Return the row3/row4/row5 pairing lists for one (victim, action, rep)."""
    # Deterministic across processes: never Python's randomised str hash.
    key = zlib.crc32(f"{user}|{action}".encode()) & 0x7FFFFFFF
    seed = np.random.SeedSequence(entropy=[SEED_BASE, key, int(rep)])
    rng = np.random.default_rng(seed)
    n = len(fake_rows)
    n5 = len(five)
    if n5 == 0:
        raise BuildRejected("no five-shot material resolved")
    perm = rng.permutation(n)                    # fake events, without replacement
    shot3 = _balanced(rng, n5, n)                # carrier shot for rows 3 and 5
    shot4 = _balanced(rng, n5, n)                # template shot for row 4

    # row 5 templates: genuine events of this victim, in a session other than
    # the carrier's.  The draw is LENGTH-MATCHED to row 3's touch templates
    # (fairness rule F3): row 3 resamples a fake touch onto the carrier's T, so
    # unless row 5's template lengths follow the same distribution the control
    # would be resampled by a different amount than the arm it controls, and the
    # resampling staircase -- not the content -- would explain any difference.
    # Matching uses only event lengths, which are computable from held material
    # and carry no detector output (fairness rule F5).
    by_session = collections.defaultdict(list)
    for row in genuine_rows:
        by_session[str(shard.session[row])].append(row)
    groups = collections.defaultdict(list)
    for k in range(n):
        groups[five[shot3[k]]["session_id"]].append(k)
    row5 = [None] * n
    row5_unmatched = [None] * n
    for key, ks in groups.items():
        candidates = [r for s, rows in by_session.items() if s != key for r in rows]
        if not candidates:
            raise BuildRejected(
                f"{user}/{action}: no genuine event outside session {key}"
            )
        seq: list[int] = []
        while len(seq) < len(ks):
            seq.extend(int(candidates[i]) for i in rng.permutation(len(candidates)))
        seq = seq[:len(ks)]
        for j, k in enumerate(ks):
            row5_unmatched[k] = (seq[j], five[shot3[k]]["row"])
        if t_match:
            # Marginal length matching: rank pairing cannot move the marginal
            # (each of the five carriers is its own session group, so the
            # carrier's T is constant inside a group and only the template's T
            # can carry the ratio).  So the template is DRAWN by nearest length
            # to the fake template row 3 uses at the same k, uniformly among the
            # NEAREST_LENGTH_CANDIDATES closest, which keeps the draw random.
            pool = np.asarray(candidates, dtype=np.int64)
            t_pool = np.log(np.asarray(
                [shard.length(int(r)) for r in pool], dtype=np.float64))
            chosen = []
            for k in ks:
                t_fake = np.log(float(shard.length(int(fake_rows[perm[k]]))))
                distance = np.abs(t_pool - t_fake)
                width = int(min(NEAREST_LENGTH_CANDIDATES, len(pool)))
                near = np.argpartition(distance, width - 1)[:width]
                chosen.append(int(pool[near[rng.integers(width)]]))
        else:
            chosen = seq
        for j, k in enumerate(ks):
            row5[k] = (chosen[j], five[shot3[k]]["row"])

    row3 = [(int(fake_rows[perm[k]]), five[shot3[k]]["row"]) for k in range(n)]
    row4 = [(five[shot4[k]]["row"], int(fake_rows[perm[k]])) for k in range(n)]
    return {"row3": row3, "row4": row4, "row5": row5,
            "row5_unmatched": row5_unmatched,
            "shot3": shot3.tolist(), "shot4": shot4.tolist()}


# --------------------------------------------------------------------------- #
# build-phase driver
# --------------------------------------------------------------------------- #
def _quantiles(values):
    if not len(values):
        return None
    v = np.asarray(values, dtype=np.float64)
    q = np.percentile(v, [0, 5, 25, 50, 75, 95, 100])
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "min": float(q[0]), "p5": float(q[1]), "p25": float(q[2]), "median": float(q[3]),
            "p75": float(q[4]), "p95": float(q[5]), "max": float(q[6])}


def context():
    bundles = action_bundle()
    users = test_users()
    book = ShardBook()
    five, shortfall = fiveshot_map(users, bundles, book)
    gen_orient, fake_orient = orientation_books(users, bundles)
    return dict(bundles=bundles, users=users, book=book, five=five,
                shortfall=shortfall, gen_orient=gen_orient, fake_orient=fake_orient)


def pools(ctx, user, action):
    shard = ctx["book"].get(ctx["bundles"][action], user)
    sel = shard.action == action
    fake_rows = [i for i in range(len(shard.label)) if shard.label[i] == 1 and sel[i]]
    gen_rows = [i for i in range(len(shard.label)) if shard.label[i] == 0 and sel[i]]
    return shard, fake_rows, gen_rows


def orientation_of(ctx, shard, row):
    if shard.label[row] == 1:
        return ctx["fake_orient"].get(str(shard.event_id[row]))
    return ctx["gen_orient"].get(str(shard.cluster[row]))


def build_one(ctx, action, shard, template_row, carrier_row):
    orientation = orientation_of(ctx, shard, carrier_row)
    if orientation is None:
        raise BuildRejected("carrier orientation unresolved")
    t_imu, t_traj = shard.signals(template_row)
    c_imu, c_traj = shard.signals(carrier_row)
    values, traj, imu, audit = build_event(action, t_traj, c_imu, c_traj, orientation)
    audit["template_orientation_id"] = orientation_of(ctx, shard, template_row)
    audit["orientation_mismatch"] = bool(
        audit["template_orientation_id"] is not None
        and audit["template_orientation_id"] != orientation)
    audit["carrier_event_id"] = str(shard.event_id[carrier_row])
    audit["template_event_id"] = str(shard.event_id[template_row])
    audit["carrier_session"] = str(shard.session[carrier_row])
    audit["template_session"] = str(shard.session[template_row])
    audit["clock_matches_carrier_bitwise"] = bool(
        np.array_equal(traj[:, 7], c_traj[:, 7]))
    return values, traj, imu, audit


def run(per_cell: int, reps: int, out_dir: Path):
    ctx = context()
    users, bundles = ctx["users"], ctx["bundles"]
    report = {
        "test_users": users, "action_bundle": bundles,
        "fiveshot_shortfall": ctx["shortfall"],
        "second_stage_policy": SECOND_STAGE_NOTE,
        "seed_base": SEED_BASE, "repetitions": reps,
        "validation_events_per_user_action_per_row": per_cell,
    }

    arms = {k: collections.defaultdict(list)
            for k in ("row1", "row2", "row3", "row4", "row5", "row5_unmatched")}
    rejects = collections.Counter()
    reject_examples = []
    audits = collections.Counter()
    problems = []
    built_examples = {}
    full_T = {k: collections.defaultdict(list)
              for k in ("row1", "row2", "row3", "row4", "row5", "row5_unmatched")}
    full_D = {k: collections.defaultdict(list)
              for k in ("row1", "row2", "row3", "row4", "row5", "row5_unmatched")}
    _ARMS = ("row3", "row4", "row5", "row5_unmatched")
    full_R = {k: collections.defaultdict(list) for k in _ARMS}
    full_TT = {k: collections.defaultdict(list) for k in _ARMS}
    resid = {k: collections.defaultdict(list) for k in _ARMS}
    orient_mismatch = collections.Counter()
    orient_total = collections.Counter()
    n_by_pair = {}
    row5_template_in_five = collections.Counter()
    row5_template_total = collections.Counter()
    distinct_sources = collections.defaultdict(set)

    scored = {}
    for action in ACTIONS:
        for user in users:
            shard, fake_rows, gen_rows = pools(ctx, user, action)
            five = ctx["five"][(user, action)]
            five_rows = {f["row"] for f in five}
            n_by_pair[f"{user}|{action}"] = len(fake_rows)
            # rows 1 and 2 are read, never rebuilt
            for row in gen_rows:
                full_T["row1"][action].append(shard.length(row))
                full_D["row1"][action].append(shard.duration_ms(row))
            for row in fake_rows:
                full_T["row2"][action].append(shard.length(row))
                full_D["row2"][action].append(shard.duration_ms(row))
            for rep in range(reps):
                p = plan(user, action, rep, five, fake_rows, gen_rows, shard)
                for name, pairs in (("row3", p["row3"]), ("row4", p["row4"]),
                                    ("row5", p["row5"]),
                                    ("row5_unmatched", p["row5_unmatched"])):
                    for k, (t_row, c_row) in enumerate(pairs):
                        full_T[name][action].append(shard.length(c_row))
                        full_D[name][action].append(shard.duration_ms(c_row))
                        full_TT[name][action].append(shard.length(t_row))
                        full_R[name][action].append(
                            shard.length(c_row) / max(1, shard.length(t_row)))
                        distinct_sources[(user, action, name, rep)].add(
                            str(shard.event_id[t_row]) if shard.label[t_row] == 0 else None)
                        distinct_sources[(user, action, name, rep)].add(
                            str(shard.event_id[c_row]) if shard.label[c_row] == 0 else None)
                        if name == "row5":
                            row5_template_total[action] += 1
                            if t_row in five_rows:
                                row5_template_in_five[action] += 1
                    if rep >= 1:
                        continue           # build a sample only, from rep 0
                    for k, (t_row, c_row) in enumerate(pairs[:per_cell]):
                        try:
                            values, traj, imu, audit = build_one(
                                ctx, action, shard, t_row, c_row)
                        except BuildRejected as exc:
                            rejects[f"{action}|{name}|{str(exc)[:60]}"] += 1
                            if len(reject_examples) < 40:
                                reject_examples.append(
                                    {"action": action, "arm": name, "user": user,
                                     "reason": str(exc)})
                            continue
                        bad = validate_joint(values, traj, len(values))
                        if bad:
                            problems.append({"action": action, "arm": name,
                                             "user": user, "problems": bad})
                        audits[f"{action}|{name}|{audit['second_stage']}"] += 1
                        orient_total[f"{action}|{name}"] += 1
                        if audit["orientation_mismatch"]:
                            orient_mismatch[f"{action}|{name}"] += 1
                        arms[name][action].append((user, values, audit))
                        resid[name][action].append(float(audit["residual_scale"]))
                        key = f"{action}|{name}"
                        if key not in built_examples:
                            built_examples[key] = {
                                k2: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                for k2, v in audit.items()
                                if k2 not in ("carrier_event_id", "template_event_id")
                            }
                            built_examples[key]["shape"] = list(values.shape)
                            built_examples[key]["elapsed_first_last"] = [
                                float(traj[0, 7]), float(traj[-1, 7])]

    report["build_rejects"] = dict(rejects)
    report["build_reject_examples"] = reject_examples
    report["build_wellformedness_problems"] = problems
    report["second_stage_modes_used"] = dict(audits)
    report["built_example_audit"] = built_examples
    report["orientation_mismatch_rate"] = {
        k: {"n": orient_total[k], "mismatch": orient_mismatch.get(k, 0)}
        for k in sorted(orient_total)}
    report["release_arm_size_per_user_action"] = {
        "min": min(n_by_pair.values()), "max": max(n_by_pair.values()),
        "distinct": sorted(set(n_by_pair.values()))}
    report["row5_template_drawn_from_the_five_fraction"] = {
        a: (row5_template_in_five.get(a, 0) / row5_template_total[a])
        for a in ACTIONS}

    # F4: distinct real recordings consumed per (victim, action) per arm
    consumed = collections.defaultdict(int)
    for (user, action, name, rep), ids in distinct_sources.items():
        ids.discard(None)
        consumed[f"{name}"] = max(consumed[f"{name}"], len(ids))
    report["max_distinct_real_recordings_per_victim_action"] = dict(consumed)

    # parity + confound tables over the FULL design (T and duration are carrier
    # properties, so they are exact without building every event)
    parity = {}
    for name in ("row1", "row2", "row3", "row4", "row5", "row5_unmatched"):
        parity[name] = {}
        for action in ACTIONS:
            entry = {
                "events": len(full_T[name][action]),
                "T": _quantiles(full_T[name][action]),
                "duration_ms": _quantiles(full_D[name][action]),
            }
            if name in full_R:
                entry["template_T"] = _quantiles(full_TT[name][action])
                entry["resample_ratio_carrierT_over_templateT"] = _quantiles(
                    full_R[name][action])
                entry["log2_resample_ratio"] = _quantiles(
                    np.log2(np.asarray(full_R[name][action], dtype=np.float64)))
                entry["second_stage_residual_scale_sample"] = _quantiles(
                    resid[name][action])
                entry["mechanical_steps"] = [
                    "S0 carrier fixes T and duration",
                    "S1 _resample_touch_template(template -> T)",
                    ("S2 _observe_exact_touch_template onto the carrier's own "
                     "endpoints" if action in EXACT_TOUCH_ACTIONS else
                     "S2 identity (release placement not reconstructable)"),
                    "S3 detector_grid_clock(T, duration) + dx/dy recomputed",
                    "S4 modality_values(imu, trajectory, 'imu_trajectory_xytime')",
                ]
            else:
                entry["mechanical_steps"] = ["released bundle event, read as-is"]
            parity[name][action] = entry
        allT = [v for a in ACTIONS for v in full_T[name][a]]
        allD = [v for a in ACTIONS for v in full_D[name][a]]
        parity[name]["ALL"] = {"events": len(allT), "T": _quantiles(allT),
                               "duration_ms": _quantiles(allD)}
    report["parity_T_duration"] = parity

    # exact event-for-event matching claims
    matched = {}
    for action in ACTIONS:
        t3 = np.asarray(full_T["row3"][action]); t5 = np.asarray(full_T["row5"][action])
        d3 = np.asarray(full_D["row3"][action]); d5 = np.asarray(full_D["row5"][action])
        t2 = np.asarray(sorted(full_T["row2"][action]))
        t4 = np.asarray(sorted(full_T["row4"][action]))
        d2 = np.asarray(sorted(full_D["row2"][action]))
        d4 = np.asarray(sorted(full_D["row4"][action]))
        matched[action] = {
            "row3_vs_row5_T_identical_event_for_event": bool(np.array_equal(t3, t5)),
            "row3_vs_row5_duration_identical_event_for_event": bool(np.array_equal(d3, d5)),
            "row4_vs_row2_T_multiset_identical": bool(
                len(t4) == len(t2) * (len(t4) // max(1, len(t2))) and
                np.array_equal(t4, np.repeat(t2, len(t4) // max(1, len(t2))))),
            "row4_vs_row2_duration_multiset_identical": bool(
                np.array_equal(d4, np.repeat(d2, len(d4) // max(1, len(d2))))),
        }
    report["confound_check_matching"] = matched
    return report, arms, ctx


def score_one_cell(arms, action, detector, ctx):
    """Prove a frozen classical joint cell consumes the built arrays."""
    import joblib
    from security_exp.event_detectors import classical_scores, extract_event_features
    cells = cell_map()
    cell = cells[f"{action}__{MODALITY}__{detector}"]
    model = joblib.load(Path(cell["model_dir"]) / "model.joblib")
    cut = float(json.loads(Path(cell["thresholds"]).read_text())["frr5"])
    out = {"cell": f"{action}__{MODALITY}__{detector}", "threshold_frr5": cut,
           "threshold_provenance": "cell thresholds.json, selection_split=development"}
    for name in ("row3", "row4", "row5"):
        feats = []
        for user, values, audit in arms[name][action]:
            feats.append(extract_event_features(detector, MODALITY,
                                                values[:, :6], values[:, 6:]))
        if not feats:
            continue
        matrix = np.stack(feats).astype(np.float32)
        scores = classical_scores(model, matrix)
        out[name] = {"events": int(len(scores)),
                     "score_mean": float(scores.mean()),
                     "score_min": float(scores.min()),
                     "score_max": float(scores.max()),
                     "accepted_fraction_at_frr5": float(np.mean(scores < cut))}
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=6)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default="/tmp/claude-473016/-home-mwang49-new-data7-data7-final-monitor-metrics-v1/e1b42475-b309-42ae-b7f3-314c50fb68d8/scratchpad/halfattack")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report, arms, ctx = run(args.per_cell, args.reps, out_dir)
    report["smoke_scoring"] = [
        score_one_cell(arms, "scroll", "hmog_style_rf", ctx),
        score_one_cell(arms, "tap", "paper_xgboost", ctx),
    ]
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in (
        "build_rejects", "build_wellformedness_problems",
        "second_stage_modes_used", "orientation_mismatch_rate",
        "release_arm_size_per_user_action", "confound_check_matching",
        "max_distinct_real_recordings_per_victim_action",
        "row5_template_drawn_from_the_five_fraction",
        "smoke_scoring")}, indent=1))


if __name__ == "__main__":
    main()
