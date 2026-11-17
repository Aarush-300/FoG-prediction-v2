"""Data layer: load -> per-block resample (60->50 Hz) -> window -> label -> subject-grouped splits.

Run `python data.py` for the dataset report and `python data.py --activity-report` for per-activity gyro RMS.
Windows never cross a (subject, session, task) block boundary, so no window mixes recordings.
"""
import sys
import numpy as np
from scipy.signal import resample_poly, butter, sosfilt

import config as C

_CACHE = C.CACHE / f"windows_{C.SENSOR}.npz"


# ---------------------------------------------------------------- loading
def load_raw():
    header = open(C.CSV_PATH).readline().strip().split(",")
    cols = ["timestamp"] + C.CHANNELS + ["activity", "fog", "fog_severity", "subjectID", "sessionID", "taskID"]
    idx = [header.index(c) for c in cols]
    arr = np.loadtxt(C.CSV_PATH, delimiter=",", skiprows=1, usecols=idx)
    return {c: arr[:, i] for i, c in enumerate(cols)}


def _blocks(d):
    # Some left-ankle rows are NaN (subjects 5, 9, 17). Break blocks at every finite/non-finite change and
    # keep only finite blocks, so no window ever contains NaN or imputed samples.
    ok = np.all(np.stack([np.isfinite(d[c]) for c in C.CHANNELS], 1), axis=1)
    key = np.stack([d["subjectID"], d["sessionID"], d["taskID"], ok.astype(float)], 1)
    brk = np.any(np.diff(key, axis=0) != 0, axis=1) | (np.diff(d["timestamp"]) > 1.5 / C.FS_RAW)
    starts = np.r_[0, np.where(brk)[0] + 1]
    ends = np.r_[starts[1:], len(key)]
    return [(s, e) for s, e in zip(starts, ends) if ok[s]]


def _nearest(lbl, n50):
    src = np.clip(np.round(np.arange(n50) * C.FS_RAW / C.FS).astype(int), 0, len(lbl) - 1)
    return lbl[src]


# ---------------------------------------------------------------- windowing
def build(force=False):
    if _CACHE.exists() and not force:
        z = np.load(_CACHE)
        return {k: z[k] for k in z.files}
    d = load_raw()
    sos = butter(2, C.HW_LPF_HZ, fs=C.FS, output="sos") if C.MATCH_HW_LPF else None
    X, subj, blk, start, frac, y = [], [], [], [], [], []
    fog50, act50, b_off, b_subj = [], [], [0], []
    for b, (s, e) in enumerate(_blocks(d)):
        sig = np.stack([d[c][s:e] for c in C.CHANNELS], 1)
        if len(sig) < 12:
            continue
        x = resample_poly(sig, C.RESAMPLE_UP, C.RESAMPLE_DOWN, axis=0, padtype="line")
        if sos is not None:
            x = sosfilt(sos, x, axis=0)           # causal, like the sensor DLPF
        n = len(x)
        fg = _nearest(d["fog"][s:e], n).astype(np.int8)
        ac = _nearest(d["activity"][s:e], n).astype(np.int8)
        subj_id = int(d["subjectID"][s])
        fog50.append(fg); act50.append(ac); b_off.append(b_off[-1] + n); b_subj.append(subj_id)
        if n < C.WIN:
            continue
        cs = np.r_[0, np.cumsum(fg)]
        for st in range(0, n - C.WIN + 1, C.STRIDE):
            f = (cs[st + C.WIN] - cs[st]) / C.WIN
            X.append(x[st:st + C.WIN]); subj.append(subj_id); blk.append(len(b_subj) - 1)
            start.append(st); frac.append(f); y.append(int(f >= C.FOG_FRAC))
    out = dict(
        X=np.asarray(X, np.float32), subj=np.asarray(subj), blk=np.asarray(blk), start=np.asarray(start),
        frac=np.asarray(frac, np.float32), y=np.asarray(y, np.int8),
        fog50=np.concatenate(fog50), act50=np.concatenate(act50),
        b_off=np.asarray(b_off), b_subj=np.asarray(b_subj),
    )
    np.savez_compressed(_CACHE, **out)
    return out


def walk_mask(act50):
    if C.WALK_ACTIVITIES is None:
        return act50 != 0
    return np.isin(act50, C.WALK_ACTIVITIES)


def walking_hours(D, subjects):
    """Real walking time (hours of 50 Hz samples, not windows) for the given subjects."""
    wm = walk_mask(D["act50"])
    tot = 0
    for b, s in enumerate(D["b_subj"]):
        if s in subjects:
            tot += wm[D["b_off"][b]:D["b_off"][b + 1]].sum()
    return tot / C.FS / 3600.0


# ---------------------------------------------------------------- features / normalization
def add_features(X):
    if C.FEATURES == "raw6":
        return X
    acc = np.linalg.norm(X[..., :3], axis=-1, keepdims=True)
    gyr = np.linalg.norm(X[..., 3:6], axis=-1, keepdims=True)
    return np.concatenate([X, acc, gyr], -1).astype(np.float32)


def n_channels():
    return 6 if C.FEATURES == "raw6" else 8


def fit_norm(X, subj, fit_subjects):
    """Per-channel mean/std from samples of `fit_subjects` only; returns (mean, std, fit_subjects)."""
    fit_subjects = sorted(set(int(s) for s in fit_subjects))
    assert set(np.unique(subj[np.isin(subj, fit_subjects)])) == set(fit_subjects)
    F = add_features(X[np.isin(subj, fit_subjects)]).reshape(-1, n_channels())
    return F.mean(0).astype(np.float32), (F.std(0) + 1e-6).astype(np.float32), tuple(fit_subjects)


def apply_norm(X, mean, std):
    return ((add_features(X) - mean) / std).astype(np.float32)


# ---------------------------------------------------------------- splits
def dev_test_subjects(D):
    all_s = sorted(set(int(s) for s in D["subj"]))
    test = sorted(C.TEST_SUBJECTS)
    dev = [s for s in all_s if s not in test]
    assert not set(dev) & set(test), "subject leakage dev/test"
    return dev, test


def make_folds(D, dev, k=C.N_FOLDS):
    """Subject-grouped folds, greedily balancing FoG-window counts (and total windows) per fold."""
    cnt = {s: int(D["y"][D["subj"] == s].sum()) for s in dev}
    tot = {s: int((D["subj"] == s).sum()) for s in dev}
    folds = [[] for _ in range(k)]
    load = [[0, 0] for _ in range(k)]
    for s in sorted(dev, key=lambda s: (-cnt[s], -tot[s])):
        j = min(range(k), key=lambda i: (load[i][0], load[i][1]))
        folds[j].append(s); load[j][0] += cnt[s]; load[j][1] += tot[s]
    return [sorted(f) for f in folds]


def assert_split(D, train_s, val_s, test_s, norm_fit_subjects):
    tr, va, te = set(train_s), set(val_s), set(test_s)
    assert not (tr & va) and not (tr & te) and not (va & te), "subject overlap between splits"
    assert set(norm_fit_subjects) <= tr, "normalization statistics must be fit on train subjects only"
    # window overlap: a block belongs to exactly one subject, so subject-disjoint => window-disjoint
    own = {}
    for b, s in zip(D["blk"], D["subj"]):
        assert own.setdefault(int(b), int(s)) == int(s), "block spans subjects"
    split_of = lambda s: "tr" if s in tr else "va" if s in va else "te" if s in te else None
    seen = {}
    for b, s in own.items():
        sp = split_of(s)
        if sp is not None:
            assert seen.setdefault(b, sp) == sp, "block appears in two splits"


def select(D, subjects):
    m = np.isin(D["subj"], list(subjects))
    return m


# ---------------------------------------------------------------- reports
def report(D):
    dev, test = dev_test_subjects(D)
    print(f"windows {len(D['y'])}  shape {D['X'].shape}  FoG(frac>={C.FOG_FRAC}) = {D['y'].mean():.3%}")
    print(f"boundary windows (0<frac<{C.FOG_FRAC}): {int(((D['frac'] > 0) & (D['frac'] < C.FOG_FRAC)).sum())}")
    print("subj  windows  fog_windows  fog%   walk_h  split")
    for s in sorted(set(D["subj"].tolist())):
        m = D["subj"] == s
        print(f"{s:4d} {m.sum():8d} {int(D['y'][m].sum()):11d} {D['y'][m].mean():6.1%} "
              f"{walking_hours(D, {s}):7.3f}  {'TEST' if s in test else 'dev'}")
    folds = make_folds(D, dev)
    for i, f in enumerate(folds):
        m = np.isin(D["subj"], f)
        print(f"fold {i}: subjects {f}  windows {m.sum()}  fog {int(D['y'][m].sum())}")
    assert sorted(sum(folds, [])) == sorted(dev)
    # leakage self-check on fold 0
    val = folds[0]; tr = [s for s in dev if s not in val]
    mean, std, fit = fit_norm(D["X"], D["subj"], tr)
    assert_split(D, tr, val, test, fit)
    print("leakage asserts passed (subjects, normalization fit, block/window disjointness)")


def activity_report():
    d = load_raw()
    g = np.stack([d[f"{C.SENSOR}_gyro_{a}"] for a in "xyz"], 1)
    print("activity  n_samples  fog_frac  ankle_gyro_RMS(deg/s units of file)")
    for a in sorted(set(d["activity"].astype(int).tolist())):
        m = d["activity"] == a
        print(f"{a:8d} {m.sum():10d} {d['fog'][m].mean():9.3f} {np.sqrt((g[m] ** 2).sum(1).mean()):10.3f}")


if __name__ == "__main__":
    if "--activity-report" in sys.argv:
        activity_report()
    else:
        report(build(force="--rebuild" in sys.argv))
