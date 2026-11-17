"""Metrics, episode scoring, post-processing FSM tuning, subject-level bootstrap CIs, plots.

Definitions
- Episode: a maximal run of consecutive positive-labelled windows inside one block (stride 25 => 0.5 s apart).
- Episode detected: the alarm FSM fires at least once while the run is active or within TOL windows after it ends.
- Detection delay: (alarm window - first episode window) * STRIDE/FS seconds, detected episodes only.
- False alarm: alarm rising edge whose window is not within TOL windows of any positive run.
- FA/hr = false alarms / real walking hours of the evaluated subjects (D.walking_hours, 50 Hz samples).

Post-processing FSM (C++-friendly): EMA(p) -> OFF --[ema>=thr_on for k consecutive windows]--> ON (alarm event)
                                                  ON --[ema<thr_off for 2 windows]--> OFF
Tuning is done on validation / OOF data only. Test data is only ever passed through the frozen parameters.
"""
import itertools
import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score, confusion_matrix, precision_recall_curve)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import data as D

TOL = 2                      # windows of slack when matching alarms to episodes
HOP_S = C.STRIDE / C.FS      # seconds between windows


# ---------------------------------------------------------------- window level
def window_metrics(y, p, thr):
    y = np.asarray(y).astype(int)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn); spec = tn / max(1, tn + fp)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    has_both = 0 < y.sum() < len(y)
    return dict(pr_auc=float(average_precision_score(y, p)) if y.sum() else float("nan"),
                roc_auc=float(roc_auc_score(y, p)) if has_both else float("nan"),
                f1=float(f1), precision=float(prec), recall=float(rec), specificity=float(spec), mcc=float(mcc),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def pick_threshold(y, p, target_recall=None):
    """Window threshold on validation data: max-F1 (or highest threshold reaching target_recall)."""
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
    if target_recall is None:
        return float(th[np.nanargmax(f1[:-1])])
    ok = np.where(rc[:-1] >= target_recall)[0]
    return float(th[ok[-1]]) if len(ok) else float(th[0])


# ---------------------------------------------------------------- post-processing FSM
def run_fsm(p, blk, ema, k, thr_on, thr_off, off_n=2):
    """Returns (alarm_event[bool], state[bool]) per window. p must be ordered by (blk, start)."""
    ev = np.zeros(len(p), bool); st = np.zeros(len(p), bool)
    prev_b, e, on, cnt, offc = None, 0.0, False, 0, 0
    for i in range(len(p)):
        if blk[i] != prev_b:
            prev_b, e, on, cnt, offc = blk[i], p[i], False, 0, 0
        else:
            e = ema * e + (1 - ema) * p[i]
        if not on:
            cnt = cnt + 1 if e >= thr_on else 0
            if cnt >= k:
                on, offc = True, 0; ev[i] = True
        else:
            offc = offc + 1 if e < thr_off else 0
            if offc >= off_n:
                on, cnt = False, 0
        st[i] = on
    return ev, st


def episodes(y, blk):
    """List of (first_idx, last_idx) of positive runs (indices into the ordered window arrays)."""
    eps, i, n = [], 0, len(y)
    while i < n:
        if y[i] == 1:
            j = i
            while j + 1 < n and y[j + 1] == 1 and blk[j + 1] == blk[i]:
                j += 1
            eps.append((i, j)); i = j + 1
        else:
            i += 1
    return eps


def episode_stats(y, blk, ev):
    """Per-subject sufficient statistics: n_eps, n_detected, delays (s), false alarms."""
    eps = episodes(y, blk)
    near = np.zeros(len(y), bool)
    det, delays = 0, []
    for a, b in eps:
        # Clip tolerance to the same block; adjacent blocks may be different trials or sessions.
        same = np.flatnonzero(blk == blk[a])
        lo_b, hi_b = int(same[0]), int(same[-1])
        lo, hi = max(lo_b, a - TOL), min(hi_b, b + TOL)
        near[lo:hi + 1] = True
        hits = [i for i in range(a, hi + 1) if ev[i] and blk[i] == blk[a]]
        if hits:
            det += 1; delays.append(max(0, hits[0] - a) * HOP_S)
    fa = int((ev & ~near).sum())
    return dict(n_eps=len(eps), n_det=det, delays=delays, fa=fa)


# ---------------------------------------------------------------- per-subject packaging
def per_subject(y, p, subj, blk, start, params, hours_of):
    """Order windows by (blk,start) within each subject, run FSM, collect stats."""
    out = {}
    for s in np.unique(subj):
        m = np.where(subj == s)[0]
        o = m[np.lexsort((start[m], blk[m]))]
        ev, _ = run_fsm(p[o], blk[o], **params)
        st = episode_stats(y[o], blk[o], ev)
        out[int(s)] = dict(y=y[o], p=p[o], hours=hours_of(int(s)), **st)
    return out


def pooled(ps, thr):
    y = np.concatenate([v["y"] for v in ps.values()]); p = np.concatenate([v["p"] for v in ps.values()])
    wm = window_metrics(y, p, thr)
    ne = sum(v["n_eps"] for v in ps.values()); nd = sum(v["n_det"] for v in ps.values())
    hrs = sum(v["hours"] for v in ps.values()); fa = sum(v["fa"] for v in ps.values())
    dl = [d for v in ps.values() for d in v["delays"]]
    return dict(**wm, ep_sens=nd / ne if ne else float("nan"), n_eps=ne,
                med_delay_s=float(np.median(dl)) if dl else float("nan"), fa_per_hr=fa / hrs if hrs > 0 else float("nan"))


def bootstrap_ci(ps, thr, n=1000, seed=0, keys=("pr_auc", "roc_auc", "f1", "precision", "recall", "specificity",
                                               "mcc", "ep_sens", "med_delay_s", "fa_per_hr")):
    """95% CI by resampling SUBJECTS with replacement."""
    rng = np.random.default_rng(seed)
    subs = list(ps.keys()); vals = {k: [] for k in keys}
    for _ in range(n):
        pick = rng.choice(len(subs), len(subs), replace=True)
        b = {j: ps[subs[i]] for j, i in enumerate(pick)}
        if sum(v["y"].sum() for v in b.values()) == 0:
            continue
        r = pooled(b, thr)
        for k in keys:
            vals[k].append(r[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) if v else (float("nan"),) * 2
            for k, v in vals.items()}


# ---------------------------------------------------------------- post-proc tuning (validation/OOF only)
def tune_postproc(y, p, subj, blk, start, thr0, hours_of, target_sens=0.9):
    """Grid search: minimise FA/hr s.t. episode sensitivity >= target. Falls back to max sensitivity."""
    best, best_key = None, None
    for ema, k, on_off in itertools.product([0.0, 0.5, 0.7], [1, 2, 3], [(thr0, thr0), (thr0, 0.7 * thr0), (1.2 * thr0, 0.8 * thr0)]):
        thr_on, thr_off = min(on_off[0], 0.99), on_off[1]
        prm = dict(ema=ema, k=k, thr_on=thr_on, thr_off=thr_off)
        r = pooled(per_subject(y, p, subj, blk, start, prm, hours_of), thr0)
        key = (r["ep_sens"] >= target_sens, -r["fa_per_hr"] if r["ep_sens"] >= target_sens else r["ep_sens"])
        if best_key is None or key > best_key:
            best, best_key = dict(params=prm, **{q: r[q] for q in ("ep_sens", "fa_per_hr", "med_delay_s")}), key
    return best


def oof_meta(Dd, seed_folds=None):
    """Window indices in the exact order train.cv concatenated OOF predictions (fold by fold, original order)."""
    dev, _ = D.dev_test_subjects(Dd)
    idx = [np.where(D.select(Dd, f))[0] for f in D.make_folds(Dd, dev)]
    idx = np.concatenate(idx)
    return Dd["subj"][idx], Dd["blk"][idx], Dd["start"][idx]


# ---------------------------------------------------------------- plots
def plot_pr(y, p, path, title="PR curve"):
    pr, rc, _ = precision_recall_curve(y, p)
    plt.figure(figsize=(4.5, 4)); plt.plot(rc, pr); plt.axhline(y.mean(), ls="--", c="gray")
    plt.xlabel("recall"); plt.ylabel("precision"); plt.title(f"{title} (AP={average_precision_score(y, p):.3f})")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def plot_confusion(m, path):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    plt.figure(figsize=(3.6, 3.4)); plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xticks([0, 1], ["no FoG", "FoG"]); plt.yticks([0, 1], ["no FoG", "FoG"])
    plt.xlabel("predicted"); plt.ylabel("true"); plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def plot_sens_fa(y, p, subj, blk, start, thr0, hours_of, path):
    pts = []
    for thr in np.linspace(0.1, 0.95, 18):
        prm = dict(ema=0.5, k=2, thr_on=thr, thr_off=0.7 * thr)
        r = pooled(per_subject(y, p, subj, blk, start, prm, hours_of), thr)
        pts.append((r["fa_per_hr"], r["ep_sens"]))
    pts = np.array(pts)
    plt.figure(figsize=(4.5, 4)); plt.plot(pts[:, 0], pts[:, 1], "o-")
    plt.xlabel("false alarms / walking hour"); plt.ylabel("episode sensitivity"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def plot_per_subject(rows, path, key="recall"):
    s = [r["subject"] for r in rows]; v = [r[key] for r in rows]
    plt.figure(figsize=(7, 3.2)); plt.bar([str(x) for x in s], v)
    plt.xlabel("subject"); plt.ylabel(key); plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def subject_table(ps, thr):
    rows = []
    for s, v in sorted(ps.items()):
        r = pooled({s: v}, thr)
        rows.append(dict(subject=s, fog_windows=int(v["y"].sum()), n_eps=r["n_eps"], recall=r["recall"],
                         precision=r["precision"], ep_sens=r["ep_sens"], fa_per_hr=r["fa_per_hr"],
                         flag=bool((v["y"].sum() > 0 and r["recall"] < 0.5) or r["fa_per_hr"] > 30)))
    return rows


if __name__ == "__main__":
    # OOF evaluation of a CV run: python evaluate.py ds_tcn   (reads outputs/oof_<name>.npz)
    import sys, json
    name = sys.argv[1] if len(sys.argv) > 1 else "ds_tcn"
    Dd = D.build()
    z = np.load(C.OUT / f"oof_{name}.npz")
    subj, blk, start = oof_meta(Dd)
    assert (subj == z["s"]).all(), "OOF ordering mismatch"
    hours_of = lambda s: D.walking_hours(Dd, {s})
    thr = pick_threshold(z["y"], z["p"])
    tuned = tune_postproc(z["y"], z["p"], subj, blk, start, thr, hours_of)
    ps = per_subject(z["y"], z["p"], subj, blk, start, tuned["params"], hours_of)
    res = dict(thr=thr, tuned=tuned, pooled=pooled(ps, thr), ci=bootstrap_ci(ps, thr), per_subject=subject_table(ps, thr))
    plot_pr(z["y"], z["p"], C.PLOTS / f"pr_oof_{name}.png", f"{name} OOF")
    plot_confusion(res["pooled"], C.PLOTS / f"cm_oof_{name}.png")
    plot_sens_fa(z["y"], z["p"], subj, blk, start, thr, hours_of, C.PLOTS / f"sens_fa_oof_{name}.png")
    plot_per_subject(res["per_subject"], C.PLOTS / f"per_subject_oof_{name}.png")
    (C.OUT / f"eval_oof_{name}.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in ("thr", "tuned", "pooled", "ci")}, indent=2, default=float))
