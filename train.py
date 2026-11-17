"""Training + subject-grouped CV.

  python train.py --models ds_tcn inception hybrid --cv          # CV for each candidate (seed 0)
  python train.py --models ds_tcn --ablate                       # augmentation ablation via CV
  python train.py --models ds_tcn --final                        # 5 seeds on all dev subjects (test untouched)

Notes
- Augmentation runs on RAW 6ch windows, then magnitude features + z-score (stats fit on train subjects only).
- If balanced_sampling is on, positives are oversampled per epoch and pos_weight is NOT applied (avoids
  double-counting the imbalance). With balanced_sampling off, pos_weight = n_neg/n_pos is used.
- Early stopping / model selection use PR-AUC on the subject-disjoint validation fold. Test subjects are never loaded here.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import argparse, json, math, random
import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score

import config as C
import data as D
import augment as A
import models as M

keras = tf.keras


def set_seed(s):
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)


def make_loss(gamma, pos_w, ls):
    def loss(y, p):
        y = tf.cast(tf.reshape(y, [-1]), tf.float32); p = tf.reshape(p, [-1])
        ys = y * (1 - ls) + 0.5 * ls
        p = tf.clip_by_value(p, 1e-6, 1 - 1e-6)
        pt = ys * p + (1 - ys) * (1 - p)
        w = y * pos_w + (1 - y)
        ce = -(ys * tf.math.log(p) + (1 - ys) * tf.math.log(1 - p))
        return tf.reduce_mean(w * tf.pow(1 - pt, gamma) * ce)
    return loss


def lr_at(ep, T):
    if ep < T["warmup_epochs"]:
        return T["lr"] * (ep + 1) / T["warmup_epochs"]
    t = (ep - T["warmup_epochs"]) / max(1, T["epochs"] - T["warmup_epochs"])
    return 1e-5 + 0.5 * (T["lr"] - 1e-5) * (1 + math.cos(math.pi * t))


def epoch_indices(y, rng, balanced):
    if not balanced:
        return rng.permutation(len(y))
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n = len(y)
    npos = min(int(0.4 * n), max(len(pos), 1) * 4)           # cap oversampling at 4x, target <=40% positives
    idx = np.r_[rng.choice(pos, npos, replace=True), rng.choice(neg, n - npos, replace=len(neg) < n - npos)]
    return rng.permutation(idx)


def prepare(Dd, train_s, val_s):
    tm, vm = D.select(Dd, train_s), D.select(Dd, val_s)
    mean, std, fit = D.fit_norm(Dd["X"], Dd["subj"], train_s)
    D.assert_split(Dd, train_s, val_s, [], fit)
    Xtr, ytr, Xva, yva = Dd["X"][tm], Dd["y"][tm], D.apply_norm(Dd["X"][vm], mean, std), Dd["y"][vm]
    if C.TRAIN_DROP_BOUNDARY:
        keep = ~((Dd["frac"][tm] > 0) & (Dd["frac"][tm] < C.FOG_FRAC))
        Xtr, ytr = Xtr[keep], ytr[keep]
    return Xtr, ytr, Xva, yva, mean, std


def train_one(name, Dd, train_s, val_s, seed, enable=A.ALL, T=None, verbose=0, builder_kw=None):
    T = {**C.TRAIN, **(T or {})}
    set_seed(seed); rng = np.random.default_rng(seed)
    Xtr, ytr, Xva, yva, mean, std = prepare(Dd, train_s, val_s)
    model = M.BUILDERS[name](qat=False, **(builder_kw or {}))
    pw = 1.0 if T["balanced_sampling"] else (float((ytr == 0).sum()) / max(1, (ytr == 1).sum()) if T["pos_weight"] else 1.0)
    model.compile(optimizer=keras.optimizers.AdamW(T["lr"], weight_decay=T["wd"]),
                  loss=make_loss(T["focal_gamma"], pw, T["label_smoothing"]))
    ema = [w.numpy().copy() for w in model.weights]
    best, best_w, bad, hist = -1.0, None, 0, []
    for ep in range(T["epochs"]):
        keras.backend.set_value(model.optimizer.learning_rate, lr_at(ep, T))
        idx = epoch_indices(ytr, rng, T["balanced_sampling"])
        Xa = D.apply_norm(A.augment_batch(Xtr[idx], rng, enable=enable), mean, std) if enable else D.apply_norm(Xtr[idx], mean, std)
        model.fit(Xa, ytr[idx].astype("float32"), batch_size=T["batch"], epochs=1, verbose=0, shuffle=True)
        for e, w in zip(ema, model.weights):
            e *= T["ema"]; e += (1 - T["ema"]) * w.numpy()
        live = [w.numpy().copy() for w in model.weights]
        for w, e in zip(model.weights, ema):
            w.assign(e)
        pva = model.predict(Xva, batch_size=512, verbose=0).ravel()
        ap = average_precision_score(yva, pva) if yva.sum() else float("nan")
        hist.append(ap)
        if verbose:
            print(f"  ep{ep:02d} val PR-AUC {ap:.4f}")
        score = ap if not np.isnan(ap) else -ep                 # fold without FoG: no signal, keep last
        if score > best:
            best, bad, best_w = score, 0, [w.numpy().copy() for w in model.weights]
        else:
            bad += 1
        for w, l in zip(model.weights, live):                     # continue training from live weights
            w.assign(l)
        if bad >= T["patience"]:
            break
    model.set_weights(best_w)
    pva = model.predict(Xva, batch_size=512, verbose=0).ravel()
    return dict(model=model, mean=mean, std=std, p_val=pva, y_val=yva, best_val_ap=best, epochs_run=len(hist))


def cv(name, Dd, seed=0, enable=A.ALL, T=None, builder_kw=None):
    dev, _ = D.dev_test_subjects(Dd)
    folds = D.make_folds(Dd, dev)
    oof_p, oof_y, oof_s, per_fold = [], [], [], []
    for i, val in enumerate(folds):
        tr = [s for s in dev if s not in val]
        r = train_one(name, Dd, tr, val, seed, enable, T, builder_kw=builder_kw)
        vm = D.select(Dd, val)
        oof_p.append(r["p_val"]); oof_y.append(r["y_val"]); oof_s.append(Dd["subj"][vm])
        per_fold.append(r["best_val_ap"])
        print(f"[{name}] fold {i} val={val} PR-AUC {r['best_val_ap']:.4f} (epochs {r['epochs_run']})", flush=True)
        keras.backend.clear_session()
    p, y, s = np.concatenate(oof_p), np.concatenate(oof_y), np.concatenate(oof_s)
    pooled = average_precision_score(y, p)
    print(f"[{name}] pooled OOF PR-AUC {pooled:.4f}  fold mean {np.nanmean(per_fold):.4f}")
    return dict(name=name, pooled_ap=float(pooled), fold_ap=[float(a) for a in per_fold],
                oof=dict(p=p, y=y, s=s))


def ablate(name, Dd, seed=0):
    rows = {"all": cv(name, Dd, seed)["pooled_ap"], "none": cv(name, Dd, seed, enable=())["pooled_ap"]}
    for a in A.ALL:
        rows[f"-{a}"] = cv(name, Dd, seed, enable=tuple(x for x in A.ALL if x != a))["pooled_ap"]
    return rows


def final(name, Dd, seeds=C.SEEDS, builder_kw=None):
    """Train each seed on one fixed subject-disjoint split so seed scores are comparable.

    This is a seed-stability run, not the final all-development-data refit. Use CV to select the
    architecture/hyperparameters; retain this fixed validation fold for early stopping and calibration.
    """
    dev, _ = D.dev_test_subjects(Dd)
    folds = D.make_folds(Dd, dev)
    val = folds[0]
    out = []
    for sd in seeds:
        tr = [s for s in dev if s not in val]
        r = train_one(name, Dd, tr, val, sd, builder_kw=builder_kw)
        path = C.OUT / f"{name}_seed{sd}.weights.h5"
        r["model"].save_weights(path)
        out.append(dict(seed=sd, val_subjects=val, val_ap=float(r["best_val_ap"]), weights=str(path),
                        mean=r["mean"].tolist(), std=r["std"].tolist()))
        print(f"[{name}] seed {sd} val={val} PR-AUC {r['best_val_ap']:.4f}", flush=True)
        keras.backend.clear_session()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ds_tcn"])
    ap.add_argument("--cv", action="store_true"); ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--final", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    Dd = D.build()
    T = {"epochs": a.epochs} if a.epochs else None
    res = {}
    for n in a.models:
        if a.cv:
            r = cv(n, Dd, a.seed, T=T); np.savez(C.OUT / f"oof_{n}.npz", **r["oof"]); res[n] = {k: r[k] for k in ("pooled_ap", "fold_ap")}
        if a.ablate:
            res[f"{n}_ablation"] = ablate(n, Dd, a.seed)
        if a.final:
            res[f"{n}_final"] = final(n, Dd)
    (C.OUT / "train_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
