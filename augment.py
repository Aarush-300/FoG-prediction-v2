"""IMU augmentation on RAW 6-channel windows (before magnitude features / z-score).

Each transform can be toggled via the `enable` set so train.py can ablate them one by one.
All functions are vectorised over the batch: X has shape (B, T, 6) = acc xyz, gyro xyz.
"""
import numpy as np

import config as C

ALL = ("rot", "scale", "noise", "warp", "shift", "ch_drop")


def _rotmats(n, max_deg, rng):
    """Random small rotations (Rodrigues) -> (n, 3, 3). Models sensor mounting / tilt variation."""
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-9
    ang = np.deg2rad(rng.uniform(-max_deg, max_deg, size=n))
    K = np.zeros((n, 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
    K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]
    s, c = np.sin(ang)[:, None, None], np.cos(ang)[:, None, None]
    return np.eye(3)[None] + s * K + (1 - c) * (K @ K)


def _time_warp(X, sigma, rng):
    B, T, _ = X.shape
    knots = 4
    speed = 1.0 + rng.normal(0, sigma, size=(B, knots + 2))
    speed = np.clip(speed, 0.5, 1.5)
    grid = np.linspace(0, 1, knots + 2)
    t = np.linspace(0, 1, T)
    out = np.empty_like(X)
    for i in range(B):
        sp = np.interp(t, grid, speed[i])
        pos = np.cumsum(sp)
        pos = (pos - pos[0]) / (pos[-1] - pos[0]) * (T - 1)
        for c in range(X.shape[2]):
            out[i, :, c] = np.interp(pos, np.arange(T), X[i, :, c])
    return out


def augment_batch(X, rng, cfg=None, enable=ALL):
    cfg = cfg or C.AUG
    X = X.copy()
    B, T, _ = X.shape
    on = rng.random(B) < cfg["p"]
    idx = np.where(on)[0]
    if len(idx) == 0:
        return X
    Xs = X[idx]
    n = len(idx)
    if "rot" in enable:
        R = _rotmats(n, cfg["rot_deg"], rng)
        Xs[..., :3] = np.einsum("bij,btj->bti", R, Xs[..., :3])
        Xs[..., 3:6] = np.einsum("bij,btj->bti", R, Xs[..., 3:6])   # same mounting rotation for gyro
    if "scale" in enable:
        Xs[..., :3] *= rng.uniform(1 - cfg["scale"], 1 + cfg["scale"], (n, 1, 1))
        Xs[..., 3:6] *= rng.uniform(1 - cfg["scale"], 1 + cfg["scale"], (n, 1, 1))
    if "warp" in enable:
        Xs = _time_warp(Xs, cfg["warp"], rng)
    if "shift" in enable:
        sh = rng.integers(-cfg["shift"], cfg["shift"] + 1, size=n)
        for k in range(n):
            ix = np.clip(np.arange(T) + sh[k], 0, T - 1)
            Xs[k] = Xs[k][ix]
    if "noise" in enable:
        sd = Xs.reshape(n, -1, 6).std(axis=1, keepdims=True) + 1e-6
        Xs += rng.normal(size=Xs.shape) * sd * cfg["noise"]
    if "ch_drop" in enable:
        drop = rng.random((n, 1, 6)) < cfg["ch_drop"]
        Xs = np.where(drop, Xs.mean(axis=1, keepdims=True), Xs)
    X[idx] = Xs
    return X
