"""QAT fine-tune -> full-INT8 TFLite -> C header, norm arrays, op list, golden vectors.

  python quantize_export.py --model ds_tcn            # seed picked by pooled subject-grouped OOF PR-AUC
  python quantize_export.py --model ds_tcn --test     # ALSO evaluate float vs INT8 on held-out test subjects (run once)

Requires `python train.py --models <name> --final` first (writes train_results.json + weights).
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import argparse, json
import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score

import config as C
import data as D
import augment as A
import models as M
import train as T
import evaluate as E

keras = tf.keras


def pick_seed(name):
    """Pick the seed with highest validation PR-AUC from the fixed-split five-seed stability run."""
    results = json.loads((C.OUT / "train_results.json").read_text())[f"{name}_final"]
    assert len(results) >= 5, "run train.py --models <name> --final first"
    sdrow = max(results, key=lambda r: r["val_ap"] if r["val_ap"] == r["val_ap"] else -1)
    Dd = D.build(); dev, _ = D.dev_test_subjects(Dd)
    val = D.make_folds(Dd, dev)[0]
    assert list(sdrow["val_subjects"]) == list(val), "final seed results must share the fixed validation fold"
    return dict(seed=sdrow["seed"], val_subjects=val, val_ap=sdrow["val_ap"], weights=sdrow["weights"],
                mean=sdrow["mean"], std=sdrow["std"])


def transfer(src, dst):
    """Copy float kernels/biases into QAT wrappers, preserving quantizer state initialized by tfmot."""
    sl = [l for l in src.layers if l.weights]
    # tfmot adds an input quantizer with state but no source-layer counterpart.
    dl = [l for l in dst.layers if (hasattr(l, "layer") and l.layer.weights) or
          (not hasattr(l, "layer") and l.weights and isinstance(l, keras.layers.Dense))]
    assert len(sl) == len(dl), ([(x.name, len(x.weights)) for x in sl], [(x.name, len(x.weights)) for x in dl])
    for a, target in zip(sl, dl):
        aw = a.get_weights()
        if hasattr(target, "layer"):
            bw = target.get_weights()
            # Wrapper weight order starts with the wrapped kernel and optional bias; trailing entries are QAT ranges.
            for i, w in enumerate(aw):
                matches = [j for j, v in enumerate(bw) if v.shape == w.shape]
                assert matches, (a.name, target.name, w.shape)
                j = matches[0] if len(matches) == 1 else next((m for m in matches if np.all(bw[m] == 0)), matches[0])
                bw[j] = w
            target.set_weights(bw)
        else:
            assert [x.shape for x in aw] == [tuple(w.shape) for w in target.get_weights()], a.name
            target.set_weights(aw)


def qat_finetune(name, sel, Dd, dev):
    val = sel["val_subjects"]
    tr = [s for s in dev if s not in val]
    mean, std = np.array(sel["mean"], np.float32), np.array(sel["std"], np.float32)
    f = M.BUILDERS[name](qat=False); f.load_weights(sel["weights"])
    q = M.apply_qat(M.BUILDERS[name](qat=True)); transfer(f, q)
    t = C.TRAIN
    q.compile(optimizer=keras.optimizers.Adam(t["qat_lr"]), loss=T.make_loss(t["focal_gamma"], 1.0, t["label_smoothing"]))
    tm, vm = D.select(Dd, tr), D.select(Dd, val)
    Xtr, ytr = Dd["X"][tm], Dd["y"][tm]
    Xva, yva = D.apply_norm(Dd["X"][vm], mean, std), Dd["y"][vm]
    rng = np.random.default_rng(sel["seed"])
    best, bw = -1, None
    for ep in range(t["qat_epochs"]):
        idx = T.epoch_indices(ytr, rng, t["balanced_sampling"])
        Xa = D.apply_norm(A.augment_batch(Xtr[idx], rng), mean, std)
        q.fit(Xa, ytr[idx].astype("float32"), batch_size=t["batch"], epochs=1, verbose=0)
        ap = average_precision_score(yva, q.predict(Xva, batch_size=512, verbose=0).ravel()) if yva.sum() else 0.0
        print(f"  QAT ep{ep} val PR-AUC {ap:.4f}")
        if ap > best:
            best, bw = ap, q.get_weights()
    q.set_weights(bw)
    # De-quantized QAT weights in the original float graph, for a like-for-like validation comparison.
    qat_float = M.BUILDERS[name](qat=False)
    src_layers = []
    for layer in q.layers:
        if hasattr(layer, "layer") and layer.layer.weights:
            src_layers.append((layer, True))
        elif isinstance(layer, keras.layers.Dense) and layer.weights:
            src_layers.append((layer, False))
    dst_layers = [layer for layer in qat_float.layers if layer.weights]
    assert len(src_layers) == len(dst_layers), (len(src_layers), len(dst_layers))
    for (src, wrapped), dst in zip(src_layers, dst_layers):
        values = src.get_weights()
        if wrapped:
            # QAT wrapper order: wrapped kernel/bias followed by quantizer state.
            values = values[:len(dst.get_weights())]
        assert [v.shape for v in values] == [v.shape for v in dst.get_weights()], (src.name, dst.name)
        dst.set_weights(values)
    return qat_float, q, mean, std, tr, val


def convert(q, rep_x):
    conv = tf.lite.TFLiteConverter.from_keras_model(q)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([x[None]] for x in rep_x)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def run_int8(tfl, X):
    it = tf.lite.Interpreter(model_content=tfl); it.allocate_tensors()
    di, do = it.get_input_details()[0], it.get_output_details()[0]
    si, zi = di["quantization"]; so, zo = do["quantization"]
    out_q, out_f = [], []
    for x in X:
        qi = np.clip(np.round(x / si + zi), -128, 127).astype(np.int8)
        it.set_tensor(di["index"], qi[None]); it.invoke()
        o = it.get_tensor(do["index"])[0, 0]
        out_q.append(int(o)); out_f.append(float((o - zo) * so))
    return np.array(out_f), np.array(out_q), qi_of(X, si, zi)


def qi_of(X, s, z):
    return np.clip(np.round(X / s + z), -128, 127).astype(np.int8)


def tflite_ops(tfl):
    it = tf.lite.Interpreter(model_content=tfl)
    return sorted({d["op_name"] for d in it._get_ops_details()})


def write_header(tfl, path):
    rows = [", ".join(f"0x{b:02x}" for b in tfl[i:i + 16]) for i in range(0, len(tfl), 16)]
    path.write_text("// auto-generated\nalignas(16) const unsigned char model_data[] = {\n  " + ",\n  ".join(rows) +
                    f"\n}};\nconst unsigned int model_data_len = {len(tfl)};\n")


def write_quantization(tfl, path):
    it = tf.lite.Interpreter(model_content=tfl)
    it.allocate_tensors()
    di, do = it.get_input_details()[0], it.get_output_details()[0]
    si, zi = di["quantization"]
    so, zo = do["quantization"]
    path.write_text(
        "// TFLite affine quantization: q = round(real / scale) + zero_point.\n"
        f"constexpr float kInputScale = {si:.9g}f;\n"
        f"constexpr int kInputZeroPoint = {zi};\n"
        f"constexpr float kOutputScale = {so:.9g}f;\n"
        f"constexpr int kOutputZeroPoint = {zo};\n")


def write_norm(mean, std, path):
    f = lambda a: ", ".join(f"{v:.8e}f" for v in a)
    features = ("acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, sqrt(ax*ax+ay*ay+az*az), sqrt(gx*gx+gy*gy+gz*gz)"
                if len(mean) == 8 else "acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z")
    path.write_text(f"// Per-channel z-score. Input channel order: {features}.\n"
                    "// For 8ch, append raw-unit magnitudes BEFORE normalization; then apply (x-mean)/std.\n"
                    f"constexpr int kNumChannels = {len(mean)};\n"
                    f"constexpr float kMean[{len(mean)}] = {{{f(mean)}}};\n"
                    f"constexpr float kStd[{len(std)}] = {{{f(std)}}};\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ds_tcn"); ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    Dd = D.build(); dev, test = D.dev_test_subjects(Dd)
    sel = pick_seed(a.model)
    print(f"deployed seed {sel['seed']} (val PR-AUC {sel['val_ap']:.4f}, val subjects {sel['val_subjects']})")
    f, q, mean, std, tr, val = qat_finetune(a.model, sel, Dd, dev)
    rng = np.random.default_rng(0)
    trm = D.select(Dd, tr)
    rep = D.apply_norm(Dd["X"][trm][rng.choice(trm.sum(), 300, replace=False)], mean, std)   # train subjects only
    tfl = convert(q, rep)
    ex = C.EXPORT; ex.mkdir(exist_ok=True)
    (ex / f"{a.model}_int8.tflite").write_bytes(tfl)
    write_header(tfl, ex / "model_data.h"); write_norm(mean, std, ex / "norm.h")
    write_quantization(tfl, ex / "quantization.h")
    ops = tflite_ops(tfl)
    (ex / "tflm_ops.txt").write_text("\n".join(ops))
    print("TFLM ops:", ops)
    b = M.budget(f)
    info = dict(model=a.model, seed=sel["seed"], params=b["params"], macs=b["macs"], int8_file_kb=len(tfl) / 1024)

    vm = D.select(Dd, val)
    Xv, yv = D.apply_norm(Dd["X"][vm], mean, std), Dd["y"][vm]
    pf = f.predict(Xv, verbose=0).ravel(); pi, _, _ = run_int8(tfl, Xv)
    info.update(val_ap_float=float(average_precision_score(yv, pf)), val_ap_int8=float(average_precision_score(yv, pi)))

    # Select threshold/FSM on this QAT model's validation probabilities (same quantized model used on device).
    # This does not touch test subjects and avoids carrying a float-model threshold into INT8 inference.
    threshold = E.pick_threshold(yv, pi)
    vm_idx = np.where(vm)[0]
    prm = E.tune_postproc(yv, pi, Dd["subj"][vm], Dd["blk"][vm], Dd["start"][vm],
                          threshold, lambda sid: D.walking_hours(Dd, {sid}), target_sens=0.7)
    info["threshold_int8_val"] = float(threshold)
    info["fsm_int8_val"] = prm
    (ex / "operating_point.h").write_text(
        "// Tune on validation only; apply to INT8 model output probability.\n"
        f"constexpr float kFogThreshold = {threshold:.9g}f;\n"
        f"constexpr int kRequireConsecutive = {prm['params']['k']};\n"
        f"constexpr float kEmaAlpha = {prm['params']['ema']:.9g}f;\n"
        f"constexpr float kThresholdOff = {prm['params']['thr_off']:.9g}f;\n"
        "constexpr int kOffConsecutive = 2;\n")

    # 10 golden vectors from the validation fold: int8 input / int8 output pairs for on-device checks
    gi = np.linspace(0, len(Xv) - 1, 10).astype(int)
    _, go, gin = run_int8(tfl, Xv[gi])
    np.savez(ex / "golden_vectors.npz", inputs_int8=gin, outputs_int8=go, val_index=gi)
    (ex / "golden_vectors.h").write_text(
        "constexpr int kGolden = 10;\nconst signed char kGoldenIn[10][%d] = {\n%s\n};\nconst signed char kGoldenOut[10] = {%s};\n"
        % (gin[0].size, ",\n".join("{" + ",".join(map(str, g.ravel())) + "}" for g in gin), ",".join(map(str, go))))

    if a.test:                                      # single, final look at the test subjects
        tm = D.select(Dd, test)
        Xt, yt = D.apply_norm(Dd["X"][tm], mean, std), Dd["y"][tm]
        pf = f.predict(Xt, verbose=0).ravel(); pi, _, _ = run_int8(tfl, Xt)
        info.update(test_ap_float=float(average_precision_score(yt, pf)), test_ap_int8=float(average_precision_score(yt, pi)),
                    test_float_int8_comparable=True)
        np.savez(C.OUT / f"test_pred_{a.model}.npz", p_float=pf, p_int8=pi, y=yt,
                 s=Dd["subj"][tm], blk=Dd["blk"][tm], start=Dd["start"][tm])
    (ex / f"{a.model}_export.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
