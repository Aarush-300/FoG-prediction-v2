"""Candidate architectures (all TFLM-safe, no BatchNorm so tfmot QAT can quantize every layer).

Every builder takes qat=False/True. With qat=True the Conv1D / DepthwiseConv1D layers are annotated with the
custom QuantizeConfigs from scratch/_qat_smoke.py (the pattern verified to convert to full-int8 TFLite);
call `apply_qat(model)` on the result. Conv layers fuse ReLU (activation="relu") so the converter fuses it too.

Candidates:
  ds_tcn      depthwise-separable residual TCN with dilations
  inception   multi-scale 1D CNN (parallel k=3/7/15 branches), no squeeze-excite
  hybrid      ds_tcn trunk + fixed band-energy branch (freeze-index style 0.5-3 Hz vs 3-8 Hz), concatenated before the head
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

import config as C
import data as D

keras = tf.keras
L = keras.layers
_qk = tfmot.quantization.keras


# ---------------------------------------------------------------- QAT plumbing
class Conv1DQuantizeConfig(_qk.QuantizeConfig):
    KERNEL_ATTR = "kernel"

    def get_weights_and_quantizers(self, layer):
        return [(getattr(layer, self.KERNEL_ATTR), _qk.quantizers.LastValueQuantizer(
            num_bits=8, per_axis=False, symmetric=True, narrow_range=True))]

    def get_activations_and_quantizers(self, layer):
        return [(layer.activation, _qk.quantizers.MovingAverageQuantizer(
            num_bits=8, per_axis=False, symmetric=False, narrow_range=False))]

    def set_quantize_weights(self, layer, qw):
        setattr(layer, self.KERNEL_ATTR, qw[0])

    def set_quantize_activations(self, layer, qa):
        layer.activation = qa[0]

    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {}


class DWConv1DQuantizeConfig(Conv1DQuantizeConfig):
    KERNEL_ATTR = "depthwise_kernel"


QAT_SCOPE = {"Conv1DQuantizeConfig": Conv1DQuantizeConfig, "DWConv1DQuantizeConfig": DWConv1DQuantizeConfig}


class ChannelSlice(L.Layer):
    """Pick one channel as (T, 1). A named layer so tfmot can round-trip the graph (a Lambda cannot)."""

    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = int(index)

    def call(self, inputs):
        i = self.index
        return inputs[..., i:i + 1]

    def get_config(self):
        cfg = super().get_config()
        cfg["index"] = self.index
        return cfg


def apply_qat(model):
    keras.utils.get_custom_objects()["ChannelSlice"] = ChannelSlice
    with _qk.quantize_scope({**QAT_SCOPE, "ChannelSlice": ChannelSlice}):
        return _qk.quantize_apply(model)


def _q(layer, qat):
    if not qat:
        return layer
    cfg = DWConv1DQuantizeConfig() if isinstance(layer, L.DepthwiseConv1D) else Conv1DQuantizeConfig()
    return _qk.quantize_annotate_layer(layer, cfg)


def conv(f, k, qat, dil=1, stride=1, act="relu", name=None):
    return _q(L.Conv1D(f, k, strides=stride, padding="same", dilation_rate=dil, activation=act, name=name), qat)


def dwconv(k, qat, dil=1, stride=1, name=None):
    return _q(L.DepthwiseConv1D(k, strides=stride, padding="same", dilation_rate=dil, activation="relu", name=name), qat)


# ---------------------------------------------------------------- blocks
def ds_block(x, f, k, dil, qat):
    """Residual depthwise-separable block: DW(k,dil)+ReLU -> PW(f)+ReLU -> Add."""
    y = dwconv(k, qat, dil)(x)
    y = conv(f, 1, qat)(y)
    if x.shape[-1] != f:
        x = conv(f, 1, qat, act=None)(x)   # linear projection shortcut
    return L.Add()([x, y])


def head(x, qat, drop=0.2):
    x = L.GlobalAveragePooling1D()(x)
    x = L.Dropout(drop)(x)
    return L.Dense(1, activation="sigmoid", name="p_fog")(x)


# ---------------------------------------------------------------- candidates
def ds_tcn(qat=False, width=32, n_blocks=5, k=5):
    inp = keras.Input((C.WIN, D.n_channels()))
    x = conv(width, k, qat, stride=2)(inp)                   # 100 -> 50
    for i in range(n_blocks):
        x = ds_block(x, width, k, 2 ** (i % 4), qat)          # dilations 1,2,4,8,1
        if i == 1:
            x = L.AveragePooling1D(2)(x)                      # 50 -> 25
    return keras.Model(inp, head(x, qat), name="ds_tcn")


def inception(qat=False, width=12, n_mod=3):
    inp = keras.Input((C.WIN, D.n_channels()))
    x = conv(2 * width, 5, qat, stride=2)(inp)
    for i in range(n_mod):
        bn = conv(width, 1, qat)(x)                           # bottleneck
        br = [conv(width, k, qat)(bn) for k in (3, 7, 15)]
        br.append(conv(width, 1, qat)(L.MaxPooling1D(3, strides=1, padding="same")(x)))
        y = L.Concatenate()(br)                               # 4*width channels
        x = conv(4 * width, 1, qat, act=None)(x) if i == 0 else x
        x = L.Add()([x, y]) if x.shape[-1] == y.shape[-1] else y
        if i < n_mod - 1:
            x = L.AveragePooling1D(2)(x)
    return keras.Model(inp, head(x, qat), name="inception")


def _band_bank(fs=C.FS, win=50):
    """Fixed (non-trainable) cos/sin filterbank for band energy. Bands: locomotor 0.5-3 Hz, freeze 3-8 Hz."""
    t = np.arange(win) / fs
    freqs = [0.75, 1.5, 2.25, 3.5, 5.0, 6.5]                 # 3 centres per band
    w = np.stack([f(2 * np.pi * fr * t) * np.hanning(win) for fr in freqs for f in (np.cos, np.sin)], 1)
    return (w / np.abs(w).sum(0, keepdims=True)).astype(np.float32)   # (win, 12)


def hybrid(qat=False, width=24, n_blocks=4, k=5):
    """Trunk on the raw window + freeze-index-style branch on |acc| (last-but-one channel).
    The band branch uses fixed filters; |.| is realised as ReLU(+) and ReLU(-) filter pairs (no Multiply op)."""
    inp = keras.Input((C.WIN, D.n_channels()))
    x = conv(width, k, qat, stride=2)(inp)
    for i in range(n_blocks):
        x = ds_block(x, width, k, 2 ** (i % 4), qat)
        if i == 1:
            x = L.AveragePooling1D(2)(x)
    trunk = L.GlobalAveragePooling1D()(x)

    # |acc| is channel 6 when FEATURES adds magnitudes; otherwise fall back to acc_z.
    mag_i = D.n_channels() - 2 if D.n_channels() == 8 else 2
    mag = ChannelSlice(mag_i, name="acc_mag")(inp)
    bank = _band_bank()
    pos = _q(L.Conv1D(bank.shape[1], bank.shape[0], strides=5, padding="same", activation="relu",
                      use_bias=False, trainable=True, name="band_pos"), qat)
    neg = _q(L.Conv1D(bank.shape[1], bank.shape[0], strides=5, padding="same", activation="relu",
                      use_bias=False, trainable=True, name="band_neg"), qat)
    bp, bn = pos(mag), neg(mag)
    be = L.Add()([L.GlobalAveragePooling1D()(bp), L.GlobalAveragePooling1D()(bn)])   # (12,) band energies
    z = L.Concatenate()([trunk, be])
    z = L.Dense(32, activation="relu")(z)
    z = L.Dropout(0.2)(z)
    model = keras.Model(inp, L.Dense(1, activation="sigmoid", name="p_fog")(z), name="hybrid")
    if not qat:   # under QAT the layers are wrapped; quantize_export.transfer() copies these weights from the float model
        model.get_layer("band_pos").set_weights([bank[:, None, :]])
        model.get_layer("band_neg").set_weights([-bank[:, None, :]])
    return model


BUILDERS = {"ds_tcn": ds_tcn, "inception": inception, "hybrid": hybrid}


# ---------------------------------------------------------------- budget accounting
def count_macs(model):
    macs = 0
    for l in model.layers:
        o = l.output_shape
        if isinstance(l, L.DepthwiseConv1D):
            # Each output element has one depthwise kernel dot product: output positions * channels * kernel length.
            macs += int(np.prod(o[1:])) * l.kernel_size[0]
        elif isinstance(l, L.Conv1D):
            macs += o[1] * l.kernel_size[0] * l.input_shape[-1] * l.filters
        elif isinstance(l, L.Dense):
            macs += l.input_shape[-1] * l.units
    return macs


def budget(model):
    p = model.count_params()
    macs = count_macs(model)
    return dict(name=model.name, params=p, macs=macs, int8_kb=p / 1024.0,
                ok=p <= C.MAX_PARAMS and macs <= C.MAX_MACS and p / 1024.0 <= C.MAX_INT8_KB)


if __name__ == "__main__":
    for n, b in BUILDERS.items():
        m = b()
        print(budget(m))
