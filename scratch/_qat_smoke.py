"""Smoke test: tfmot QAT of Conv1D/DepthwiseConv1D -> full-integer TFLite."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

keras = tf.keras
L = keras.layers
quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
quantize_apply = tfmot.quantization.keras.quantize_apply
QuantizeConfig = tfmot.quantization.keras.QuantizeConfig


class Conv1DQuantizeConfig(QuantizeConfig):
    KERNEL_ATTR = "kernel"

    def get_weights_and_quantizers(self, layer):
        return [(getattr(layer, self.KERNEL_ATTR), tfmot.quantization.keras.quantizers.LastValueQuantizer(
            num_bits=8, per_axis=False, symmetric=True, narrow_range=True))]

    def get_activations_and_quantizers(self, layer):
        return [(layer.activation, tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
            num_bits=8, per_axis=False, symmetric=False, narrow_range=False))]

    def set_quantize_weights(self, layer, quantize_weights):
        setattr(layer, self.KERNEL_ATTR, quantize_weights[0])

    def set_quantize_activations(self, layer, quantize_activations):
        layer.activation = quantize_activations[0]

    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {}


class DWConv1DQuantizeConfig(Conv1DQuantizeConfig):
    KERNEL_ATTR = "depthwise_kernel"


def q(layer):
    cfg = DWConv1DQuantizeConfig() if isinstance(layer, L.DepthwiseConv1D) else Conv1DQuantizeConfig()
    return quantize_annotate_layer(layer, cfg)


inp = keras.Input((100, 6))
x = q(L.Conv1D(16, 5, padding="same", activation="relu"))(inp)
y = q(L.DepthwiseConv1D(3, padding="same", activation="relu"))(x)
y = q(L.Conv1D(16, 1, padding="same", activation="relu"))(y)
x = L.Add()([x, y])
x = L.GlobalAveragePooling1D()(x)
out = L.Dense(1, activation="sigmoid")(x)
m = keras.Model(inp, out)
with tfmot.quantization.keras.quantize_scope({"Conv1DQuantizeConfig": Conv1DQuantizeConfig, "DWConv1DQuantizeConfig": DWConv1DQuantizeConfig}):
    qm = quantize_apply(m)
qm.compile(optimizer="adam", loss="binary_crossentropy")
rng = np.random.default_rng(0)
X = rng.normal(size=(64, 100, 6)).astype("float32")
y = (rng.random(64) > 0.7).astype("float32")
qm.fit(X, y, epochs=1, verbose=0)

def rep():
    for i in range(32):
        yield [X[i:i + 1]]

converter = tf.lite.TFLiteConverter.from_keras_model(qm)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite = converter.convert()
print("bytes", len(tflite))
interp = tf.lite.Interpreter(model_content=tflite)
interp.allocate_tensors()
di = interp.get_input_details()[0]
do = interp.get_output_details()[0]
print("in", di["dtype"], di["quantization"])
print("out", do["dtype"], do["quantization"])
scale, zp = di["quantization"]
q_in = np.clip(np.round(X[0] / scale + zp), -128, 127).astype(np.int8)
interp.set_tensor(di["index"], q_in[None])
interp.invoke()
print("pred", interp.get_tensor(do["index"]))
print("OK")
