"""Verify the exported INT8 TFLite artifact against generated golden vectors.

Run from this directory:
  python verify_export.py
This is an export-integrity check only; it does not load or evaluate the held-out test set.
"""
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

import config as C


def verify(model_path, golden_path, atol=1):
    model_path, golden_path = Path(model_path), Path(golden_path)
    assert model_path.is_file(), f"missing TFLite model: {model_path}"
    assert golden_path.is_file(), f"missing golden vectors: {golden_path}"
    golden = np.load(golden_path)
    inputs = golden["inputs_int8"]
    expected = golden["outputs_int8"].reshape(-1)
    assert inputs.dtype == np.int8, inputs.dtype
    expected = expected.astype(np.int16)
    assert len(inputs) == len(expected) == 10, (inputs.shape, expected.shape)

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    assert inp["dtype"] == np.int8 and out["dtype"] == np.int8
    assert tuple(inp["shape"]) == (1, C.WIN, 8), inp["shape"]
    assert tuple(out["shape"]) == (1, 1), out["shape"]
    assert inputs.shape == (10, C.WIN, 8), inputs.shape

    actual = []
    for x in inputs:
        interpreter.set_tensor(inp["index"], x[None, ...])
        interpreter.invoke()
        actual.append(interpreter.get_tensor(out["index"]).reshape(-1)[0])
    actual = np.asarray(actual, dtype=np.int8)
    delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    assert np.all(delta <= atol), f"golden mismatch; max abs int8 delta={delta.max()}"
    print(f"PASS: {len(inputs)} golden vectors; max int8 delta={delta.max()}")
    print(f"input: {tuple(inp['shape'])} {inp['dtype'].__name__} scale={inp['quantization'][0]:.9g} "
          f"zero_point={inp['quantization'][1]}")
    print(f"output: {tuple(out['shape'])} {out['dtype'].__name__} scale={out['quantization'][0]:.9g} "
          f"zero_point={out['quantization'][1]}")
    print(f"operators: {', '.join(sorted({op['op_name'] for op in interpreter._get_ops_details()}))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(C.EXPORT / "ds_tcn_int8.tflite"))
    parser.add_argument("--golden", default=str(C.EXPORT / "golden_vectors.npz"))
    parser.add_argument("--atol", type=int, default=1)
    args = parser.parse_args()
    verify(args.model, args.golden, args.atol)
