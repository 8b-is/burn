"""CPU regression for the Keras-to-ONNX export used by train.py."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import tensorflow as tf

from train import export_model


class ExportTest(unittest.TestCase):
    def test_keras_model_exports_and_runs(self):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(1,)),
            tf.keras.layers.Dense(2, activation="relu", kernel_initializer="ones"),
            tf.keras.layers.Dense(1, kernel_initializer="ones"),
        ])
        inputs = np.array([[0], [1], [2]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            export_model(model, path)
            exported = onnx.load(path)
            onnx.checker.check_model(exported)
            actual = ReferenceEvaluator(exported).run(None, {"input": inputs})[0]
            np.testing.assert_array_equal(actual, model(inputs, training=False).numpy())


if __name__ == "__main__":
    unittest.main()
