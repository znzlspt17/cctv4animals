import sys

sys.path.insert(0, ".")

# 1) TensorFlow backend check
import tensorflow as tf

print("=== TensorFlow ===")
print("Version:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPU devices:", gpus)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print()

# 2) DeepFace model/detector info
from server.config import settings

print("=== DeepFace Config ===")
print("Model:", settings.DEEPFACE_MODEL)
print("Detector (register):", settings.DEEPFACE_DETECTOR)
print("Detector (realtime):", settings.DEEPFACE_DETECTOR_REALTIME)
print()

# 3) Check what backend DeepFace actually uses
import numpy as np
from deepface import DeepFace

dummy = np.zeros((224, 224, 3), dtype=np.uint8)
print("=== Running DeepFace.represent() ===")
r = DeepFace.represent(
    dummy,
    model_name=settings.DEEPFACE_MODEL,
    detector_backend="skip",
    enforce_detection=False,
)
emb = r[0]["embedding"]
print("Embedding length:", len(emb))
print()

# 4) Check tf device placement
print("=== TF Device Placement ===")
with tf.device("/CPU:0"):
    print("CPU device available: YES")
try:
    with tf.device("/GPU:0"):
        t = tf.constant([1.0])
        print("GPU device test: SUCCESS -", t.device)
except RuntimeError as e:
    print("GPU device test: FAIL -", e)

print()
print("=== CONCLUSION ===")
if gpus:
    print("DeepFace is using GPU")
else:
    print("DeepFace is using CPU (no GPU available to TensorFlow)")
