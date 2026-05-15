import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("Importing TensorFlow...")
import tensorflow as tf
print(f"TensorFlow {tf.__version__} OK")
