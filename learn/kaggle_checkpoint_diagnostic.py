# Diagnostik saja: tidak menulis artefak dan tidak menjalankan fit.
import json
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf

assert "datasets" in globals(), "Jalankan sel TRANSFORMASI Phase 3 dahulu."
probe_run = fgd6_staged / "runs/LWDN/20261028T151525_228962Z"
probe_config = json.loads((probe_run / "config.json").read_text())
print("Lingkungan asal:", probe_config["tensorflow_version"], probe_config["keras_version"], probe_config["gpu_name"], probe_config["precision_policy"])
print("Lingkungan Kaggle:", tf.__version__, tf.keras.__version__, [tf.config.experimental.get_device_details(g).get("device_name") for g in tf.config.list_physical_devices("GPU")])
assert tf.config.list_physical_devices("GPU"), "GPU tidak tersedia."
assert probe_config["precision_policy"] == "float32", "Tinjau precision sebelum melanjutkan."
tf.keras.mixed_precision.set_global_policy("float32")
with warnings.catch_warnings(record=True) as probe_warnings:
    warnings.simplefilter("always")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probe_model = tf.keras.models.load_model(probe_run / "finetune_best.keras")
for warning in probe_warnings:
    print("Peringatan load:", warning.message)
assert not probe_warnings, "Tinjau peringatan pemulihan sebelum melanjutkan."
assert probe_model.optimizer is not None and probe_model.count_params() == 1644938
probe_val = datasets["DenseNet121"]["val"]
probe_log = pd.read_csv(probe_run / "finetune.csv")
probe_best = probe_log.loc[probe_log["val_loss"].idxmin()]
with tf.device("/GPU:0"):
    probe_images, _ = next(iter(probe_val))
    probe_output = probe_model(probe_images, training=False)
print("Forward pass:", probe_output.device)
assert "GPU:0" in probe_output.device.upper(), "Forward pass tidak berjalan di GPU."
probe_metrics = probe_model.evaluate(probe_val, return_dict=True, verbose=0)
print("Checkpoint terbaik:", probe_metrics, "| log:", dict(probe_best[["val_loss", "val_accuracy"]]))
assert np.isclose(probe_metrics["loss"], probe_best["val_loss"], rtol=1e-3, atol=1e-4)
assert np.isclose(probe_metrics["accuracy"], probe_best["val_accuracy"], atol=1e-5)
backup_dir = probe_run / "finetune_resume_backup"
backup_meta = json.loads((backup_dir / "training_metadata.json").read_text())
with warnings.catch_warnings(record=True) as probe_warnings:
    warnings.simplefilter("always")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probe_model.load_weights(backup_dir / "latest.weights.h5")
for warning in probe_warnings:
    print("Peringatan backup:", warning.message)
assert not probe_warnings, "State backup perlu ditinjau."
expected_steps = int(backup_meta["epoch"]) * ((N_IMAGES["train"] + BATCH - 1) // BATCH)
print("Backup epoch:", backup_meta["epoch"], "| optimizer iterations:", int(probe_model.optimizer.iterations.numpy()), "| harapan:", expected_steps)
assert int(probe_model.optimizer.iterations.numpy()) == expected_steps
backup_log = pd.read_csv(probe_run / "finetune_resume.csv")
backup_row = backup_log.loc[backup_log["epoch"] == int(backup_meta["epoch"]) - 1].iloc[-1]
backup_metrics = probe_model.evaluate(probe_val, return_dict=True, verbose=0)
print("Backup:", backup_metrics, "| log:", dict(backup_row[["val_loss", "val_accuracy"]]))
assert np.isclose(backup_metrics["loss"], backup_row["val_loss"], rtol=1e-3, atol=1e-4)
assert np.isclose(backup_metrics["accuracy"], backup_row["val_accuracy"], atol=1e-5)
print("Diagnostik lulus; perbedaan versi dan perangkat tetap harus dicatat sebelum resume.")
