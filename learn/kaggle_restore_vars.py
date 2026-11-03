# Memulihkan variabel handoff yang sudah diperiksa tanpa mengubah data Phase 3.
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

WORK_DIR = Path("/kaggle/working")
fgd6_staged = WORK_DIR / "FGD6_staged"
handoff_root = fgd6_staged / "handoff/20261028T104133_556236Z"
mapped_table = fgd6_staged / "df_final_kaggle.csv"
if not mapped_table.is_file():
    raise RuntimeError("Jalankan staging dan pemeriksaan manifest terlebih dahulu.")
PREP_DIR = handoff_root / "images"
metadata = json.loads((handoff_root / "metadata.json").read_text(encoding="utf-8"))
df_final = pd.read_csv(mapped_table)
train_df = pd.read_csv(handoff_root / "train_df.csv")
CLASS_NAMES = metadata["classes"]
IMG_SIZE, BATCH, SEED = metadata["image_size"], metadata["batch"], metadata["seed"]
FILE_ID, SPLIT = metadata["dataset_file_id"], tuple(metadata["split"])
checks = metadata["checks"]

if not checks or not all(row[1] is True for row in checks):
    raise ValueError("Pemeriksaan QC pada handoff belum lulus.")
if len(CLASS_NAMES) != 10 or set(train_df["label"]) != set(CLASS_NAMES):
    raise ValueError("Kelas pada data train tidak cocok dengan metadata handoff.")
if not all((PREP_DIR / split / cls).is_dir() for split in ("train", "val", "test") for cls in CLASS_NAMES):
    raise ValueError("Folder citra tidak sesuai dengan kelas dan split handoff.")

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
augmenter = tf.keras.models.model_from_json(metadata["augmenter"])
print("Variabel handoff siap. Jalankan sel TRANSFORMASI Phase 3 asli.")
print("Arsip asli dan identitas handoff untuk recovery model belum diverifikasi di sel ini.")
