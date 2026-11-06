"""Inferensi LWDN untuk foto daun."""

from io import BytesIO
import json
import logging
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st


MAX_FILES = 20
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
IMAGE_SIZE = 224
PREVIEW_SIZE = 1000
# shortcut: skor softmax bisa tinggi pada foto asing; kalibrasi dengan foto berlabel di dalam dan di luar 10 kelas.
MIN_LEAF_SCORE = 0.85
MODEL_DIR = Path(__file__).with_name("model")
CLASS_NAMES = json.loads((MODEL_DIR / "classes.json").read_text(encoding="utf-8"))
PLANT_INFO = json.loads((MODEL_DIR / "plant_info.json").read_text(encoding="utf-8"))
if set(CLASS_NAMES) != set(PLANT_INFO):
    raise ValueError("Informasi khasiat tidak sesuai dengan kelas model.")


def read_image(data):
    """Validasi foto, lalu hasilkan input model dan pratinjau dari satu dekode."""
    if len(data) > MAX_BYTES:
        raise ValueError("Ukuran foto melebihi 10 MB. Pilih foto yang lebih kecil.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format not in {"PNG", "JPEG"}:
                    raise ValueError("Format foto harus PNG atau JPG.")
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError("Resolusi foto terlalu besar. Gunakan maksimal 20 megapiksel.")
                image.verify()
            with Image.open(BytesIO(data)) as image:
                ImageOps.exif_transpose(image, in_place=True)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                scale = min(1, PREVIEW_SIZE / max(image.size))
                preview = image if scale == 1 else image.resize(
                    tuple(max(1, round(side * scale)) for side in image.size), reducing_gap=2
                )
                preview_bytes = BytesIO()
                preview.save(preview_bytes, format="JPEG")

                image.thumbnail((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white")
                canvas.paste(image, ((IMAGE_SIZE - image.width) // 2, (IMAGE_SIZE - image.height) // 2))
                # Ekspor Kaggle mengharapkan RGB 0–255; normalisasi sudah ada di model.
                return np.asarray(canvas, dtype=np.float32)[None, ...], preview_bytes.getvalue()
    except (UnidentifiedImageError, OSError, SyntaxError) as error:
        raise ValueError("Foto tidak dapat dibaca. Coba unggah ulang atau pilih foto lain.") from error
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
        raise ValueError("Resolusi foto terlalu besar. Gunakan maksimal 20 megapiksel.") from error


@st.cache_resource(show_spinner=False)
def load_model():
    import tensorflow as tf

    # Paket ekspor FGD-6: 20261106T034140Z_1fda11bd.
    model = tf.keras.models.load_model(MODEL_DIR / "lwdn_v4.keras", compile=False)
    if model.input_shape != (None, IMAGE_SIZE, IMAGE_SIZE, 3) or model.output_shape != (None, len(CLASS_NAMES)):
        raise ValueError("Bentuk input atau output model tidak sesuai dengan aplikasi.")

    @tf.function(input_signature=[tf.TensorSpec((1, IMAGE_SIZE, IMAGE_SIZE, 3), tf.float32)], autograph=False)
    def forward(batch):
        return model(batch, training=False)

    return forward


def predict(batch, forward):
    """Berikan label hanya jika skor kelas tertinggi mencapai ambang."""
    start = perf_counter()
    probabilities = forward(batch).numpy()[0]
    elapsed_ms = (perf_counter() - start) * 1000
    if probabilities.shape != (len(CLASS_NAMES),) or not np.isfinite(probabilities).all():
        raise ValueError("Keluaran model tidak valid.")
    index = int(np.argmax(probabilities))
    score = float(probabilities[index])
    return CLASS_NAMES[index] if score >= MIN_LEAF_SCORE else None, score, elapsed_ms


def main():
    st.set_page_config(page_title="HerbLeaf AI", page_icon=":material/eco:", layout="wide")
    st.html("""
    <style>
    .stApp { background: #f6f8f3; }
    .stMainBlockContainer { max-width: 1100px; padding-top: 2.2rem; padding-bottom: 2rem; }
    h1 { font-size: clamp(2rem, 5vw, 2.8rem) !important; letter-spacing: -.05em; padding: 0 0 .5rem !important; }
    h2 { font-size: 1.25rem !important; letter-spacing: -.025em; margin: 0 !important; padding: 0 0 .3rem !important; }
    h3 { font-size: 1.05rem !important; }
    p, li { line-height: 1.6; }
    .brand { display: flex; align-items: center; gap: .65rem; color: #315c36; font-size: .9rem; font-weight: 650; margin-bottom: 1rem; }
    .brand small { margin-left: auto; color: #687765; font-weight: 400; }
    .intro { color: #60705e; max-width: 570px; margin: 0 0 .7rem; font-size: 1rem; }
    .st-key-upload, .st-key-results { background: #fff; border: 1px solid #dce5d5; border-radius: 16px; padding: 1.5rem; gap: .75rem; }
    .st-key-upload { margin-bottom: .7rem; }
    [data-testid="stFileUploaderDropzone"] { background: #f7faf4; border: 1px dashed #a7ba98; border-radius: 10px; }
    [data-testid="stFileUploaderDropzone"] button { background: white; }
    [data-testid="stBaseButton-primary"] { min-height: 46px; font-weight: 600; }
    [data-testid="stRadio"] { margin-bottom: .25rem; }
    [data-testid="stCaptionContainer"] p { color: #64715e; }
    .empty { text-align: center; padding: 1.3rem .5rem 1.5rem; }
    .empty strong { display: block; color: #4e6246; font-weight: 600; margin-bottom: .35rem; }
    .empty p { color: #64715e; font-size: .9rem; margin: 0; }
    .footer { color: #64715e; font-size: .75rem; text-align: center; margin-top: 1.2rem; }
    @media (max-width: 600px) {
        .stMainBlockContainer { padding: 1.2rem 1rem; }
        .st-key-upload, .st-key-results { padding: 1.1rem; }
        .brand { margin-bottom: .7rem; }
        .brand small { font-size: .7rem; }
    }
    </style>
    """)

    st.html('<div class="brand">HerbLeaf AI <small>Tanaman obat Indonesia</small></div>')
    st.title("HerbLeaf AI")
    st.html('<p class="intro">Unggah foto satu daun untuk melihat prediksi LWDN dan informasi tanamannya.</p>')
    st.caption("Jenis daun yang didukung: " + ", ".join(CLASS_NAMES) + ".")

    with st.container(key="upload"):
        st.html('<h2>Masukan foto daun</h2>')
        mode = st.radio("Cara unggah", ["Foto", "Folder"], horizontal=True, label_visibility="collapsed")
        if mode == "Foto":
            photo = st.file_uploader("Unggah foto", type=["png", "jpg", "jpeg"], max_upload_size=10, key="photo")
            files = [photo] if photo else []
        else:
            files = st.file_uploader("Unggah folder", type=["png", "jpg", "jpeg"], accept_multiple_files="directory", max_upload_size=10, key="folder")
        st.caption("PNG atau JPG · Maks. 10 MB per foto · Maks. 20 foto per proses")

        # ponytail: proses dibatasi 20 foto; gunakan paginasi bila kebutuhan batch bertambah.
        too_many = len(files) > MAX_FILES
        if too_many:
            st.warning("Folder berisi lebih dari 20 foto. Pilih folder yang lebih kecil.")
        signature = (mode, tuple((file.file_id, file.name, file.size) for file in files))
        submitted = st.button("Prediksi", type="primary", icon=":material/eco:", width="stretch", disabled=not files or too_many)
        if submitted or st.session_state.get("processed") != signature:
            st.session_state.pop("processed", None)
            st.session_state.pop("predictions", None)
        if submitted:
            with st.status("Memproses foto...", expanded=True) as status:
                try:
                    forward = None
                    predictions = {}
                    for i, file in enumerate(files, start=1):
                        status.update(label=f"Memproses foto {i} dari {len(files)}...")
                        try:
                            batch, preview = read_image(file.getvalue())
                        except ValueError as error:
                            predictions[file.file_id] = str(error)
                            continue
                        if forward is None:
                            forward = load_model()
                        predictions[file.file_id] = (preview, *predict(batch, forward))
                    st.session_state["predictions"] = predictions
                    st.session_state["processed"] = signature
                    status.update(label="Prediksi selesai", state="complete", expanded=False)
                except Exception:
                    logging.exception("Model inference failed")
                    status.update(label="Prediksi gagal", state="error", expanded=True)
                    st.error("Model tidak dapat dijalankan. Periksa file model dan dependensi aplikasi.")

    with st.container(key="results"):
        st.html('<h2>Hasil pengenalan</h2>')
        if not files or too_many or st.session_state.get("processed") != signature:
            st.html('<div class="empty"><strong>Hasil akan muncul di sini</strong><p>Pilih foto, lalu tekan tombol Prediksi.</p></div>')
        else:
            for file in files:
                prediction = st.session_state["predictions"][file.file_id]
                st.subheader(Path(file.name).name)
                if isinstance(prediction, str):
                    st.error(prediction)
                    continue
                preview, label, score, elapsed_ms = prediction
                image_column, result_column = st.columns([2, 3], gap="large")
                with image_column:
                    st.image(BytesIO(preview), output_format="JPEG", width="stretch")
                with result_column:
                    if label is None:
                        st.warning("Foto tidak dikenali. Foto mungkin di luar 10 jenis daun yang didukung, atau daunnya kurang jelas.")
                        st.caption("Coba unggah foto satu daun yang tajam dengan latar polos.")
                    else:
                        st.markdown(f"**Nama daun:** {label}")
                        st.markdown(f"**Nama latin:** *{PLANT_INFO[label]['latin']}*")
                        st.markdown("**Khasiat yang diteliti:**")
                        st.markdown("\n".join(f"- {benefit}" for benefit in PLANT_INFO[label]["benefits"]))
                        st.caption("Ringkasan penelitian dari tim; bagian tanaman dan tingkat bukti berbeda. Bukan saran medis.")
                    st.caption(f"Skor model: {score:.1%} · Waktu inferensi: {elapsed_ms:.1f} ms")
                st.divider()

    st.html('<p class="footer">FGD Kelompok 6 · Pengenalan daun tanaman obat</p>')


if __name__ == "__main__":
    main()
