"""Jalankan: python self_check.py (tanpa framework tes tambahan)."""

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

from app import CLASS_NAMES, MAX_BYTES, PLANT_INFO, load_models, predict, read_image


def image_bytes(image_format="PNG", size=(40, 20), exif=None):
    buffer = BytesIO()
    kwargs = {"exif": exif} if exif else {}
    Image.new("RGB", size, "green").save(buffer, format=image_format, **kwargs)
    return buffer.getvalue()


def rejects(data, message):
    try:
        read_image(data)
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError("Input seharusnya ditolak")


assert read_image(image_bytes()).size == (40, 20)
assert read_image(image_bytes("JPEG")).mode == "RGB"
assert read_image(image_bytes(size=(2000, 1000))).size == (1000, 500)
exif = Image.Exif()
exif[274] = 6
assert read_image(image_bytes("JPEG", exif=exif)).size == (20, 40)
rejects(b"ini bukan gambar", "tidak dapat dibaca")
rejects(image_bytes()[:30], "tidak dapat dibaca")
rejects(image_bytes("GIF"), "PNG atau JPG")
rejects(b"0" * (MAX_BYTES + 1), "10 MB")
rejects(image_bytes(size=(5000, 4001)), "20 megapiksel")
assert set(CLASS_NAMES) == set(PLANT_INFO)
assert all(info["latin"] and len(info["benefits"]) == 3 for info in PLANT_INFO.values())

models = load_models()
assert tuple(models) == ("LWDN", "DenseNet-121")
predictions = predict(image_bytes(), models)
for label, confidence, elapsed_ms in predictions.values():
    assert label in CLASS_NAMES and 0 <= confidence <= 1 and elapsed_ms >= 0

app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run(timeout=120)
assert not app.exception, app.exception
assert app.button[0].disabled
assert app.radio[0].value == "Satu foto"
app.radio[0].set_value("Satu folder").run()
assert not app.exception, app.exception
assert app.button[0].disabled
assert app.get("file_uploader")[0].proto.max_upload_size_mb == 10

photo_data = image_bytes()
photo = SimpleNamespace(file_id="test-photo", name='foto"<uji>.png', size=len(photo_data), getvalue=lambda: photo_data)
with patch("streamlit.file_uploader", return_value=photo):
    app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run(timeout=120)
    app.button[0].click().run(timeout=120)
    assert not app.exception, app.exception
    assert app.selectbox[0].value == "Belum diisi"
    assert any("Keyakinan:" in item.value for item in app.caption)
    assert len(app.get("expander")) == 2
    assert any(PLANT_INFO[predictions["LWDN"][0]]["benefits"][0] in item.value for item in app.get("markdown"))
    timing_captions = [item.value for item in app.caption if "Waktu inferensi:" in item.value]
    app.selectbox[0].set_value(predictions["LWDN"][0]).run(timeout=120)
    assert not app.exception, app.exception
    assert [item.value for item in app.caption if "Waktu inferensi:" in item.value] == timing_captions
    assert any("Benar: 100.0% · Salah: 0.0%" in item.value for item in app.sidebar.get("markdown"))
    app.selectbox[0].set_value("Di luar 10 kelas").run(timeout=120)
    assert not app.exception, app.exception
    assert sum("Benar: 0.0% · Salah: 100.0%" in item.value for item in app.sidebar.get("markdown")) == 2
print("LULUS: validasi foto, kedua model, prediksi nyata, dan penilaian label di sidebar.")
